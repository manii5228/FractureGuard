from __future__ import annotations

import hmac
import json
import math
import re
from datetime import timedelta
from io import BytesIO
from time import perf_counter

from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.db.models import Max, Q
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST

from .imaging import VIEWS, PreparedImage, prepare_upload
from .models import Asset, Case, Controls, Job, Notice, Report, Revision, User
from .reports import make_pdf
from .security import (
    audit,
    canonical,
    consume_otp,
    digest,
    over_limit,
    read_payload,
    role_required,
    seal,
    unseal,
)

S = Case.Status
R = User.Role
ACTIVE_QUEUE = (S.REVIEW, S.REVIEWED, S.SIGNED, S.FAILED)
STATUS_LABELS = {S.REVIEWED: "Awaiting reviewer approval", S.SIGNED: "Awaiting release"}


def touch(case, *fields):
    case.save(update_fields=[*fields, "updated"])


def audit_access(action, actor, case, **fields):
    now = timezone.now()
    Case.objects.filter(pk=case.pk, updated__lt=now - timedelta(minutes=5)).exclude(
        status=S.DELETED,
    ).update(updated=now)
    audit(action, actor, case, **fields)


def notify(user, case, event, key):
    Notice.objects.get_or_create(user=user, case=case, event_key=key, defaults={"event": event})


def owner_case(request, case_id, *, lock=False):
    query = Case.objects.select_for_update() if lock else Case.objects.all()
    return get_object_or_404(
        query.exclude(status=S.DELETED),
        pk=case_id,
        owner=request.user,
        owner__is_active=True,
        owner__role=R.DEMO,
    )


def reviewer_case(request, case_id, *, lock=False):
    query = Case.objects.select_for_update() if lock else Case.objects.all()
    return get_object_or_404(
        query.exclude(status=S.DELETED),
        pk=case_id,
        reviewer=request.user,
        reviewer__is_active=True,
        reviewer__role=R.REVIEWER,
    )


def current_revision(case):
    return get_object_or_404(Revision, case=case, number=case.revision)


def matching_review(request, case):
    return request.POST.get("study_revision") == str(case.revision) and request.POST.get(
        "edit_version"
    ) == str(case.edit_version)


def account_notifications(user):
    query = Notice.objects.filter(user=user).exclude(case__status=S.DELETED)
    if user.role == R.DEMO:
        query = query.filter(case__owner=user)
    elif user.role == R.REVIEWER:
        query = query.filter(Q(case__reviewer=user) | Q(case__reviewer__isnull=True, event="ready"))
    else:
        return []
    return query.order_by("-created")[:12]


@role_required(R.DEMO, R.REVIEWER, R.ADMIN)
def home(request):
    if request.user.role == R.REVIEWER:
        return redirect("queue")
    if request.user.role == R.ADMIN:
        return redirect("operations")
    items = (
        Case.objects.filter(owner=request.user)
        .exclude(status=S.DELETED)
        .annotate(
            last_submitted=Max("revisions__submitted"),
        )
        .order_by("-created")[:100]
    )
    return render(
        request, "cases.html", {"cases": items, "notices": account_notifications(request.user)}
    )


@role_required(R.DEMO)
@require_POST
@transaction.atomic
def new_case(request):
    if over_limit("draft", request.user.pk, 20, 3600):
        messages.error(request, "The hourly case limit was reached. Try again later.")
        return redirect("home")
    case = Case.objects.create(owner=request.user)
    audit("case_created", request.user, case)
    return redirect("case_detail", case_id=case.pk)


def owner_label(case):
    return STATUS_LABELS.get(case.status, case.get_status_display())


@role_required(R.DEMO)
@require_http_methods(["GET", "POST"])
def case_detail(request, case_id):
    case = owner_case(request, case_id)
    context = {
        "case": case,
        "label": owner_label(case),
        "controls": Controls.objects.get_or_create(pk=1)[0],
    }
    if case.return_note:
        context["return_note"] = unseal(case.return_note, f"return:{case.pk}").decode()
    context["upload_open"] = case.status in (S.DRAFT, S.RETURNED)
    if request.method == "GET":
        audit_access("case_access", request.user, case)
        return render(request, "case.html", context)
    if over_limit("submission", request.user.pk, 20, 3600):
        context["error"] = "The hourly submission limit was reached. Try again later."
        return render(request, "case.html", context, status=429)
    start = perf_counter()
    images, errors = {}, {}
    prior_assets = (
        {asset.view: asset for asset in current_revision(case).assets.all()}
        if case.status == S.RETURNED
        else {}
    )
    if request.POST.get("side") not in {"L", "R"} or request.POST.get("attest") != "yes":
        context["error"] = "Select a side and confirm that the study is de-identified."
    for view in VIEWS:
        try:
            files = request.FILES.getlist(view)
            if not files and view in prior_assets:
                asset = prior_assets[view]
                images[view] = PreparedImage(
                    unseal(asset.png, f"image:{asset.pk}"),
                    unseal(asset.preview, f"preview:{asset.pk}"),
                    asset.width,
                    asset.height,
                    asset.fingerprint,
                )
                continue
            if len(files) != 1:
                raise ValueError("Choose exactly one PNG for this view.")
            images[view] = prepare_upload(files[0])
        except ValueError as problem:
            errors[view] = str(problem)
    if len(images) == 2 and images["frontal"].fingerprint == images["lateral"].fingerprint:
        errors["lateral"] = "Use a different image for the lateral view."
    if errors or context.get("error"):
        context["errors"] = errors
        audit("upload", request.user, case, result="invalid")
        return render(request, "case.html", context, status=400)
    side = request.POST["side"]
    fingerprint = digest(canonical([side, *(images[view].fingerprint for view in VIEWS)]))
    with transaction.atomic():
        case = owner_case(request, case_id, lock=True)
        previous = case.revisions.filter(number=case.revision).first()
        if case.status not in (S.DRAFT, S.RETURNED):
            if previous and previous.fingerprint == fingerprint:
                return redirect("case_detail", case_id=case.pk)
            raise PermissionDenied
        if prior_assets and any(
            asset.revision_id != previous.pk for asset in prior_assets.values()
        ):
            return HttpResponse("Reload the case before resubmitting.", status=409)
        if previous and previous.fingerprint == fingerprint:
            context["error"] = "Replace at least one image before resubmitting."
            return render(request, "case.html", context, status=400)
        if not Controls.objects.get(pk=1).model_enabled:
            context["error"] = (
                "New processing is paused. Your draft is available to try again later."
            )
            return render(request, "case.html", context, status=409)
        revision = Revision.objects.create(
            case=case,
            number=case.revision + 1,
            side=side,
            fingerprint=fingerprint,
            validation_ms=round((perf_counter() - start) * 1000),
        )
        for view, image in images.items():
            asset = Asset(
                revision=revision,
                view=view,
                width=image.width,
                height=image.height,
                fingerprint=image.fingerprint,
            )
            asset.png = seal(image.png, f"image:{asset.pk}")
            asset.preview = seal(image.preview, f"preview:{asset.pk}")
            asset.save(force_insert=True)
        Job.objects.create(revision=revision)
        case.status, case.side, case.revision = S.QUEUED, side, revision.number
        case.edit_version, case.priority, case.return_note = 0, False, b""
        touch(case, "status", "side", "revision", "edit_version", "priority", "return_note")
        audit("upload", request.user, case)
    return redirect("case_detail", case_id=case.pk)


@role_required(R.DEMO)
def case_status(request, case_id):
    case = owner_case(request, case_id)
    return JsonResponse({"status": case.status, "label": owner_label(case)})


@role_required(R.DEMO)
@require_POST
@transaction.atomic
def request_deletion(request, case_id):
    case = owner_case(request, case_id, lock=True)
    if not case.deletion_requested:
        case.deletion_requested = True
        touch(case, "deletion_requested")
        audit("deletion_requested", request.user, case)
    return redirect("case_detail", case_id=case.pk)


@role_required(R.REVIEWER)
def queue(request):
    items = Case.objects.filter(status__in=ACTIVE_QUEUE).select_related("reviewer")
    state, assignment = request.GET.get("status", ""), request.GET.get("assignment", "")
    history = (S.RELEASED, S.WITHDRAWN, S.RETURNED)
    if state in history:
        items = Case.objects.filter(reviewer=request.user, status=state).select_related("reviewer")
    elif state in ACTIVE_QUEUE:
        items = items.filter(status=state)
    if assignment == "mine":
        items = items.filter(reviewer=request.user)
    elif assignment == "unassigned":
        items = items.filter(reviewer__isnull=True)
    items = items.annotate(last_submitted=Max("revisions__submitted"))
    return render(
        request,
        "queue.html",
        {
            "cases": items.order_by("-priority", "last_submitted")[:100],
            "chosen_status": state,
            "assignment": assignment,
            "statuses": [(item.value, item.label) for item in (*ACTIVE_QUEUE, *history)],
            "notices": account_notifications(request.user),
        },
    )


@role_required(R.REVIEWER)
@require_POST
@transaction.atomic
def claim(request, case_id):
    case = get_object_or_404(Case.objects.select_for_update(), pk=case_id, status__in=ACTIVE_QUEUE)
    if case.reviewer_id not in (None, request.user.pk):
        return HttpResponse("This case already has a reviewer.", status=409)
    if case.reviewer_id is None:
        case.reviewer = request.user
        touch(case, "reviewer")
        audit("claimed", request.user, case)
        notify(request.user, case, "assigned", f"claim:{case.revision}:{request.user.pk}")
    return redirect("review", case_id=case.pk)


def review_data(revision):
    prediction = read_payload(revision.prediction, f"prediction:{revision.pk}", {})
    initial = {
        "boxes": sorted(prediction.get("boxes", []), key=lambda box: box["score"], reverse=True)[
            :120
        ],
        "findings": "",
        "notes": "",
        "limitations": "",
        "impression": "",
    }
    return prediction, read_payload(revision.findings, f"findings:{revision.pk}", initial)


@role_required(R.REVIEWER)
def review(request, case_id):
    case = reviewer_case(request, case_id)
    revision = current_revision(case)
    prediction, findings = review_data(revision)
    audit_access("case_access", request.user, case)
    signed = revision.reports.order_by("-number").first()
    context = {
        "case": case,
        "revision": revision,
        "prediction": prediction,
        "findings": findings,
        "views": list(revision.assets.order_by("view")),
        "signed": signed,
        "editable": case.status in (S.REVIEW, S.REVIEWED, S.WITHDRAWN),
        "needs_otp": bool(request.user.otp_secret),
    }
    return render(request, "review.html", context)


def validated_boxes(source, assets):
    if not isinstance(source, list) or len(source) > 120:
        raise ValueError("Provide a valid annotation list (up to 120 boxes).")
    boxes, seen = [], set()
    for box in source:
        if not isinstance(box, dict):
            raise ValueError("Each annotation must contain a view and box coordinates.")
        key, view, status = box.get("id"), box.get("view"), box.get("status")
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"model-(?:frontal|lateral)-[0-9]{1,3}|manual-[a-f0-9-]{36}", key)
            or key in seen
            or view not in VIEWS
            or status not in ("preliminary", "approved", "rejected")
        ):
            raise ValueError("An annotation has an invalid identity, view, or status.")
        coordinates = box.get("coords")
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != 4
            or not all(
                type(value) in (int, float) and math.isfinite(value) for value in coordinates
            )
        ):
            raise ValueError("Each box requires four finite coordinates.")
        x1, y1, x2, y2 = coordinates
        width, height = assets[view]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Keep each annotation within its radiograph.")
        seen.add(key)
        boxes.append({"id": key, "view": view, "status": status, "coords": coordinates})
    return boxes


@role_required(R.REVIEWER)
@require_POST
@transaction.atomic
def save_review(request, case_id):
    case = reviewer_case(request, case_id, lock=True)
    if case.status not in (S.REVIEW, S.REVIEWED, S.WITHDRAWN):
        raise PermissionDenied
    if not matching_review(request, case):
        return HttpResponse(
            "A newer revision is available. Reload the case before editing.", status=409
        )
    revision = current_revision(case)
    _, previous = review_data(revision)
    assets = {asset.view: (asset.width, asset.height) for asset in revision.assets.all()}
    try:
        boxes = validated_boxes(json.loads(request.POST.get("boxes", "[]")), assets)
        fields = {
            name: request.POST.get(name, "").strip()
            for name in ("findings", "limitations", "notes", "impression")
        }
        if any(len(value) > 2000 for value in fields.values()):
            raise ValueError("Keep each report field within 2,000 characters.")
        if not fields["findings"] or not fields["impression"]:
            raise ValueError("Reviewer findings and a final impression are required.")
        if request.POST.get("verify_views") != "yes":
            raise ValueError("Confirm the view labels and side before saving findings.")
    except (ValueError, TypeError) as problem:
        messages.error(request, str(problem))
        return redirect("review", case_id=case.pk)
    revision.findings = seal(canonical({**fields, "boxes": boxes}), f"findings:{revision.pk}")
    revision.save(update_fields=["findings"])
    old = {box["id"]: box for box in previous["boxes"]}
    new = {box["id"]: box for box in boxes}
    for key in sorted(old.keys() | new.keys()):
        if old.get(key) != new.get(key):
            audit(
                "annotation_changed",
                request.user,
                case,
                detail={
                    "box": key,
                    "change": new.get(key, {}).get("status", "removed"),
                },
            )
    case.status, case.edit_version = S.REVIEWED, case.edit_version + 1
    touch(case, "status", "edit_version")
    audit("review_saved", request.user, case)
    return redirect("report_preview", case_id=case.pk)


@role_required(R.REVIEWER)
@require_POST
@transaction.atomic
def return_case(request, case_id):
    case = reviewer_case(request, case_id, lock=True)
    if case.status not in (S.REVIEW, S.REVIEWED, S.FAILED):
        raise PermissionDenied
    reason = request.POST.get("reason", "").strip()
    if not 1 <= len(reason) <= 500:
        messages.error(request, "Enter a short replacement reason (up to 500 characters).")
        return redirect("review", case_id=case.pk)
    case.status, case.return_note = S.RETURNED, seal(reason.encode(), f"return:{case.pk}")
    touch(case, "status", "return_note")
    notify(case.owner, case, "returned", f"returned:{case.revision}")
    audit("returned", request.user, case)
    return redirect("queue")


@role_required(R.REVIEWER)
@require_POST
@transaction.atomic
def retry_case(request, case_id):
    case = reviewer_case(request, case_id, lock=True)
    job = current_revision(case).job
    if case.status != S.FAILED or job.state != "failed" or job.attempt >= 3:
        raise PermissionDenied
    job.state, job.queued = "queued", timezone.now()
    job.save(update_fields=["state", "queued"])
    case.status = S.QUEUED
    touch(case, "status")
    audit("retry_queued", request.user, case)
    return redirect("queue")


@role_required(R.REVIEWER)
def asset_preview(request, asset_id):
    asset = get_object_or_404(Asset.objects.select_related("revision__case"), pk=asset_id)
    case = asset.revision.case
    if (
        case.reviewer_id != request.user.pk
        or case.status == S.DELETED
        or case.revision != asset.revision.number
    ):
        raise Http404
    audit_access("image_access", request.user, case, version=asset.revision.number)
    return HttpResponse(unseal(asset.preview, f"preview:{asset.pk}"), content_type="image/png")


def report_context(case, revision, findings, *, signer, timestamp, number):
    return {
        **{key: findings[key] for key in ("findings", "limitations", "notes", "impression")},
        "boxes": [box for box in findings["boxes"] if box["status"] == "approved"],
        "reference": case.reference,
        "revision": revision.number,
        "version": number,
        "side": "Left" if revision.side == "L" else "Right",
        "signer": signer.username,
        "signer_id": signer.pk,
        "signed_at": timestamp,
    }


def signed_snapshot(report):
    data = unseal(report.snapshot, f"report:{report.revision_id}:{report.number}")
    if not hmac.compare_digest(digest(data), report.digest):
        raise Http404
    return json.loads(data)


@role_required(R.REVIEWER)
def report_preview(request, case_id, *, pdf=False):
    case = reviewer_case(request, case_id)
    revision = current_revision(case)
    latest = revision.reports.order_by("-number").first()
    if case.status == S.REVIEWED:
        snapshot = report_context(
            case,
            revision,
            review_data(revision)[1],
            signer=request.user,
            timestamp="Recorded when signed",
            number=1 if not latest else latest.number + 1,
        )
    elif case.status in (S.SIGNED, S.RELEASED, S.WITHDRAWN) and latest:
        snapshot = signed_snapshot(latest)
    else:
        raise Http404
    audit_access("report_preview", request.user, case)
    if pdf:
        if case.status == S.REVIEWED:
            sources = {
                asset.view: unseal(asset.preview, f"preview:{asset.pk}")
                for asset in revision.assets.all()
            }
            content = make_pdf(snapshot, sources)
        else:
            content = unseal(latest.pdf, f"report:{revision.pk}:{latest.number}:pdf")
            if not hmac.compare_digest(digest(content), latest.pdf_digest):
                raise Http404
        return FileResponse(
            BytesIO(content), content_type="application/pdf", filename=f"{case.reference}.pdf"
        )
    return render(
        request,
        "report.html",
        {
            "case": case,
            "snapshot": snapshot,
            "staff": True,
            "needs_otp": bool(request.user.otp_secret),
        },
    )


@role_required(R.REVIEWER)
@require_POST
def sign_report(request, case_id):
    if over_limit("sign", request.user.pk, 10):
        return HttpResponse("Try signing again later.", status=429)
    with transaction.atomic():
        case = reviewer_case(request, case_id, lock=True)
        if case.status != S.REVIEWED or not matching_review(request, case):
            return HttpResponse("Review the current version before signing.", status=409)
        signer = User.objects.get(pk=request.user.pk)
        if not signer.check_password(request.POST.get("password", "")[:256]) or (
            signer.otp_secret and not consume_otp(signer, request.POST.get("code", ""))
        ):
            audit("sign", request.user, case, result="denied")
            messages.error(
                request, "Signing requires your current password and authenticator code if enabled."
            )
            return redirect("report_preview", case_id=case.pk)
        revision = current_revision(case)
        latest = revision.reports.order_by("-number").first()
        number = 1 if not latest else latest.number + 1
        now = timezone.now()
        snapshot = report_context(
            case,
            revision,
            review_data(revision)[1],
            signer=signer,
            timestamp=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            number=number,
        )
        encoded = canonical(snapshot)
        context = f"report:{revision.pk}:{number}"
        start = perf_counter()
        previews = {
            asset.view: unseal(asset.preview, f"preview:{asset.pk}")
            for asset in revision.assets.all()
        }
        pdf = make_pdf(snapshot, previews)
        Report.objects.create(
            revision=revision,
            number=number,
            signer=signer,
            signed_at=now,
            digest=digest(encoded),
            snapshot=seal(encoded, context),
            pdf=seal(pdf, context + ":pdf"),
            pdf_digest=digest(pdf),
            pdf_ms=round((perf_counter() - start) * 1000),
        )
        case.status = S.SIGNED
        touch(case, "status")
        audit("sign", signer, case, detail={"report_version": number})
    return redirect("report_preview", case_id=case.pk)


@role_required(R.REVIEWER)
@require_POST
@transaction.atomic
def release_report(request, case_id):
    case = reviewer_case(request, case_id, lock=True)
    if case.status != S.SIGNED or request.POST.get("confirm") != "yes":
        raise PermissionDenied
    report = current_revision(case).reports.order_by("-number").first()
    if report is None:
        raise Http404
    if request.POST.get("study_revision") != str(case.revision) or request.POST.get(
        "report_version"
    ) != str(report.number):
        return HttpResponse("Reload the signed report before releasing it.", status=409)
    signed_snapshot(report)
    case.released_report, case.status = report, S.RELEASED
    touch(case, "released_report", "status")
    notify(case.owner, case, "released", f"release:{report.pk}")
    audit("release", request.user, case, detail={"report_version": report.number})
    return redirect("report_preview", case_id=case.pk)


@role_required(R.REVIEWER)
@require_POST
@transaction.atomic
def withdraw_report(request, case_id):
    case = reviewer_case(request, case_id, lock=True)
    if case.status != S.RELEASED or request.POST.get("confirm") != "yes":
        raise PermissionDenied
    if request.POST.get("study_revision") != str(case.revision) or request.POST.get(
        "report_version"
    ) != str(case.released_report.number):
        return HttpResponse("Reload the released report before withdrawing it.", status=409)
    case.released_report, case.status = None, S.WITHDRAWN
    touch(case, "released_report", "status")
    audit("withdraw", request.user, case)
    return redirect("review", case_id=case.pk)


@role_required(R.DEMO)
def released_report(request, case_id, *, download=False):
    case = owner_case(request, case_id)
    report = case.released_report
    if case.status != S.RELEASED or report is None or report.revision.case_id != case.pk:
        raise Http404
    snapshot = signed_snapshot(report)
    if download:
        pdf = unseal(report.pdf, f"report:{report.revision_id}:{report.number}:pdf")
        if not hmac.compare_digest(digest(pdf), report.pdf_digest):
            raise Http404
        audit_access(
            "report_download",
            request.user,
            case,
            detail={"report_version": report.number},
        )
        return FileResponse(BytesIO(pdf), as_attachment=True, filename=f"{case.reference}.pdf")
    audit_access("report_access", request.user, case, detail={"report_version": report.number})
    return render(request, "report.html", {"case": case, "snapshot": snapshot})
