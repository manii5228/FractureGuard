from __future__ import annotations

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.core.exceptions import ValidationError
from django.db import transaction
from django.http import HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from .accounts import issue_code
from .cases import S, notify, touch
from .models import Audit, Case, Controls, Job, User
from .retention import purge_case
from .security import audit, role_required


def dashboard(request, **extra):
    logs = Audit.objects.select_related("actor").order_by("-time")
    action = request.GET.get("event", "")[:40]
    reference = request.GET.get("reference", "")[:16]
    if action:
        logs = logs.filter(action=action)
    if reference:
        logs = logs.filter(reference__iexact=reference)
    return render(
        request,
        "operations.html",
        {
            "controls": Controls.objects.get_or_create(pk=1)[0],
            "users": User.objects.order_by("username")[:100],
            "logs": logs[:80],
            "roles": User.Role.choices,
            "cases": Case.objects.exclude(status__in=[S.DRAFT, S.DELETED]).select_related(
                "reviewer"
            )[:100],
            "deletions": Case.objects.filter(deletion_requested=True).exclude(status=S.DELETED)[
                :100
            ],
            "reviewers": User.objects.filter(is_active=True, role=User.Role.REVIEWER).order_by(
                "username"
            ),
            "jobs": {
                state: Job.objects.filter(state=state).count()
                for state in ("queued", "running", "failed")
            },
            "storage_bytes": (settings.WEB_ROOT / "fractureguard.sqlite3").stat().st_size
            if (settings.WEB_ROOT / "fractureguard.sqlite3").is_file()
            else 0,
            "model_name": settings.MODEL_NAME,
            "model_sha": settings.MODEL_SHA,
            "model_files_present": settings.MODEL_PATH.is_file() and settings.POLICY_PATH.is_file(),
            "job_worker_ms": list(
                Job.objects.filter(state="complete")
                .order_by("-finished")
                .values_list("worker_ms", flat=True)[:10]
            ),
            "filter_event": action,
            "filter_reference": reference,
            **extra,
        },
    )


@role_required(User.Role.ADMIN)
@require_http_methods(["GET", "POST"])
def operations(request):
    if request.method == "GET":
        return dashboard(request)
    action = request.POST.get("action")
    token = None
    with transaction.atomic():
        if action == "invite":
            username = request.POST.get("username", "").strip().lower()
            try:
                UnicodeUsernameValidator()(username)
            except ValidationError:
                return HttpResponseBadRequest("Choose a valid account name.")
            if not 3 <= len(username) <= 150 or User.objects.filter(username=username).exists():
                return HttpResponseBadRequest(
                    "Choose an unused account name between 3 and 150 characters."
                )
            user = User(username=username, is_active=False)
            user.set_unusable_password()
            user.save()
            token = issue_code(user, "invite", 24)
            audit("invite", request.user, detail={"account": user.pk})
        elif action in {"account", "reset", "reinvite"}:
            user_id = request.POST.get("user_id", "")
            if not user_id.isdecimal() or len(user_id) > 12:
                return HttpResponseBadRequest("Choose a valid account.")
            user = get_object_or_404(User.objects.select_for_update(), pk=user_id)
            if user.pk == request.user.pk:
                return HttpResponseBadRequest(
                    "Use a separate administrator to change your account."
                )
            if action == "reset":
                if not user.is_active:
                    return HttpResponseBadRequest(
                        "Enable the account before issuing a recovery code."
                    )
                token = issue_code(user, "reset", 1)
                audit("recovery_issued", request.user, detail={"account": user.pk})
            elif action == "reinvite":
                if user.has_usable_password() or user.is_active:
                    return HttpResponseBadRequest("Use account recovery for an activated account.")
                token = issue_code(user, "invite", 24)
                audit("invite_reissued", request.user, detail={"account": user.pk})
            else:
                role = request.POST.get("role")
                if role not in User.Role.values:
                    return HttpResponseBadRequest("Choose a valid role.")
                user.role = role
                user.is_active = request.POST.get("enabled") == "yes" and user.has_usable_password()
                user.session_epoch += 1
                user.save(update_fields=["role", "is_active", "session_epoch"])
                audit(
                    "account_changed",
                    request.user,
                    detail={"account": user.pk, "role": role, "enabled": user.is_active},
                )
                if not user.is_active or role != User.Role.REVIEWER:
                    for case in user.assignments.exclude(status__in=[S.DELETED, S.RELEASED]):
                        case.reviewer = None
                        touch(case, "reviewer")
                        audit("assignment_cleared", request.user, case)
        elif action == "settings":
            days = request.POST.get("retention", "")
            if not days.isdecimal() or len(days) > 2 or not 1 <= int(days) <= 90:
                return HttpResponseBadRequest("Choose retention between 1 and 90 days.")
            controls = Controls.objects.get_or_create(pk=1)[0]
            controls.retention_days = int(days)
            controls.model_enabled = request.POST.get("enabled") == "yes"
            controls.save(update_fields=["retention_days", "model_enabled"])
            audit(
                "settings_changed",
                request.user,
                detail={
                    "model_enabled": controls.model_enabled,
                    "retention_days": controls.retention_days,
                },
            )
        elif action in {"assign", "purge"}:
            from uuid import UUID

            try:
                case_id = UUID(request.POST.get("case_id", ""))
            except ValueError:
                return HttpResponseBadRequest("Choose a valid case.")
            case = get_object_or_404(
                Case.objects.select_for_update().exclude(status=S.DELETED), pk=case_id
            )
            if action == "purge":
                if not case.deletion_requested or request.POST.get("confirm") != "yes":
                    return HttpResponseBadRequest("Confirm the outstanding deletion request.")
                purge_case(case, request.user)
            else:
                reviewer_id = request.POST.get("reviewer_id", "")
                if not reviewer_id.isdecimal() or len(reviewer_id) > 12:
                    return HttpResponseBadRequest("Choose a valid reviewer.")
                reviewer = get_object_or_404(
                    User, pk=reviewer_id, role=User.Role.REVIEWER, is_active=True
                )
                if case.status == S.DRAFT:
                    return HttpResponseBadRequest("A draft cannot be assigned.")
                if case.reviewer_id != reviewer.pk:
                    case.reviewer = reviewer
                    touch(case, "reviewer")
                    event = audit("assigned", request.user, case, detail={"account": reviewer.pk})
                    notify(reviewer, case, "assigned", f"assign:{event.pk}")
        else:
            return HttpResponseBadRequest("Choose a supported administrative action.")
    if token:
        return dashboard(request, token=token, issued_for=user.username)
    messages.success(request, "Change saved.")
    return redirect("operations")
