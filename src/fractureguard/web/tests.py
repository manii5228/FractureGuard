import csv
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import timedelta
from io import BytesIO
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pyotp
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, SimpleTestCase, TestCase, override_settings, tag
from django.urls import reverse
from django.utils import timezone
from PIL import Image

from fractureguard.paths import PROJECT_ROOT

from .cases import review_data, signed_snapshot
from .models import Audit, Case, Job, Notice, Report, User
from .retention import expire_cases, tombstones
from .worker import SupervisedInference, claim_job, finish_job, process_one

PASSWORD = "Workflow-fixture.pass.7819"
S = Case.Status
FIRST_RELEASE = {"confirm": "yes", "study_revision": 1, "report_version": 1}


def image(seed=1):
    pixels = np.random.default_rng(seed).integers(0, 65536, (325, 320), dtype=np.uint16)
    data = BytesIO()
    Image.fromarray(pixels).save(data, format="PNG")
    return upload(data.getvalue())


def upload(data):
    return SimpleUploadedFile("fixture.png", data, content_type="image/png")


def synthetic_prediction(outcome="suspected_fracture"):
    return {
        "boxes": [
            {
                "id": "model-frontal-0",
                "view": "frontal",
                "coords": [10, 20, 45, 54],
                "status": "preliminary",
                "score": 0.83,
            }
        ],
        "outcome": outcome,
        "warnings": [],
        "label": outcome,
        "score_label": "83%",
        "model_sha256": settings.MODEL_SHA,
        "policy_sha256": settings.POLICY_SHA,
    }


def url(name, case=None):
    return reverse(name, args=[case.pk] if case else [])


class Workflow(TestCase):
    @classmethod
    def setUpTestData(cls):
        for name, role in (
            ("owner", "demo"),
            ("bystander", "demo"),
            ("reader", "reviewer"),
            ("otherreader", "reviewer"),
            ("operator", "admin"),
        ):
            User.objects.create_user(username=name, password=PASSWORD, role=role)

    def signed_in(self, name):
        client = Client()
        response = client.post(url("sign_in"), {"username": name, "password": PASSWORD})
        self.assertRedirects(response, "/", fetch_redirect_response=False)
        return client

    def new(self, owner):
        response = owner.post(url("new_case"))
        return Case.objects.get(pk=response.url.strip("/").split("/")[-1])

    def submit(self, owner, case, **images):
        return owner.post(url("case_detail", case), {"side": "L", "attest": "yes", **images})

    def findings(self, reader, case, *, number=0):
        manual = {
            "id": f"manual-{uuid4()}",
            "view": "frontal",
            "status": "approved",
            "coords": [15, 20, 61, 63],
        }
        previous_box = {
            "id": "model-frontal-0",
            "view": "frontal",
            "status": "rejected",
            "coords": [10, 20, 45, 54],
        }
        return reader.post(
            url("save_review", case),
            {
                "boxes": json.dumps([manual, previous_box]),
                "edit_version": number,
                "study_revision": 1,
                "verify_views": "yes",
                "findings": "Demonstration reviewer finding <review>.",
                "impression": "Demo only.",
            },
        )

    def test_release_transaction_and_isolation(self):
        owner, reader = self.signed_in("owner"), self.signed_in("reader")
        bystander, outsider = self.signed_in("bystander"), self.signed_in("otherreader")
        admin = self.signed_in("operator")
        self.assertContains(admin.get(url("operations")), "Audit log")
        case = self.new(owner)
        self.assertContains(owner.get(url("case_detail", case)), "Frontal PNG")
        self.assertEqual(
            self.submit(owner, case, frontal=image(), lateral=image(2)).status_code, 302
        )
        self.assertEqual(
            self.submit(owner, case, frontal=image(), lateral=image(2)).status_code, 302
        )
        self.assertEqual(case.revisions.count(), 1)
        asset = case.revisions.get().assets.get(view="frontal")
        preview_url = reverse("asset_preview", args=[asset.pk])
        self.assertFalse(bytes(asset.png).startswith(b"\x89PNG"))
        self.assertEqual(bystander.get(url("case_status", case)).status_code, 404)
        self.assertEqual(bystander.get(url("case_detail", case)).status_code, 404)
        self.assertEqual(owner.get(preview_url).status_code, 403)
        self.assertTrue(process_one(lambda views: synthetic_prediction()))
        self.assertContains(reader.get(url("queue")), "Ready in the review queue")
        self.assertEqual(reader.get(url("review", case)).status_code, 404)
        self.assertEqual(reader.post(url("claim", case)).status_code, 302)
        self.assertEqual(outsider.post(url("claim", case)).status_code, 409)
        self.assertEqual(outsider.get(preview_url).status_code, 404)
        self.assertEqual(admin.get(url("review", case)).status_code, 403)
        preview = reader.get(preview_url)
        self.assertTrue(preview.content.startswith(b"\x89PNG"))
        self.assertEqual(preview["Cache-Control"], "private, no-store")
        self.assertContains(reader.get(url("review", case)), "Preliminary model result")
        self.assertEqual(owner.get(url("download", case)).status_code, 404)
        self.assertEqual(self.findings(reader, case).status_code, 302)
        self.assertEqual(self.findings(reader, case).status_code, 409)
        self.assertContains(reader.get(url("report_preview", case)), "&lt;review&gt;")
        self.assertEqual(reader.get(url("staff_pdf", case))["Content-Type"], "application/pdf")
        signed = reader.post(
            url("sign_report", case),
            {
                "password": PASSWORD,
                "edit_version": 1,
                "study_revision": 1,
            },
        )
        self.assertEqual(signed.status_code, 302)
        self.assertEqual(owner.get(url("download", case)).status_code, 404)
        self.assertEqual(reader.post(url("release_report", case)).status_code, 403)
        self.assertEqual(
            reader.post(
                url("release_report", case),
                {
                    **FIRST_RELEASE,
                    "report_version": 2,
                },
            ).status_code,
            409,
        )
        self.assertEqual(reader.post(url("release_report", case), FIRST_RELEASE).status_code, 302)
        self.assertEqual(reader.post(url("release_report", case), FIRST_RELEASE).status_code, 403)
        self.assertEqual(Notice.objects.filter(case=case, event="released").count(), 1)
        self.assertEqual(bystander.get(url("download", case)).status_code, 404)
        owner_report = owner.get(url("released_report", case))
        self.assertContains(owner_report, "Demo only.")
        self.assertNotContains(owner_report, "Preliminary model result")
        pdf = b"".join(owner.get(url("download", case)).streaming_content)
        self.assertTrue(pdf.startswith(b"%PDF"))
        first = Report.objects.get()
        self.assertFalse(bytes(first.pdf).startswith(b"%PDF"))
        self.assertEqual(len(signed_snapshot(first)["boxes"]), 1)
        self.assertEqual(reader.post(url("withdraw_report", case), FIRST_RELEASE).status_code, 302)
        self.assertEqual(owner.get(url("download", case)).status_code, 404)
        self.assertContains(reader.get("/review/?status=withdrawn"), case.reference)
        self.assertEqual(self.findings(reader, case, number=1).status_code, 302)
        self.assertEqual(
            reader.post(
                url("sign_report", case),
                {"password": PASSWORD, "edit_version": 2, "study_revision": 1},
            ).status_code,
            302,
        )
        self.assertEqual(
            list(Report.objects.order_by("number").values_list("number", flat=True)), [1, 2]
        )
        audit_events = Audit.objects.filter(reference=case.reference)
        self.assertTrue(
            {"upload", "inference", "sign", "release", "report_download", "withdraw"}
            <= set(audit_events.values_list("action", flat=True))
        )
        self.assertNotIn(
            "Demonstration reviewer finding", repr(list(audit_events.values_list("detail")))
        )
        self.assertEqual(
            admin.post(
                url("operations"),
                {
                    "action": "account",
                    "role": "demo",
                    "user_id": User.objects.get(username="reader").pk,
                    "enabled": "yes",
                },
            ).status_code,
            302,
        )
        self.assertEqual(reader.get(url("review", case)).url, url("sign_in"))

    def test_upload_return_retry_and_deletion(self):
        owner, reader = self.signed_in("owner"), self.signed_in("reader")
        admin = self.signed_in("operator")
        case = self.new(owner)
        self.assertContains(
            self.submit(owner, case, frontal=image()), "exactly one", status_code=400
        )
        self.assertContains(
            self.submit(owner, case, frontal=upload(b"bad"), lateral=image(2)),
            "valid PNG",
            status_code=400,
        )
        with override_settings(MAX_UPLOAD_BYTES=10):
            self.assertContains(
                self.submit(owner, case, frontal=image(), lateral=image(2)),
                "no larger",
                status_code=400,
            )
        self.assertContains(
            self.submit(owner, case, frontal=image(), lateral=image()),
            "different image",
            status_code=400,
        )
        self.submit(owner, case, frontal=image(), lateral=image(2))
        old_job, old_lease = claim_job()
        self.assertTrue(finish_job(old_job, old_lease, None, 100))
        self.assertEqual(reader.post(url("claim", case)).status_code, 302)
        self.assertContains(reader.get(url("review", case)), "no preliminary result")
        self.assertEqual(reader.post(url("retry_case", case)).status_code, 302)
        self.assertTrue(process_one(lambda views: synthetic_prediction("uncertain_manual_review")))
        self.assertEqual(
            reader.post(
                url("return_case", case), {"reason": "Improve the lateral view."}
            ).status_code,
            302,
        )
        self.assertContains(owner.get(url("case_detail", case)), "Improve the lateral view")
        self.assertEqual(self.submit(owner, case, lateral=image(3)).status_code, 302)
        self.assertFalse(finish_job(old_job, old_lease, synthetic_prediction(), 1980))
        self.assertEqual(case.revisions.count(), 2)
        self.assertTrue(process_one(lambda views: synthetic_prediction("no_fracture_detected")))
        self.assertEqual(owner.post(url("request_deletion", case)).status_code, 302)
        with TemporaryDirectory() as folder, override_settings(DELETION_ROOT=Path(folder)):
            response = admin.post(
                url("operations"),
                {
                    "action": "purge",
                    "case_id": case.pk,
                    "confirm": "yes",
                },
            )
            self.assertEqual(response.status_code, 302)
            self.assertIn(case.pk, tombstones(Path(folder)))
            self.assertFalse(case.revisions.exists())
            self.assertEqual(owner.get(url("case_status", case)).status_code, 404)
            aged = self.new(owner)
            Case.objects.filter(pk=aged.pk).update(updated=timezone.now() - timedelta(days=31))
            self.assertEqual(expire_cases(), 1)
            self.assertIn(aged.pk, tombstones(Path(folder)))

    def test_csrf_and_shared_staff_mfa(self):
        csrf = Client(enforce_csrf_checks=True)
        self.assertEqual(
            csrf.post(url("sign_in"), {"username": "owner", "password": PASSWORD}).status_code, 403
        )
        response = csrf.get(url("sign_in"))
        self.assertIn("frame-ancestors 'none'", response["Content-Security-Policy"])
        self.assertEqual(response["X-Frame-Options"], "DENY")
        secret = pyotp.random_base32()
        with (
            override_settings(SHARED=True, SESSION_COOKIE_SECURE=True),
            patch("fractureguard.web.accounts.pyotp.random_base32", return_value=secret),
            patch("fractureguard.web.security.timezone.now", return_value=timezone.now()),
        ):
            client = Client()
            data = {"username": "reader", "password": PASSWORD}
            self.assertEqual(client.post(url("sign_in"), data).url, url("second_factor"))
            self.assertNotIn(secret, str(client.session["pending"]))
            generator = pyotp.TOTP(secret)
            code = generator.generate_otp(int(timezone.now().timestamp()) // generator.interval)
            response = client.post(url("second_factor"), {"code": code})
            self.assertEqual(response.url, "/")
            self.assertTrue(response.cookies["sessionid"]["secure"])
            self.assertTrue(
                User.objects.get(username="reader").password.startswith("argon2$argon2id")
            )
            self.assertEqual(client.post(url("sign_out")).status_code, 302)
            client.post(url("sign_in"), data)
            self.assertContains(
                client.post(url("second_factor"), {"code": code}), "fresh six-digit code"
            )

    @tag("real_model")
    def test_frozen_public_pairs_through_web_and_worker(self):
        from fractureguard.data.training_data import _resolve_source_image

        root = PROJECT_ROOT / "data"
        with (root / "manifests" / "grazpedwri_m5_test_cohort.csv").open(newline="") as handle:
            cohort = sorted(csv.DictReader(handle), key=lambda row: row["study_id"])
        views = {key: list(rows) for key, rows in groupby(cohort, key=lambda row: row["study_id"])}
        owner = self.signed_in("owner")
        reviewers = (self.signed_in("reader"), self.signed_in("otherreader"))
        timings, fixtures, outcomes = [], [], Counter()
        engine = SupervisedInference()
        self.addCleanup(engine.close)
        engine.start()
        table = PROJECT_ROOT / "reports/predictions/m6/m6_onnx_fp32_studies.csv"
        with table.open(newline="") as handle:
            for reference in csv.DictReader(handle):
                study = views[reference["study_id"]]
                outcome = reference["triage_decision"]
                if (
                    outcomes[outcome] >= 4
                    or len(study) != 2
                    or not all(row["projection"] in {"frontal", "lateral"} for row in study)
                    or study[0]["projection"] == study[1]["projection"]
                ):
                    continue
                paths = {
                    row["projection"]: _resolve_source_image(
                        root / "raw/grazpedwri-dx", row["image_path"]
                    )
                    for row in study
                }
                case = self.new(owner)
                response = self.submit(
                    owner, case, **{view: upload(path.read_bytes()) for view, path in paths.items()}
                )
                self.assertEqual(
                    response.status_code, 302, str(response.context and response.context["errors"])
                )
                self.assertTrue(process_one(engine.predict))
                revision = case.revisions.get()
                job = Job.objects.get(revision=revision)
                prediction, _ = review_data(revision)
                self.assertEqual(prediction["model_sha256"], settings.MODEL_SHA)
                self.assertAlmostEqual(
                    prediction["fused_max_score"], float(reference["fused_max_score"]), delta=0.0005
                )
                self.assertEqual(prediction["outcome"], outcome)
                reader = reviewers[len(timings) % len(reviewers)]
                self.assertEqual(reader.post(url("claim", case)).status_code, 302)
                self.assertEqual(self.findings(reader, case).status_code, 302)
                self.assertEqual(
                    reader.post(
                        url("sign_report", case),
                        {
                            "password": PASSWORD,
                            "edit_version": 1,
                            "study_revision": 1,
                        },
                    ).status_code,
                    302,
                )
                self.assertEqual(
                    reader.post(url("release_report", case), FIRST_RELEASE).status_code, 302
                )
                self.assertTrue(
                    b"".join(owner.get(url("download", case)).streaming_content).startswith(b"%PDF")
                )
                outcomes[outcome] += 1
                fixtures.append([reference["study_id"], outcome])
                timings.append(
                    [
                        revision.validation_ms,
                        job.queue_ms,
                        job.worker_ms,
                        revision.reports.get().pdf_ms,
                    ]
                )
                if len(timings) == 12:
                    break
        self.assertEqual(set(outcomes.values()), {4})
        self.assertEqual(len(outcomes), 3)
        print("M7 public fixtures:", json.dumps(fixtures))
        print(
            "M7 web p95 ms (validation, queue, worker, PDF):",
            np.quantile(timings, 0.95, axis=0).tolist(),
        )
        self.assertLessEqual(float(np.quantile(np.array(timings)[:, 2], 0.95)), 5000)


class RestoreWorkflow(SimpleTestCase):
    def test_encrypted_restore_and_concurrent_transitions(self):
        script = """
import django
django.setup()
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from django.conf import settings
from django.core.management import call_command
from django.db import connection, connections, transaction
from django.test import Client
from django.urls import reverse
from django.utils import timezone
from fractureguard.web.models import Case, Notice, Report, Revision, User
from fractureguard.web.retention import purge_case
from fractureguard.web.security import digest, seal

call_command("migrate", interactive=False, verbosity=0)
owner = User.objects.create_user(username="owner")
readers = [User.objects.create_user(username=name, role="reviewer", password="fixture-only")
           for name in ("reader-a", "reader-b")]
case = Case.objects.create(owner=owner, status=Case.Status.REVIEW, revision=1)
revision = Revision.objects.create(case=case, number=1, side="L", fingerprint="fixture")
clients = [Client(HTTP_HOST="localhost") for _ in readers]
for client, reader in zip(clients, readers, strict=True):
    result = client.post(
        reverse("sign_in"), {"username": reader.username, "password": "fixture-only"}
    )
    assert result.status_code == 302
barrier = Barrier(2, timeout=10)
def post_together(client, action, data):
    try:
        barrier.wait()
        return client.post(reverse(action, args=[case.pk]), data).status_code
    finally:
        connections.close_all()
with ThreadPoolExecutor(max_workers=2) as pool:
    pending = [pool.submit(post_together, client, "claim", {}) for client in clients]
    assert sorted(future.result() for future in pending) == [302, 409]
case.refresh_from_db()
assert case.reviewer_id in {reviewer.pk for reviewer in readers}
pdf = b"%PDF-1.4 dummy test document"
key = f"report:{revision.pk}:1"
Report.objects.create(
    revision=revision, number=1, signer=case.reviewer, signed_at=timezone.now(),
    digest=digest(b"{}"), snapshot=seal(b"{}", key), pdf=seal(pdf, key + ":pdf"),
    pdf_digest=digest(pdf),
)
case.status = Case.Status.SIGNED
case.save(update_fields=["status"])
releasers = [Client(HTTP_HOST="localhost"), Client(HTTP_HOST="localhost")]
for client in releasers:
    client.post(
        reverse("sign_in"), {"username": case.reviewer.username, "password": "fixture-only"}
    )
with ThreadPoolExecutor(max_workers=2) as pool:
    pending = [pool.submit(post_together, client, "release_report", {
        "confirm": "yes", "study_revision": 1, "report_version": 1,
    }) for client in releasers]
    assert sorted(future.result() for future in pending) == [302, 403]
assert Notice.objects.filter(case=case, event="released").count() == 1
case.refresh_from_db()
assert case.status == Case.Status.RELEASED
call_command("backup")
backup = next((settings.WEB_ROOT / "backups").glob("*.fgbackup"))
assert not backup.read_bytes().startswith(b"SQLite format 3")
with transaction.atomic():
    purge_case(case)
connection.close()
call_command("restorebackup", backup, offline=True)
case.refresh_from_db()
assert case.status == Case.Status.DELETED
assert not case.revisions.exists()
assert clients[0].get(reverse("queue")).url == reverse("sign_in")
connection.close()
"""
        with TemporaryDirectory() as folder:
            location = Path(folder)
            data = location / "database"
            data.mkdir()
            environment = {
                **os.environ,
                "FG_SHARED": "0",
                "FG_WEB_ROOT": str(data),
                "FG_DELETION_DIR": str(location / "deletions"),
                "FG_DJANGO_KEY": settings.SECRET_KEY,
                "FG_DATA_KEY": settings.DATA_KEY,
            }
            result = subprocess.run(
                [sys.executable, "-B", "-c", script],
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
