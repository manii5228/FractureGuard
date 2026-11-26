from __future__ import annotations

import uuid

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    class Role(models.TextChoices):
        DEMO = "demo", "Demo user"
        REVIEWER = "reviewer", "Reviewer"
        ADMIN = "admin", "Administrator"

    role = models.CharField(max_length=12, choices=Role, default=Role.DEMO)
    session_epoch = models.PositiveIntegerField(default=0)
    otp_secret = models.BinaryField(blank=True, default=bytes)
    otp_step = models.BigIntegerField(default=-1)


class AccessCode(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    digest = models.CharField(max_length=64, unique=True)
    purpose = models.CharField(max_length=12, choices=[("invite", "Invite"), ("reset", "Reset")])
    expires = models.DateTimeField()
    used = models.DateTimeField(null=True)


class RateWindow(models.Model):
    key = models.CharField(primary_key=True, max_length=64)
    starts = models.DateTimeField()
    hits = models.PositiveIntegerField(default=0)


class Case(models.Model):
    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        QUEUED = "queued", "Submitted"
        PROCESSING = "processing", "Processing"
        REVIEW = "review_required", "Awaiting review"
        REVIEWED = "reviewed", "Reviewed"
        SIGNED = "signed", "Signed"
        RELEASED = "released", "Report ready"
        FAILED = "failed", "Processing failed"
        RETURNED = "returned", "Replacement requested"
        WITHDRAWN = "withdrawn", "Report withdrawn"
        DELETED = "deleted", "Deleted"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(User, on_delete=models.PROTECT, related_name="cases")
    reviewer = models.ForeignKey(
        User, on_delete=models.PROTECT, related_name="assignments", null=True, blank=True
    )
    status = models.CharField(max_length=20, choices=Status, default=Status.DRAFT, db_index=True)
    side = models.CharField(max_length=1, choices=[("L", "Left"), ("R", "Right")], blank=True)
    revision = models.PositiveIntegerField(default=0)
    edit_version = models.PositiveIntegerField(default=0)
    priority = models.BooleanField(default=False, db_index=True)
    return_note = models.BinaryField(blank=True, default=bytes)
    deletion_requested = models.BooleanField(default=False)
    released_report = models.ForeignKey(
        "Report", on_delete=models.SET_NULL, null=True, related_name="active_in_cases"
    )
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, db_index=True)

    @property
    def reference(self):
        return f"FG-{self.id.hex[:12].upper()}"


class Revision(models.Model):
    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name="revisions")
    number = models.PositiveIntegerField()
    side = models.CharField(max_length=1)
    fingerprint = models.CharField(max_length=64)
    prediction = models.BinaryField(blank=True, default=bytes)
    findings = models.BinaryField(blank=True, default=bytes)
    submitted = models.DateTimeField(auto_now_add=True)
    validation_ms = models.PositiveIntegerField(default=0)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["case", "number"], name="case_revision")]


class Asset(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    revision = models.ForeignKey(Revision, on_delete=models.CASCADE, related_name="assets")
    view = models.CharField(max_length=8, choices=[("frontal", "AP/PA"), ("lateral", "Lateral")])
    png = models.BinaryField()
    preview = models.BinaryField()
    width = models.PositiveIntegerField()
    height = models.PositiveIntegerField()
    fingerprint = models.CharField(max_length=64)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["revision", "view"], name="revision_view")]


class Job(models.Model):
    revision = models.OneToOneField(Revision, on_delete=models.CASCADE, related_name="job")
    state = models.CharField(max_length=12, default="queued", db_index=True)
    attempt = models.PositiveSmallIntegerField(default=0)
    lease = models.UUIDField(null=True)
    started = models.DateTimeField(null=True)
    finished = models.DateTimeField(null=True)
    queued = models.DateTimeField(auto_now_add=True)
    queue_ms = models.PositiveIntegerField(default=0)
    worker_ms = models.PositiveIntegerField(default=0)


class Report(models.Model):
    revision = models.ForeignKey(Revision, on_delete=models.CASCADE, related_name="reports")
    number = models.PositiveIntegerField()
    signer = models.ForeignKey(User, on_delete=models.PROTECT)
    signed_at = models.DateTimeField()
    digest = models.CharField(max_length=64)
    snapshot = models.BinaryField()
    pdf = models.BinaryField()
    pdf_digest = models.CharField(max_length=64)
    pdf_ms = models.PositiveIntegerField(default=0)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["revision", "number"], name="report_number")]


class Audit(models.Model):
    time = models.DateTimeField(auto_now_add=True, db_index=True)
    actor = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    action = models.CharField(max_length=40, db_index=True)
    reference = models.CharField(max_length=16, blank=True, db_index=True)
    version = models.PositiveIntegerField(null=True)
    result = models.CharField(max_length=32, default="ok")
    detail = models.JSONField(default=dict)


class Notice(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    case = models.ForeignKey(Case, on_delete=models.CASCADE)
    event = models.CharField(max_length=20)
    event_key = models.CharField(max_length=80)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["user", "case", "event_key"], name="notice_event_once")
        ]


class Controls(models.Model):
    model_enabled = models.BooleanField(default=True)
    retention_days = models.PositiveSmallIntegerField(default=30)
