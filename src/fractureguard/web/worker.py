from __future__ import annotations

import multiprocessing
import uuid
from datetime import timedelta
from time import perf_counter

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .cases import notify, touch
from .models import Case, Controls, Job, User
from .security import audit, canonical, seal, unseal
from .worker_child import inference_child

S = Case.Status


class SupervisedInference:
    def __init__(self):
        self.channel = self.process = None

    def start(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        configuration = (
            settings.MODEL_PATH,
            settings.POLICY_PATH,
            settings.MODEL_SHA,
            settings.POLICY_SHA,
            settings.PREPROCESSING,
        )
        self.channel = parent
        self.process = context.Process(
            target=inference_child, args=(child, configuration), daemon=True
        )
        self.process.start()
        child.close()
        try:
            if not parent.poll(60) or parent.recv() is not True:
                raise RuntimeError("Model initialization failed")
        except (RuntimeError, OSError, EOFError):
            self.close()
            raise

    def close(self):
        if self.channel is not None:
            self.channel.close()
        if self.process is not None:
            self.process.join(0.2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            self.process.close()
        self.channel = self.process = None

    def predict(self, images):
        if self.channel is None:
            self.start()
        try:
            self.channel.send(images)
            if not self.channel.poll(20):
                raise TimeoutError("Model processing limit reached")
            return self.channel.recv()
        except (TimeoutError, EOFError, OSError):
            self.close()
            raise


@transaction.atomic
def claim_job():
    now = timezone.now()
    for expired in (
        Job.objects.select_for_update()
        .filter(
            state="running",
            started__lt=now - timedelta(seconds=110),
        )
        .select_related("revision__case")[:30]
    ):
        _finish(expired, None, 0)
    if not Controls.objects.get_or_create(pk=1)[0].model_enabled:
        return None
    job = (
        Job.objects.select_for_update()
        .filter(
            state="queued",
            revision__case__status=S.QUEUED,
            attempt__lt=3,
        )
        .select_related("revision__case")
        .order_by("queued")
        .first()
    )
    if job is None:
        return None
    case = job.revision.case
    if case.revision != job.revision.number:
        job.state = "stale"
        job.save(update_fields=["state"])
        return None
    job.state, job.started, job.lease = "running", now, uuid.uuid4()
    job.attempt += 1
    job.queue_ms = max(0, round((now - job.queued).total_seconds() * 1000))
    job.save(update_fields=["state", "lease", "started", "attempt", "queue_ms"])
    case.status = S.PROCESSING
    touch(case, "status")
    audit("inference_started", case=case, detail={"attempt": job.attempt})
    return job.pk, job.lease


def _finish(job, result, elapsed):
    revision, case = job.revision, job.revision.case
    job.finished, job.worker_ms = timezone.now(), elapsed
    if case.revision != revision.number or case.status != S.PROCESSING:
        job.state = "stale"
    elif result is None:
        job.state, case.status = "failed", S.FAILED
        case.priority = True
        touch(case, "status", "priority")
        audit("inference", case=case, result="processing_failed")
    else:
        revision.prediction = seal(canonical(result), f"prediction:{revision.pk}")
        revision.save(update_fields=["prediction"])
        job.state, case.status = "complete", S.REVIEW
        case.priority = result["outcome"] == "uncertain_manual_review" or bool(result["warnings"])
        touch(case, "status", "priority")
        audit("inference", case=case, detail={"model_sha256": result["model_sha256"]})
    job.save(update_fields=["state", "finished", "worker_ms"])
    if job.state in {"complete", "failed"}:
        listeners = User.objects.filter(role=User.Role.REVIEWER, is_active=True)
        if case.reviewer_id:
            listeners = listeners.filter(pk=case.reviewer_id)
        for user in listeners:
            notify(user, case, "ready", f"ready:{revision.number}:{job.attempt}")


@transaction.atomic
def finish_job(job_id, lease, result, elapsed):
    job = (
        Job.objects.select_for_update()
        .filter(
            pk=job_id,
            lease=lease,
            state="running",
        )
        .select_related("revision__case")
        .first()
    )
    if job is None:
        return False
    _finish(job, result, elapsed)
    return job.state != "stale"


def process_one(predict):
    claimed = claim_job()
    if not claimed:
        return False
    job_id, lease = claimed
    start = perf_counter()
    try:
        job = Job.objects.get(pk=job_id)
        images = {
            asset.view: unseal(asset.png, f"image:{asset.pk}")
            for asset in job.revision.assets.all()
        }
        result = predict(images)
    except Exception:
        result = None
    finish_job(job_id, lease, result, round((perf_counter() - start) * 1000))
    return True
