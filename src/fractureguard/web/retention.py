import os
import uuid
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.utils.crypto import salted_hmac

from .cases import touch
from .models import Case, Controls, Notice, RateWindow
from .security import audit, seal, unseal


def record_tombstone(case_id):
    root = settings.DELETION_ROOT
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    text_id = str(case_id)
    name = salted_hmac("fractureguard.deletion", text_id, algorithm="sha256").hexdigest()
    path = root / f"{name}.deleted"
    if path.exists():
        return
    temporary = root / f"{uuid.uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        handle.write(seal(text_id.encode(), f"tombstone:{name}"))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def tombstones(root):
    return [
        uuid.UUID(unseal(path.read_bytes(), f"tombstone:{path.stem}").decode())
        for path in Path(root).glob("*.deleted")
    ]


def purge_case(case, actor=None, *, record=True):
    if case.status == Case.Status.DELETED:
        return
    if record:
        record_tombstone(case.pk)
    case.released_report = case.reviewer = None
    case.status, case.side, case.return_note = Case.Status.DELETED, "", b""
    case.deletion_requested, case.priority = False, False
    touch(
        case,
        "released_report",
        "reviewer",
        "status",
        "return_note",
        "side",
        "deletion_requested",
        "priority",
    )
    case.revisions.all().delete()
    Notice.objects.filter(case=case).delete()
    audit("case_deleted", actor, case)


@transaction.atomic
def expire_cases():
    today = timezone.now()
    controls = Controls.objects.get_or_create(pk=1)[0]
    deadline = today - timedelta(days=controls.retention_days)
    items = (
        Case.objects.select_for_update()
        .exclude(status=Case.Status.DELETED)
        .filter(
            updated__lt=deadline,
        )
        .order_by("updated")[:20]
    )
    for case in items:
        purge_case(case)
    RateWindow.objects.filter(starts__lt=today - timedelta(days=2)).delete()
    return len(items)


def expire_backups():
    cutoff = (timezone.now() - timedelta(days=7)).timestamp()
    for artifact in (settings.WEB_ROOT / "backups").glob("fractureguard-*.fgbackup"):
        if artifact.is_file() and not artifact.is_symlink() and artifact.stat().st_mtime < cutoff:
            artifact.unlink(missing_ok=True)
