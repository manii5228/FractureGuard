import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.exceptions import InvalidTag
from django.conf import settings
from django.contrib.sessions.models import Session
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

from ...models import AccessCode, Case
from ...retention import purge_case, tombstones
from ...security import audit, unseal


class Command(BaseCommand):
    help = "Restore an encrypted backup with the web service and worker stopped."

    def add_arguments(self, parser):
        parser.add_argument("backup", type=Path)
        parser.add_argument("--offline", action="store_true")

    def handle(self, *args, **options):
        if not options["offline"]:
            raise CommandError("Stop the web service and worker, then confirm with --offline.")
        backup = options["backup"]
        if backup.suffix != ".fgbackup":
            raise CommandError("Choose an encrypted .fgbackup file.")
        original = connection.settings_dict["NAME"]
        target = Path(original)
        if any(Path(str(target) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
            raise CommandError("The database is in use or requires SQLite recovery.")
        try:
            removed = tombstones(settings.DELETION_ROOT)
            content = unseal(backup.read_bytes(), "database-backup:v1")
            with TemporaryDirectory(dir=target.parent) as folder:
                staged = Path(folder) / target.name
                with (
                    closing(sqlite3.connect(":memory:")) as snapshot,
                    closing(sqlite3.connect(staged)) as disk,
                ):
                    snapshot.deserialize(content)
                    if snapshot.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        raise ValueError("Backup integrity check failed")
                    snapshot.backup(disk)
                connection.close()
                connection.settings_dict["NAME"] = staged
                try:
                    with transaction.atomic():
                        for case in (
                            Case.objects.select_for_update().filter(pk__in=removed).iterator()
                        ):
                            purge_case(case, record=False)
                        Session.objects.all().delete()
                        AccessCode.objects.all().delete()
                        audit("backup_restored")
                finally:
                    connection.close()
                    connection.settings_dict["NAME"] = original
                staged.replace(target)
        except (ValueError, OSError, sqlite3.DatabaseError, InvalidTag) as problem:
            raise CommandError("Backup authentication or restoration failed.") from problem
        self.stdout.write("Backup restored with deletion receipts applied.")
