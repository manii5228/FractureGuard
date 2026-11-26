import os
import sqlite3
from contextlib import closing

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from django.utils import timezone

from ...retention import expire_backups
from ...security import seal


class Command(BaseCommand):
    help = "Make an encrypted SQLite backup. Backups expire after seven days."

    def handle(self, *args, **options):
        path = settings.WEB_ROOT / "backups"
        path.mkdir(parents=True, exist_ok=True)
        artifact = path / f"fractureguard-{timezone.now():%Y%m%dT%H%M%S%f}.fgbackup"
        source = connection.settings_dict["NAME"]
        if not os.path.isfile(source):
            raise CommandError("Run migrate before backing up.")
        with (
            closing(sqlite3.connect(source)) as live,
            closing(sqlite3.connect(":memory:")) as snapshot,
        ):
            live.backup(snapshot)
            content = seal(snapshot.serialize(), "database-backup:v1")
        with artifact.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        expire_backups()
        self.stdout.write(f"Encrypted backup created as {artifact.name}.")
