import base64
import os
import secrets
import sys
from pathlib import Path

from django.core.management import execute_from_command_line

from fractureguard.paths import PROJECT_ROOT


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fractureguard.web.settings")
    if os.getenv("FG_SHARED") != "1":
        root = Path(os.getenv("FG_WEB_ROOT", PROJECT_ROOT / "work" / "m7"))
        keys = Path(os.getenv("FG_WEB_KEYS", PROJECT_ROOT / "work" / "m7-keys"))
        root.mkdir(parents=True, exist_ok=True)
        keys.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name, value in (
            ("django", secrets.token_urlsafe(48)),
            ("storage", base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")),
        ):
            path = keys / name
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w", encoding="ascii") as handle:
                handle.write(value)
    execute_from_command_line(["fractureguard.web", *sys.argv[1:]])


if __name__ == "__main__":
    main()
