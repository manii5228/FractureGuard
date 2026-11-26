from datetime import timedelta
from time import monotonic, sleep

from django.contrib.sessions.models import Session
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from ...models import AccessCode, Controls
from ...retention import expire_backups, expire_cases
from ...worker import SupervisedInference, process_one


class Command(BaseCommand):
    help = "Run the bounded CPU inference worker and scheduled retention checks."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true")

    def handle(self, *args, **options):
        engine = SupervisedInference()
        maintenance = 0
        try:
            if Controls.objects.get_or_create(pk=1)[0].model_enabled:
                engine.start()
            self.stdout.write("Case worker ready.")
            while True:
                if monotonic() >= maintenance:
                    while expire_cases() == 20:
                        pass
                    expire_backups()
                    now = timezone.now()
                    Session.objects.filter(expire_date__lt=now).delete()
                    AccessCode.objects.filter(expires__lt=now - timedelta(days=7)).delete()
                    maintenance = monotonic() + 3600
                worked = process_one(engine.predict)
                if options["once"]:
                    return
                if not worked:
                    sleep(0.5)
        except KeyboardInterrupt:
            pass
        except (RuntimeError, TimeoutError, EOFError, OSError) as problem:
            raise CommandError("Worker initialization or storage failed.") from problem
        finally:
            engine.close()
