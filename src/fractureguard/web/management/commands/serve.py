from django.conf import settings
from django.core.management.base import BaseCommand
from waitress import serve

from ...wsgi import application


class Command(BaseCommand):
    help = "Serve FractureGuard on loopback. Shared deployments require a local TLS proxy."

    def add_arguments(self, parser):
        parser.add_argument("--port", type=int, default=8765)

    def handle(self, *args, **options):
        self.stdout.write(f"Serving on 127.0.0.1:{options['port']}")
        trusted = getattr(settings, "SECURE_PROXY_SSL_HEADER", None)
        serve(
            application,
            host="127.0.0.1",
            port=options["port"],
            threads=4,
            max_request_body_size=settings.MAX_BODY_BYTES,
            trusted_proxy="127.0.0.1" if trusted else None,
            trusted_proxy_headers={"x-forwarded-proto"} if trusted else (),
        )
