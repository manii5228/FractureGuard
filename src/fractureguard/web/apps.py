from django.apps import AppConfig
from django.db.backends.signals import connection_created


def configure_sqlite(sender, connection, **kwargs):
    if connection.vendor == "sqlite":
        with connection.cursor() as cursor:
            cursor.execute("PRAGMA secure_delete = ON")


class WebConfig(AppConfig):
    name = "fractureguard.web"

    def ready(self):
        connection_created.connect(configure_sqlite, dispatch_uid="fractureguard.secure_delete")
