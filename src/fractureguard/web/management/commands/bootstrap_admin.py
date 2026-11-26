from getpass import getpass

from django.contrib.auth.password_validation import validate_password
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from ...models import User
from ...security import audit


class Command(BaseCommand):
    help = "Create the first FractureGuard administrator."

    def add_arguments(self, parser):
        parser.add_argument("username")

    def handle(self, *args, **options):
        name = options["username"].lower()
        user = User(username=name, role=User.Role.ADMIN)
        if User.objects.filter(role=User.Role.ADMIN).exists():
            raise CommandError("An administrator has already been created.")
        try:
            UnicodeUsernameValidator()(name)
            if not 3 <= len(name) <= 150:
                raise ValidationError("Use an account name between 3 and 150 characters.")
            password = getpass("Password: ")
            if password != getpass("Confirm password: ") or len(password) > 256:
                raise ValidationError("Enter matching passwords of at most 256 characters.")
            validate_password(password, user)
        except ValidationError as problem:
            raise CommandError(" ".join(problem.messages)) from problem
        with transaction.atomic():
            if (
                User.objects.filter(role=User.Role.ADMIN).exists()
                or User.objects.filter(username=name).exists()
            ):
                raise CommandError("An administrator or this account already exists.")
            user.set_password(password)
            user.save()
            audit("bootstrap_admin", user)
        self.stdout.write("Administrator created.")
