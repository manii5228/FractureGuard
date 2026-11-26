from __future__ import annotations

import secrets
from datetime import timedelta

import pyotp
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import transaction
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST

from .models import AccessCode, User
from .security import audit, consume_otp, digest, over_limit, seal, unseal


def complete_login(request, user):
    login(request, user, backend="django.contrib.auth.backends.ModelBackend")
    now = int(timezone.now().timestamp())
    request.session.update({"epoch": user.session_epoch, "last_activity": now, "signed_in_at": now})
    audit("sign_in", user)
    return redirect("home")


@require_http_methods(["GET", "POST"])
def sign_in(request):
    if request.user.is_authenticated:
        return redirect("home")
    error = ""
    if request.method == "POST":
        name = request.POST.get("username", "").strip().lower()[:150]
        peer = request.META.get("REMOTE_ADDR", "local")
        if over_limit("login_ip", peer, 30) or over_limit("login_user", name, 8):
            return render(request, "auth.html", {"error": "Try again in ten minutes."}, status=429)
        password = request.POST.get("password", "")
        user = (
            authenticate(request, username=name, password=password)
            if len(password) <= 256
            else None
        )
        if user:
            if user.role != User.Role.DEMO and (settings.SHARED or user.otp_secret):
                request.session.cycle_key()
                request.session["pending"] = {
                    "user": user.pk,
                    "epoch": user.session_epoch,
                    "expires": int(timezone.now().timestamp()) + 300,
                    "enrollment": None
                    if user.otp_secret
                    else seal(pyotp.random_base32().encode(), f"enroll:{user.pk}").hex(),
                }
                return redirect("second_factor")
            return complete_login(request, user)
        audit("sign_in", result="denied")
        error = "The account name or password was not accepted."
    return render(request, "auth.html", {"error": error})


@require_http_methods(["GET", "POST"])
def second_factor(request):
    pending = request.session.get("pending")
    if not pending or pending["expires"] < timezone.now().timestamp():
        request.session.pop("pending", None)
        return redirect("sign_in")
    user = User.objects.filter(
        pk=pending["user"], is_active=True, session_epoch=pending["epoch"]
    ).first()
    if user is None:
        request.session.pop("pending", None)
        return redirect("sign_in")
    enrollment = pending["enrollment"]
    secret = unseal(bytes.fromhex(enrollment), f"enroll:{user.pk}").decode() if enrollment else None
    error = ""
    if request.method == "POST":
        if over_limit("mfa", user.pk, 8):
            request.session.pop("pending", None)
            return redirect("sign_in")
        if consume_otp(user, request.POST.get("code", ""), enrollment=secret):
            request.session.pop("pending", None)
            return complete_login(request, user)
        audit("second_factor", user, result="denied")
        error = "Enter a fresh six-digit code from your authenticator."
    return render(request, "auth.html", {"mode": "mfa", "secret": secret, "error": error})


@require_POST
def sign_out(request):
    if request.user.is_authenticated:
        audit("sign_out", request.user)
    logout(request)
    return redirect("sign_in")


def issue_code(user, purpose, hours):
    now = timezone.now()
    AccessCode.objects.filter(user=user, purpose=purpose, used__isnull=True).update(used=now)
    token = secrets.token_urlsafe(32)
    AccessCode.objects.create(
        user=user,
        digest=digest(token.encode()),
        purpose=purpose,
        expires=now + timedelta(hours=hours),
    )
    return token


@require_http_methods(["GET", "POST"])
def recover(request):
    if request.user.is_authenticated:
        return redirect("home")
    error = ""
    if request.method == "POST":
        if over_limit("recovery", request.META.get("REMOTE_ADDR", "local"), 12):
            return render(
                request, "auth.html", {"mode": "recover", "error": "Try again later."}, status=429
            )
        token = request.POST.get("token", "").strip()[:100]
        password = request.POST.get("password", "")
        if password != request.POST.get("confirm") or len(password) > 256:
            error = "Use matching passwords of at most 256 characters."
        else:
            with transaction.atomic():
                code = (
                    AccessCode.objects.select_for_update()
                    .filter(
                        digest=digest(token.encode()),
                        expires__gt=timezone.now(),
                        used__isnull=True,
                    )
                    .select_related("user")
                    .first()
                )
                if code is None or (not code.user.is_active and code.purpose == "reset"):
                    error = "This access code is unavailable. Ask your administrator for a new one."
                else:
                    user = code.user
                    try:
                        validate_password(password, user)
                    except ValidationError as problem:
                        error = " ".join(problem.messages)
                    else:
                        user.set_password(password)
                        user.is_active = True
                        user.session_epoch += 1
                        user.save(update_fields=["password", "is_active", "session_epoch"])
                        code.used = timezone.now()
                        code.save(update_fields=["used"])
                        audit(
                            "account_activated" if code.purpose == "invite" else "password_reset",
                            user,
                        )
                        messages.success(request, "Password saved. Sign in to continue.")
                        return redirect("sign_in")
    return render(request, "auth.html", {"mode": "recover", "error": error})
