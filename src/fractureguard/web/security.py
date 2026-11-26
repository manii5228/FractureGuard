from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import timedelta
from functools import lru_cache, wraps

import pyotp
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from django.conf import settings
from django.contrib.auth import logout
from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.http import HttpResponse
from django.shortcuts import redirect
from django.utils import timezone
from django.utils.crypto import salted_hmac

from .models import Audit, RateWindow, User


@lru_cache(maxsize=1)
def cipher(key):
    return AESGCM(base64.urlsafe_b64decode(key))


def seal(data: bytes, context: str) -> bytes:
    nonce = os.urandom(12)
    return nonce + cipher(settings.DATA_KEY).encrypt(nonce, data, context.encode())


def unseal(data: bytes, context: str) -> bytes:
    return cipher(settings.DATA_KEY).decrypt(data[:12], data[12:], context.encode())


def canonical(data) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_payload(data, context, default=None):
    return json.loads(unseal(data, context)) if data else default


def audit(action, actor=None, case=None, *, result="ok", version=None, detail=None):
    return Audit.objects.create(
        actor=actor,
        action=action,
        reference=case.reference if case else "",
        result=result,
        version=version if version is not None else (case.revision if case else None),
        detail=detail or {},
    )


@transaction.atomic
def over_limit(bucket, value, limit, seconds=600):
    key = salted_hmac("fractureguard.web.rate", f"{bucket}:{value}", algorithm="sha256").hexdigest()
    now = timezone.now()
    row, _ = RateWindow.objects.get_or_create(key=key, defaults={"starts": now})
    if row.starts <= now - timedelta(seconds=seconds):
        row.starts, row.hits = now, 0
    if row.hits >= limit:
        return True
    row.hits += 1
    row.save(update_fields=["hits", "starts"])
    return False


def role_required(*roles):
    def decorate(view):
        @wraps(view)
        def check(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect("sign_in")
            if request.user.role not in roles:
                raise PermissionDenied
            return view(request, *args, **kwargs)

        return check

    return decorate


@transaction.atomic
def consume_otp(user, code, *, enrollment=None):
    current = User.objects.select_for_update().get(pk=user.pk)
    secret = (
        unseal(current.otp_secret, f"otp:{current.pk}").decode()
        if current.otp_secret
        else enrollment
    )
    if not secret or len(code) != 6 or not code.isascii() or not code.isdigit():
        return False
    generator = pyotp.TOTP(secret)
    step = int(timezone.now().timestamp()) // generator.interval
    if step <= current.otp_step or not hmac.compare_digest(generator.generate_otp(step), code):
        return False
    current.otp_secret = seal(secret.encode(), f"otp:{current.pk}")
    current.otp_step = step
    current.save(update_fields=["otp_secret", "otp_step"])
    return True


class ResponseAndSizeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            length = int(request.META.get("CONTENT_LENGTH") or 0)
        except ValueError:
            return HttpResponse("Invalid request size.", status=400)
        if length < 0 or length > settings.MAX_BODY_BYTES:
            return HttpResponse("Upload exceeds the allowed request size.", status=413)
        response = self.get_response(request)
        response["Cache-Control"] = "private, no-store"
        response["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
            "connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        response["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


class SessionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            now = int(timezone.now().timestamp())
            last = request.session.get("last_activity", 0)
            start = request.session.get("signed_in_at", 0)
            if (
                request.session.get("epoch") != request.user.session_epoch
                or now - last > 1800
                or now - start > settings.SESSION_COOKIE_AGE
            ):
                logout(request)
            elif now - last >= 60:
                request.session["last_activity"] = now
        return self.get_response(request)
