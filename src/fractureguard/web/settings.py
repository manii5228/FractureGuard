import base64
import binascii
import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

from fractureguard.paths import PROJECT_ROOT

SHARED = os.getenv("FG_SHARED") == "1"
DEBUG = False
WEB_ROOT = Path(os.getenv("FG_WEB_ROOT", PROJECT_ROOT / "work" / "m7"))
DELETION_ROOT = Path(os.getenv("FG_DELETION_DIR", PROJECT_ROOT / "work" / "m7-deletions"))
key_root = Path(os.getenv("FG_WEB_KEYS", PROJECT_ROOT / "work" / "m7-keys"))


def secret(variable, filename):
    if value := os.getenv(variable):
        return value
    if SHARED:
        raise ImproperlyConfigured(f"{variable} is required for shared deployment")
    try:
        return (key_root / filename).read_text(encoding="ascii").strip()
    except FileNotFoundError as error:
        raise ImproperlyConfigured("Run python -m fractureguard.web migrate first") from error


SECRET_KEY = secret("FG_DJANGO_KEY", "django")
DATA_KEY = secret("FG_DATA_KEY", "storage")
try:
    if len(base64.b64decode(DATA_KEY, altchars=b"-_", validate=True)) != 32:
        raise ValueError
except (binascii.Error, ValueError) as error:
    raise ImproperlyConfigured("FG_DATA_KEY must encode exactly 32 random bytes") from error
ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("FG_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",")
    if host.strip()
]
if SHARED and (
    os.getenv("FG_ENCRYPTED_VOLUME") != "1"
    or not (os.getenv("FG_ALLOWED_HOSTS") and ALLOWED_HOSTS)
    or not os.getenv("FG_DELETION_DIR")
    or "*" in ALLOWED_HOSTS
    or DELETION_ROOT.resolve().is_relative_to(WEB_ROOT.resolve())
    or len(SECRET_KEY) < 50
):
    raise ImproperlyConfigured(
        "Shared mode requires named hosts, strong keys and separate encrypted storage"
    )
SECURE_SSL_REDIRECT = SHARED
SECURE_HSTS_SECONDS = 31536000 if SHARED else 0
if SHARED and os.getenv("FG_TRUST_TLS_PROXY") == "1":
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = CSRF_COOKIE_SECURE = SHARED
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = CSRF_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_AGE = 8 * 60 * 60
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"

ROOT_URLCONF = "fractureguard.web.urls"
WSGI_APPLICATION = "fractureguard.web.wsgi.application"
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "fractureguard.web.apps.WebConfig",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "fractureguard.web.security.ResponseAndSizeMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "fractureguard.web.security.SessionMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": WEB_ROOT / "fractureguard.sqlite3",
        "OPTIONS": {"timeout": 15, "transaction_mode": "IMMEDIATE"},
    }
}
AUTH_USER_MODEL = "web.User"
PASSWORD_HASHERS = ["django.contrib.auth.hashers.Argon2PasswordHasher"]
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": f"django.contrib.auth.password_validation.{name}"}
    for name in (
        "UserAttributeSimilarityValidator",
        "MinimumLengthValidator",
        "CommonPasswordValidator",
        "NumericPasswordValidator",
    )
]
LOGIN_URL = "sign_in"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
USE_TZ = True
TIME_ZONE = "UTC"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BODY_BYTES = 2 * MAX_UPLOAD_BYTES + 65536
FILE_UPLOAD_HANDLERS = ["django.core.files.uploadhandler.MemoryFileUploadHandler"]
FILE_UPLOAD_MAX_MEMORY_SIZE = MAX_BODY_BYTES
DATA_UPLOAD_MAX_MEMORY_SIZE = 65536
DATA_UPLOAD_MAX_NUMBER_FILES = 2

MODEL_NAME = "YOLO26n FP32"
MODEL_PATH = PROJECT_ROOT / "reports" / "models" / "m6" / "yolo26n_fp32.onnx"
MODEL_SHA = "165118a5d394f964c447e5d0116a1982bff678b65637dda9467765c329110de2"
POLICY_PATH = PROJECT_ROOT / "reports" / "predictions" / "m4" / "m4_frozen_triage_policy.json"
POLICY_SHA = "3b7ea8e68562b039ec89a872d62029ac43a4f617c93e4ed20aca3415b17c2527"
PREPROCESSING = "percentile_1_99_rect640_v1"
