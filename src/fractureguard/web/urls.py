from pathlib import Path

from django.http import FileResponse, Http404
from django.shortcuts import render
from django.urls import path

from . import accounts, cases, operations


def static_file(request, name):
    types = {"style.css": "text/css", "app.js": "application/javascript"}
    if name not in types:
        raise Http404
    source = Path(__file__).parent / "static" / name
    return FileResponse(source.open("rb"), content_type=types[name])


def client_error(request, exception=None, status=400):
    labels = {
        400: "The request could not be read.",
        403: "You do not have access to this action.",
        404: "This page is unavailable.",
        500: "The request could not be completed.",
    }
    return render(request, "error.html", {"error": labels[status]}, status=status)


handler400 = client_error
handler403 = lambda request, exception: client_error(request, exception, 403)  # noqa: E731
handler404 = lambda request, exception: client_error(request, exception, 404)  # noqa: E731
handler500 = lambda request: client_error(request, status=500)  # noqa: E731

urlpatterns = [
    path("", cases.home, name="home"),
    path("login/", accounts.sign_in, name="sign_in"),
    path("login/verify/", accounts.second_factor, name="second_factor"),
    path("account/access/", accounts.recover, name="recover"),
    path("logout/", accounts.sign_out, name="sign_out"),
    path("cases/new/", cases.new_case, name="new_case"),
    path("cases/<uuid:case_id>/", cases.case_detail, name="case_detail"),
    path("cases/<uuid:case_id>/status/", cases.case_status, name="case_status"),
    path("cases/<uuid:case_id>/deletion/", cases.request_deletion, name="request_deletion"),
    path("cases/<uuid:case_id>/report/", cases.released_report, name="released_report"),
    path(
        "cases/<uuid:case_id>/report.pdf",
        cases.released_report,
        {"download": True},
        name="download",
    ),
    path("review/", cases.queue, name="queue"),
    path("review/<uuid:case_id>/claim/", cases.claim, name="claim"),
    path("review/<uuid:case_id>/", cases.review, name="review"),
    path("review/<uuid:case_id>/save/", cases.save_review, name="save_review"),
    path("review/<uuid:case_id>/return/", cases.return_case, name="return_case"),
    path("review/<uuid:case_id>/retry/", cases.retry_case, name="retry_case"),
    path("review/<uuid:case_id>/report/", cases.report_preview, name="report_preview"),
    path("review/<uuid:case_id>/report.pdf", cases.report_preview, {"pdf": True}, name="staff_pdf"),
    path("review/<uuid:case_id>/sign/", cases.sign_report, name="sign_report"),
    path("review/<uuid:case_id>/release/", cases.release_report, name="release_report"),
    path("review/<uuid:case_id>/withdraw/", cases.withdraw_report, name="withdraw_report"),
    path("media/<uuid:asset_id>/", cases.asset_preview, name="asset_preview"),
    path("operations/", operations.operations, name="operations"),
    path("static/<str:name>", static_file, name="static_file"),
]
