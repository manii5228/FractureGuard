from io import BytesIO
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as PDFImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

NOTICE = (
    "University research prototype. For demonstration only. Do not use for diagnosis or treatment."
)


def make_pdf(snapshot, preview_by_view):
    output = BytesIO()
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#153d43")
    story = [Paragraph("FractureGuard", styles["Title"])]
    fields = (
        ("Case", snapshot["reference"]),
        (
            "Version",
            f"Study revision {snapshot['revision']} | Report version {snapshot['version']}",
        ),
        ("Examination", "Wrist radiographs | AP/PA and lateral | " + snapshot["side"]),
        ("Technical limitations", snapshot["limitations"] or "None recorded"),
        ("Reviewer findings", snapshot["findings"]),
        ("Reviewer notes", snapshot["notes"] or "None recorded"),
        ("Final impression", snapshot["impression"]),
        ("Signed by", snapshot["signer"] + " | " + snapshot["signed_at"]),
    )
    for label, value in fields:
        story.extend(
            [
                Paragraph(label, styles["Heading3"]),
                Paragraph(escape(value).replace("\n", "<br/>"), styles["BodyText"]),
            ]
        )
    story.append(Spacer(1, 12))
    story.append(Paragraph(escape(NOTICE), styles["Italic"]))
    for view, label in (("frontal", "AP/PA"), ("lateral", "Lateral")):
        approved = [box for box in snapshot["boxes"] if box["view"] == view]
        if approved:
            with Image.open(BytesIO(preview_by_view[view])) as source:
                image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            for box in approved:
                draw.rectangle(box["coords"], outline="#47e8d2", width=max(2, image.width // 200))
            scale = min(240 / image.width, 290 / image.height)
            rendered = BytesIO()
            image.save(rendered, format="PNG")
            rendered.seek(0)
            story.extend(
                [
                    Paragraph(f"{label} reviewer annotations", styles["Heading3"]),
                    PDFImage(rendered, width=image.width * scale, height=image.height * scale),
                ]
            )
    SimpleDocTemplate(output, title=f"FractureGuard {snapshot['reference']}").build(story)
    return output.getvalue()
