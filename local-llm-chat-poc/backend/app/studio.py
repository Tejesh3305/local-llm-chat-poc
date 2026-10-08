"""AI Studio: turn text into PDF, Word and PowerPoint files, draw demo images, and describe documents.

None of this needs a model. The model only drafts or analyses text (see main.py); everything here
is plain Python, so the files come out the same with or without one.
"""

import hashlib
import io
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document as WordDocument
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor
from fpdf import FPDF
from pptx import Presentation
from pptx.dml.color import RGBColor as SlideColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Emu, Inches
from pptx.util import Pt as SlidePt

from .corpus import STOP_WORDS

BRAND = "SAINIK AI"
UNIT = "14 Corps Battlefield Intelligence System"
FOOTER = "Generated on this machine by SAINIK AI. Sample output for demonstration."

OLIVE = (0x4B, 0x5A, 0x2A)
DARK = (0x2B, 0x34, 0x20)
BRASS = (0xD3, 0xBD, 0x6E)
INK = (0x1B, 0x1F, 0x16)
MUTED = (0x69, 0x6F, 0x5C)

FORMATS = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "png": "image/png",  # pictures drawn in the chat
    "svg": "image/svg+xml",
    "jpg": "image/jpeg",  # photos sent with a chat message
    "webp": "image/webp",
}


# ---------- text -> outline ----------


@dataclass
class Section:
    heading: str
    blocks: list[tuple[str, str]] = field(default_factory=list)  # ("p" | "bullet", text)


@dataclass
class Outline:
    title: str
    sections: list[Section]

    def as_dict(self) -> dict:
        return {"title": self.title, "sections": [{"heading": s.heading, "blocks": [list(b) for b in s.blocks]} for s in self.sections]}


def outline(title: str, content: str) -> Outline:
    """Markdown-ish text as title and sections: '#'/'##' lines start sections, '-', '*' and '1.' lines are bullets."""
    sections: list[Section] = []
    paragraph: list[str] = []

    def current() -> Section:
        if not sections:
            sections.append(Section(""))
        return sections[-1]

    def flush() -> None:
        if paragraph:
            current().blocks.append(("p", " ".join(paragraph)))
            paragraph.clear()

    for raw in content.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        bullet = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)$", line)
        if not line:
            flush()
        elif heading:
            flush()
            text = heading.group(2).strip().strip("*")
            if len(heading.group(1)) == 1 and not title.strip() and not sections:
                title = text  # a leading "# Title" is the document title
            else:
                sections.append(Section(text))
        elif bullet:
            flush()
            current().blocks.append(("bullet", bullet.group(1).strip()))
        elif re.fullmatch(r"-{3,}|\*{3,}|_{3,}", line):
            flush()
        else:
            paragraph.append(line)
    flush()
    sections = [s for s in sections if s.heading or s.blocks]
    return Outline(title.strip() or "Untitled document", sections)


def runs(text: str) -> list[tuple[str, bool]]:
    """Inline text as (text, bold) pieces; other markdown marks are dropped."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)([^*]+)\*(?!\*)", r"\1", text)  # *italic*
    pieces = re.split(r"\*\*([^*]+)\*\*", text)
    return [(piece, index % 2 == 1) for index, piece in enumerate(pieces) if piece]


def plain(text: str) -> str:
    return "".join(piece for piece, _ in runs(text))


def stamp() -> str:
    return datetime.now().strftime("%d %b %Y, %H:%M")


def filename(title: str, extension: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")[:60] or "document"
    return f"{slug}.{extension}"


# ---------- PDF ----------

# A Unicode font from the system, so text beyond Latin prints; Devanagari needs text shaping, which this demo leaves out.
FONT_CANDIDATES = [
    (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    (Path("/System/Library/Fonts/Supplemental/Arial.ttf"), Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")),
]


class _Report(FPDF):
    def __init__(self, title: str):
        super().__init__(format="A4")
        self.doc_title = title
        self.family = "Helvetica"
        for regular, bold in FONT_CANDIDATES:
            if regular.exists() and bold.exists():
                self.add_font("Body", "", str(regular))
                self.add_font("Body", "B", str(bold))
                self.family = "Body"
                break
        self.set_auto_page_break(True, margin=22)
        self.set_margins(20, 26, 20)

    def text_ok(self, text: str) -> str:
        return text if self.family == "Body" else text.encode("latin-1", "replace").decode("latin-1")

    def header(self):
        self.set_fill_color(*DARK)
        self.rect(0, 0, self.w, 14, "F")
        self.set_fill_color(*BRASS)
        self.rect(0, 14, self.w, 1.2, "F")
        self.set_xy(20, 4.2)
        self.set_text_color(255, 255, 255)
        self.set_font(self.family, "B", 10)
        self.cell(40, 6, BRAND)
        self.set_text_color(*BRASS)
        self.set_font(self.family, "", 7.5)
        self.cell(0, 6, UNIT.upper(), align="R")
        self.set_y(26)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(*MUTED)
        self.line(20, self.get_y(), self.w - 20, self.get_y())
        self.set_text_color(*MUTED)
        self.set_font(self.family, "", 7.5)
        self.cell(0, 8, self.text_ok(f"{FOOTER}   {stamp()}"), align="L")
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", align="R")


def to_pdf(doc: Outline) -> bytes:
    pdf = _Report(doc.title)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_text_color(*INK)
    pdf.set_font(pdf.family, "B", 20)
    pdf.multi_cell(0, 9, pdf.text_ok(doc.title))
    pdf.set_draw_color(*OLIVE)
    pdf.set_line_width(0.8)
    pdf.line(20, pdf.get_y() + 2, 45, pdf.get_y() + 2)
    pdf.set_line_width(0.2)
    pdf.ln(7)
    pdf.set_text_color(*MUTED)
    pdf.set_font(pdf.family, "", 9)
    headed = sum(1 for section in doc.sections if section.heading)
    pdf.cell(0, 5, pdf.text_ok(f"Prepared {stamp()}  ·  {headed} {'section' if headed == 1 else 'sections'}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for section in doc.sections:
        if section.heading:
            pdf.ln(2)
            pdf.set_text_color(*OLIVE)
            pdf.set_font(pdf.family, "B", 13)
            pdf.multi_cell(0, 7, pdf.text_ok(plain(section.heading)))
            pdf.ln(1)
        pdf.set_text_color(*INK)
        pdf.set_font(pdf.family, "", 10.5)
        for kind, text in section.blocks:
            body = pdf.text_ok(re.sub(r"`([^`]*)`", r"\1", text))
            if kind == "bullet":
                pdf.set_x(24)
                pdf.cell(5, 5.6, "•" if pdf.family == "Body" else "-")
                pdf.multi_cell(0, 5.6, body, markdown=True)
            else:
                pdf.multi_cell(0, 5.6, body, markdown=True)
                pdf.ln(1.5)
        pdf.ln(2)
    return bytes(pdf.output())


# ---------- Word ----------


def _word_runs(paragraph, text: str) -> None:
    for piece, bold in runs(text):
        paragraph.add_run(piece).bold = bold


def to_docx(doc: Outline) -> bytes:
    word = WordDocument()
    styles = word.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"].font.size = Pt(11)
    for name, size in (("Title", 24), ("Heading 1", 14)):
        styles[name].font.color.rgb = RGBColor(*OLIVE)
        styles[name].font.size = Pt(size)

    section = word.sections[0]
    head = section.header.paragraphs[0]
    head.text = f"{BRAND}  ·  {UNIT}"
    head.runs[0].font.size = Pt(8)
    head.runs[0].font.color.rgb = RGBColor(*MUTED)
    foot = section.footer.paragraphs[0]
    foot.text = f"{FOOTER}  {stamp()}"
    foot.runs[0].font.size = Pt(8)
    foot.runs[0].font.color.rgb = RGBColor(*MUTED)

    word.add_paragraph(doc.title, style="Title")
    meta = word.add_paragraph(f"Prepared {stamp()}")
    meta.runs[0].font.color.rgb = RGBColor(*MUTED)
    meta.runs[0].font.size = Pt(9)

    for part in doc.sections:
        if part.heading:
            word.add_heading(plain(part.heading), level=1)
        for kind, text in part.blocks:
            paragraph = word.add_paragraph(style="List Bullet" if kind == "bullet" else None)
            _word_runs(paragraph, text)
            if kind == "p":
                paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    buffer = io.BytesIO()
    word.save(buffer)
    return buffer.getvalue()


# ---------- PowerPoint ----------

BULLETS_PER_SLIDE = 6


def slides(doc: Outline) -> list[dict]:
    """The deck as data: a title slide, then one slide per section (long sections are split)."""
    deck = [{"kind": "title", "title": doc.title, "subtitle": f"{UNIT}  ·  {stamp()}"}]
    for section in doc.sections:
        points = [plain(text) for _, text in section.blocks] or [""]
        chunks = [points[i : i + BULLETS_PER_SLIDE] for i in range(0, len(points), BULLETS_PER_SLIDE)]
        for index, chunk in enumerate(chunks):
            heading = plain(section.heading) or doc.title
            deck.append({"kind": "content", "title": heading + (f" ({index + 1}/{len(chunks)})" if len(chunks) > 1 else ""), "points": [p for p in chunk if p]})
    deck.append({"kind": "end", "title": "Questions", "subtitle": FOOTER})
    return deck


def _fill(shape, colour) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = SlideColor(*colour)
    shape.line.fill.background()


def to_pptx(doc: Outline) -> bytes:
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
    width, height = deck.slide_width, deck.slide_height
    blank = deck.slide_layouts[6]

    def text(slide, left, top, w, h, value, size, colour, bold=False):
        frame = slide.shapes.add_textbox(left, top, w, h).text_frame
        frame.word_wrap = True
        run = frame.paragraphs[0].add_run()
        run.text = value
        run.font.size, run.font.bold = SlidePt(size), bold
        run.font.color.rgb = SlideColor(*colour)
        return frame

    for number, item in enumerate(slides(doc), start=1):
        slide = deck.slides.add_slide(blank)
        if item["kind"] in ("title", "end"):
            _fill(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, width, height), DARK)
            _fill(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(3.55), Inches(1.6), Emu(60000)), BRASS)
            text(slide, Inches(0.8), Inches(1.0), Inches(11), Inches(0.5), BRAND, 16, BRASS, bold=True)
            text(slide, Inches(0.8), Inches(1.9), Inches(11.5), Inches(1.6), item["title"], 40, (255, 255, 255), bold=True)
            text(slide, Inches(0.8), Inches(3.8), Inches(11.5), Inches(0.8), item["subtitle"], 16, (0xD2, 0xD3, 0xBF))
            continue
        _fill(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, width, Inches(1.1)), DARK)
        _fill(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.1), width, Emu(50000)), BRASS)
        text(slide, Inches(0.6), Inches(0.22), Inches(11), Inches(0.7), item["title"], 26, (255, 255, 255), bold=True)
        body = slide.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.0)).text_frame
        body.word_wrap = True
        for index, point in enumerate(item["points"]):
            paragraph = body.paragraphs[0] if index == 0 else body.add_paragraph()
            paragraph.space_after = SlidePt(12)
            marker = paragraph.add_run()
            marker.text = "▪  "
            marker.font.size, marker.font.color.rgb = SlidePt(20), SlideColor(*OLIVE)
            run = paragraph.add_run()
            run.text = point
            run.font.size, run.font.color.rgb = SlidePt(20), SlideColor(*INK)
        text(slide, Inches(0.6), Inches(6.9), Inches(9), Inches(0.4), f"{BRAND}  ·  {doc.title}", 10, MUTED)
        text(slide, Inches(11.8), Inches(6.9), Inches(1), Inches(0.4), str(number), 10, MUTED)

    buffer = io.BytesIO()
    deck.save(buffer)
    return buffer.getvalue()


RENDERERS = {"pdf": to_pdf, "docx": to_docx, "pptx": to_pptx}


# ---------- demo images ----------

PALETTES = {
    "snow": ("#dfe8f0", "#9fb3c8", "#5b7491", "#2f4259", "#f7fafc"),
    "night": ("#0b1320", "#1d2a3a", "#2f4255", "#4b6278", "#e8e4c9"),
    "desert": ("#f3d9a4", "#d9a85b", "#b77b39", "#7d4f22", "#fff4d8"),
    "forest": ("#cfe0c3", "#7fa36b", "#4b6e3b", "#2a4222", "#f2f6ea"),
    "default": ("#e9e3cf", "#b8b18a", "#7d7a52", "#4b5a2a", "#faf7ea"),
}
KEYWORDS = {
    "snow": ("snow", "winter", "ice", "glacier", "himalaya", "cold", "frost", "ladakh", "siachen"),
    "night": ("night", "dark", "moon", "midnight", "stars"),
    "desert": ("desert", "sand", "dune", "thar", "arid"),
    "forest": ("forest", "jungle", "green", "valley", "river", "monsoon"),
}
STYLES = ("terrain map", "landscape", "poster")


def palette_for(prompt: str) -> str:
    words = prompt.lower()
    return next((name for name, hints in KEYWORDS.items() if any(hint in words for hint in hints)), "default")


def _blob(rng: random.Random, cx: float, cy: float, radius: float, points: int = 28) -> str:
    """A closed, irregular outline around (cx, cy)."""
    phase = [rng.uniform(0, math.tau) for _ in range(3)]
    path = []
    for i in range(points):
        angle = math.tau * i / points
        r = radius * (1 + 0.18 * math.sin(2 * angle + phase[0]) + 0.1 * math.sin(3 * angle + phase[1]) + 0.06 * math.sin(5 * angle + phase[2]))
        path.append(f"{cx + r * math.cos(angle):.1f},{cy + r * math.sin(angle):.1f}")
    return "M" + " L".join(path) + " Z"


def _ridge(rng: random.Random, base: float, height: float, width: int) -> str:
    x, points = 0.0, [f"0,{base:.0f}"]
    while x < width:
        x += rng.uniform(30, 90)
        points.append(f"{min(x, width):.0f},{base - rng.uniform(0.25, 1) * height:.0f}")
    return "M" + " L".join(points) + f" L{width},{base + 400} L0,{base + 400} Z"


def demo_image(prompt: str, style: str) -> tuple[str, dict]:
    """A drawing that follows the prompt's words: same prompt, same picture. Returns the SVG and what was used."""
    width, height = 1024, 640
    seed = int(hashlib.sha256(f"{style}|{prompt.strip().lower()}".encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)
    name = palette_for(prompt)
    light, mid, deep, darkest, paper = PALETTES[name]
    parts: list[str] = []

    if style == "terrain map":
        parts.append(f'<rect width="{width}" height="{height}" fill="{paper}"/>')
        peaks = [(rng.uniform(150, 870), rng.uniform(120, 520), rng.uniform(120, 260)) for _ in range(rng.randint(2, 4))]
        shades = [light, mid, deep, darkest]
        for cx, cy, size in peaks:
            for ring in range(7):
                r = size * (1 - ring / 7)
                fill = shades[min(ring // 2, 3)]
                parts.append(f'<path d="{_blob(rng, cx, cy, r)}" fill="{fill}" fill-opacity="{0.18 + ring * 0.06:.2f}" stroke="{darkest}" stroke-opacity=".35" stroke-width="1"/>')
        for i in range(1, 8):  # grid squares with references
            x, y = i * width / 8, i * height / 8
            parts.append(f'<line x1="{x:.0f}" y1="0" x2="{x:.0f}" y2="{height}" stroke="{darkest}" stroke-opacity=".18"/>')
            parts.append(f'<line x1="0" y1="{y:.0f}" x2="{width}" y2="{y:.0f}" stroke="{darkest}" stroke-opacity=".18"/>')
            parts.append(f'<text x="{x + 4:.0f}" y="14" font-size="11" fill="{darkest}" fill-opacity=".6">{40 + i}</text>')
        route = " ".join(f"{20 + i * 140 + rng.uniform(-30, 30):.0f},{height * 0.75 - i * 55 + rng.uniform(-40, 40):.0f}" for i in range(8))
        parts.append(f'<polyline points="{route}" fill="none" stroke="#a8281f" stroke-width="4" stroke-dasharray="14 8" stroke-linecap="round"/>')
        for _ in range(4):  # position markers
            x, y = rng.uniform(80, 940), rng.uniform(80, 560)
            parts.append(f'<rect x="{x - 14:.0f}" y="{y - 9:.0f}" width="28" height="18" fill="{paper}" stroke="#2c5d9c" stroke-width="2.5"/>')
            parts.append(f'<line x1="{x - 14:.0f}" y1="{y - 9:.0f}" x2="{x + 14:.0f}" y2="{y + 9:.0f}" stroke="#2c5d9c" stroke-width="2"/>')
    elif style in ("landscape", "photo"):  # the demo has no photos; a landscape is the closest it draws
        parts.append(f'<defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{darkest if name == "night" else mid}"/><stop offset="1" stop-color="{light}"/></linearGradient></defs>')
        parts.append(f'<rect width="{width}" height="{height}" fill="url(#sky)"/>')
        parts.append(f'<circle cx="{rng.uniform(150, 870):.0f}" cy="{rng.uniform(90, 200):.0f}" r="{rng.uniform(30, 55):.0f}" fill="{paper}" fill-opacity=".9"/>')
        for layer, colour in enumerate((mid, deep, darkest)):
            base = height * (0.55 + layer * 0.14)
            parts.append(f'<path d="{_ridge(rng, base, 180 - layer * 40, width)}" fill="{colour}" fill-opacity="{0.75 + layer * 0.1:.2f}"/>')
        if name == "snow":
            parts.append(f'<path d="{_ridge(rng, height * 0.5, 60, width)}" fill="#ffffff" fill-opacity=".55"/>')
    else:  # poster
        parts.append(f'<rect width="{width}" height="{height}" fill="{darkest}"/>')
        for _ in range(9):
            parts.append(f'<circle cx="{rng.uniform(0, width):.0f}" cy="{rng.uniform(0, height):.0f}" r="{rng.uniform(40, 220):.0f}" fill="{rng.choice((light, mid, deep, "#d3bd6e"))}" fill-opacity="{rng.uniform(.15, .55):.2f}"/>')
        for i in range(5):
            parts.append(f'<rect x="0" y="{height * (0.62 + i * 0.05):.0f}" width="{width}" height="6" fill="#d3bd6e" fill-opacity="{0.5 - i * 0.08:.2f}"/>')

    caption = escape(prompt.strip()[:90] or "Untitled")
    parts.append(f'<rect x="0" y="{height - 46}" width="{width}" height="46" fill="#1c2216" fill-opacity=".82"/>')
    parts.append(f'<text x="20" y="{height - 18}" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#f2f1e3">{caption}</text>')
    parts.append(f'<text x="{width - 20}" y="{height - 18}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#d3bd6e">DEMO RENDER · NOT A GENERATIVE MODEL</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">{"".join(parts)}</svg>'
    return svg, {"style": style, "palette": name, "seed": seed % 100000, "width": width, "height": height}


# ---------- document insights ----------

DATE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})\b", re.I)


def insights(text: str) -> dict:
    """Figures that need no model: length, reading time, frequent terms, dates and numbers mentioned."""
    words = re.findall(r"[A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F'-]+", text)
    terms = Counter(w.lower() for w in words if len(w) > 3 and w.lower() not in STOP_WORDS)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return {
        "words": len(words),
        "characters": len(text),
        "sentences": len(sentences),
        "reading_minutes": max(1, round(len(words) / 220)),
        "keywords": [word for word, _ in terms.most_common(12)],
        "dates": list(dict.fromkeys(m.group(0) for m in DATE.finditer(text)))[:10],
        "numbers": len(re.findall(r"\b\d+(?:\.\d+)?\b", text)),
    }


# ---------- what the model is asked ----------

DRAFT_KINDS = {
    "report": "a formal written report with an introduction, findings and recommendations",
    "letter": "a formal letter or memorandum, with a subject line, body and closing",
    "briefing": "a briefing presentation: short sections of 3 to 6 bullet points each, no long paragraphs",
}

DRAFT_SYSTEM = (
    "You are SAINIK AI, a staff assistant running offline. Write clear, professional text in Markdown: "
    "start with '# ' and a title, use '## ' for section headings, '- ' for bullet points and **bold** for key terms. "
    "Do not invent precise figures, names or dates that the user did not give; write [placeholder] instead. "
    "Write in the language the request is written in."
)

ANALYSIS_TASKS = {
    "summary": ("Executive summary", "Write an executive summary of the document in 5 to 8 sentences, then a one-line bottom line."),
    "key_points": ("Key points", "List the 6 to 10 most important points of the document as bullet points, most important first."),
    "entities": ("People, places and dates", "List the people, organisations, places, dates and figures the document mentions, grouped under '## ' headings. Only list what is in the text."),
    "actions": ("Actions and risks", "List the actions, deadlines and responsibilities the document sets out, then any risks or open questions, under two '## ' headings."),
    "hindi": ("Summary in Hindi", "Write a summary of the document in Hindi (Devanagari script), 5 to 8 sentences."),
}

ANALYSIS_SYSTEM = (
    "You are SAINIK AI, analysing a document for a staff officer. Use only the document text you are given; "
    "if something is not in it, say so. Answer in Markdown."
)

IMAGE_TASKS = {
    "describe": ("Describe", "Describe this image in detail: what it shows, the setting, notable objects and anything unusual. Use short paragraphs."),
    "summary": ("Report summary", "Write a short report-style summary of this image: 3 to 5 bullet points a staff officer could paste into a report."),
    "text": ("Read text", "Read out any text, numbers, signs or labels visible in this image, exactly as written. If there is none, say so."),
}

CLASSIFY_PROMPT = (
    "Classify this image. Answer in this form:\n"
    "**Category:** one of: military vehicle, civilian vehicle, aircraft, watercraft, weapon, military personnel, civilians, "
    "military installation, building or infrastructure, terrain or landscape, document or map, animal, other.\n"
    "**Type:** the specific kind (for example main battle tank, armoured personnel carrier, pickup truck, attack helicopter).\n"
    "**Confidence:** high, medium or low, and why.\n"
    "**Key features:** 2 to 4 visible details that support the answer.\n"
    "Do not guess an exact model or country unless markings make it clear; say if it cannot be told from the picture."
)
IMAGE_TASKS["classify"] = ("Classify", CLASSIFY_PROMPT)

VISION_SYSTEM = (
    "You are SAINIK AI, analysing pictures for a staff officer. Say only what you can see in the picture; "
    "when unsure, say so and give your confidence. Be concise and use Markdown."
)
