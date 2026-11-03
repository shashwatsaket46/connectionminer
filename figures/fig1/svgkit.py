"""Minimal SVG writer used by fig1.py (copied from the paper build's svgkit.py; rendering/capture helpers removed).

Mirrors the PIL composers: same pixel canvas, PIL text anchors converted to SVG baselines with the real Liberation Sans
metrics, raster tiles as base64 <image>, matplotlib tiles captured as SVG (svg.fonttype none -> live <text>) and
inlined in their own <g> with ids prefixed so several tiles can live in one document.
"""
import base64
import contextlib
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import ImageFont

FD = Path(__file__).resolve().parent.parent / "fonts"      # figure_kit/fonts
FAMILY = "Liberation Sans, Arial, Helvetica, sans-serif"
SVGNS, XLINK = "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVGNS)
ET.register_namespace("xlink", XLINK)


def _font(size, bold=False, italic=False):
    s = {(False, False): "Regular", (True, False): "Bold", (False, True): "Italic", (True, True): "BoldItalic"}
    return ImageFont.truetype(str(FD / f"LiberationSans-{s[(bold, italic)]}.ttf"), size)


def rgb(c):
    return c if isinstance(c, str) else "#%02x%02x%02x" % tuple(c[:3])


def png_b64(im, fmt="PNG"):
    buf = io.BytesIO()
    if fmt == "JPEG":
        im.convert("RGB").save(buf, "JPEG", quality=95)
    else:
        im.save(buf, "PNG", optimize=True)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


class Doc:
    def __init__(self, w, h, title=""):
        self.w, self.h, self.out, self.n = w, h, [], 0
        self.title = title

    # ---------------------------------------------------------------- structure
    @contextlib.contextmanager
    def group(self, gid, transform=None):
        t = f' transform="{transform}"' if transform else ""
        self.out.append(f'<g id="{gid}"{t}>')
        yield
        self.out.append("</g>")

    def raw(self, s):
        self.out.append(s)

    # ---------------------------------------------------------------- shapes
    def frame(self, x, y, w, h, radius=26, width=4, color=(110, 110, 110), gid=None):
        """PIL rounded_rectangle([x+3, y+3, x+w-4, y+h-4], radius, outline, width): stroke drawn inward"""
        i = 3 + width / 2
        idt = f' id="{gid}"' if gid else ""
        self.out.append(f'<rect{idt} x="{x + i:.2f}" y="{y + i:.2f}" width="{w - 2 * i - 0:.2f}" height="{h - 2 * i:.2f}" '
                        f'rx="{radius - width / 2:.2f}" fill="none" stroke="{rgb(color)}" stroke-width="{width}"/>')

    def rounded(self, x0, y0, x1, y1, radius=26, width=4, color=(110, 110, 110), gid=None):
        """PIL rounded_rectangle([x0, y0, x1, y1]) with an inward stroke"""
        i = width / 2
        idt = f' id="{gid}"' if gid else ""
        self.out.append(f'<rect{idt} x="{x0 + i:.2f}" y="{y0 + i:.2f}" width="{x1 + 1 - x0 - width:.2f}" '
                        f'height="{y1 + 1 - y0 - width:.2f}" rx="{radius - i:.2f}" fill="none" stroke="{rgb(color)}" '
                        f'stroke-width="{width}"/>')

    def rect(self, x0, y0, x1, y1, fill="white", gid=None):
        idt = f' id="{gid}"' if gid else ""
        self.out.append(f'<rect{idt} x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="{rgb(fill)}"/>')

    def line(self, pts, color, width):
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.out.append(f'<polyline points="{p}" fill="none" stroke="{rgb(color)}" stroke-width="{width}"/>')

    def polygon(self, pts, fill):
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.out.append(f'<polygon points="{p}" fill="{rgb(fill)}"/>')

    def ellipse(self, x0, y0, x1, y1, fill):
        self.out.append(f'<ellipse cx="{(x0 + x1 + 1) / 2:.2f}" cy="{(y0 + y1 + 1) / 2:.2f}" rx="{(x1 + 1 - x0) / 2:.2f}" '
                        f'ry="{(y1 + 1 - y0) / 2:.2f}" fill="{rgb(fill)}"/>')

    # ---------------------------------------------------------------- text
    def text(self, x, y, s, size, fill=(25, 25, 25), bold=False, italic=False, anchor="la", gid=None):
        """PIL ImageDraw.text((x, y), s, anchor=anchor) -> SVG <text> at the same place.
        Horizontal l/m/r -> text-anchor; vertical a (ascender) / m (middle) / s (baseline) -> explicit baseline y."""
        f = _font(size, bold, italic)
        asc, desc = f.getmetrics()
        h, v = anchor[0], anchor[1]
        base = {"a": y + asc, "m": y + (asc - desc) / 2, "s": y, "d": y - desc}[v]
        ta = {"l": "start", "m": "middle", "r": "end"}[h]
        idt = f' id="{gid}"' if gid else ""
        st = (' font-weight="bold"' if bold else "") + (' font-style="italic"' if italic else "")
        self.out.append(f'<text{idt} x="{x:.2f}" y="{base:.2f}" font-family="{FAMILY}" font-size="{size}"{st} '
                        f'fill="{rgb(fill)}" text-anchor="{ta}" xml:space="preserve">{escape(s)}</text>')

    @staticmethod
    def textlength(s, size, bold=False, italic=False):
        return _font(size, bold, italic).getlength(s)

    # ---------------------------------------------------------------- images
    def image(self, im, x, y, w=None, h=None, gid=None, fmt="PNG"):
        w = im.width if w is None else w
        h = im.height if h is None else h
        idt = f' id="{gid}"' if gid else ""
        self.out.append(f'<image{idt} x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                        f'preserveAspectRatio="none" xlink:href="{png_b64(im, fmt)}"/>')

    def mpl(self, svg_bytes, x, y, scale, prefix, crop=None, gid=None):
        """inline a matplotlib SVG: its point coordinates * scale, origin at (x, y); crop=(x0, y0, x1, y1) in points
        is the part of the figure placed at (x, y) (the trim). No clipping: the figure background is transparent."""
        root = ET.fromstring(svg_bytes)
        for m in root.findall(f"{{{SVGNS}}}metadata"):
            root.remove(m)
        ids = {}
        for el in root.iter():
            if "id" in el.attrib:
                new = f"{prefix}_{el.attrib['id']}"
                ids[el.attrib["id"]] = new
                el.attrib["id"] = new
        body = "".join(ET.tostring(c, encoding="unicode") for c in root)

        def rep(m):
            return m.group(1) + ids.get(m.group(2), m.group(2)) + m.group(3)
        body = re.sub(r'(url\(#)([^)]+)(\))', rep, body)
        body = re.sub(r'(href="#)([^"]+)(")', rep, body)
        body = body.replace("font-family: 'Liberation Sans'", f"font-family: {FAMILY}")
        body = body.replace("font-family: 'DejaVu Sans'", f"font-family: {FAMILY}")
        body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.S)        # global *{} rule: inline below
        cx, cy = (crop[0], crop[1]) if crop else (0, 0)
        t = f"translate({x:.3f} {y:.3f}) scale({scale:.6f}) translate({-cx:.3f} {-cy:.3f})"
        self.out.append(f'<g id="{gid or prefix}" transform="{t}" style="stroke-linejoin: round; stroke-linecap: butt">'
                        f"{body}</g>")

    # ---------------------------------------------------------------- output
    def save(self, path):
        head = (f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="{SVGNS}" xmlns:xlink="{XLINK}" version="1.1" '
                f'width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">\n'
                f"<title>{escape(self.title)}</title>\n"
                f'<rect id="background" x="0" y="0" width="{self.w}" height="{self.h}" fill="#ffffff"/>\n')
        Path(path).write_text(head + "\n".join(self.out) + "\n</svg>\n", encoding="utf-8")
