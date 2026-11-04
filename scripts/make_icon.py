"""Draw the ViPr app icon.  Needs Pillow.   python scripts/make_icon.py

Same look as the brand mark inside the app: the soft accent blue (#8AB4F8 family) rounded square with the
dark waveform from the sidebar icon. Output (assets/icon/):
  vipr.svg          vector source of the same drawing
  vipr-1024.png     master
  vipr-256.png      used as the window / taskbar icon inside the app
  vipr.ico          Windows icon with 16, 24, 32, 48, 64, 128 and 256 px (embedded into vipr.exe by build.rs)

Small sizes get thicker bars so the glyph stays readable at 16 px.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "assets" / "icon"

TOP = (160, 196, 252)  # gradient top    (#A0C4FC)
BOTTOM = (104, 148, 232)  # gradient bottom (#6894E8)
INK = (21, 23, 28)  # same dark as the app background (#15171C)

# the sidebar waveform: bars at x = 2, 6, 10, 14, 18, 22 with these (top, bottom) in a 24 x 24 grid
BARS = [(10, 13), (6, 17), (3, 21), (8, 16), (5, 18), (10, 13)]
BAR_X = [2, 6, 10, 14, 18, 22]


def lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def draw(size: int) -> Image.Image:
    """Render one icon at `size` px (4x supersampled, then reduced)."""
    ss = 4
    s = size * ss
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))

    # gradient square, clipped to a rounded rectangle
    grad = Image.new("RGBA", (s, s))
    gp = grad.load()
    for y in range(s):
        t = y / (s - 1)
        row = (
            lerp(TOP[0], BOTTOM[0], t),
            lerp(TOP[1], BOTTOM[1], t),
            lerp(TOP[2], BOTTOM[2], t),
            255,
        )
        for x in range(s):
            gp[x, y] = row
    mask = Image.new("L", (s, s), 0)
    margin = s * 0.03
    ImageDraw.Draw(mask).rounded_rectangle(
        [margin, margin, s - margin, s - margin], radius=s * 0.225, fill=255
    )
    img.paste(grad, (0, 0), mask)

    # waveform
    d = ImageDraw.Draw(img)
    glyph_w = s * (0.60 if size >= 64 else 0.66)
    unit = glyph_w / 22.0
    thick = (2.7 if size >= 64 else 3.5) * unit
    cx, cy = s / 2, s / 2
    for x, (y0, y1) in zip(BAR_X, BARS):
        bx = cx + (x - 12) * unit
        half = ((y1 - y0) / 2 + 1.1) * unit  # round caps add a little length
        mid = cy + ((y0 + y1) / 2 - 12) * unit
        d.rounded_rectangle(
            [bx - thick / 2, mid - half, bx + thick / 2, mid + half],
            radius=thick / 2,
            fill=INK + (255,),
        )

    return img.resize((size, size), Image.LANCZOS)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def write_ico(path: Path, sizes: list[int]) -> None:
    """ICO with PNG-compressed entries (Windows Vista and newer)."""
    blobs = [(n, png_bytes(draw(n))) for n in sizes]
    head = struct.pack("<HHH", 0, 1, len(blobs))
    offset = 6 + 16 * len(blobs)
    entries = b""
    data = b""
    for n, blob in blobs:
        entries += struct.pack(
            "<BBBBHHII", n % 256, n % 256, 0, 0, 1, 32, len(blob), offset + len(data)
        )
        data += blob
    path.write_bytes(head + entries + data)


def svg() -> str:
    unit = 600 / 22.0
    bars = []
    for x, (y0, y1) in zip(BAR_X, BARS):
        bx = 512 + (x - 12) * unit
        half = ((y1 - y0) / 2 + 1.1) * unit
        mid = 512 + ((y0 + y1) / 2 - 12) * unit
        bars.append(
            f'<line x1="{bx:.1f}" y1="{mid - half + 1.35 * unit:.1f}" x2="{bx:.1f}" y2="{mid + half - 1.35 * unit:.1f}"/>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#A0C4FC"/><stop offset="1" stop-color="#6894E8"/></linearGradient></defs>'
        '<rect x="31" y="31" width="962" height="962" rx="230" fill="url(#g)"/>'
        f'<g stroke="#15171C" stroke-width="{2.7 * unit:.1f}" stroke-linecap="round">{"".join(bars)}</g></svg>\n'
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vipr-1024.png").write_bytes(png_bytes(draw(1024)))
    (OUT / "vipr-256.png").write_bytes(png_bytes(draw(256)))
    write_ico(OUT / "vipr.ico", [16, 24, 32, 48, 64, 128, 256])
    (OUT / "vipr.svg").write_text(svg(), encoding="utf-8")
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:16} {f.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
