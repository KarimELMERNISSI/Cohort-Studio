"""
Build the images of the documentation site (docs/assets/) from app screenshots and the brand assets.

    uv run python dev_scripts/make_site_assets.py <screenshots-dir>

The screenshots come from dev_scripts/site_screenshots.py: the real application on the tutorial cohort, a
1440x900 viewport captured at 2x. Crop boxes below are in CSS pixels of that viewport.

Outputs (docs/assets/):
    brand/wordmark-light.png, brand/wordmark-dark.png   the application wordmarks (assets/brand)
    brand/favicon-32.png, brand/icon-192.png, brand/apple-touch-icon.png   browser icons
    brand/icon-256.webp                                 the logo on the page
    shots/*.webp                                        crops used on the page
    og-image.png                                        1200x630 social preview
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.theme import APP_TAGLINE, BRAND  # noqa: E402

LOGO = ROOT / "assets" / "karim-app-logo.png"
BRAND_DIR = ROOT / "assets" / "brand"
OUT = ROOT / "docs" / "assets"
VIEWPORT_WIDTH = 1440

# output name -> (screenshot name, crop box (left, top, right, bottom) in CSS px or None, output width in px)
SHOTS: dict[str, tuple[str, tuple[int, int, int, int] | None, int]] = {
    "hero-survival": ("survival", None, 2240),
    "population": ("population", (650, 277, 1145, 792), 990),
    "quality": ("quality", (338, 68, 1376, 765), 1600),
    "mi-diagnostics": ("mi_diagnostics", (346, 88, 1382, 801), 1600),
    "plan": ("plan", (338, 104, 1376, 900), 1600),
    "cox": ("cox", (344, 134, 1381, 782), 1600),
    "dossier": ("dossier_results", (338, 403, 1376, 900), 1600),
    "mi-setup": ("mi_setup", (338, 216, 1376, 864), 1200),
    "comparison": ("comparison", (346, 144, 1382, 792), 1200),
}


def _font(size: int) -> ImageFont.FreeTypeFont:
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    return ImageFont.truetype(path, size)


def _square_logo() -> Image.Image:
    logo = Image.open(LOGO).convert("RGBA")
    logo = logo.crop(logo.getbbox())
    side = max(logo.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(logo, ((side - logo.width) // 2, (side - logo.height) // 2))
    return square


def brand() -> None:
    out = OUT / "brand"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("wordmark_light", "wordmark_dark"):
        Image.open(BRAND_DIR / f"{name}.png").save(
            out / f"{name.replace('_', '-')}.png", optimize=True
        )
    logo = _square_logo()
    for size, name in ((32, "favicon-32.png"), (192, "icon-192.png")):
        logo.resize((size, size), Image.LANCZOS).save(out / name, optimize=True)
    logo.resize((256, 256), Image.LANCZOS).save(out / "icon-256.webp", "WEBP", quality=90, method=6)
    touch = Image.new("RGBA", (180, 180), (255, 255, 255, 255))  # iOS shows transparency as black
    inner = logo.resize((148, 148), Image.LANCZOS)
    touch.paste(inner, (16, 16), inner)
    touch.convert("RGB").save(out / "apple-touch-icon.png", optimize=True)


def shots(source: Path) -> dict[str, tuple[int, int]]:
    out = OUT / "shots"
    out.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, tuple[int, int]] = {}
    for name, (shot, box, width) in SHOTS.items():
        image = Image.open(source / f"{shot}.png").convert("RGB")
        scale = image.width / VIEWPORT_WIDTH
        if box is not None:
            image = image.crop(tuple(round(v * scale) for v in box))
        height = round(image.height * width / image.width)
        image = image.resize((width, height), Image.LANCZOS)
        image.save(out / f"{name}.webp", "WEBP", quality=82, method=6)
        sizes[name] = (width, height)
    return sizes


def _rounded(image: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, image.width - 1, image.height - 1), radius, fill=255
    )
    rounded = image.convert("RGBA")
    rounded.putalpha(mask)
    return rounded


def og_image(source: Path) -> None:
    width, height = 1200, 630
    navy = tuple(int(BRAND["navy"][i : i + 2], 16) for i in (1, 3, 5))
    deep = (14, 64, 74)
    canvas = Image.new("RGB", (width, height), navy)
    gradient = ImageDraw.Draw(canvas)
    for y in range(height):  # vertical navy -> deep teal gradient
        t = y / (height - 1)
        gradient.line(
            [(0, y), (width, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(navy, deep))
        )
    canvas = canvas.convert("RGBA")

    shot = Image.open(source / "survival.png").convert("RGB")
    shot = shot.resize((760, round(shot.height * 760 / shot.width)), Image.LANCZOS)
    shot = _rounded(shot, 18)
    shadow = Image.new("RGBA", (shot.width + 80, shot.height + 80), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (40, 50, shot.width + 40, shot.height + 50), 18, fill=(0, 0, 0, 150)
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(22))
    canvas.alpha_composite(shadow, (560 - 40, 150 - 40))
    canvas.alpha_composite(shot, (560, 150))

    wordmark = Image.open(BRAND_DIR / "wordmark_dark.png").convert("RGBA")
    wordmark = wordmark.resize((round(wordmark.width * 64 / wordmark.height), 64), Image.LANCZOS)
    canvas.alpha_composite(wordmark, (64, 56))

    draw = ImageDraw.Draw(canvas)
    title = _font(40)
    lines, line = [], ""
    for word in APP_TAGLINE.split():  # wrap the tagline to the left column
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=title) > 450 and line:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    y = 170
    for text in lines:
        draw.text((64, y), text, font=title, fill=(255, 255, 255))
        y += 52
    mint = tuple(int(BRAND["mint"][i : i + 2], 16) for i in (1, 3, 5))
    draw.text((64, y + 24), "Open source · local-first · Apache 2.0", font=_font(22), fill=mint)
    canvas.convert("RGB").save(OUT / "og-image.png", optimize=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python dev_scripts/make_site_assets.py <screenshots-dir>")
    source = Path(sys.argv[1])
    brand()
    sizes = shots(source)
    og_image(source)
    for name, (w, h) in sizes.items():
        size_kb = (OUT / "shots" / f"{name}.webp").stat().st_size / 1024
        print(f"shots/{name}.webp  {w}x{h}  {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
