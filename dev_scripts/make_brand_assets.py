"""
Generate the brand assets used by the app shell from the application logo.

    python dev_scripts/make_brand_assets.py

Outputs (assets/brand/):
    icon.png            square logo, transparent background (favicon, collapsed header, login card)
    wordmark_light.png  logo + "CohortStudio" for light themes (navy + teal lettering)
    wordmark_dark.png   the same for dark themes (white + mint lettering)

The lettering uses DejaVu Sans Bold, which ships with matplotlib (free licence), so the assets can be
regenerated offline. Colours come from ui/theme.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.theme import BRAND  # noqa: E402

LOGO = ROOT / "assets" / "karim-app-logo.png"
OUT = ROOT / "assets" / "brand"
HEIGHT = 160  # rendered at 32 px in the header: 5x for sharp high-DPI display


def _font(size: int) -> ImageFont.FreeTypeFont:
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    return ImageFont.truetype(path, size)


def _icon(size: int) -> Image.Image:
    logo = Image.open(LOGO).convert("RGBA")
    logo = logo.crop(logo.getbbox())
    side = max(logo.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(logo, ((side - logo.width) // 2, (side - logo.height) // 2))
    return square.resize((size, size), Image.LANCZOS)


def _wordmark(first_colour: str, second_colour: str) -> Image.Image:
    icon = _icon(HEIGHT)
    font = _font(int(HEIGHT * 0.46))
    first, second = "Cohort", "Studio"
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    w1 = probe.textlength(first, font=font)
    w2 = probe.textlength(second, font=font)
    gap = int(HEIGHT * 0.16)
    width = HEIGHT + gap + int(w1 + w2) + int(HEIGHT * 0.06)
    canvas = Image.new("RGBA", (width, HEIGHT), (0, 0, 0, 0))
    canvas.paste(icon, (0, 0), icon)
    draw = ImageDraw.Draw(canvas)
    ascent, descent = font.getmetrics()
    y = (HEIGHT - (ascent + descent)) // 2 + int(HEIGHT * 0.02)
    x = HEIGHT + gap
    draw.text((x, y), first, font=font, fill=first_colour)
    draw.text((x + w1, y), second, font=font, fill=second_colour)
    return canvas


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _icon(256).save(OUT / "icon.png", optimize=True)
    _wordmark(BRAND["navy"], BRAND["teal"]).save(OUT / "wordmark_light.png", optimize=True)
    _wordmark("#FFFFFF", BRAND["mint"]).save(OUT / "wordmark_dark.png", optimize=True)
    for name in ("icon.png", "wordmark_light.png", "wordmark_dark.png"):
        with Image.open(OUT / name) as im:
            print(f"{OUT / name}: {im.size[0]}x{im.size[1]}")


if __name__ == "__main__":
    main()
