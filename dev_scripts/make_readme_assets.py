"""
Images of the README (assets/readme/), from screenshots of the real application.

    uv run python dev_scripts/make_readme_assets.py [screenshots-dir]

The screenshots come from the library, assets/screenshots (see dev_scripts/keep_screenshots.py), or from the
directory of a capture run of dev_scripts/site_screenshots.py: the application on the tutorial cohort, a
1440x900 viewport captured at 2x.

tour.webp, describe.webp
    Animations of screens, each in a browser window on a background in the brand colours, with a numbered
    caption and progress dots: the tour of a study at the top of the README, and the description of the data.
    Between two screens the caption fades out, the next screen sweeps in behind a soft edge, and the next caption
    fades in. Each frame of the sweep changes only a band of the window and WebP stores only what changed, which
    keeps the files well under the 2 MB of the large-file hook. Frames are drawn at twice the output size and then
    reduced, which smooths every edge. The animations loop; GitHub shows animated WebP in every current browser.
workflow.webp
    The workflow map of the Start page, at the end of the tutorial walk-through.
plots/<name>.webp
    The Visualize figures on their own, trimmed of their empty margins and centred on a 3:2 canvas.
"""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "assets" / "screenshots"
OUT = ROOT / "assets" / "readme"

Step = tuple[str, str, str]  # (screenshot, URL path, caption)
TOUR: list[Step] = [  # in the order of a study
    ("start_workflow", "", "Start: one workflow from a raw file to results"),
    ("quality", "validate", "Check data quality: one score, every check behind it"),
    ("population", "validate", "Tick the criteria: one population, with its participant flow"),
    ("explore_stats", "explore", "Describe the data: statistics, fill rates, normality tests"),
    ("visualize", "visualize", "Plot the data: 14 plot types, p-values between groups"),
    ("mi_diagnostics", "enrich", "Impute missing values for inference, with diagnostics"),
    ("survival", "survival", "Survival curves pooled across 10 imputed datasets"),
    ("cox", "survival", "Models pooled with Rubin's rules, FMI for every term"),
    ("dossier_results", "dossier", "Hand over a Word study dossier, each result traced"),
]
DESCRIBE: list[Step] = [
    (
        "explore_types",
        "explore",
        "Every column sorted by type: quantitative, binary, categorical, date",
    ),
    ("explore_stats", "explore", "Descriptive statistics with fill rates and normality tests"),
    ("explore_matrix", "explore", "Correlations as a matrix, or as a network"),
]
# output name -> screenshot (plot_<name>)
PLOTS = {
    "box": "box",
    "violin": "violin",
    "scatter": "scatter",
    "correlation": "corr",
    "clustermap": "clustermap",
    "sunburst": "sunburst",
}
PLOT_WIDTH, PLOT_RATIO = 1200, 1.5
WORKFLOW_WIDTH = 1800

S = 2  # frames are drawn at twice the output size, then reduced
W, H = 1600 * S, 1120 * S
WIN_X, WIN_Y = 80 * S, 40 * S
BAR = 44 * S  # browser title bar
SHOT_W, SHOT_H = 1440 * S, 900 * S
CAPTION_Y, DOTS_Y = 1030 * S, 1084 * S  # centres
STRIP_Y, STRIP_H = CAPTION_Y - 32 * S, 64 * S  # band that holds the caption pill
EDGE = 160 * S  # soft edge of the sweep
HOLD_MS = 2800
FADE_OUT, SWEEP, FADE_IN = (3, 50), (6, 55), (2, 50)  # (frames, milliseconds per frame)

NAVY, TEAL, IDLE_DOT = (11, 43, 54), (14, 124, 134), (163, 194, 195)
CHROME, CHROME_LINE, ADDRESS = (241, 245, 246), (225, 234, 236), (90, 116, 128)


def screenshot(source: Path, name: str) -> Image.Image:
    """A screenshot by name: the PNG of a capture run or the WebP of the library."""
    for suffix in (".png", ".webp"):
        if (path := source / f"{name}{suffix}").exists():
            return Image.open(path).convert("RGB")
    raise FileNotFoundError(f"no screenshot {name!r} in {source}")


@cache
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Segoe UI where it is installed, DejaVu Sans elsewhere."""
    segoe = Path("C:/Windows/Fonts") / ("seguisb.ttf" if bold else "segoeui.ttf")
    if segoe.exists():
        return ImageFont.truetype(str(segoe), size)
    props = font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    return ImageFont.truetype(font_manager.findfont(props), size)


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


@cache
def _background() -> Image.Image:
    """A diagonal of brand tints, a white glow behind the window and the window's shadow."""
    y, x = np.ogrid[0:H, 0:W]
    t = ((x / W + y / H) / 2).astype(np.float32)[..., None]
    start, end = np.array((220, 245, 235), np.float32), np.array((204, 236, 240), np.float32)
    image = Image.fromarray((start + (end - start) * t).round().astype(np.uint8), "RGB")
    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).ellipse((W * 0.12, H * 0.02, W * 0.88, H * 0.82), fill=150)
    image.paste((255, 255, 255), (0, 0, W, H), glow.filter(ImageFilter.GaussianBlur(120 * S)))
    shadow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shadow).rounded_rectangle(
        (WIN_X + 10 * S, WIN_Y + 28 * S, WIN_X + SHOT_W - 10 * S, WIN_Y + BAR + SHOT_H + 22 * S),
        18 * S,
        fill=95,
    )
    image.paste(NAVY, (0, 0, W, H), shadow.filter(ImageFilter.GaussianBlur(26 * S)))
    return image


def _window(shot: Image.Image, path: str) -> Image.Image:
    """A browser window around one screenshot, with the page's address."""
    window = Image.new("RGBA", (SHOT_W, BAR + SHOT_H), (*CHROME, 255))
    draw = ImageDraw.Draw(window)
    draw.line((0, BAR - S, SHOT_W, BAR - S), fill=CHROME_LINE, width=S)
    for i, colour in enumerate(((242, 102, 94), (245, 190, 79), (91, 195, 109))):
        cx, cy, r = (26 + 22 * i) * S, BAR // 2, 6 * S
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=colour)
    x0, width = (SHOT_W - 520 * S) / 2, 520 * S
    draw.rounded_rectangle(
        (x0, 9 * S, x0 + width, BAR - 9 * S),
        8 * S,
        fill=(255, 255, 255),
        outline=CHROME_LINE,
        width=S,
    )
    draw.text(
        (SHOT_W / 2, BAR / 2),
        f"localhost:8501/{path}",
        font=_font(17 * S),
        fill=ADDRESS,
        anchor="mm",
    )
    if shot.size != (SHOT_W, SHOT_H):
        shot = shot.resize((SHOT_W, SHOT_H), Image.LANCZOS)
    window.paste(shot, (0, BAR))
    corners = Image.new("L", window.size, 0)
    ImageDraw.Draw(corners).rounded_rectangle(
        (0, 0, window.width - 1, window.height - 1), 18 * S, fill=255
    )
    window.putalpha(corners)
    return window


def _sweep(old: Image.Image, new: Image.Image, t: float) -> Image.Image:
    """The new window over the old one, left of an edge at progress t, with a soft edge."""
    edge = _ease(t) * (SHOT_W + EDGE)
    ramp = np.clip((edge - np.arange(SHOT_W)) / EDGE, 0, 1)
    mask = Image.fromarray(
        np.repeat((ramp * 255).astype(np.uint8)[None, :], old.height, axis=0), "L"
    )
    return Image.composite(new, old, mask)


def _caption(number: int, text: str) -> Image.Image:
    """A numbered caption pill on a transparent band (STRIP_H high) across the canvas."""
    strip = Image.new("RGBA", (W, STRIP_H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(strip)
    font, height = _font(27 * S, bold=True), 56 * S
    width = draw.textlength(text, font=font) + 100 * S
    x0, y0, cy = (W - width) / 2, (STRIP_H - height) / 2, STRIP_H / 2
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), height / 2, fill=(255, 255, 255, 255))
    cx, r = x0 + 31 * S, 18 * S
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*TEAL, 255))
    draw.text(
        (cx, cy + S),
        str(number),
        font=_font(21 * S, bold=True),
        fill=(255, 255, 255, 255),
        anchor="mm",
    )
    draw.text((x0 + 64 * S, cy + S), text, font=font, fill=(*NAVY, 255), anchor="lm")
    return strip


def _frame(
    window: Image.Image,
    caption: Image.Image | None,
    alpha: float,
    dots: tuple[int, int, int, float],
) -> Image.Image:
    """One frame at output size. dots = (count, old step, new step, progress from one to the other)."""
    canvas = _background().copy()
    canvas.paste(window, (WIN_X, WIN_Y), window)
    if caption is not None and alpha > 0:
        canvas.paste(
            caption, (0, STRIP_Y), caption.getchannel("A").point(lambda v: round(v * alpha))
        )
    count, old, new, t = dots
    draw = ImageDraw.Draw(canvas)
    for k in range(count):
        active = (1 - t) * (k == old) + t * (k == new)
        colour = tuple(round(a + (b - a) * active) for a, b in zip(IDLE_DOT, TEAL))
        cx, r = W / 2 + (k - (count - 1) / 2) * 22 * S, (4 + 1.5 * active) * S
        draw.ellipse((cx - r, DOTS_Y - r, cx + r, DOTS_Y + r), fill=colour)
    return canvas.resize((W // S, H // S), Image.LANCZOS)


def animate(steps: list[Step], source: Path) -> tuple[list[Image.Image], list[int]]:
    """Every frame of the loop through the steps and its duration in milliseconds."""
    windows = [_window(screenshot(source, name), path) for name, path, _ in steps]
    captions = [_caption(number, text) for number, (_, _, text) in enumerate(steps, start=1)]
    n = len(steps)
    frames, durations = [], []

    def add(image: Image.Image, ms: int) -> None:
        frames.append(image)
        durations.append(ms)

    for old in range(n):
        new = (old + 1) % n
        add(_frame(windows[old], captions[old], 1, (n, old, old, 0)), HOLD_MS)
        count, ms = FADE_OUT
        for k in range(1, count + 1):
            add(_frame(windows[old], captions[old], 1 - k / count, (n, old, old, 0)), ms)
        count, ms = SWEEP
        for k in range(1, count + 1):
            add(
                _frame(_sweep(windows[old], windows[new], k / count), None, 0, (n, old, old, 0)), ms
            )
        count, ms = FADE_IN
        for k in range(1, count + 1):
            t = k / (count + 1)
            add(_frame(windows[new], captions[new], t, (n, old, new, _ease(t))), ms)
    corners = Image.new("L", (W, H), 0)
    ImageDraw.Draw(corners).rounded_rectangle((0, 0, W - 1, H - 1), 36 * S, fill=255)
    corners = corners.resize((W // S, H // S), Image.LANCZOS)
    for image in frames:  # transparent outside the rounded corners
        image.putalpha(corners)
    return frames, durations


def save_animation(steps: list[Step], source: Path, path: Path) -> None:
    frames, durations = animate(steps, source)
    frames[0].save(  # kmax=0: no forced key frames, so each frame stores only what changed
        path,
        "WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        quality=86,
        method=4,
        kmin=0,
        kmax=0,
    )
    print(f"{path.relative_to(ROOT)}: {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB")


def plot(image: Image.Image) -> Image.Image:
    """A figure without its empty margins, centred on a 3:2 canvas of its own background colour."""
    background = image.getpixel((4, 4))
    difference = ImageChops.difference(image, Image.new("RGB", image.size, background)).convert("L")
    image = image.crop(difference.point(lambda v: 255 if v > 12 else 0).getbbox())
    width, height = image.width + 96, image.height + 96
    width, height = max(width, round(height * PLOT_RATIO)), max(height, round(width / PLOT_RATIO))
    canvas = Image.new("RGB", (width, height), background)
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas.resize((PLOT_WIDTH, round(PLOT_WIDTH / PLOT_RATIO)), Image.LANCZOS)


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit("usage: python dev_scripts/make_readme_assets.py [screenshots-dir]")
    source = Path(sys.argv[1]) if len(sys.argv) == 2 else LIBRARY
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    save_animation(TOUR, source, OUT / "tour.webp")
    save_animation(DESCRIBE, source, OUT / "describe.webp")

    card = screenshot(source, "workflow")
    card = card.resize(
        (WORKFLOW_WIDTH, round(card.height * WORKFLOW_WIDTH / card.width)), Image.LANCZOS
    )
    card.save(path := OUT / "workflow.webp", "WEBP", quality=90, method=6)
    print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1e3:.0f} kB")
    for name, shot in PLOTS.items():
        plot(screenshot(source, f"plot_{shot}")).save(
            path := OUT / "plots" / f"{name}.webp", "WEBP", quality=88, method=6
        )
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
