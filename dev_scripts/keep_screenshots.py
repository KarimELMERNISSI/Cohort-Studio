"""
Add the screenshots of a capture run to the library (assets/screenshots), as lossless WebP.

    uv run python dev_scripts/keep_screenshots.py <screenshots-dir>

dev_scripts/site_screenshots.py writes PNG files. Lossless WebP keeps every pixel in less than half the space,
and the library lets dev_scripts/make_site_assets.py and dev_scripts/make_readme_assets.py (and any later image,
such as another animation) be rebuilt without running the application. A screenshot replaces the library file of
the same name; debug screenshots are left out.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "assets" / "screenshots"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python dev_scripts/keep_screenshots.py <screenshots-dir>")
    shots = [p for p in sorted(Path(sys.argv[1]).glob("*.png")) if not p.stem.startswith("debug")]
    if not shots:
        raise SystemExit(f"no screenshots in {sys.argv[1]}")
    LIBRARY.mkdir(parents=True, exist_ok=True)
    for path in shots:
        target = LIBRARY / f"{path.stem}.webp"
        Image.open(path).convert("RGB").save(target, "WEBP", lossless=True, quality=100, method=6)
        print(f"{target.relative_to(ROOT)}: {target.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
