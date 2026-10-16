"""
The participant flow as a static picture (PNG) for the Word dossier: the same boxes and geometry as the app's
diagram (core.attrition.flow_layout), drawn with matplotlib's object API (no pyplot state, safe in a server).
"""

from __future__ import annotations

from io import BytesIO
from textwrap import wrap

from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from core.attrition import ANALYSIS, POPULATION, FlowExit, ParticipantFlow, flow_layout

BOX_W, BOX_H = 1.0, 0.56
FILL = {"exit": "#fbf1e4", POPULATION: "#d9ecf5", ANALYSIS: "#e6f3ea"}
LINE = {"exit": "#b36b00", POPULATION: "#1f5f8b", ANALYSIS: "#2e7d4f"}
DEFAULT_FILL, DEFAULT_LINE = "#f2f5f8", "#5b7083"


def flow_png(flow: ParticipantFlow, *, dpi: int = 200) -> bytes:
    pos = flow_layout(flow)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    width, height = max(xs) - min(xs) + 1.6, max(ys) - min(ys) + 1.0
    fig = Figure(figsize=(max(5.0, 1.9 * width), max(3.0, 1.9 * height)))
    ax = fig.subplots()
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "-|>", "color": DEFAULT_LINE, "lw": 1.0},
        )

    for item in [*flow.main, *flow.exits, *flow.analyses]:
        x, y = pos[item.id]
        kind = "exit" if isinstance(item, FlowExit) else item.kind
        w = BOX_W * (1.1 if isinstance(item, FlowExit) else 1.0)
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - BOX_H / 2),
                w,
                BOX_H,
                boxstyle="round,pad=0.02",
                facecolor=FILL.get(kind, DEFAULT_FILL),
                edgecolor=LINE.get(kind, DEFAULT_LINE),
                linewidth=1.0,
            )
        )
        label = "\n".join(wrap(item.label, 24))
        ax.text(x, y + 0.05, label, ha="center", va="center", fontsize=6.5)
        ax.text(x, y - BOX_H / 2 + 0.08, f"n = {item.n:,}", ha="center", va="center", fontsize=6.5, weight="bold")

    for before, after in zip(flow.main, flow.main[1:], strict=False):
        (x0, y0), (x1, y1) = pos[before.id], pos[after.id]
        arrow(x0, y0 - BOX_H / 2, x1, y1 + BOX_H / 2)
    for exit_ in flow.exits:
        x, y = pos[exit_.id]
        arrow(0.0, y, x - BOX_W * 0.55, y)
    population = next((b for b in flow.main if b.kind == POPULATION), flow.main[-1])
    px, py = pos[population.id]
    for box in flow.analyses:
        x, y = pos[box.id]
        arrow(px, py - BOX_H / 2, x, y + BOX_H / 2)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    return buffer.getvalue()


__all__ = ["flow_png"]
