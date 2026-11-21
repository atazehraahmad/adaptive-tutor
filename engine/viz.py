"""Concept-graph rendering as inline SVG.

The graph is the product's defining object, so it is drawn rather than described:
chapters become columns, prerequisite depth runs down the page, and each node is
filled by mastery so a learner (or a teacher) reads progress as a shape.
"""
from __future__ import annotations

from .graph import ConceptGraph

INK = "#16202B"
RULE = "#D7E0E4"
EDGE = "#B6C3CB"
EDGE_CROSS = "#DCE5EA"
PAPER = "#FFFFFF"
ACCENT = "#2F3BB3"

# Mastery is diverging, not sequential: below par / at par / above par, with a
# neutral (hueless) midpoint. Shared with the Altair charts so one colour means
# the same thing everywhere in the product.
MASTERY_LOW = "#B0432F"
MASTERY_MID = "#9E978C"
MASTERY_HIGH = "#1F7A58"
STOPS = [(0.0, (176, 67, 47)), (0.5, (158, 151, 140)), (1.0, (31, 122, 88))]


def mastery_colour(v: float) -> str:
    v = max(0.0, min(1.0, v))
    for i in range(len(STOPS) - 1):
        a, ca = STOPS[i]
        b, cb = STOPS[i + 1]
        if a <= v <= b:
            t = 0 if b == a else (v - a) / (b - a)
            r, g, bl = (round(ca[j] + (cb[j] - ca[j]) * t) for j in range(3))
            return f"#{r:02X}{g:02X}{bl:02X}"
    return "#3E8E6E"


def _depths(graph: ConceptGraph) -> dict[str, int]:
    """Longest path from a root, so prerequisites always sit above dependents."""
    depth: dict[str, int] = {}
    for cid in graph.topo_order():
        pre = graph.concepts[cid].prereq
        depth[cid] = 0 if not pre else 1 + max(depth[p] for p in pre)
    return depth


def _wrap(text: str, width: int = 22) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:2]


def _order_layers(graph: ConceptGraph, layers: list[list[str]]) -> list[list[str]]:
    """Barycentre heuristic: pull each node toward the mean row of its neighbours."""
    index = {cid: i for layer in layers for i, cid in enumerate(layer)}
    for _ in range(4):
        for li in range(1, len(layers)):
            layers[li].sort(key=lambda c: (
                sum(index[p] for p in graph.concepts[c].prereq) /
                len(graph.concepts[c].prereq)) if graph.concepts[c].prereq else index[c])
            index.update({cid: i for i, cid in enumerate(layers[li])})
        for li in range(len(layers) - 2, -1, -1):
            kids = {cid: [d for d in graph.dependents(cid)] for cid in layers[li]}
            layers[li].sort(key=lambda c: (
                sum(index[d] for d in kids[c]) / len(kids[c])) if kids[c] else index[c])
            index.update({cid: i for i, cid in enumerate(layers[li])})
    return layers


def _short(name: str, n: int = 21) -> str:
    return name if len(name) <= n else name[: n - 1].rstrip() + "\u2026"


def concept_graph_svg(graph: ConceptGraph, mastery: dict[str, float],
                      highlight: set[str] | None = None,
                      chip_w: int = 142, gap_x: int = 30, row_h: int = 44) -> str:
    """Prerequisite DAG, left to right, each node filled by mastery."""
    highlight = highlight or set()
    depth = _depths(graph)
    n_layers = max(depth.values()) + 1
    layers = [[c for c in graph.topo_order() if depth[c] == d] for d in range(n_layers)]
    layers = _order_layers(graph, layers)

    chip_h = 30
    top = 46
    rows = max(len(l) for l in layers)
    width = n_layers * chip_w + (n_layers - 1) * gap_x + 24
    height = top + rows * row_h + 20

    pos: dict[str, tuple[float, float]] = {}
    for li, layer in enumerate(layers):
        x = 12 + li * (chip_w + gap_x)
        # centre short layers vertically
        offset = (rows - len(layer)) * row_h / 2
        for ri, cid in enumerate(layer):
            pos[cid] = (x, top + offset + ri * row_h)

    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" aria-label="Prerequisite graph coloured by mastery" '
        f'style="font-family:\'IBM Plex Sans\',system-ui,sans-serif">',
        '<defs><pattern id="gp" width="24" height="24" patternUnits="userSpaceOnUse">'
        f'<path d="M24 0H0V24" fill="none" stroke="{RULE}" stroke-width="0.6"/>'
        '</pattern></defs>',
        f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
        f'<rect width="{width}" height="{height}" fill="url(#gp)" opacity="0.45"/>',
        f'<text x="12" y="22" fill="{INK}" font-size="12" font-weight="600">'
        f'Prerequisites flow left to right</text>',
        f'<text x="12" y="38" fill="#5B6B77" font-size="10.5">'
        f'Answering one question moves belief along these edges</text>',
    ]

    for cid, c in graph.concepts.items():
        x2, y2 = pos[cid]
        for pr in c.prereq:
            x1, y1 = pos[pr]
            sx, sy = x1 + chip_w, y1 + chip_h / 2
            ex, ey = x2, y2 + chip_h / 2
            mx = (sx + ex) / 2
            lit = cid in highlight and pr in highlight
            o.append(
                f'<path d="M{sx:.0f},{sy:.0f} C{mx:.0f},{sy:.0f} {mx:.0f},{ey:.0f} '
                f'{ex:.0f},{ey:.0f}" fill="none" '
                f'stroke="{ACCENT if lit else EDGE}" '
                f'stroke-width="{2 if lit else 1.1}" '
                f'opacity="{1 if lit else 0.75}"/>')

    for cid, c in graph.concepts.items():
        x, y = pos[cid]
        v = mastery.get(cid, 0.0)
        col = mastery_colour(v)
        on = cid in highlight
        o.append(
            f'<g><title>{c.name} \u2014 {c.chapter} \u2014 {v:.0%} mastery</title>'
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{chip_w}" height="{chip_h}" rx="6" '
            f'fill="#FFFFFF" stroke="{ACCENT if on else "#CBD6DD"}" '
            f'stroke-width="{1.8 if on else 1}"/>'
            f'<rect x="{x:.0f}" y="{y:.0f}" width="4.5" height="{chip_h}" '
            f'fill="{col}" clip-path="inset(0 0 0 0 round 6px 0 0 6px)"/>'
            f'<text x="{x+13:.0f}" y="{y+19:.0f}" fill="{INK}" font-size="9.6" '
            f'font-weight="{600 if on else 450}">{_short(c.name)}</text></g>')

    o.append("</svg>")
    return "".join(o)


def legend_svg() -> str:
    cells = "".join(
        f'<rect x="{i*34}" y="0" width="34" height="9" fill="{mastery_colour(i/9)}"/>'
        for i in range(10))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 9" '
            f'width="170" height="9">{cells}</svg>')
