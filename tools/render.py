"""把地图画成 PNG（调试和预览用，正式画面在前端 P2 再做）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine.hexmap.coords import EDGE_CORNERS, SQRT3, corner_offset, offset_to_pixel
from engine.hexmap.worldgen import World
from engine.rules.schema import Ruleset

RIVER_COLOR = (40, 110, 200)
START_COLORS = [
    (220, 50, 47),
    (38, 139, 210),
    (133, 153, 0),
    (211, 54, 130),
    (181, 137, 0),
    (42, 161, 152),
    (108, 113, 196),
    (203, 75, 22),
]


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


_CJK_FONTS = (
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf",
)
_LATIN_FONTS = (
    "/usr/share/fonts/google-noto-vf/NotoSans-VF.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)


def _font(size: int, cjk: bool = False) -> ImageFont.ImageFont:
    for path in _CJK_FONTS if cjk else _LATIN_FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_world(
    world: World,
    rules: Ruleset,
    hex_size: float = 9.0,
    show_resources: bool = True,
    show_legend: bool = True,
) -> Image.Image:
    g = world.grid
    margin = hex_size * 1.5
    map_w = hex_size * SQRT3 * (g.width + 0.5) + 2 * margin
    map_h = hex_size * (1.5 * (g.height - 1) + 2) + 2 * margin
    legend_w = 90 if show_legend else 0
    img = Image.new("RGB", (int(map_w + legend_w), int(map_h)), (24, 28, 36))
    draw = ImageDraw.Draw(img)

    corners = [corner_offset(i, hex_size) for i in range(6)]
    terrain_rgb = [_hex_to_rgb(t.color) for t in rules.terrain.values()]

    def center(i: int) -> tuple[float, float]:
        col, row = g.col_row(i)
        x, y = offset_to_pixel(col, row, hex_size)
        return x + margin + hex_size * SQRT3 / 2, y + margin + hex_size

    centers = [center(i) for i in range(g.size)]

    # 地形
    for i in range(g.size):
        cx, cy = centers[i]
        poly = [(cx + dx, cy + dy) for dx, dy in corners]
        draw.polygon(poly, fill=terrain_rgb[g.terrain[i]], outline=(0, 0, 0, 40))

    # 河流（每条边只画一次：只从 idx 较小的一侧画）
    width = max(1, round(hex_size / 4))
    for i in np.flatnonzero(g.river_edges):
        i = int(i)
        cx, cy = centers[i]
        for d in range(6):
            j = int(g.neighbors[i, d])
            if g.has_river(i, d) and (j > i or j < 0):
                a, b = EDGE_CORNERS[d]
                p = (cx + corners[a][0], cy + corners[a][1])
                q = (cx + corners[b][0], cy + corners[b][1])
                draw.line([p, q], fill=RIVER_COLOR, width=width)

    # 资源
    if show_resources:
        r = hex_size * 0.28
        for i in np.flatnonzero(g.resource):
            res = rules.resource_at(int(g.resource[i]))
            cx, cy = centers[int(i)]
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r], fill=_hex_to_rgb(res.color), outline=(0, 0, 0)
            )

    # 起始点
    font = _font(max(9, int(hex_size)))
    r = hex_size * 0.75
    for n, s in enumerate(world.starts):
        cx, cy = centers[s]
        color = START_COLORS[n % len(START_COLORS)]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255), outline=color, width=3)
        draw.text((cx, cy), str(n + 1), fill=color, font=font, anchor="mm")

    if show_legend:
        _legend(draw, rules, int(map_w) + 10, 10, _font(12, cjk=True))
    return img


def _legend(draw: ImageDraw.ImageDraw, rules: Ruleset, x: int, y: int, font) -> None:
    draw.text((x, y), "地形", fill=(230, 230, 230), font=font)
    y += 18
    for t in rules.terrain.values():
        draw.rectangle([x, y, x + 12, y + 12], fill=_hex_to_rgb(t.color))
        draw.text((x + 18, y - 1), t.name, fill=(220, 220, 220), font=font)
        y += 16
    y += 8
    draw.line([x, y, x + 24, y], fill=RIVER_COLOR, width=3)
    draw.text((x + 30, y - 7), "河流", fill=(220, 220, 220), font=font)
    y += 18
    draw.ellipse([x, y, x + 12, y + 12], fill=(255, 255, 255), outline=START_COLORS[0], width=2)
    draw.text((x + 18, y - 1), "起始点", fill=(220, 220, 220), font=font)
    y += 22
    draw.text((x, y), "资源", fill=(230, 230, 230), font=font)
    y += 18
    for res in rules.resources.values():
        draw.ellipse([x + 2, y + 2, x + 10, y + 10], fill=_hex_to_rgb(res.color), outline=(0, 0, 0))
        draw.text((x + 18, y - 1), res.name, fill=(220, 220, 220), font=font)
        y += 16
