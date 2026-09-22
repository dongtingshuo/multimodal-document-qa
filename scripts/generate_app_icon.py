#!/usr/bin/env python3
"""Create the project macOS icon at a large size for later downsampling.

The icon is intentionally generated with Pillow rather than relying on a
machine-specific design application. This keeps the App icon reproducible
when the project is copied to another Mac.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "app_icon_1024.png"
SIZE = 1024
SCALE = 4
CANVAS = SIZE * SCALE


def _scaled(value: float) -> int:
    return round(value * SCALE)


def _rgba_gradient(width: int, height: int) -> Image.Image:
    """Build a deep indigo background with a restrained cyan center glow."""

    image = Image.new("RGBA", (width, height))
    pixels = image.load()
    cx, cy = width * 0.52, height * 0.43
    max_distance = math.hypot(width * 0.76, height * 0.72)
    for y in range(height):
        for x in range(width):
            t = y / max(1, height - 1)
            distance = math.hypot(x - cx, y - cy) / max_distance
            glow = max(0.0, 1.0 - distance) ** 2.1
            r = int(13 + 28 * t + 11 * glow)
            g = int(22 + 16 * t + 38 * glow)
            b = int(58 + 36 * (1 - t) + 54 * glow)
            pixels[x, y] = (r, g, b, 255)
    return image


def _bezier_points(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    count: int = 80,
):
    for index in range(count + 1):
        t = index / count
        inverse = 1 - t
        x = inverse * inverse * start[0] + 2 * inverse * t * control[0] + t * t * end[0]
        y = inverse * inverse * start[1] + 2 * inverse * t * control[1] + t * t * end[1]
        yield _scaled(x), _scaled(y)


def _draw_retrieval_arc(base: Image.Image) -> None:
    points = list(_bezier_points((165, 760), (425, 104), (870, 260)))

    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.line(points, fill=(50, 229, 220, 150), width=_scaled(28), joint="curve")
    glow = glow.filter(ImageFilter.GaussianBlur(_scaled(20)))
    base.alpha_composite(glow)

    beam = Image.new("RGBA", base.size, (0, 0, 0, 0))
    beam_draw = ImageDraw.Draw(beam)
    beam_draw.line(points, fill=(105, 246, 234, 225), width=_scaled(8), joint="curve")
    beam_draw.line(points, fill=(222, 255, 250, 210), width=_scaled(2), joint="curve")
    base.alpha_composite(beam)

    node_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    node_draw = ImageDraw.Draw(node_layer)
    for position, color, radius in [
        ((250, 590), (99, 248, 231, 255), 17),
        ((508, 300), (122, 212, 255, 255), 15),
        ((760, 246), (255, 194, 91, 255), 16),
    ]:
        x, y = _scaled(position[0]), _scaled(position[1])
        r = _scaled(radius)
        node_draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        inner = max(2, _scaled(radius - 7))
        node_draw.ellipse((x - inner, y - inner, x + inner, y + inner), fill=(238, 255, 252, 230))
    base.alpha_composite(node_layer)


def _draw_document(base: Image.Image) -> None:
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    box = (_scaled(306), _scaled(210), _scaled(744), _scaled(814))
    shadow_draw.rounded_rectangle(box, radius=_scaled(54), fill=(0, 4, 22, 180))
    shadow = shadow.filter(ImageFilter.GaussianBlur(_scaled(30)))
    base.alpha_composite(shadow)

    sheet = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sheet_draw = ImageDraw.Draw(sheet)
    sheet_draw.rounded_rectangle(
        box,
        radius=_scaled(54),
        fill=(244, 251, 255, 255),
        outline=(198, 233, 242, 255),
        width=_scaled(5),
    )

    # Folded page corner.
    fold = [(_scaled(620), _scaled(210)), (_scaled(744), _scaled(334)), (_scaled(638), _scaled(334))]
    sheet_draw.polygon(fold, fill=(198, 239, 242, 255))
    sheet_draw.line(
        [(_scaled(620), _scaled(210)), (_scaled(620), _scaled(318)), (_scaled(638), _scaled(334))],
        fill=(187, 220, 231, 255),
        width=_scaled(4),
    )
    sheet_draw.line(
        [(_scaled(620), _scaled(210)), (_scaled(744), _scaled(334))],
        fill=(175, 214, 225, 255),
        width=_scaled(4),
    )

    # Multimodal content: a small visual tile and structured text lines.
    tile = (_scaled(366), _scaled(350), _scaled(684), _scaled(505))
    sheet_draw.rounded_rectangle(tile, radius=_scaled(24), fill=(225, 246, 248, 255))
    sheet_draw.ellipse((_scaled(398), _scaled(378), _scaled(454), _scaled(434)), fill=(53, 199, 203, 255))
    sheet_draw.polygon(
        [
            (_scaled(385), _scaled(480)),
            (_scaled(468), _scaled(421)),
            (_scaled(538), _scaled(477)),
            (_scaled(600), _scaled(402)),
            (_scaled(672), _scaled(480)),
        ],
        fill=(90, 192, 220, 255),
    )
    sheet_draw.line([(_scaled(397), _scaled(535)), (_scaled(655), _scaled(535))], fill=(39, 121, 161, 255), width=_scaled(13))
    sheet_draw.line([(_scaled(397), _scaled(582)), (_scaled(610), _scaled(582))], fill=(92, 178, 202, 255), width=_scaled(11))
    sheet_draw.line([(_scaled(397), _scaled(629)), (_scaled(666), _scaled(629))], fill=(170, 211, 220, 255), width=_scaled(10))
    sheet_draw.rounded_rectangle((_scaled(397), _scaled(693), _scaled(532), _scaled(734)), radius=_scaled(16), fill=(35, 177, 188, 255))
    sheet_draw.rounded_rectangle((_scaled(548), _scaled(693), _scaled(665), _scaled(734)), radius=_scaled(16), fill=(248, 184, 81, 255))

    halo = Image.new("RGBA", base.size, (0, 0, 0, 0))
    halo_draw = ImageDraw.Draw(halo)
    halo_draw.rounded_rectangle(box, radius=_scaled(54), outline=(64, 238, 229, 130), width=_scaled(15))
    halo = halo.filter(ImageFilter.GaussianBlur(_scaled(16)))
    base.alpha_composite(halo)
    base.alpha_composite(sheet)


def generate() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    icon = _rgba_gradient(CANVAS, CANVAS)

    sheen = Image.new("RGBA", icon.size, (0, 0, 0, 0))
    sheen_draw = ImageDraw.Draw(sheen)
    sheen_draw.rounded_rectangle(
        (_scaled(28), _scaled(28), _scaled(996), _scaled(996)),
        radius=_scaled(214),
        outline=(145, 213, 255, 90),
        width=_scaled(4),
    )
    sheen_draw.arc((_scaled(82), _scaled(40), _scaled(956), _scaled(560)), 195, 348, fill=(255, 255, 255, 34), width=_scaled(8))
    icon.alpha_composite(sheen)

    # Add a restrained lower vignette before the foreground mark is composited.
    vignette = Image.new("RGBA", icon.size, (0, 0, 0, 0))
    vignette_draw = ImageDraw.Draw(vignette)
    vignette_draw.ellipse((_scaled(120), _scaled(680), _scaled(930), _scaled(1160)), fill=(3, 5, 22, 75))
    vignette = vignette.filter(ImageFilter.GaussianBlur(_scaled(60)))
    icon.alpha_composite(vignette)

    _draw_retrieval_arc(icon)
    _draw_document(icon)
    icon.resize((SIZE, SIZE), Image.Resampling.LANCZOS).save(OUTPUT, "PNG", optimize=True)
    return OUTPUT


if __name__ == "__main__":
    print(generate())
