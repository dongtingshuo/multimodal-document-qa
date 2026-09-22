#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."
icon_source="assets/app_icon_1024.png"
iconset="assets/AppIcon.iconset"
icon_file="assets/AppIcon.icns"

if [[ ! -f "$icon_source" ]]; then
  .venv/bin/python scripts/generate_app_icon.py
fi

mkdir -p "$iconset"

ICON_SOURCE="$icon_source" ICONSET="$iconset" .venv/bin/python - <<'PY'
import os
from pathlib import Path

from PIL import Image, ImageCms

source = Image.open(os.environ["ICON_SOURCE"]).convert("RGBA")
icc_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
iconset = Path(os.environ["ICONSET"])
sizes = (16, 32, 128, 256, 512)
for size in sizes:
    source.resize((size, size), Image.Resampling.LANCZOS).save(iconset / f"icon_{size}x{size}.png", "PNG", optimize=True, icc_profile=icc_profile)
    source.resize((size * 2, size * 2), Image.Resampling.LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png", "PNG", optimize=True, icc_profile=icc_profile)
PY

ICONSET="$iconset" ICON_FILE="$icon_file" .venv/bin/python - <<'PY'
"""Pack the PNG renditions into an ICNS container.

The system iconutil shipped on some macOS releases can unpack ICNS files but
reject otherwise valid iconsets while building them. The ICNS container format
is a small big-endian record stream, so packing the PNG renditions directly is
more portable and keeps this build step deterministic.
"""

import os
import struct
from pathlib import Path

iconset = Path(os.environ["ICONSET"])
output = Path(os.environ["ICON_FILE"])
renditions = [
    ("ic11", "icon_16x16@2x.png"),   # 32 px, Retina 16 px
    ("ic12", "icon_32x32@2x.png"),   # 64 px, Retina 32 px
    ("ic07", "icon_128x128.png"),
    ("ic13", "icon_128x128@2x.png"),
    ("ic08", "icon_256x256.png"),
    ("ic14", "icon_256x256@2x.png"),
    ("ic09", "icon_512x512.png"),
    ("ic10", "icon_512x512@2x.png"),
]
records = []
for type_code, filename in renditions:
    payload = (iconset / filename).read_bytes()
    records.append(type_code.encode("ascii") + struct.pack(">I", len(payload) + 8) + payload)
body = b"".join(records)
output.write_bytes(b"icns" + struct.pack(">I", len(body) + 8) + body)
PY

echo "$icon_file"
