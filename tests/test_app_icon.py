from __future__ import annotations

import plistlib
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_macos_app_template_declares_valid_icon() -> None:
    info_path = ROOT / "assets" / "Info.plist"
    icon_path = ROOT / "assets" / "AppIcon.icns"

    info = plistlib.loads(info_path.read_bytes())
    assert info["CFBundleIconFile"] == "AppIcon"
    assert icon_path.is_file()

    data = icon_path.read_bytes()
    assert data[:4] == b"icns"
    assert struct.unpack(">I", data[4:8])[0] == len(data)

    offset = 8
    types = set()
    while offset < len(data):
        entry_type = data[offset : offset + 4]
        entry_length = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
        assert entry_length >= 8
        assert offset + entry_length <= len(data)
        types.add(entry_type)
        offset += entry_length
    assert {b"ic07", b"ic08", b"ic09", b"ic10"}.issubset(types)
