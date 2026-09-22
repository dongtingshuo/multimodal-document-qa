import json
import zipfile

from scripts.package_source import package


def test_source_snapshot_excludes_credentials_runtime_and_symlinks(tmp_path):
    root = tmp_path / "project"
    (root / "docqa").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "models").mkdir()
    (root / "docqa/api.py").write_text("# source")
    (root / ".env").write_text("private credential")
    (root / ".env.example").write_text("DOCQA_API_KEY=")
    (root / "data/private.pdf").write_text("private document")
    (root / "models/weights.bin").write_bytes(b"weights")
    (root / "docqa/linked.py").symlink_to(root / ".env")
    output = tmp_path / "source.zip"
    assert package(root, output) == 2
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"docqa/api.py", ".env.example", "SOURCE_MANIFEST.json"}
        manifest = json.loads(archive.read("SOURCE_MANIFEST.json"))
        assert len(manifest) == 2
