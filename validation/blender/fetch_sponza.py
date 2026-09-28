"""Download Khronos glTF-Sample-Assets Sponza (glTF + bin + textures, ~53 MB) for the M0 timing run.

  python3 validation/blender/fetch_sponza.py [--out validation/assets/downloaded/sponza] [--ref main]

Lists the folder via the GitHub contents API (one call), downloads each file from
raw.githubusercontent.com and verifies size and git blob SHA-1. Existing verified files are skipped.
License: CC-BY 3.0 (Crytek), see the upstream README; the folder is gitignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = "KhronosGroup/glTF-Sample-Assets"
FOLDER = "Models/Sponza/glTF"


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "restirpt-validation"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("validation/assets/downloaded/sponza"))
    ap.add_argument("--ref", default="main")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    listing = json.loads(fetch(f"https://api.github.com/repos/{REPO}/contents/{FOLDER}?ref={args.ref}"))
    files = [e for e in listing if e["type"] == "file"]
    got = 0
    for e in files:
        dst = args.out / e["name"]
        if dst.exists() and dst.stat().st_size == e["size"] and git_blob_sha1(dst.read_bytes()) == e["sha"]:
            continue
        data = fetch(f"https://raw.githubusercontent.com/{REPO}/{args.ref}/{FOLDER}/{e['name']}")
        if len(data) != e["size"] or git_blob_sha1(data) != e["sha"]:
            print(f"integrity mismatch: {e['name']}", file=sys.stderr)
            return 1
        dst.write_bytes(data)
        got += 1
    manifest = {e["name"]: {"size": e["size"], "git_sha1": e["sha"]} for e in files}
    (args.out / "manifest.json").write_text(json.dumps({"repo": REPO, "ref": args.ref, "folder": FOLDER, "files": manifest}, indent=1) + "\n")
    print(f"sponza: {len(files)} files ({sum(e['size'] for e in files) / 1e6:.1f} MB), downloaded {got}, in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
