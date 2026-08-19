#!/usr/bin/env python3
"""
Download the trained checkpoint.

Model weights are not in git — they are ~70 MB of binary that changes every
training run, which is what bloated this repo's history to 163 MB before they
were removed. They live on a GitHub Release instead.

    python backend/model/fetch_weights.py

Stdlib only, so this runs before `pip install -r requirements.txt`.

Publishing a new checkpoint
---------------------------
    gh release create weights-v2 backend/model/best_model.pth \
        --title "Model weights v2" --notes "EfficientNet-B4, calibrated"

then update weights.json with the new url / sha256 / size:

    python backend/model/fetch_weights.py --manifest backend/model/best_model.pth
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "weights.json"
TARGET = HERE / "best_model.pth"
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.is_file():
        sys.exit(f"No manifest at {MANIFEST}. Cannot verify a download without it.")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def verify(path: Path, manifest: dict) -> bool:
    """
    True when the file is present and matches the manifest checksum.

    An empty checksum means the manifest has not been filled in yet. That is
    reported as unverified rather than passing quietly — "checksum OK" when
    nothing was checked is the kind of message that hides a corrupt download.
    """
    if not path.is_file():
        return False
    expected = manifest.get("sha256")
    if not expected:
        print("  no sha256 in weights.json — cannot verify this file")
        print(f"  record it with: python {Path(__file__).name} --manifest {path.name}")
        return False
    actual = sha256_of(path)
    if actual != expected:
        print(f"  checksum mismatch\n    expected {expected}\n    actual   {actual}")
        return False
    return True


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            got = 0
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                if total:
                    pct = got * 100 / total
                    print(f"\r  {got/1e6:6.1f} / {total/1e6:.1f} MB  ({pct:5.1f}%)", end="", flush=True)
            print()
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        sys.exit(
            f"HTTP {exc.code} fetching the checkpoint.\n"
            f"Check the url in {MANIFEST.name}, or set WEIGHTS_URL to override it."
        )
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        sys.exit(f"Could not reach the download host: {exc.reason}")
    tmp.replace(dest)


def write_manifest(path: Path) -> None:
    """Regenerate weights.json from a local checkpoint before publishing it."""
    manifest = load_manifest() if MANIFEST.is_file() else {}
    manifest.update({
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    })
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"updated {MANIFEST.name}:")
    print(f"  size   {manifest['size_bytes'] / 1e6:.1f} MB")
    print(f"  sha256 {manifest['sha256']}")
    print(f"\nSet 'url' in {MANIFEST.name} to the release asset URL before publishing.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download or verify the trained checkpoint.")
    ap.add_argument("--manifest", metavar="CHECKPOINT",
                    help="Regenerate weights.json from a local checkpoint instead of downloading.")
    ap.add_argument("--force", action="store_true", help="Re-download even if a valid file exists.")
    args = ap.parse_args()

    if args.manifest:
        return write_manifest(Path(args.manifest).resolve())

    manifest = load_manifest()
    url = os.getenv("WEIGHTS_URL") or manifest.get("url")

    if TARGET.is_file() and not args.force:
        print(f"checking existing {TARGET.name} ({TARGET.stat().st_size / 1e6:.1f} MB)")
        if verify(TARGET, manifest):
            print("checksum OK — nothing to do.")
            return
        if not url:
            # An unverifiable local file is still usable; there is just nothing
            # to re-download from. Say so instead of failing.
            print("keeping the existing file (unverified, and no download url configured).")
            return
        print("  re-downloading")

    if not url:
        sys.exit(
            f"No download url configured.\n\n"
            f"The server runs in demo mode without weights, so this is optional for\n"
            f"frontend and clinical-logic work. To enable real inference either:\n"
            f"  - set 'url' in {MANIFEST.name} to a release asset, or\n"
            f"  - set WEIGHTS_URL=<url>, or\n"
            f"  - train your own (see backend/model/TRAINING.md) and place the\n"
            f"    checkpoint at {TARGET}"
        )

    download(url, TARGET)
    if not verify(TARGET, manifest):
        sys.exit("Downloaded file failed checksum verification. Refusing to use it.")
    print(f"saved {TARGET} — restart the server to leave demo mode.")


if __name__ == "__main__":
    main()
