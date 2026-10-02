"""Copy the audio a manifest references into DEST (e.g. a Kaggle Dataset folder), verified.

Usage: python scripts/pack_audio.py manifests/bonafide_raw.jsonl /kaggle/working/indispoof-audio
       python scripts/pack_audio.py manifests/bonafide_raw.jsonl --verify-only
"""

import sys

from indispoof.data.manifest import read_manifest
from indispoof.data.storage import pack, verify

if __name__ == "__main__":
    rows = read_manifest(sys.argv[1])
    if sys.argv[2] == "--verify-only":
        problems = verify(rows)
        print("\n".join(problems[:20]) or f"all {len(rows)} files present and unaltered")
        sys.exit(1 if problems else 0)
    print(f"copied {pack(rows, sys.argv[2])} of {len(rows)} files to {sys.argv[2]}")
