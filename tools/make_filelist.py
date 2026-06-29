from __future__ import annotations

import argparse
from pathlib import Path


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in VALID_EXTS])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in files), encoding="utf-8")
    print(f"Wrote {len(files)} paths to {out}")


if __name__ == "__main__":
    main()
