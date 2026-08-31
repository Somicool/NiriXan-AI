"""Fetch the CodeFormer ONNX face-restoration model used by Face Enhancement.

    python scripts/download_face_model.py

Run once. ~377 MB, saved to backend/data/models/codeformer.onnx (gitignored).
Face Enhancement works without it - it falls back to selecting the clearest real
frame from the person's track - but the "AI-Enhanced" stage stays empty until the
model is present.

WHY ONNX AND NOT THE PYTORCH PACKAGE
------------------------------------
gfpgan and codeformer both depend on `basicsr`, which imports
torchvision.transforms.functional_tensor. That module was removed in torchvision
0.17 and this project runs 0.20.1, so those packages cannot be imported without
downgrading the torch stack that YOLO, CLIP, OSNet and InsightFace all rely on.
An ONNX build runs on onnxruntime (already installed, CUDA-capable) and touches
none of it.

The download resumes if interrupted, and the file size is verified before use -
a truncated model fails with an opaque protobuf error otherwise.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config                                          # noqa: E402

URL = "https://huggingface.co/bluefoxcreation/Codeformer-ONNX/resolve/main/codeformer.onnx"
UA = {"User-Agent": "Mozilla/5.0 nirixan-ai"}
MAX_ATTEMPTS = 12


def remote_size() -> int:
    req = urllib.request.Request(URL, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)


def main() -> int:
    dest = Path(config.FACE_ENH_CODEFORMER)
    dest.parent.mkdir(parents=True, exist_ok=True)

    want = remote_size()
    if want <= 0:
        print("could not determine the remote file size; aborting")
        return 1
    print(f"CodeFormer ONNX: {want / 1e6:.0f} MB -> {dest}")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= want:
            break
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        print(f"  attempt {attempt}/{MAX_ATTEMPTS}, from {have / 1e6:.0f} MB")
        try:
            req = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(dest, "ab" if have else "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except Exception as e:                    # network drops are expected; resume
            print(f"   interrupted: {type(e).__name__}: {e}")

    have = dest.stat().st_size if dest.exists() else 0
    if have < want:
        print(f"incomplete ({have / 1e6:.0f}/{want / 1e6:.0f} MB) - re-run to resume")
        return 1

    from app import face_enhance
    ok = face_enhance.codeformer_available()
    print(f"downloaded {have / 1e6:.0f} MB; model loads: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
