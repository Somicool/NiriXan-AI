"""Focused checks for night-vision / black-and-white footage support.

    python scripts/verify_colorless.py

A  real colour video          -> colorless=False, colour extraction unchanged
B  grayscale video            -> colorless=True, colours only black/white/None
C  dark grayscale clothing    -> black, high confidence
D  bright grayscale clothing  -> white, high confidence
E  ambiguous / unreadable     -> None (mid-grey, glare, deep shadow, too small)
F  explicit colour filter     -> black/white filter on B&W video; hues don't zero it;
                                 colour-video filtering unchanged
Read-only: nothing is written to the database.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, database                                    # noqa: E402
from app.ingestion import attribute_extractor as ax, embedder        # noqa: E402
from app.models.schemas import SearchFilters                         # noqa: E402
from app.search import filters                                       # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def person_crop(garment_v: int, bg_v: int = 120, h: int = 220, w: int = 90, noise=6):
    """Synthetic grayscale person crop: garment fills the centre, background edges."""
    rng = np.random.default_rng(0)
    img = np.full((h, w), bg_v, np.float32)
    img[int(h * .05):int(h * .95), int(w * .15):int(w * .85)] = garment_v
    img += rng.normal(0, noise, img.shape)
    g = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def gray_reencode(src: Path, dst: Path, max_frames=240):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while n < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        out.write(cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
        n += 1
    cap.release(); out.release()


def real_person_crops(k=12):
    with database.get_conn() as c:
        rows = c.execute("SELECT crop_path FROM detections WHERE class_label='person' "
                         "AND crop_path IS NOT NULL LIMIT 60").fetchall()
    imgs = [cv2.imread(r["crop_path"]) for r in rows]
    return [i for i in imgs if i is not None and i.size][:k]


def main() -> int:
    database.init_db()
    colour_video = config.VIDEO_DIR / "test1.mp4"
    crops = real_person_crops()
    pid = sorted(config.PERSON_CLASSES)[0]

    print("A. normal colour video")
    c, s = ax.probe_colorless(colour_video)
    check("A1 colour video not flagged", not c, f"median sat {s}")
    embs = embedder.embed_crops(crops)
    ids = [pid] * len(crops)
    before = ax._extract_batch(crops, ids, embs, region_split=True)
    after = ax.extract_batch(crops, ids, embs, region_split=True)          # colorless default
    check("A2 extract_batch(colorless=False) identical to original", before == after,
          f"{len(crops)} real crops")
    check("A3 no color_mode marker on colour footage",
          all("color_mode" not in r for r in after))

    print("B. grayscale video")
    with tempfile.TemporaryDirectory() as td:
        g = Path(td) / "gray.mp4"
        gray_reencode(colour_video, g)
        c, s = ax.probe_colorless(g)
        check("B1 grayscale re-encode flagged colorless", c, f"median sat {s}")
    gcrops = [cv2.cvtColor(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
              for i in crops]
    gembs = embedder.embed_crops(gcrops)
    vid_cls = sorted(config.VEHICLE_CLASSES)[0]
    res = ax.extract_batch(gcrops + gcrops[:3], [pid] * len(gcrops) + [vid_cls] * 3,
                           np.vstack([gembs, gembs[:3]]), region_split=True, colorless=True)
    allowed = {"black", "white", None}
    vals = [r.get(k) for r in res for k in ("upper_color", "lower_color", "color") if k in r]
    check("B2 only black/white/None produced", set(vals) <= allowed,
          f"values seen: {sorted({str(v) for v in vals})}")
    check("B3 accessories still extracted for persons",
          all("accessories" in r for r in res[:len(gcrops)]))
    check("B4 vehicle_type still extracted for vehicles",
          all(r.get("vehicle_type") for r in res[len(gcrops):]))

    print("C/D/E. black / white / unknown classification")
    col, conf = ax.bw_color(person_crop(25)[:88], person_crop(25))
    check("C dark clothing -> black", col == "black" and conf >= 0.8, f"{col} {conf}")
    col, conf = ax.bw_color(person_crop(225)[:88], person_crop(225))
    check("D bright clothing -> white", col == "white" and conf >= 0.8, f"{col} {conf}")
    col, conf = ax.bw_color(person_crop(125)[:88], person_crop(125))
    check("E1 mid-grey (ambiguous) -> None", col is None and conf == 0.0, f"{col} {conf}")
    glare = person_crop(255, noise=0)
    col, conf = ax.bw_color(glare[:88], glare)
    check("E2 blown-out / IR glare -> None", col is None, f"{col} {conf}")
    shadow = person_crop(8, bg_v=12, noise=2)
    col, conf = ax.bw_color(shadow[:88], shadow)
    check("E3 heavily shadowed scene -> None", col is None, f"{col} {conf}")
    tiny = person_crop(25, h=20, w=8)
    col, conf = ax.bw_color(tiny[:8], tiny)
    check("E4 too small -> None", col is None, f"{col} {conf}")
    half = person_crop(125)
    half[:, : half.shape[1] // 2] = 20
    col, conf = ax.bw_color(half[:88], half)
    check("E5 half dark / half mid-grey -> None", col is None, f"{col} {conf}")

    print("F. explicit colour filter")
    BW_VID, COL_VID = 9001, 9002
    bw_ids = {BW_VID}
    det_bw_black = {"video_id": BW_VID, "class_label": "person",
                    "attributes": {"upper_color": "black", "lower_color": None}}
    det_bw_white = {"video_id": BW_VID, "class_label": "person",
                    "attributes": {"upper_color": "white", "lower_color": "white"}}
    det_col_red = {"video_id": COL_VID, "class_label": "person",
                   "attributes": {"upper_color": "red", "lower_color": "blue"}}

    def m(det, colours):
        return filters.match(det, SearchFilters(colors=colours), bw_ids)

    check("F1 'black' keeps black person on B&W video", m(det_bw_black, ["black"]))
    check("F2 'black' drops white person on B&W video", not m(det_bw_white, ["black"]))
    check("F3 'white' keeps white person on B&W video", m(det_bw_white, ["white"]))
    check("F4 'blue' does NOT zero B&W video results",
          m(det_bw_black, ["blue"]) and m(det_bw_white, ["blue"]))
    check("F5 'blue'+'black' on B&W -> black still enforced",
          m(det_bw_black, ["blue", "black"]) and not m(det_bw_white, ["blue", "black"]))
    check("F6 colour video: 'red' matches red (unchanged)", m(det_col_red, ["red"]))
    check("F7 colour video: 'green' still rejects (unchanged)", not m(det_col_red, ["green"]))
    check("F8 no colorless set -> legacy behaviour",
          not filters.match(det_bw_black, SearchFilters(colors=["blue"])))

    passed = sum(ok for _n, ok, _d in RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
