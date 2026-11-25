"""Reuse the historical 11/10 native frames and exact historical contact sheets."""
import argparse
from pathlib import Path

from PIL import Image

from if15_research.audit import checks_for
from if17_research.catalog import read, ref, sha
from if19_research.inputs import ANSWER_KEY, native_rows
from if20_research.direct import inspect_png
from p1.io import dump

from .protocol import HERE, ROOT, protocol


OFF = ROOT / "if19_research_20261121"


PIXEL_KEY = {
    "short": {
        3720: "Red ball separate from gripper on white tabletop to the left of blue tray.",
        87720: "Red ball on tabletop left of tray. Gripper above/near it; report only visible contact evidence.",
        153720: "Red ball under/near gripping fingers, to the left of the tray. No clear separated late-type gap.",
        165720: "Red ball under the gripper and near the tray direction. Fingers and ball appear near/overlapping in projection.",
        189720: "Ball between finger tips by tray rim; silhouettes adjoin/overlap without a clear visible gap; force/contact in 3D not established.",
        195720: "Red ball close to or under the fingers at the tray; perspective/contact may be ambiguous.",
        201720: "Ball at/in near-left tray; finger tips close above in projection. Judge only the visible relation.",
        207720: "Ball in blue tray with a clear visible gap from the open gripper tips above.",
        213720: "Ball in blue tray with a clear visible gap from the gripper tips above-left.",
        219720: "Ball in blue tray with a clear visible gap from the gripper tips above-left.",
        303720: "Ball in blue tray clearly apart from the gripper, which is elevated and to the left."
    },
    "g3": {
        2400: "Cup on table. No clearly discernible separate red-white target on either side in this sample.",
        2640: "Distinct blurry red-white object is on the image right of the cup, near the right edge.",
        2700: "Red-white target is on the image right of the cup.",
        2800: "Red-white target is on the image right of the cup, near cup edge.",
        2860: "Neither side of cup shows a discernible red-white target in this sample.",
        3000: "Neither side of cup shows a discernible red-white target in this sample.",
        3160: "No discernible red-white target in this sample.",
        3200: "Target's position near/left of the cup boundary can be ambiguous; avoid exact continuous trajectory.",
        3240: "Red-white object clearly on the image left of cup.",
        3300: "Red-white object clearly on the image left of cup."
    }
}


def clip_frame(src, folder, historical, row, position, sheet):
    pts = row["pts"]
    path = Path(row["native_image"])
    native = inspect_png(path.read_bytes())
    if native["sha256"] != row["native_image_sha256"] or (native["width"], native["height"]) != (src["probe"]["width"], src["probe"]["height"]):
        raise ValueError(f"Source picture mismatch at PTS {pts}")
    if any(row[k] != position[k] for k in ("source_frame", "pts", "time_base")) or abs(row["t"] - position["seconds"]) > 1e-6:
        raise ValueError(f"Source time mismatch at PTS {pts}")
    scaled = folder / f"cell_{pts}.png"
    with Image.open(path) as original:
        img = original.convert("RGB")
        img.thumbnail((340, 191))  # identical Pillow operation to frozen rendered_page
        x = (position["column"] - 1) * 340
        y = 26 + (position["row"] - 1) * (191 + 58)
        box = [x, y, x + img.width, y + img.height]
        if sheet.crop(box).tobytes() != img.tobytes():
            raise ValueError(f"Legacy sheet pixels do not match generated tile {pts}")
        img.save(scaled)
    return {"pts": pts, "t": row["t"], "time_base": row["time_base"], "source_frame": row["source_frame"],
            "native_png": ref(path) | {"image": native},
            "cell_png": ref(scaled) | {"image": inspect_png(scaled.read_bytes())},
            "sheet_cell_xyxy": box, "legacy_display_ordinal": position["ordinal"],
            "source_sha256": src["sha256"], "previous_delivery": {"history": historical,
                 "call": row["prior_call"], "attachment_sha256": row["prior_attachment_sha256"]}}


def prepare(out, card):
    folder = out / "input"
    folder.mkdir(exist_ok=False, parents=True)
    packets = {}
    for case in ("short", "g3"):
        old = OFF / "fixed_input" / case / "input.json"
        frozen = read(old)
        src, rows, prior = native_rows(case)
        audited = checks_for(src, rows, [], [], [], len(rows))
        if not audited["pass"] or sha(src["path"]) != src["sha256"]:
            raise ValueError(f"Original source validation failed: {case}")
        card_dir = folder / case
        card_dir.mkdir()
        sheet_path = Path(frozen["image"]["path"])
        if sha(sheet_path) != frozen["image"]["sha256"]:
            raise ValueError(f"Legacy sheet hash changed: {case}")
        with Image.open(sheet_path) as png:
            sheet = png.convert("RGB")
            frames = [clip_frame(src, card_dir, prior, row, pos, sheet)
                      for row, pos in zip(rows, frozen["positions"], strict=True)]
        packets[case] = {"source": src, "frames": frames, "sheet": ref(sheet_path) | {"image": inspect_png(sheet_path.read_bytes())},
                         "legacy_input": ref(old), "old_position_table": frozen["positions"],
                         "original_source_audit": audited}
    dump(folder / "frames.json", packets)
    # Reference data stay outside all predictor packets.
    key = out / "evaluation_only" / "reference.json"
    dump(key, {"per_frame": PIXEL_KEY, "historical_reference": ANSWER_KEY,
               "criteria": "Score explicit source relations; absence of a mention is omitted. Exact release, hidden path and identity are unknown. Use original per-frame pixels and delivery receipts for adjudication.",
               "reference_sources": {c: ref(OFF / "evaluation_only" / f"{c}_reference.json") for c in packets}})
    dump(out / "protocol_lock.json", {"study": protocol(), "card": ref(card), "packets": ref(folder / "frames.json"),
                                     "evaluation_only": ref(key), "code": [ref(path) for path in sorted(HERE.glob("*.py"))]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--card", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.out.resolve(), args.card.resolve())
