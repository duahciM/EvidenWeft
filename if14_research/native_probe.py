"""One locked native-versus-320 SIFT control on the same F01 roots."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
from identity_catalog_matching.common import read,sha
from if12_research.contracts import reference
from if13_research.evaluate import pair_panel
from if13_research.geometry import SIFT, local, reciprocal, triangles, wide
from if13_research.integrate import pair_data
from if13_research.landmarks import checked_chart
from if13_research.run import V
from p1.io import command,dump
from tempfile import TemporaryDirectory
from v05.engine import select_expression

from .observations import IF13,RULES


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--decode-gray",action="store_true")
    a=p.parse_args()
    config=RULES["native"]
    h=IF13/"traditional"/"diagnostics"/config["diagnostic_group"]
    stamps_file=IF13/"source_pair_audit/native_manifest.json"
    root=a.out/("M1_native_gray" if a.decode_gray else "M1_native_resolution")
    root.mkdir(exist_ok=False)
    stamp={f["sample"]:f for f in read(stamps_file)["frames"]}
    old=pair_data(h)
    source=read(IF13/"traditional/cases/F01/scope_locked.json")
    assert sha(source["source_path"])==source["source"]
    dump(root/"lock.json",{"scope":reference(h/"scope_locked.json"),"native_frames":reference(stamps_file),
        "baseline_roots":reference(h/"roots.jsonl"),"source_hash":source["source"],
        "protocol":reference(Path(__file__).with_name("protocol.json")),"code":sha(__file__),"configuration":config,
        "preprocessing":"native ffmpeg decode to gray" if a.decode_gray else "audited native RGB PNG to OpenCV gray"})
    cv2.setNumThreads(2)
    begin=perf_counter()
    sift=cv2.SIFT_create(nfeatures=config["feature_cap"])
    frame_ids=sorted({x for k in old for x in k})
    grays={}
    if a.decode_gray:
        with Image.open(stamp[frame_ids[0]]["native_image"]) as image:
            w,h=image.size
        expr=select_expression([stamp[j]["source_frame"] for j in frame_ids])
        with TemporaryDirectory(prefix="if14_gray_") as tmp:
            script=Path(tmp)/"gray_filter.txt"
            script.write_text(f"select='{expr}',format=gray",encoding="utf8")
            data=command(["ffmpeg","-v","error","-noautorotate","-i",source["source_path"],"-/filter:v",script,
                          "-fps_mode","passthrough","-f","rawvideo","-pix_fmt","gray","pipe:1"])
        array=np.frombuffer(data,np.uint8).reshape(-1,h,w)
        assert array.shape[0]==len(frame_ids)
        grays=dict(zip(frame_ids,array))
    feats={}
    for j in frame_ids:
        im=stamp[j]
        if sha(im["native_image"])!=im["native_image_sha256"]:
            raise ValueError(f"native reference {j} failed SHA")
        if a.decode_gray:
            gray=grays[j]
        else:
            with Image.open(im["native_image"]) as raw:
                gray=cv2.cvtColor(np.asarray(raw.convert("RGB")),cv2.COLOR_RGB2GRAY)
        kp,desc=sift.detectAndCompute(gray,None)
        layout=next(r["coordinate_space"] for r in old.values() if j in r["pair"])
        scale=np.float32([layout["analysis_width"]/gray.shape[1],layout["analysis_height"]/gray.shape[0]])
        xy=np.float32([k.pt for k in kp]).reshape(-1,2)
        feats[j]=((xy+.5)*scale-.5,desc,[k.response for k in kp])
    elapsed_features=perf_counter()-begin
    rows={}
    for (i,j),r in old.items():
        h,w=(r["coordinate_space"][k] for k in ("analysis_height","analysis_width"))
        xy,bid=reciprocal(feats[i],feats[j])
        phase={
            "E0":wide(xy,(h,w),f"{i}:{j}",r["legacy"]["quality"]),
            "E1":local(xy,f"{i}:{j}",r["legacy"]["quality"],V)}
        rows[i,j]=r|{"arms":phase,"candidate_sets":{"0":xy,"1":xy},
             "native_measurement":{"native_proposals":len(xy),"reciprocal_direction_counts":list(bid),
                 "old_320_proposals":len(r["candidate_sets"]),
                 "keypoints_per_image":[len(feats[k][0]) for k in (i,j)],
                 "old_320_reason":{k:z["reason"] for k,z in r["arms"].items()}}}
    charts={}
    for arm in ("E0","E1"):
        proposed=triangles(rows,arm,V,"M1_native")
        normalized={p:r|{"candidate_sets":r["candidate_sets"]["0"]} for p,r in rows.items()}
        charts[arm]=[checked_chart(z["observations"],normalized,arm,z) for z in proposed]
    dump(root/"heldout_charts.json",charts)
    out=[]
    for key,r in rows.items():
        rr=r|{"candidate_sets":r["candidate_sets"]["0"]}
        if "requested_development_root" in r["planned_role"]:
            for arm in charts:
                rr.setdefault("third_view_validated",{})[arm]=[c["id"] for c in charts[arm]
                           if set(key)<=set(c["frames"]) and c["status"]=="validated_local_chart"]
            out.append(rr)
            if rr["candidate_sets"]:
                pair_panel(rr,stamp,root/f"{key[0]}_{key[1]}.png",("E0","E1"),"same F01 native source")
    with (root/"roots.jsonl").open("x",encoding="utf8") as f:
        for r in rows.values():
            f.write(json.dumps(r|{"candidate_sets":r["candidate_sets"]["0"]},allow_nan=False)+"\n")
    report={"scope":"F01 isolated oracle roots; not used to select/query/label","complete_roots":len(rows),
        "target_roots":len(out),"native_candidate_points":sum(len(x["candidate_sets"]) for x in out),
        "analysis_candidate_points":sum(x["native_measurement"]["old_320_proposals"] for x in out),
        "native_geometry_support":{k:sum(x["arms"][k]["support"] for x in out) for k in ("E0","E1")},
        "native_failure_reason":{k:dict(Counter(x["arms"][k]["reason"] for x in out)) for k in ("E0","E1")},
        "target_charts":{k:sum(bool(x["third_view_validated"][k]) for x in out) for k in ("E0","E1")},
        "failures":[],"feature_seconds":elapsed_features,"wall_seconds":perf_counter()-begin,
        "native_decoded_images_reused":len(feats) if not a.decode_gray else 0,
        "actual_native_gray_source_frames":len(grays),"new_source_pts":0,"physical_opening_landmark":None}
    dump(root/"target_root_summary.json",report)
    print(report,flush=True)


if __name__=="__main__":
    main()
