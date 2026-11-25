"""Third-view verification with common-feature points excluded from both fits."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from identity_catalog_matching.common import read,sha
from p1.io import dump

from .geometry import apply,points
from .integrate import pair_data
from .run import RULES

V=RULES["validation"]


def good_rows(edge,method):
    return [edge["candidate_sets"][i] for i in edge["arms"][method]["inliers"]]


def fit(rows,local):
    if len(rows)<(6 if local else 8):
        return None
    a,b=points(rows)
    if np.linalg.matrix_rank(np.c_[a,np.ones(len(a))])<3:
        return None
    if np.linalg.matrix_rank(np.c_[b,np.ones(len(b))])<3:
        return None
    if local:
        coef=np.linalg.lstsq(np.c_[a,np.ones(len(a))],b,rcond=None)[0]
        H=np.vstack([coef.T,[0,0,1]])
    else:
        H,_=cv2.findHomography(a,b,0)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-10:
        return None
    return H


def checked_chart(frames, bank, method, reference):
    a,b,c=frames
    edges=[bank[k] for k in [(a,b),(b,c),(a,c)]]
    p,q,r=[good_rows(e,method) for e in edges]
    bc={j["features"][0]:j for j in q}
    ac={j["features"][0]:j for j in r}
    consistent=[]
    conflicts=[]
    for x in p:
        ai,bi=x["features"]
        if bi not in bc or ai not in ac:
            continue
        z,y=bc[bi],ac[ai]
        if z["features"][1]!=y["features"][1]:
            conflicts.append({"observations":[x["a_xy"],x["b_xy"],z["b_xy"],y["b_xy"]],
                              "reason":"contradictory_direct_third_feature_id"})
        else:
            consistent.append((x,z,y))
    held=sorted(consistent,key=lambda t:tuple(t[0]["features"]))[::3]
    excludedA={x["features"][0] for x,_,_ in held}
    excludedB={z["features"][0] for _,z,_ in held}
    train_ab=[x for x in p if x["features"][0] not in excludedA]
    train_bc=[z for z in q if z["features"][0] not in excludedB]
    # AC is disjoint as a pair and is never used for fitting either transform.
    H1,H2=(fit(z,method in ("E1","E3")) for z in (train_ab,train_bc))
    row={"id":reference["id"],"frames":frames,"method":method,"anchor_image_sample":a,
         "fit_roots":[e["root"] for e in edges[:2]],"independent_check_root":edges[2]["root"],
         "training_sizes":[len(train_ab),len(train_bc)],"excluded_point_counts":[len(excludedA),len(excludedB)],
         "model_affinity":"local_affine" if method in ("E1","E3") else "projective_2d",
         "third_view_was_not_fit":True,"competing_matches":conflicts,
         "metric_position":None,"camera_pose":None,"stationarity":"unresolved",
         "status":"no_heldout_fit"}
    if H1 is None or H2 is None or len(held)<V["minimum_track_cycles"]:
        return row
    acc=[]
    for x,z,y in held:
        b1=apply(H1,[x["a_xy"]])
        c1=apply(H2,b1) if b1 is not None else None
        if b1 is None or c1 is None:
            continue
        e1=float(np.linalg.norm(b1[0]-x["b_xy"]))
        e2=float(np.linalg.norm(c1[0]-y["b_xy"]))
        lid=sha256(f'{edges[0]["root"].split(":pair:")[0]}:{method}:{a}:{x["features"][0]}'.encode()).hexdigest()[:16]
        acc.append({"local_landmark_id":f"IMG:{lid}","track_feature_ids":[*x["features"],y["features"][1]],
               "observed_a_b_c_xy":[x["a_xy"],x["b_xy"],y["b_xy"]],
               "predicted_third_xy":c1[0].tolist(),"error_intermediate_px":e1,"error_third_px":e2,
               "support":max(e1,e2)<=V["third_view_snap_px"]})
    actual=sum(v["support"] for v in acc)
    retain=actual>=V["minimum_track_cycles"] and actual>=V["minimum_track_cycles"]*len(acc)/4 and not conflicts
    row.update(status="validated_local_chart" if retain else "competing_or_heldout_failed",
               predictions=acc,independent_supported_tracks=actual,
               transforms={"from_anchor_to_neighbor":H1.ravel().tolist(),"neighbor_to_test":H2.ravel().tolist()},
               no_transitive_merge=True)
    return row


def generate(run):
    cases=[]
    for cat in ("cases","diagnostics"):
        for path in sorted((run/cat).glob("*/image_cycles.json")):
            root=path.parent
            candidates=pair_data(root)
            results={}
            for mode,rows in read(path).items():
                results[mode]=[checked_chart(row["observations"],candidates,mode,row) for row in rows]
            dest=root/"heldout_charts.json"
            dump(dest,results)
            cases.append({"scope":root.name,"kind":cat,
                          "states":{mode:dict(Counter(r["status"] for r in rows)) for mode,rows in results.items()}})
    dump(run/"heldout_chart_manifest.json",{"cases":cases,"code":sha(__file__),
         "source_scope":"already matched triangle proposals; no added frame or pair budget",
         "retention_rule":"at least 3 tracks and 75% heldout, both residuals <=3 px; conflict abstains",
         "only_2D_local_charts":True})
    print(run.name,"checked",len(cases),"groups",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mode",choices=("traditional","learned"),required=True)
    a=p.parse_args()
    generate(a.out.resolve()/a.mode)
