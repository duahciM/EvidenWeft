"""IF13 offline correspondence and local reference experiments."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

import cv2
import numpy as np
from identity_catalog_matching.common import read, sha
from if07_research.run import peak_mb
from if10_research import spatial
from if12_research import modular as m
from if12_research.contracts import IF10, IF11, SPATIAL, reference
from p1.io import dump
from v05.engine import FEATURE_CFG

from . import geometry as g

HERE=Path(__file__).resolve().parent
RULES=read(HERE/"protocol.json")
OLD_IF12=HERE.parent/"if12_research_20261119"
R=RULES["source"]
V=RULES["validation"]


def auto_plan(n, views):
    pairs={}
    width=R["targets_per_reference_set"]
    def add(a,b,why):
        if a!=b:
            pairs.setdefault(tuple(sorted((a,b))),set()).add(why)
    for h,_,_ in views[:R["max_reference_sets"]]:
        ref=h["endpoints"]["A"]
        if not ref:
            continue
        pivot=ref[len(ref)//2]
        for a,b in combinations(ref,2):
            add(a,b,"A_multiview_reference")
        for i in np.linspace(0,n-2,min(width,n-1)).round().astype(int):
            i=int(i)
            j=min(n-1,i+R["third_view_offset_coarse_frames"])
            add(i,j,"held_independent_source_neighbor")
            for k in (i,j):
                add(k,pivot,"direct_A_comparison")
    return [{"samples":list(k),"roles":sorted(v)} for k,v in sorted(pairs.items())]


def diagnostic_plan(views, branch):
    if branch=="oracle":
        h=views[0][0]
    else:
        h=next(a for a,_,_ in views if a["arm"]=="auto_2")
    refs=h["endpoints"]["A"]
    # Isolated diagnostic scope only. No plate times, bands or positions go into any detector.
    indices=(94,98,102,106,110,112,114,116)
    pairs={tuple(sorted((i,a))):["requested_development_root",f"query:{i}",f"A:{a}"]
           for i in indices for a in refs}
    for a,b in combinations(refs,2):
        pairs[tuple(sorted((a,b)))]=["independent_reference_pair"]
    return [{"samples":list(k),"roles":v} for k,v in sorted(pairs.items())]


def coordinates(f):
    return {k:f[k] for k in ("sample","source_frame","pts","time_base","t","image","sharpness")}


def matcher(out, learned):
    params=FEATURE_CFG["sift"]
    if not learned:
        algo=cv2.SIFT_create(nfeatures=params["nfeatures"])
        def f(frame):
            kp,d=algo.detectAndCompute(frame["gray"],None)
            return np.float32([p.pt for p in kp]).reshape(-1,2),d,[p.response for p in kp]
        return f,{"method":"SIFT","opencv":cv2.__version__,"equivalent_legacy_candidate_gate":True}
    from .learned import backend
    return backend(out,RULES["learning"])


def compute(fs,bank,digest,scale,plan,extract,mode):
    frames=sorted({x for item in plan for x in item["samples"]})
    start=perf_counter()
    failures=getattr(extract,"failures",[])
    before=len(failures)
    features={i:extract(fs[i]) for i in frames}
    feature_wall=perf_counter()-start
    edges={}
    differences=[]
    for item in plan:
        i,j=item["samples"]
        first,second=coordinates(fs[i]),coordinates(fs[j])
        q=min(first["sharpness"],second["sharpness"])>=SPATIAL["reference_quality"]
        old=bank[i,j]
        key=f"{i}:{j}"
        candidates, counts=g.reciprocal(features[i],features[j])
        e0=g.wide(candidates,fs[i]["gray"].shape,key,q)
        e1=g.local(candidates,key,q,V)
        arm_code=("E2","E3") if mode=="learned" else ("E0","E1")
        if mode=="traditional":
            err=[]
            if old["verified"]!=e0["verified_p14"]:
                err.append("verified")
            if old["count"]!=len(e0["inliers"]):
                err.append("inlier_count")
            expected=old["xy"]
            if old["verified"] and old["applies"] and expected[0].shape[0]:
                actual=g.points([candidates[k] for k in e0["inliers"]])
                if any(not np.array_equal(x,y) for x,y in zip(actual,expected)):
                    err.append("inlier_coordinates")
            if err:
                differences.append({"samples":[i,j],"fields":err,"frozen_count":old["count"],"fresh_count":len(e0["inliers"])})
        edges[i,j]={"root":f'{digest}:pair:{first["source_frame"]}:{second["source_frame"]}',
            "pair":[i,j],"source_frames":[first,second],"planned_role":item["roles"],"pre_geometry_counts":dict(zip(("forward","backward"),counts)),
            "coordinate_space":scale,"candidate_sets":{arm_code[0][1]:candidates},
            "arms":dict(zip(arm_code,(e0,e1))),
            "legacy":{"root":old["id"],"quality":old["quality"],"verified":old["verified"],
                       "if10_eligible":spatial.valid(old,SPATIAL),"coordinates_complete":old["applies"],
                       "frozen_geometry_count":old["count"],"frozen_hull":old["hull"],"source_distribution":old["distribution"]}}
        edges[i,j]["candidate_sets"][arm_code[1][1]]=candidates
    return edges,{"pairs_visited":len(plan),"distinct_coarse_frames":len(frames),"new_match_inference_frames":len(frames),
                   "new_source_frame_pts":0,"failed_feature_inputs":failures[before:],"feature_seconds":feature_wall,"total_compute_seconds":perf_counter()-start,
                   "peak_process_MiB":peak_mb(),"legacy_reconstruction_disagreements":differences}


def stage(folder,fs,src,bank,scale,plan,extract,mode,scope):
    edges,meta=compute(fs,bank,src["sha256"],scale,plan,extract,mode)
    arms=("E0","E1") if mode=="traditional" else ("E2","E3")
    cycles={arm:g.triangles(edges,arm,V,scope) for arm in arms}
    with (folder/"roots.jsonl").open("x",encoding="utf8") as stream:
        for row in edges.values():
            packed=row|{"candidate_sets":next(iter(row["candidate_sets"].values()))}
            stream.write(json.dumps(packed,allow_nan=False)+"\n")
    dump(folder/"image_cycles.json",cycles)
    summary={a:{"proposal_points":sum(r["arms"][a]["candidate_count"] for r in edges.values()),
                "geometric_pairs":sum(r["arms"][a]["support"] for r in edges.values()),
                "cycle_states":dict(Counter(x["state"] for x in cycles[a]))} for a in arms}
    obj={"scope":scope,"mode":mode,"arms":summary,"cost":meta,"plan_count":len(plan),"no_metric_map":True,
         "geometry_candidates_not_same_instance":True}
    dump(folder/"summary.json",obj)
    return obj


def run_one(out,run,case,src,extract,mode):
    base=run/"cases"/case
    base.mkdir(parents=True,exist_ok=False)
    (fs,meta),_=m.read_pool(case,src,base)
    (bank,_),_=m.observations(case,fs)
    scale=read(base/"M0_frames.json")["coordinates"]
    views,_=m.frozen_views(case,"auto",len(fs))
    p=auto_plan(len(fs),views)
    dump(base/"scope_locked.json",{"case":case,"auto_plan":p,"source":meta["source"]["sha256"],"source_path":src})
    if mode=="learned":
        assert read(out/"traditional"/"cases"/case/"scope_locked.json")["auto_plan"]==p
    res=stage(base,fs,meta["source"],bank,scale,p,extract,mode,case)
    if case=="F01":
        oracle,_=m.frozen_views(case,"oracle",len(fs))
        for branch, refviews in (("oracle",oracle),("auto2_diagnostic",views)):
            b=run/"diagnostics"/branch
            b.mkdir(parents=True,exist_ok=False)
            plan=diagnostic_plan(refviews,branch)
            dump(b/"scope_locked.json",{"case":case,"diagnostic_only":True,"same_frames_and_points_for_validation_arms":True,
                                         "source_hash":meta["source"]["sha256"],"plan":plan})
            if mode=="learned":
                assert read(out/"traditional"/"diagnostics"/branch/"scope_locked.json")["plan"]==plan
            special=stage(b,fs,meta["source"],bank,scale,plan,extract,mode,branch)
            res[branch]=special
    return res


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mode",choices=("traditional","learned"),required=True)
    args=p.parse_args()
    out=args.out.resolve()
    h=read(out/"history_lock.json")
    srcs={r["sha256"]:r["path"] for r in h["media"]+list(h["historical_sources"].values())}
    run=out/args.mode
    run.mkdir(exist_ok=False)
    dump(run/"protocol_locked.json",RULES)
    tracker={"mode":args.mode,"history":reference(out/"history_lock.json"),
             "code":{p.name:sha(p) for p in (HERE/"run.py",HERE/"geometry.py",HERE/"protocol.json")},
             "baseline_manifest":reference(IF11/"run_manifest.json"),"status":"initializing","cases":[],"failures":[]}
    if args.mode=="learned":
        tracker["code"]["learned.py"]=sha(HERE/"learned.py")
        tracker["resolved_runtime_lock"]=reference(out/"model_backend_locked.json")
        tracker["exact_traditional_scope_manifest"]=reference(out/"traditional/manifest.json")
    dump(run/"lock.json",tracker)
    cv2.setNumThreads(1)
    try:
        extract,device=matcher(out,args.mode=="learned")
    except Exception:
        tracker["status"]="backend_unavailable"
        tracker["failures"].append({"error":traceback.format_exc()})
        dump(run/"manifest.json",tracker)
        raise
    tracker["backend"]=device
    order=read(IF11/"run_manifest.json")["cases"]
    order=sorted(order,key=lambda row:row["case"]!="F01")
    for row in order:
        case=row["case"]
        try:
            hsh=row["source"]["sha256"]
            src=srcs[hsh]
            if sha(src)!=hsh:
                raise ValueError(f"source content changed: {case}")
            res=run_one(out,run,case,src,extract,args.mode)
            tracker["cases"].append(res)
            print(args.mode,case,res["arms"],len(res["cost"]["legacy_reconstruction_disagreements"]),flush=True)
        except Exception:
            tracker["failures"].append({"case":case,"error":traceback.format_exc()})
            print(tracker["failures"][-1]["error"],flush=True)
        tracker["status"]="running"
        dump(run/"manifest.json",tracker)
    tracker["status"]="completed" if not tracker["failures"] else "completed_with_failures"
    dump(run/"manifest.json",tracker)


if __name__=="__main__":
    main()
