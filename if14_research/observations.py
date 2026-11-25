"""Compare appearance and environment on frozen direct measurements."""
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from identity_catalog_matching.common import read
from if10_research import spatial
from if12_research.contracts import SPATIAL
from if13_research.geometry import apply
from p14_scene_prototypes.features import hull_coverage

from .judgement import empty,evidence,identifiers,sid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
IF13=ROOT/"if13_research_20261120"
IF09=ROOT/"if09_research_20261119"
RULES=read(HERE/"protocol.json")
Q=RULES["anchor"]


def frame_ref(frame):
    return {k:frame[k] for k in ("sample","source_frame","pts","time_base","t","image")}


def bbox(anchor,shape):
    h,w=shape
    x,y=map(float,anchor)
    return [max(0.,x-Q["patch_half_width"]),max(0.,y-Q["patch_half_height"]),
            min(w-1.,x+Q["patch_half_width"]),min(h-1.,y+Q["patch_half_height"])]


def measure_regions(p, shape):
    if not spatial.valid(p,SPATIAL):
        return {"status":"no_qualified_frozen_pair","families":[],
                "legacy_reason":"unavailable_quality_or_insufficient_geometric_support"}
    a,b=p["xy"]
    n=len(a)
    s=np.linspace(0,n-1,min(n,Q["patch_seeds"])).round().astype(int)
    held=np.arange(n)%Q["held_every"]==0
    axis=np.float32([Q["patch_half_width"],Q["patch_half_height"]])
    family=[]
    for j in s:
        i=int(j)
        near_a=np.all(np.abs(a-a[i])<=axis,axis=1)
        near_b=np.all(np.abs(b-b[i])<=axis,axis=1)
        inn=near_a&near_b
        ext=~(near_a|near_b)
        train=ext&~held
        test=ext&held
        row={"seed":i,"region_a":bbox(a[i],shape),"region_b":bbox(b[i],shape),
             "inside":int(inn.sum()),"outside_train":int(train.sum()),"outside_held":int(test.sum())}
        if int(inn.sum())<Q["min_inside"]:
            family.append(row|{"state":"insufficient_target_points"})
            continue
        if int(train.sum())<Q["min_outside_train"] or int(test.sum())<Q["min_outside_held"]:
            family.append(row|{"state":"insufficient_excluded_context"})
            continue
        if min(hull_coverage(a[train],shape),hull_coverage(b[train],shape))<.008:
            family.append(row|{"state":"degenerate_anchor_domain"})
            continue
        P=np.c_[a[train],np.ones(int(train.sum()))]
        model=np.vstack([np.linalg.lstsq(P,b[train],rcond=None)[0].T,[0,0,1]])
        if np.linalg.det(model[:2,:2])<=0:
            family.append(row|{"state":"unusable_or_reflecting_context_fit"})
            continue
        predicted=apply(model,a)
        if predicted is None:
            family.append(row|{"state":"no_valid_context_projection"})
            continue
        error=np.linalg.norm(predicted-b,axis=1)
        mid=float(np.median(error[test]))
        tail=float(np.quantile(error[test],.9))
        target_error=float(np.median(error[inn]))
        ok=mid<=Q["median_px"] and tail<=Q["p90_px"]
        if not ok:
            state="context_holdout_unreliable"
        elif target_error<=Q["median_px"]:
            state="context_and_local_slot_compatible"
        else:
            state="target_geometry_or_view_competes"
        family.append(row|{"state":state,"context_held_median_px":mid,"context_held_p90_px":tail,
          "target_projection_median_px":target_error,"model":model.ravel().tolist(),
          "target_points_a":a[inn].tolist(),"target_points_b":b[inn].tolist(),
          "held_context_a":a[test].tolist(),"held_context_b":b[test].tolist()})
    ok=[a for a in family if a["state"]=="context_and_local_slot_compatible"]
    review=sorted(ok,key=lambda q:(-q["inside"],q["context_held_median_px"],q["seed"]))
    return {"status":"local_image_compatible" if review else "no_safe_context_slot_comparison",
            "families":family,"best":review[0] if review else None,"target_excluded_from_context_fit":True,
            "visual_identity_uniqueness_measured":False,"shared_pair_root":p["id"]}


def choose_pairs(k_role,shadow,bank,fs):
    group=sorted({f["sample"] for f in k_role+shadow})
    all_pairs=list(combinations(group,2))
    old=sorted(all_pairs,key=lambda p:(-(fs[p[1]]["t"]-fs[p[0]]["t"]),p))
    eligible=[p for p in old if spatial.valid(bank.get(p),SPATIAL)]
    unknown=next((p for p in old if p not in eligible),None)
    nearby=min(eligible,key=lambda p:(fs[p[1]]["t"]-fs[p[0]]["t"],p),default=None)
    order=list(dict.fromkeys(p for p in [unknown,nearby]+eligible+old if p is not None))
    return order[:RULES["source"]["per_clip_pair_cap"]]


def pair_packet(case,meta,fs,pair,bank):
    a,b=pair
    item=bank.get(pair)
    h=measure_regions(item,fs[a]["gray"].shape)
    ref=meta["source"]["sha256"]
    root=f"{ref}:{item['id']}" if item else None
    fs_pair=[frame_ref(fs[i]) for i in pair]
    roots=[root] if root else []
    X=evidence("X","partially_compatible",roots,group="unidentified local image features") if spatial.valid(item,SPATIAL) else empty("X")
    if h.get("best"):
        G=evidence("G","image_slot_compatible",roots,space="unscaled image-space context registration")
        N=evidence("N","context_compatible",roots,space="outside-patch heldout context")
    else:
        G,N=empty("G"),empty("N")
    T=empty("T")|{"reason":"two sampled endpoints do not verify the intervening target"}
    return {"comparison_id":sid("comparison",ref,*pair),"origin":"real_frozen_frame_pair",
            "case":case,"case_scope":"seen_development","source_sha256":ref,
            "observations":identifiers(ref,fs_pair),"source_samples":list(pair),
            "unobserved_open_intervals":[[f["t"] for f in fs_pair]],"physical_scene_id":None,
            "replacement_truth":"unknown","evidence":[X,G,N,T],"visual_measurement":h,
            "anchor_dependency":"X G N share one frozen SIFT root when available"}


def relative_motion(case,meta,fs):
    file=IF09/"P2_04"/case/"report_evidence.json"
    if not file.is_file():
        return None
    d=read(file)
    match=next((s for s in d["S2"] if s["state"]=="supports_specific_relative_image_motion" and
                s["initial_middle_final"] and s["initial_middle_final"].get("middle") is not None),None)
    if not match:
        return None
    trace=match["initial_middle_final"]
    selected=[trace[k] for k in ("before","middle","after")]
    ledger={x["sample"]:x for x in match["points"]}
    stamps=[frame_ref(fs[i]) for i in selected]
    points=[ledger[i] for i in selected]
    ref=meta["source"]["sha256"]
    roots=[f"{ref}:{x['background']['root']}" for x in points if x["background"].get("root")]
    chain=[frame_ref(fs[i]) for i in range(selected[0],selected[-1]+1) if
           i in ledger and ledger[i]["center_relative_to_reference_xy"] is not None]
    missing=[[x["t"],y["t"]] for x,y in zip(chain,chain[1:]) if y["source_frame"]>x["source_frame"]+1]
    return {"comparison_id":sid("relative_displacement",ref,*selected),"origin":"real_rereadable_frozen_S2",
       "case":case,"case_scope":"seen_development","source_sha256":ref,"observations":identifiers(ref,stamps),
       "source_samples":selected,"unobserved_open_intervals":missing,"physical_scene_id":None,
       "replacement_truth":"unknown","visual_measurement":{"roles":trace,"local_regions":points,"matched_chain_frames":chain,
                                              "source_trace":str(file),"physical_cause":None},
       "evidence":[evidence("X","partially_compatible",roots,reason="candidate region appearance, no verified individual discriminator"),
                   empty("G")|{"reason":"relative position changes, association of a physical slot never claimed"},
                   evidence("N","context_compatible",roots,reason="independent-in-pixel background fit; reuses this pair ledger"),
                   evidence("T","observed_displacement",roots,reason="signed relative image displacement across actual sampled points",
                      source_samples=selected)]}
