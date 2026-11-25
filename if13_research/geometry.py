"""The same reciprocal candidate gate, frozen homography and held-out local affine."""
from hashlib import sha256
from itertools import combinations
from math import isfinite

import cv2
import numpy as np
from p14_scene_prototypes.features import hull_coverage
from v05.engine import FEATURE_CFG

SIFT=FEATURE_CFG["sift"]


def reciprocal(a,b):
    qa,da,sa=a
    qb,db,sb=b
    if da is None or db is None or min(len(da),len(db))<2:
        return [],(0,0)
    m=cv2.BFMatcher(cv2.NORM_L2)
    def neighbors(x,y,flip=False):
        accepted={}
        for two in m.knnMatch(x,y,k=2):
            if len(two)!=2 or two[0].distance>=SIFT["ratio_test"]*two[1].distance:
                continue
            p,q=two
            key=(p.trainIdx,p.queryIdx) if flip else (p.queryIdx,p.trainIdx)
            accepted[key]=float(p.distance/max(q.distance,1e-12))
        return accepted
    fw,bw=neighbors(da,db),neighbors(db,da,True)
    found=[{"features":[i,j],"a_xy":qa[i].tolist(),"b_xy":qb[j].tolist(),
            "relative_nn_distance":[fw[i,j],bw[i,j]],"detector_response":[float(sa[i]),float(sb[j])]}
           for i,j in sorted(fw.keys()&bw.keys())]
    return found,(len(fw),len(bw))


def points(cands):
    return [np.float32([p[k] for p in cands]).reshape(-1,2) for k in ("a_xy","b_xy")]


def apply(matrix, pts):
    H=np.asarray(matrix,dtype=np.float64).reshape(3,3)
    q=np.c_[np.asarray(pts),np.ones(len(pts))] @ H.T
    if not len(q) or np.min(np.abs(q[:,2]))<1e-9:
        return None
    q=q[:,:2]/q[:,2,None]
    return q if np.isfinite(q).all() else None


def seed(key):
    return (SIFT["seed"]+int(sha256(key.encode()).hexdigest()[:7],16))%(2**31-1)


def wide(cands,shape,key,quality):
    n=len(cands)
    info={"candidate_count":n,"support":False,"inliers":[],"model":None,"quality":quality,
          "reason":"insufficient_mutual_matches","verified_p14":False}
    if n<SIFT["minimum_mutual_matches"]:
        return info
    a,b=points(cands)
    cv2.setRNGSeed(seed(key))
    H,mask=cv2.findHomography(a,b,cv2.RANSAC,SIFT["ransac_threshold_px"],
              maxIters=SIFT["ransac_max_iterations"],confidence=SIFT["ransac_confidence"])
    if H is None or mask is None or not np.isfinite(H).all():
        return info|{"reason":"geometry_failed"}
    keep=np.flatnonzero(mask.ravel())
    hull=[hull_coverage(p[keep],shape) for p in (a,b)]
    frac=len(keep)/n
    why=next((name for check,name in (
        (len(keep)<SIFT["minimum_inliers"],"too_few_inliers"),
        (frac<SIFT["minimum_inlier_ratio"],"low_inlier_fraction"),
        (min(hull)<SIFT["minimum_hull_coverage_each_image"],"insufficient_spatial_coverage")) if check),None)
    state=why or ("quality_ineligible" if not quality else "accepted")
    return info|{"support":state=="accepted","verified_p14":why is None,"reason":state,
                 "p14_reason":why or "geometrically_verified_local_correspondence",
                 "inliers":keep.tolist(),"model":H.ravel().tolist(),"hull":hull,"fraction":frac}


def local(cands,key,quality,rules):
    n=len(cands)
    reason="quality_ineligible" if not quality else "insufficient_mutual_candidates" if n<rules["minimum_local_candidates"] else None
    if reason:
        return {"support":False,"reason":reason,"hypotheses":[],"candidate_count":n,
                "inliers":[],"model":None}
    a,b=points(cands)
    ids=np.arange(n)
    held=(ids%rules["held_period"])==0
    eligible=ids[~held]
    num=min(12,len(eligible))
    centers=eligible[np.linspace(0,len(eligible)-1,num).round().astype(int)]
    radius=rules["local_radius_px"]
    seen=set()
    hypotheses=[]
    for c in centers:
        near=(np.linalg.norm(a-a[c],axis=1)<=radius)&(np.linalg.norm(b-b[c],axis=1)<=radius)
        unique=tuple(ids[near])
        if unique in seen:
            continue
        seen.add(unique)
        fit=ids[near & ~held]
        test=ids[near & held]
        row={"center_match":int(c),"near_candidates":list(map(int,unique)),"train":fit.tolist(),"held":test.tolist()}
        if len(fit)<6 or len(test)<3:
            hypotheses.append(row|{"state":"too_few_neighborhood_points"})
            continue
        if min(hull_coverage(a[fit],(int(radius*2),int(radius*2))),
               hull_coverage(b[fit],(int(radius*2),int(radius*2))))<.008:
            hypotheses.append(row|{"state":"degenerate_training_footprint"})
            continue
        cv2.setRNGSeed(seed(f"{key}:local:{int(c)}"))
        M,mask=cv2.estimateAffine2D(a[fit],b[fit],method=cv2.RANSAC,
            ransacReprojThreshold=SIFT["ransac_threshold_px"],
            maxIters=SIFT["ransac_max_iterations"],confidence=SIFT["ransac_confidence"])
        if M is None or mask is None or int(mask.sum())<6:
            hypotheses.append(row|{"state":"local_fit_rejected"})
            continue
        l=np.linalg.svd(M[:,:2],compute_uv=False)
        if l[1]<=1e-6 or l[0]/l[1]>rules["max_local_condition"] or np.linalg.det(M[:,:2])<=0:
            hypotheses.append(row|{"state":"degenerate_transform"})
            continue
        H=np.vstack([M,[0,0,1]])
        P=apply(H,a[test])
        Q=apply(np.linalg.inv(H),b[test])
        if P is None or Q is None:
            hypotheses.append(row|{"state":"non_finite_local_projection"})
            continue
        errors=np.maximum(np.linalg.norm(P-b[test],axis=1),np.linalg.norm(Q-a[test],axis=1))
        p50,p90=float(np.median(errors)),float(np.quantile(errors,.9))
        ok=p50<=rules["held_median_px"] and p90<=rules["held_p90_px"]
        inlier=sorted(set(fit[mask.ravel()>0].tolist()+test[errors<=rules["held_p90_px"]].tolist()))
        hypotheses.append(row|{"state":"passed" if ok else "heldout_failed","median_px":p50,"p90_px":p90,
                            "held_errors_px":errors.tolist(),"model":H.ravel().tolist(),"inliers":inlier})
    accepted=sorted((h for h in hypotheses if h["state"]=="passed"),
                    key=lambda h:(-len(h["held"]),h["median_px"],-len(h["inliers"]),h["center_match"]))
    first=accepted[0] if accepted else {}
    return {"support":bool(first),"reason":"accepted_local_only" if first else "no_heldout_local_model",
            "candidate_count":n,"inliers":first.get("inliers",[]),"model":first.get("model"),
            "held":first.get("held",[]),"hypotheses":hypotheses}


def triangles(edges,method,rules,scope):
    graph={}
    for key,e in edges.items():
        result=e["arms"].get(method)
        if result and result["support"]:
            a,b=key
            graph.setdefault(a,set()).add(b)
    out=[]
    for a,links in sorted(graph.items()):
        for b,c in combinations(sorted(links),2):
            if c not in graph.get(b,set()):
                continue
            roles=[edges[k] for k in ((a,b),(b,c),(a,c))]
            one,two,three=[{p["features"][0]:p for i,p in enumerate(x["candidate_sets"][method[1]]) if i in
                            set(x["arms"][method]["inliers"])} for x in roles]
            tested,tracks=[],[]
            for i,x in one.items():
                j=x["features"][1]
                if j not in two or i not in three:
                    continue
                y,z=two[j],three[i]
                closed=y["features"][1]==z["features"][1]
                checked={"ids":[i,j,y["features"][1]],"direct_final_id":z["features"][1],
                         "actual_xy":[x["a_xy"],x["b_xy"],y["b_xy"]],"confirmed":False}
                if closed:
                    H1,H2,H3=[q["arms"][method]["model"] for q in roles]
                    A=apply(H1,[x["a_xy"]])
                    B=apply(H2,A) if A is not None else None
                    C=apply(H3,[x["a_xy"]])
                    if B is not None and C is not None:
                        distance=float(max(np.linalg.norm(B[0]-C[0]),np.linalg.norm(B[0]-y["b_xy"])))
                        checked.update(image_cycle_error_px=distance,confirmed=distance<=rules["third_view_snap_px"])
                tested.append(checked)
                if checked["confirmed"]:
                    tracks.append(checked)
            extra=len(tracks)>=rules["minimum_track_cycles"]
            conflicting=sum(not r["confirmed"] for r in tested)
            out.append({"id":f"{scope}:{method}:{a}:{b}:{c}","observations":[a,b,c],
                        "pair_roots":[q["root"] for q in roles],"state":"local_image_cycle" if extra and not conflicting else
                            "competing_or_inconsistent" if conflicting else "insufficient_third_view",
                        "supported_cycle_count":len(tracks),"conflicting_cycle_count":conflicting,
                        "tracks":tested,"map_position":None,"scale":None,"scene_id":None,
                        "independent_heldout_new_view_observed":True})
    return out
