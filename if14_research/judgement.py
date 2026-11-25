"""Conservative, non-transitive identity judgements based on explicit evidence roots."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json

ARMS=("R1","R2","R3")
HYPOTHESES=("same_instance","same_slot_replacement","move_return_or_occlusion","association_or_environment_change")


def sid(prefix, *parts):
    key=sha256(json.dumps(parts,sort_keys=True,ensure_ascii=True).encode()).hexdigest()[:16]
    return f"{prefix}:{key}"


def evidence(channel,state,roots=(),**detail):
    return {"channel":channel,"state":state,"roots":sorted(set(roots))}|detail


def empty(channel):
    return evidence(channel,"unknown")


def case_claim(packet,arm):
    if arm not in ARMS:
        raise ValueError(f"unknown identity arm {arm}")
    use={"R1":{"X"},"R2":{"X","G","N"},"R3":{"X","G","N","T"}}[arm]
    supplied={x["channel"]:x for x in packet["evidence"]}
    obs={k:supplied.get(k,empty(k)) if k in use else evidence(k,"not_in_arm") for k in ("X","G","N","T")}
    for k, item in obs.items():
        if item["state"] not in ("unknown","not_in_arm") and not item["roots"]:
            obs[k]=empty(k)|{"reason":"asserted evidence carried no observation root"}
    x,g,n,t=[obs[k]["state"] for k in ("X","G","N","T")]
    # Compatibility in a local image chart is weaker than fixed-world location.
    slot={"state":g if g not in ("not_in_arm","unknown") else "unknown","physical_position":None}
    scene={"state":("local_context_supported" if n=="context_compatible" else
                    "context_change_unresolved" if n=="context_conflict" else "unknown"),
           "scene_assignment":None}
    comparable_detail=x in ("individually_compatible","individual_conflict")
    independently_observed=t=="observed_sampled_continuation"
    handoff=t=="observed_distinct_handoff"
    certified_handoff=handoff and g=="slot_verified_in_comparable_coordinates" and x=="individual_conflict"
    if x=="individual_conflict":
        inst="different_instance_candidate"
    elif x=="individually_compatible" and independently_observed:
        inst="individual_association_supported_at_observed_points"
    elif comparable_detail or x in ("partially_compatible","shared_appearance"):
        inst="appearance_compatible_identity_unresolved"
    else:
        inst="unknown"
    if certified_handoff:
        repl="observed_distinct_handoff_at_verified_slot"
    elif x=="individual_conflict" and g in ("slot_verified_in_comparable_coordinates","image_slot_compatible"):
        repl="same_slot_replacement_candidate"
    else:
        repl="unknown"
    intervals=deepcopy(packet.get("unobserved_open_intervals",[]))
    hypothesis={h:"open" for h in HYPOTHESES}
    if x=="individual_conflict":
        hypothesis["same_instance"]="strong_contradiction"
    if repl!="unknown":
        hypothesis["same_slot_replacement"]="supported_observed" if certified_handoff else "supported_candidate"
    if t=="observed_displacement":
        hypothesis["move_return_or_occlusion"]="relative_motion_observed"
    if n=="context_conflict":
        hypothesis["association_or_environment_change"]="conflicting_context_requires_review"
    return {"arm":arm,"claim_id":packet["comparison_id"],"instance":{"state":inst,"physical_id_merge":False},
            "replacement":repl,"slot":slot,"scene":scene,"episode":{
                "state":t if t not in ("not_in_arm","unknown") else "unknown","unobserved_open_intervals":intervals,
                "absence_of_replacement_between_samples":None},
            "evidence":obs,"hypotheses":hypothesis,"can_support_same_instance_as_fact":False,
            "replacement_truth":packet["replacement_truth"],"ablation":rankings(obs,inst)}


def rank_score(obs,weights):
    positive={"individually_compatible","observed_sampled_continuation","image_slot_compatible",
              "slot_verified_in_comparable_coordinates","context_compatible"}
    veto="individual_conflict"==obs["X"]["state"] or obs["T"]["state"]=="observed_distinct_handoff"
    counted=set()
    used=[]
    omitted=[]
    missing=[]
    for k in ("X","T","G","N"):
        r=obs[k]
        v=r.get("roots",[])
        if r["state"] not in positive or not v or weights[k]==0:
            missing.append({"channel":k,"state":r["state"],"by_weight_disabled":weights[k]==0})
            continue
        # A channel earns at most one weight. Any shared root makes this derived channel non-independent.
        if counted.intersection(v):
            omitted.append({"channel":k,"shared_roots":sorted(counted.intersection(v))})
            continue
        counted.update(v)
        used.append({"channel":k,"weight":weights[k],"roots":v})
    score=sum(q["weight"] for q in used) if used else None
    return {"usable_association_rank":None if veto else score,"uncalibrated_aux_total":score,
            "individual_conflict_blocks_positive_rank":veto,"counted":used,"de_duplicated":omitted,
            "not_counted_without_measurement":missing,"used_as_identity_probability":False}


def rankings(obs,verdict):
    w={"X":4,"T":4,"G":2,"N":1}
    return {"tiered":{"primary":[{"channel":i,"state":obs[i]["state"]} for i in ("X","T")],
                       "auxiliary":[{"channel":i,"state":obs[i]["state"]} for i in ("G","N")],"verdict":verdict},
            "fixed_4_4_2_1":rank_score(obs,w),"fixed_4_4_2_0":rank_score(obs,w|{"N":0}),
            "ranking_cannot_change_verdict":True}


def identifiers(prefix, frames):
    # Each observation gets an independent hypothesized slot/instance. No union-find over pair support.
    return [{"observation_id":sid("obs",prefix,f["source_frame"]),
             "appearance_id":sid("X",prefix,f["source_frame"]),"slot_id":sid("slot",prefix,f["source_frame"]),
             "context_id":sid("N",prefix,f["source_frame"]),"instance_candidate_id":sid("instance",prefix,f["source_frame"]),
             "track_id":sid("track",prefix,f["source_frame"]),"frame":f} for f in frames]
