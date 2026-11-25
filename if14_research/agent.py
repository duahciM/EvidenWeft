"""P3-05 developer smoke with one source-root-backed hint and unchanged query loop."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from identity_catalog_matching.common import read,sha
from if11_research.agent import CLI
from if12_research.contracts import IF11,reference
from if13_research import agent as if13
from p1.io import dump

from .observations import HERE,RULES


def select_hint(out,case):
    d=out/"R_real"/case
    p=d/"M8_shadow_hints.json"
    h=next((x for x in read(p)["hints"] if x["both_available_in_agent_initial"]),None)
    if h:
        return h|{"source_packet":str(p),"hint_level":"same local chart proposal; physical slot is unverified"}
    initial=read(IF11/"cases"/case/"selection.json")["arms"][RULES["agent"]["initial_arm"]]
    available={r["sample"] for r in initial}
    file=d/"packets.json"
    options=[r for r in read(file) if r["origin"]=="real_frozen_frame_pair" and
              set(r["source_samples"])<=available and r["evidence"][0]["state"]=="partially_compatible"
              and r["unobserved_open_intervals"][0][1]-r["unobserved_open_intervals"][0][0]>=RULES["anchor"]["minimum_gap_s"]]
    if not options:
        return None
    r=max(options,key=lambda q:q["unobserved_open_intervals"][0][1]-q["unobserved_open_intervals"][0][0])
    return {"comparison_id":r["comparison_id"],"root_ids":r["evidence"][0]["roots"],
          "unverified_target_endpoints_gap":r["unobserved_open_intervals"][0],
          "endpoint_stamps":[x["frame"] for x in r["observations"]],"source_packet":str(file),
          "hint_level":"partial pretrained-free visual match; same slot not established",
          "physical_slot_verified":False,"instance_or_replacement_verified":False,
          "both_available_in_agent_initial":True,"direct_image_evidence":False}


def observe(out,case,cli):
    hint=select_hint(out,case)
    trace=out/f"P3_{case}_intervention_lock.json"
    cfg=RULES["agent"]
    assert all(if13.RULES[k]==cfg[k] for k in ("provider","model","reasoningEffort","maxTokens","initial_arm"))
    if not hint:
        dump(trace,{"case":case,"state":"not_run","reason":"no source-root-backed hypothesis on two delivered endpoints"})
        print(case,"no eligible hint",flush=True)
        return
    a,b=(r["t"] for r in hint["endpoint_stamps"])
    note=(f"Optional retrieval hint, NOT a visual fact: an algorithm found partial local visual similarity "
          f"between delivered samples at {a:.6f}s and {b:.6f}s. The hint does not establish what the "
          f"patch is, whether it occupies the same physical slot, or whether the instance was unchanged, "
          f"moved, occluded, or replaced. That association is not verified in the intervening observations. "
          f"Use only delivered frames for direct claims. You may ignore this hint or freely request a useful "
          f"source window; no automatic query is required.")
    dump(trace,{"case":case,"state":"will_run","new_input_only":note,"hint":hint,"model":cfg,
                "same_prior_engine":reference(if13.__file__),"wrapper_code":sha(__file__),
                "old_no_hint_run":str(HERE.parent/"if13_research_20261120"),
                "scope":"seen development diagnosis, no experimental ground truth provided"})
    with patch.object(if13,"ADD",if13.ADD+"\n"+note):
        if13.observe(out,case,cli,"P3")
    result=read(out/"P3"/case/"result.json")
    dump(out/"P3"/case/"hint_receipt.json",{"first_prompt_contains_exact_hint":note in result["first_prompt"],
          "second_prompt_contains_exact_hint":note in result.get("second_prompt",""),"selection_unchanged":len(result["initial"])==8,
          "hint_state":"unverified_retrieval_only","attempted_free_windows":len(result.get("reads",[])),
          "coverage_status":result["status"],"hint_reference":reference(trace), "directly_delivered_offline_M8_images":False})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--case",choices=RULES["agent"]["cases"])
    p.add_argument("--cli",type=Path,default=CLI)
    a=p.parse_args()
    for case in [a.case] if a.case else RULES["agent"]["cases"]:
        observe(a.out.resolve(),case,a.cli)


if __name__=="__main__":
    main()
