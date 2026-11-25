"""R0–R4 entry; judgement, selection and original reading remain separate stages."""
import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter
import traceback

from identity_catalog_matching.common import read,sha
from if10_research.run import export_selection
from if12_research import modular as m
from if12_research.contracts import IF11,reference
from if13_research.integrate import overlap
from p1.io import dump

from .judgement import ARMS,case_claim
from .observations import HERE,ROOT,IF13,RULES,choose_pairs,pair_packet,relative_motion

IF12=ROOT/"if12_research_20261119"
PROTECTED={"initial","interaction_or_middle","final","clip_initial_visual_state","clip_final_visual_state"}


def bounded_window(center,start,end):
    width=RULES["query"]["interval_s"]
    a=max(start,min(center-width/2,end-width))
    return [a,min(end,a+width)]


def make_hints(packets, K_initial, source):
    selected={r["sample"] for r in K_initial}
    choices=[]
    start,end=(source[k] for k in ("start","end"))
    for packet in packets:
        h={r["channel"]:r for r in packet["evidence"]}
        if not (h["X"]["state"]=="partially_compatible" and h["G"]["state"]=="image_slot_compatible"
                and h["T"]["state"]=="unknown"):
            continue
        gap=packet["unobserved_open_intervals"][0]
        if gap[1]-gap[0]<RULES["anchor"]["minimum_gap_s"]:
            continue
        choices.append({"comparison_id":packet["comparison_id"],"root_ids":h["G"]["roots"],
           "source_samples":packet["source_samples"],"endpoint_stamps":[x["frame"] for x in packet["observations"]],
           "observation_link":"locally compatible image patches and surrounding anchors from a shared root",
           "physical_slot_verified":False,"instance_or_replacement_verified":False,
           "unverified_target_endpoints_gap":gap,"shadow_query":bounded_window(sum(gap)/2,start,end),
           "both_available_in_agent_initial":set(packet["source_samples"])<=selected,
           "allocated":False,"direct_image_evidence":False})
    return sorted(choices,key=lambda x:(not x["both_available_in_agent_initial"],
                                        -(x["unverified_target_endpoints_gap"][1]-x["unverified_target_endpoints_gap"][0]),
                                        x["source_samples"]))


def shadow_pick(case,fs,src,folder,old,r0,motion,R3):
    original=r0["shadow"]
    ids=[p["sample"] for p in original]
    oldroles={int(k):set(v) for k,v in old["roles_by_sample"].items()}
    p2=read(IF12/"regions/cases"/case/"P2_07.json")
    roots_by_sample={int(k):{v["root"] for v in vs} for k,vs in p2["verified_image_relations"].items()}
    start,end=src["start"],src["end"]
    def evidence_coverage(picks):
        return (set().union(*(oldroles.get(p,set()) for p in picks)),
            set().union(*(roots_by_sample.get(p,set()) for p in picks)),
            {min(3,int(4*(fs[p]["t"]-start)/max(.01,end-start))) for p in picks})
    proposals=[]
    adopted=None
    selected=original
    if motion and R3["episode"]["state"]=="observed_displacement":
        v=motion["visual_measurement"]
        a,k,b=motion["source_samples"]
        chain={x["sample"] for x in v["matched_chain_frames"]}
        already=sorted(i for i in ids if a<i<b and i in chain)
        roots=[s["background"]["root"] for s in v["local_regions"] if s["background"].get("root")]
        gain={"sample":k,"source":motion["observations"][1]["frame"],"new_role":"observed_intermediate_relative_state",
              "already_covered_by_actual_intermediate_samples":already,"supporting_roots":roots,
              "eligible":not already and k not in ids and len(set(roots))>=2}
        proposals.append(gain)
        if gain["eligible"]:
            before=evidence_coverage(ids)
            for j in ids:
                if j in (a,b) or oldroles.get(j,set())&PROTECTED:
                    continue
                k8=sorted(set(ids)-{j}|{k})
                losses=[sorted(x-y) for x,y in zip(before,evidence_coverage(k8))]
                if not any(losses):
                    adopted={"added":k,"displaced":j,"lost_independent_roles":[],"new_role":gain["new_role"],
                             "comparison_id":motion["comparison_id"]}
                    selected=export_selection(fs,src,{"IF14_shadow":k8},folder/"shadow_images")["IF14_shadow"]
                    break
    times=sorted({r["t"] for r in selected}|{start,end})
    largest=max(zip(times,times[1:]),key=lambda x:x[1]-x[0])
    result={"case":case,"candidate_pool":"same IF13 coarse pool","confirmed_visible_handoff_count":0,
      "budget":8,"baseline":original,"shadow":selected,"candidate_motion_roles":proposals,"swap":adopted,
      "historical_same_pool_refs":[reference(IF11/"cases"/case/"selection.json"),
                                  reference(IF12/"regions/cases"/case/"P2_07.json"),
                                  reference(IF13/"connections"/case/"M7_selection.json")],
      "largest_gap_between_delivered_samples":list(largest),"replacement_truth":"unknown"}
    assert len(result["shadow"])==8
    dump(folder/"M7_R4_selection.json",result)
    return result


def phase_evidence(out,case,path,row):
    root=out/"R_real"/case
    root.mkdir(parents=True,exist_ok=False)
    start=perf_counter()
    (fs,meta),m0=m.read_pool(case,path,root)
    source=meta["source"]|{"path":path}
    (bank,_),m1=m.observations(case,fs)
    old=read(IF11/"cases"/case/"selection.json")
    k_role=old["arms"]["K_roles"]
    baseline=IF13/"connections"/case
    r0=read(baseline/"M7_selection.json")
    pairs=choose_pairs(k_role,r0["shadow"],bank,fs)
    p=[pair_packet(case,meta,fs,selected,bank) for selected in pairs]
    motion=relative_motion(case,meta,fs)
    if motion:
        p.append(motion)
    table={v["comparison_id"]:{k:case_claim(v,k) for k in ARMS} for v in p}
    dump(root/"packets.json",p)
    dump(root/"judgements.json",table)
    dump(root/"R0_frozen.json",{"if13":reference(IF13/"traditional/cases"/case/"summary.json"),
        "image_chart_bank":reference(IF13/"traditional/cases"/case/"heldout_charts.json"),
        "if13_M4":reference(baseline/"M4_candidates.json"),"if13_M7":reference(baseline/"M7_selection.json"),
        "if12_M6":reference(IF12/"replay/cases"/case/"M6_judgements.json"),
        "whole_default_unchanged":True,"R0_physical_replacement_truth":"unknown"})
    hints=make_hints(p,k_role,source)
    dump(root/"M8_shadow_hints.json",{"hints":hints,"actual_queries_from_this_module":0,
                    "physical_replacement_assertion":None,"may_be_sent_as_unverified_search_hint":True})
    picks=shadow_pick(case,fs,source,root,old,r0,motion,table[motion["comparison_id"]]["R3"] if motion else None)
    return {"case":case,"status":"completed","replacement_truth":"unknown","packets":len(p),
       "pairs_count":len(pairs),"state_motion_packet":bool(motion),
       "patch_status_counts":dict(Counter(x["visual_measurement"]["status"] for x in p if x["origin"]=="real_frozen_frame_pair")),
       "instance_R3":dict(Counter(x["R3"]["instance"]["state"] for x in table.values())),
       "hypotheses_R3":dict(Counter(x["R3"]["replacement"] for x in table.values())),
       "environment_veto_duplicate_count":sum(any(y["channel"]=="N" for y in
           x["R3"]["ablation"]["fixed_4_4_2_1"]["de_duplicated"]) for x in table.values()),
       "hints":len(hints),"agent_initial_hints":sum(h["both_available_in_agent_initial"] for h in hints),
       "M7_changed":bool(picks["swap"]),"wall_seconds":perf_counter()-start,
       "M1_input_ledger":m1["artifact_refs"],"M0_input":m0["artifact_refs"]}


def phase_queries(out,case,path,row):
    root=out/"M8_protected"/case
    root.mkdir(parents=True,exist_ok=False)
    meta=read(m.pool_path(case))
    source=meta["source"]|{"path":path}
    old=IF13/"M8_budget"/case/"query.json"
    archive=read(old)
    wanted=[q["request"] for q in archive["executed"]]
    assert len(wanted)==RULES["query"]["windows"]
    assert not overlap(*wanted)
    again=m.execute_queries(source,meta,wanted,root,"query_")
    exact=[]
    for i,q in enumerate(again,1):
        current=root/f"query_{i}"/"keyframes.json"
        previous=old.parent/f"source_read_{i}"/"keyframes.json"
        scan=lambda f:[(v["source_frame"],v["pts"],v["time_base"],v["native_image_sha256"]) for v in read(f)["frames"]]
        ok=q["state"]=="read_original" and scan(current)==scan(previous)
        exact.append(ok)
    result={"case":case,"status":"completed" if all(exact) else "query_failed_or_changed",
           "verified_schedule":reference(old),"generation":"frozen IF13 candidates","allocation":"protected IF13 schedule",
           "executed":again,"returned_images_exact_to_IF13":exact,"agent_direct_route_still_available":True,
           "freely_selected_agent_queries_are_separate":True,"new_identity_shadow_requests_read_here":0}
    dump(root/"query.json",result)
    return result


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",required=True,type=Path)
    a.add_argument("--phase",required=True,choices=("evidence","queries"))
    opt=a.parse_args()
    out=opt.out.resolve()
    lock=read(out/"history_lock.json")
    paths={r["sha256"]:r["path"] for r in lock["media"]+list(lock["historical_sources"].values())}
    target=out/("R_real" if opt.phase=="evidence" else "M8_protected")
    target.mkdir(exist_ok=False)
    dump(target/"lock.json",{"history":reference(out/"history_lock.json"),"protocol":reference(HERE/"protocol.json"),
         "code":{p.name:sha(p) for p in (HERE/"run.py",HERE/"judgement.py",HERE/"observations.py")},
         "manual_truth_read":False,"review_mode":"seen_development"})
    notes=[]
    for row in read(IF11/"run_manifest.json")["cases"]:
        case=row["case"]
        if opt.phase=="queries" and case not in RULES["query"]["cases"]:
            continue
        f=phase_evidence if opt.phase=="evidence" else phase_queries
        try:
            path=paths[row["source"]["sha256"]]
            if sha(path)!=row["source"]["sha256"]:
                raise ValueError("could not resolve frozen real video")
            result=f(out,case,path,row)
        except Exception:
            result={"case":case,"status":"failed","error":traceback.format_exc()}
        notes.append(result)
        print(case,result["status"],result.get("instance_R3",result.get("returned_images_exact_to_IF13")),result.get("error",""),flush=True)
        dump(target/"manifest.json",{"phase":opt.phase,"cases":notes,"replacement_truth":"unknown"})


if __name__=="__main__":
    main()
