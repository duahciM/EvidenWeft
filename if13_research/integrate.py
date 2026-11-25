"""IF13 additive M4/M7 evidence and bounded M8 coverage allocation."""
import argparse
from collections import defaultdict
from pathlib import Path

from identity_catalog_matching.common import read,sha
from if10_research.run import export_selection
from if12_research import modular as m
from if12_research.contracts import IF10, IF11, PRIOR, reference
from p1.io import dump

from .run import OLD_IF12, RULES

FOCUS=PRIOR["p2"]["focus"]


def overlap(x,y):
    a,b=x["window"]
    c,d=y["window"]
    return min(b,d)-max(a,c)>1e-7


def protected_offer(q0, first, fs, start, end, selected):
    pools=q0["pool"]
    ranked=[x["request"] for x in q0["executed"]]
    if not ranked:
        return [],{"status":"no_historical_allocation","protected_offers":[]}
    primary=ranked[0]
    fallback=[x for x in pools if x["reason"]=="independent timeline fallback"]
    usable=[x for x in fallback if not overlap(x,primary)]
    gaps=[]
    if first:
        for r in first["XB"]:
            i=r["A_prior_or_coview_sample"]
            j=r["first_B_sample"]
            if i is not None and fs[i]["t"]<fs[j]["t"]:
                gaps.append((fs[i]["t"],fs[j]["t"]))
    if not gaps:
        times=sorted({f["t"] for f in selected}|{start,end})
        gaps=list(zip(times,times[1:]))
    a,b=max(gaps,key=lambda e:(e[1]-e[0],-e[0]))
    mid=(a+b)/2
    trace={"protected_offers":fallback,"usable_with_primary":usable,
           "coverage_gap":[a,b],"gap_mid_s":mid,"history":ranked,"primary_evidence":primary}
    if not usable:
        return ranked,trace|{"status":"protected_slot_unavailable","displaced":[]}
    pick=min(usable,key=lambda x:(abs(fs[x["sample"]]["t"]-mid),x["sample"]))
    requests=[primary,pick]
    return requests,trace|{"status":"protected_slot_used","selected_fallback":pick,
                           "displaced":[x for x in ranked if x not in requests]}


def run_budget(out, case, src):
    folder=out/"M8_budget"/case
    folder.mkdir(parents=True,exist_ok=False)
    (fs,meta),receipt=m.read_pool(case,src,folder)
    source=meta["source"]|{"path":src}
    base=IF11/"cases"/case
    q0=read(base/"query.json")
    first=read(base/"auto_1.json") if (base/"auto_1.json").exists() else None
    selected=read(base/"selection.json")["arms"]["K_roles"]
    wants,trace=protected_offer(q0,first,fs,source["start"],source["end"],selected)
    assert len(wants)<=RULES["query"]["windows"] and all(
        0<r["window"][1]-r["window"][0]<=RULES["query"]["width_s"]+1e-7 for r in wants)
    readings=m.execute_queries(source,meta,wants,folder,pattern="source_read_")
    data={"module_id":"M8","execution_mode":"live_compute","source_hash":source["sha256"],
          "budget":RULES["query"],"candidate_pool_changed":False,"trace":trace,"executed":readings,
          "new_structures_do_not_compete_for_coverage":True,"agent_received_these_images":False,
          "input_refs":[reference(base/"query.json"),reference(base/"selection.json")],
          "record":receipt["artifact_refs"],"code":sha(__file__)}
    dump(folder/"query.json",data)
    return {"case":case,"selected":wants,"read_status":[r["state"] for r in readings],
            "protected":trace["status"]}


def pair_data(folder):
    import json
    with (folder/"roots.jsonl").open(encoding="utf8") as f:
        return {tuple(r["pair"]):r for r in map(json.loads,f)}


def new_relationships(out,case):
    base=out/"traditional"/"cases"/case
    traditional=pair_data(base)
    sources=[(base,"E1",traditional)]
    learned=out/"learned"/"cases"/case
    if (learned/"image_cycles.json").is_file():
        alt=pair_data(learned)
        sources.extend((learned,method,alt) for method in ("E2","E3"))
    answer=[]
    observed={}
    for folder,method,bank in sources:
        cycles=read(folder/"heldout_charts.json")[method]
        for r in cycles:
            if r["status"]!="validated_local_chart":
                continue
            i,j,k=r["frames"]
            pairs=((i,j),(i,k),(j,k))
            historical=[e for e in pairs if traditional[e]["arms"]["E0"]["support"]]
            newcomers=[e for e in pairs if e not in historical and bank[e]["arms"][method]["support"]]
            if not newcomers:
                continue
            r={"id":r["id"],"method":method,"source_frames":[i,j,k],"root_ids":[*r["fit_roots"],r["independent_check_root"]],
               "new_supported_roots":[bank[p]["root"] for p in newcomers],
               "historically_supported_roots":[bank[p]["root"] for p in historical],
               "image_points":r["predictions"],"confirmed_point_cycles":r["independent_supported_tracks"],
               "provenance_path":str(folder/"heldout_charts.json"),"world_identity":None}
            key=(i,j,k,tuple(sorted(r["new_supported_roots"])))
            if key not in observed or observed[key]["confirmed_point_cycles"]<r["confirmed_point_cycles"]:
                observed[key]=r
    answer.extend(observed.values())
    return sorted(answer,key=lambda a:(-len(a["new_supported_roots"]),-a["confirmed_point_cycles"],a["source_frames"]))


def apply_shadow(out,case,src):
    folder=out/"connections"/case
    folder.mkdir(parents=True,exist_ok=False)
    (fs,meta),frame_receipt=m.read_pool(case,src,folder)
    source=meta["source"]|{"path":src}
    paths=OLD_IF12/"regions/cases"/case
    historical=IF11/"cases"/case
    p2=read(paths/"P2_07.json")
    old=read(historical/"selection.json")
    children=[reference(p) for p in paths.glob("*_IF12.json")]
    parents=[reference(p) for p in historical.glob("auto_*.json")]
    new=new_relationships(out,case)
    baseline=p2["arm"]
    baseline_ids=[r["sample"] for r in baseline]
    roles={int(i):v for i,v in old["roles_by_sample"].items()}
    existing_rels={int(i):{q["root"] for q in value} for i,value in p2["verified_image_relations"].items()}
    start,end=source["start"],source["end"]
    def facts(picks):
        x=set(picks)
        return ({f for i in picks for f in roles.get(i,[])},
                {v for i in picks for v in existing_rels.get(i,set())},
                {min(3,int(4*(fs[i]["t"]-start)/max(.001,end-start))) for i in picks},
                {r["id"] for r in new if len(x.intersection(r["source_frames"]))>=2})
    original=facts(baseline_ids)
    considered=[]
    for i in sorted({i for r in new for i in r["source_frames"]}-set(baseline_ids)):
        for j in baseline_ids:
            after=sorted(set(baseline_ids)-{j}|{i})
            seen=facts(after)
            lost=[sorted(x-y) for x,y in zip(original,seen)]
            if set(roles.get(j,[]))&{"initial","interaction_or_middle","final","clip_initial_visual_state","clip_final_visual_state"}:
                continue
            gains=seen[3]-original[3]
            considered.append({"add":i,"remove":j,"lost":lost,"gains":sorted(gains),"admissible":bool(gains and not any(lost))})
    eligible=[r for r in considered if r["admissible"]]
    winner=max(eligible,key=lambda a:(len(a["gains"]),-a["add"],-a["remove"])) if eligible else None
    selected=baseline
    if winner is not None:
        k=sorted(set(baseline_ids)-{winner["remove"]}|{winner["add"]})
        selected=export_selection(fs,source,{"IF13_shadow":k},folder/"shadow")["IF13_shadow"]
    hints=[]
    for r in new[:2]:
        i=r["source_frames"][1]
        t=fs[i]["t"]
        width=RULES["query"]["width_s"]
        a=max(start,min(t-width/2,end-width))
        hints.append({"source_sample":i,"window":[a,min(a+width,end)],"relation_id":r["id"],
                      "allocation":"shadow_only","root_ids":r["new_supported_roots"]})
    dump(folder/"M4_candidates.json",{"correspondence_only_appended":new,"I0_parent_refs":parents,
                 "IF12_child_refs":children,"existing_candidate_digests_kept":True,
                 "physical_Y":None,"physical_identity":None,"authoritative_partition":None})
    dump(folder/"M7_selection.json",{"same_source_pool":True,"budget":8,
      "references":[reference(historical/"selection.json"),reference(paths/"P2_07.json")],
      "baseline":baseline,"shadow":selected,"active_swap":winner,
      "considered_swaps":considered,"image_cycle_relation_is_not_semantic_value":True})
    dump(folder/"M8_structure_hints.json",{"proposals":hints,"actual_reads":0,"protected_M8_coverage_unaffected":True})
    return {"case":case,"most_recent_continuity_changes":len(new),"shadow_query_hints":len(hints),
            "shadow_selection_changed":bool(winner),"prior_S2_valid":old["valid_five_arm_comparison"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--phase",required=True,choices=("budget","connections"))
    args=p.parse_args()
    o=args.out.resolve()
    (o/("M8_budget" if args.phase=="budget" else "connections")).mkdir(exist_ok=False)
    h=read(o/"history_lock.json")
    srcs={r["sha256"]:r["path"] for r in h["media"]+list(h["historical_sources"].values())}
    if args.phase=="connections":
        t=read(o/"traditional/manifest.json")
        if t["status"]!="completed":
            raise SystemExit("Traditional arm has not completed successfully")
    f=run_budget if args.phase=="budget" else apply_shadow
    notes=[]
    for r in read(IF11/"run_manifest.json")["cases"]:
        case=r["case"]
        if args.phase=="budget" and case not in FOCUS:
            continue
        src=srcs[r["source"]["sha256"]]
        note=f(o,case,src)
        notes.append(note)
        print(note,flush=True)
    dump(o/f"{args.phase}_manifest.json",{"cases":notes,"protocol":reference(Path(__file__).with_name("protocol.json")),
                                        "implementation":reference(__file__)})


if __name__=="__main__":
    main()
