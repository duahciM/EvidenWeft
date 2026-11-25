"""Post-run evaluation, real video evidence panels, and independent source checks."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
from identity_catalog_matching.common import read,sha
from if12_research import modular as m
from if12_research.contracts import IF11
from p1.io import command,dump
from v05.engine import export_native
from v07.audit import audit_images

from .observations import IF13


def score(out):
    dirs=out/"R_real"
    cases=read(dirs/"manifest.json")["cases"]
    stats={mode:{"instance":Counter(),"slot":Counter(),"context":Counter(),"track":Counter(),"replacement":Counter()}
           for mode in ("R1","R2","R3")}
    exact_environment=Counter()
    budget_omitted=checked=0
    details=[]
    for r in cases:
        if r["status"]!="completed":
            continue
        case=r["case"]
        folder=dirs/case
        judged=read(folder/"judgements.json")
        packets=read(folder/"packets.json")
        old=read(IF11/"cases"/case/"selection.json")
        r0=read(IF13/"connections"/case/"M7_selection.json")
        indices={f["sample"] for f in old["arms"]["K_roles"]+r0["shadow"]}
        all_points=set(combinations(sorted(indices),2))
        examined={tuple(x["source_samples"]) for x in packets if x["origin"]=="real_frozen_frame_pair"}
        unrun=sorted(all_points-examined)
        checked+=len(examined)
        budget_omitted+=len(unrun)
        dump(folder/"budget_review.json",{"pairs_total_from_selected_bank":len(all_points),"assessed":sorted(examined),
               "not_assessed_due_to_budget":unrun,"empty_scores_are_not_negative_evidence":True})
        for val in judged.values():
            for mode, q in val.items():
                s=stats[mode]
                s["instance"][q["instance"]["state"]]+=1
                s["slot"][q["slot"]["state"]]+=1
                s["context"][q["scene"]["state"]]+=1
                s["track"][q["episode"]["state"]]+=1
                s["replacement"][q["replacement"]]+=1
                assert q["scene"]["scene_assignment"] is None
                assert not q["instance"]["physical_id_merge"]
                assert q["episode"]["absence_of_replacement_between_samples"] is None
            a=val["R3"]["ablation"]
            c=a["fixed_4_4_2_1"]; x=a["fixed_4_4_2_0"]
            if c["usable_association_rank"]!=x["usable_association_rank"]:
                exact_environment["ranking_changed"]+=1
            if any(y["channel"]=="N" for y in c["de_duplicated"]):
                exact_environment["environment_shared_root_blocked"]+=1
            if c["usable_association_rank"] is None:
                exact_environment["no_rank_evidence"]+=1
        details.append({"case":case,"reviewed":len(judged),"shadow_hints":r["hints"],"not_assessed_pairs":len(unrun),
                 "changed_selection":r["M7_changed"],"registered_motion_candidate":r["state_motion_packet"],
                 "replacement_truth":"unknown"})
    f=read(out/"R_logic_fixtures/results.json")
    c=f["same_slot_visible_discriminator_conflict"]
    handoff=f["visible_handoff_with_distinguishing_details"]
    separate={
        "R1_cannot_infer_same_slot_without_G":c["R1"]["replacement"]=="unknown",
        "R2_conditional_same_slot_candidate":c["R2"]["replacement"]=="same_slot_replacement_candidate",
        "R2_cannot_observe_handoff":handoff["R2"]["replacement"]!="observed_distinct_handoff_at_verified_slot",
        "R3_observed_handoff_from_supplied_native_premises":handoff["R3"]["replacement"]=="observed_distinct_handoff_at_verified_slot",
        "fixed_numeric_environment_can_change_rank_on_distinct_roots":
             f["lookalike_across_gap"]["R3"]["ablation"]["fixed_4_4_2_1"]["usable_association_rank"]==7 and
             f["lookalike_across_gap"]["R3"]["ablation"]["fixed_4_4_2_0"]["usable_association_rank"]==6,
    }
    assert all(separate.values())
    protected=read(out/"M8_protected/manifest.json")["cases"]
    assert all(r["status"]=="completed" for r in protected)
    exact_replays=[ok for r in protected for ok in r["returned_images_exact_to_IF13"]]
    assert all(exact_replays)
    room=out/"M8_protected/room_corridor_dev/query_2/keyframes.json"
    near=[{"pts":f["pts"],"t":f["t"],"source_frame":f["source_frame"],"native_image":f["native_image"]}
          for f in read(room)["frames"] if 10.325<=f["t"]<=10.550]
    result={"seen_cases_completed":len(details),"all_case_outputs":details,"judgements_by_arm":stats,
        "assessed_pair_count":checked,"budget_unassessed_pair_count":budget_omitted,
        "environment_ablation":exact_environment,"logic_additional_checks":separate,
        "real_replacement_truth_denominator":0,"real_replacement_accuracy":None,
        "protected_budget_exact_real_rereads":len(exact_replays),"room_protected_returns":near,
        "true_replacement_false_merge_or_split_rates":None}
    dump(out/"evaluation.json",result)
    print("evaluated",len(details),"old source clips,",checked,"pairs,",budget_omitted,"budget not evaluated,",len(near),"protected near frames",flush=True)


def draw_story(out):
    root=out/"evidence_panels"
    root.mkdir(exist_ok=False)
    entries=[]
    index={r["case"]:r["source"] for r in read(IF11/"run_manifest.json")["cases"]}
    for case in ("F01","local_object"):
        target=root/case
        target.mkdir()
        (fs,meta),_=m.read_pool(case,index[case]["path"],target)
        source=meta["source"]|index[case]
        packets=read(out/"R_real"/case/"packets.json")
        if case=="F01":
            packet=next(r for r in packets if r["source_samples"]==[10,122])
            (bank,_),_=m.observations(case,fs)
            raw=bank[tuple(packet["source_samples"])]["xy"]
        else:
            packet=next(r for r in packets if r["origin"]=="real_rereadable_frozen_S2")
            raw=None
        path=target/"native"
        obs=export_native(source,fs,packet["source_samples"],path)
        entries.append({"case":case,"frames":obs})
        w,h=440,248
        W=w*len(obs)
        im=Image.new("RGB",(W,h+49),"#173042")
        draw=ImageDraw.Draw(im)
        scales=[]
        fh,fw=fs[0]["gray"].shape
        for idx,f in enumerate(obs):
            with Image.open(f["native_image"]) as p:
                p=p.convert("RGB")
                p.thumbnail((w,h))
                im.paste(p,(idx*w,0))
                scales.append((idx*w,p.width/fw,p.height/fh))
            draw.text((idx*w+7,h+4),f'{f["t"]:.6f} s - PTS {f["pts"]} - source frame {f["source_frame"]}',fill="white")
        if raw:
            for A,B in zip(*raw):
                ends=[(dx+sx*v[0],sy*v[1]) for (dx,sx,sy),v in zip(scales,(A,B))]
                draw.line(ends,fill="#6affd8",width=1)
                for x,y in ends:
                    draw.ellipse((x-2,y-2,x+2,y+2),outline="#fff5ce")
            note="Geometry-supported local view correspondences; no verified slot or identity across the interval."
        else:
            for k,z in enumerate(packet["visual_measurement"]["local_regions"]):
                a,b,c,d=z["continuity_candidate"]["bbox"]
                dx,sx,sy=scales[k]
                draw.rectangle((dx+a*sx,b*sy,dx+(a+c)*sx,(b+d)*sy),outline="#61ecca",width=3)
            note="Actual sampled relative movement, stable local reference; intervening native times not continuously observed."
        draw.text((6,h+25),note,fill="white")
        im.save(target/"observations.png")
    dump(root/"native_manifest.json",entries)
    print("Actual-source overlays",len(entries),flush=True)


def source_audit(out):
    registry={r["case"]:read(m.pool_path(r["case"]))["source"]|r["source"]
              for r in read(IF11/"run_manifest.json")["cases"]}
    bank=defaultdict(dict)
    def add(c, frames):
        for f in frames:
            bank[c][f["native_image"]]=f
    for c in registry:
        for f in (out/"M8_protected"/c).glob("**/keyframes.json"):
            add(c,read(f)["frames"])
        f=out/"P3"/c/"result.json"
        if f.is_file():
            add(c,read(f)["initial"])
            for p in f.parent.glob("**/keyframes.json"):
                add(c,read(p)["frames"])
        f=out/"R_real"/c/"M7_R4_selection.json"
        if f.is_file():
            add(c,read(f)["shadow"])
    for rec in read(out/"evidence_panels/native_manifest.json"):
        add(rec["case"],rec["frames"])
    add("F01",read(IF13/"source_pair_audit/native_manifest.json")["frames"])
    pts,native,errors=[],[],[]
    for case,by_path in bank.items():
        source=registry[case]
        try:
            p=json.loads(command(["ffprobe","-v","error","-show_frames","-select_streams","v:0","-show_entries",
                                "frame=pts,pts_time:stream=time_base","-of","json",source["path"]]))
            imgs=list(by_path.values())
            raw=p["frames"]
            matches=all(f["pts"]==raw[f["source_frame"]]["pts"] and f["time_base"]==p["streams"][0]["time_base"] and
                        f["t"]==float(raw[f["source_frame"]]["pts_time"]) for f in imgs)
            pts.append({"case":case,"count":len(imgs),"matching_pts_and_time_base":matches,
                       "source_hash_verified":sha(source["path"])==source["sha256"]})
            for start in range(0,len(imgs),24):
                native.extend(audit_images(source,imgs[start:start+24]))
        except Exception as exc:
            errors.append({"case":case,"error":repr(exc)})
        print("source check",case,len(by_path),flush=True)
    ok=bool(native) and not errors and all(x["matching_pts_and_time_base"] and x["source_hash_verified"] for x in pts) and all(
       x["exact_source_rgb_equal"] and x["file_hash_equal"] for x in native)
    dump(out/"native_source_audit.json",{"pass":ok,"PTS":pts,"native":native,"errors":errors})
    print("source audit",ok,len(native),len(pts),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--what",required=True,choices=("score","panels","audit"))
    a=p.parse_args()
    {"score":score,"panels":draw_story,"audit":source_audit}[a.what](a.out.resolve())


if __name__=="__main__":
    main()
