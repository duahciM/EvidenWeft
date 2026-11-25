"""Post-prediction audit and visualization of missed source pair roots."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from PIL import Image, ImageDraw
from identity_catalog_matching.common import read,sha
from if12_research import modular as m
from if12_research.contracts import IF11, IF10, ROOT, reference
from p1.io import command,dump
from v05.engine import export_native
from v07.audit import audit_images

from .integrate import pair_data
from .run import OLD_IF12


def pair_panel(row, stamps, path, code, title):
    w,h=440,248
    sheet=Image.new("RGB",(w*2,h+60),"#152331")
    draw=ImageDraw.Draw(sheet)
    dims=row["coordinate_space"]
    canvas_scale=[]
    for n,(i,stamp) in enumerate(zip(row["pair"],row["source_frames"])):
        with Image.open(stamps[i]["native_image"]) as original:
            image=original.convert("RGB")
            image.thumbnail((w,h))
            cw,ch=image.size
            sheet.paste(image,(n*w,0))
            canvas_scale.append((n*w,cw/dims["analysis_width"],ch/dims["analysis_height"]))
            draw.text((n*w+4,h+3),f'{stamp["t"]:.6f}s  PTS {stamp["pts"]}  sample {i}',fill="white")
    global_code, local_code=code
    hits=set(row["arms"][global_code]["inliers"])
    local=set(row["arms"][local_code]["inliers"])
    for i,c in enumerate(row["candidate_sets"]):
        z=[(origin+v[0]*sx,v[1]*sy) for (origin,sx,sy),v in zip(canvas_scale,[c["a_xy"],c["b_xy"]])]
        color="#5bf1c6" if i in local else "#ffbe56" if i in hits else "#657789"
        draw.line((z[0],z[1]),fill=color,width=2 if i in local else 1)
        for x,y in z:
            draw.ellipse((x-2,y-2,x+2,y+2),outline=color)
    a,b=[row["arms"][k] for k in code]
    draw.text((4,h+20),f'{title}: {global_code} {a["reason"]}; {local_code} {b["reason"]}',fill="white")
    draw.text((4,h+37),f'Mutual candidates {len(row["candidate_sets"])}; amber global inliers; cyan local inliers; gray proposals',fill="white")
    sheet.save(path)


def diagnostic(o):
    base=o/"traditional/diagnostics"
    record=read(o/"traditional/cases/F01/scope_locked.json")
    folder=o/"source_pair_audit"
    folder.mkdir(exist_ok=False)
    (fs,meta),_=m.read_pool("F01",record["source_path"],folder)
    source=meta["source"]|{"path":record["source_path"]}
    refs={k:pair_data(base/k) for k in ("oracle","auto2_diagnostic")}
    samples=sorted({i for groups in refs.values() for row in groups.values() for i in row["pair"]})
    imgs={r["sample"]:r for r in export_native(source,fs,samples,folder/"native")}
    dump(folder/"native_manifest.json",{"case":"F01","frames":list(imgs.values())})
    results={}
    for mode,code in (("traditional",("E0","E1")),("learned",("E2","E3"))):
        root=o/mode
        if not (root/"manifest.json").is_file() or not (root/"diagnostics/oracle/roots.jsonl").is_file():
            continue
        for b in refs:
            pair_file=root/"diagnostics"/b
            bank=pair_data(pair_file)
            charts=read(pair_file/"heldout_charts.json") if (pair_file/"heldout_charts.json").exists() else {}
            selected=[row for row in bank.values() if "requested_development_root" in row["planned_role"]]
            output=folder/f"{mode}_{b}"
            output.mkdir()
            findings=[]
            for row in selected:
                a,c=row["pair"]
                target=next(int(k.split(":")[1]) for k in row["planned_role"] if k.startswith("query:"))
                confirms={}
                for arm,data in charts.items():
                    confirms[arm]=[v["id"] for v in data if {"validated_local_chart"}.issuperset({v["status"]})
                                   and set(row["pair"])<=set(v["frames"])]
                r={"root":row["root"],"pair":row["pair"],"target_sample":target,"pts":[i["pts"] for i in row["source_frames"]],
                   "pair_reprojection_and_full_coordinates":str(pair_file/"roots.jsonl"),
                   "history":row["legacy"],"point_count":len(row["candidate_sets"]),
                   "fit":{k:{name:row["arms"][k].get(name) for name in ("reason","support","inliers","hull")} for k in code},
                   "independent_direct_chart_ids":confirms}
                if row["candidate_sets"]:
                    pair_panel(row,imgs,output/f"{a}_{c}.png",code,b)
                    r["panel"]=str(output/f"{a}_{c}.png")
                findings.append(r)
            summary={"roots":findings,"target_roots":len(findings),
              "stages":{k:dict(Counter(r["fit"][k]["reason"] for r in findings)) for k in code},
              "chart_rescued_target_pairs":{k:sum(not r["history"]["verified"] and bool(r["independent_direct_chart_ids"].get(k)) for r in findings)
                                            for k in code}}
            results[f"{mode}_{b}"]=summary
            dump(output/"review.json",summary)
    dump(folder/"F01_root_audit.json",results)
    print("root review",[(k,v["stages"],v["chart_rescued_target_pairs"]) for k,v in results.items()],flush=True)


def query_score(o):
    case="room_corridor_dev"
    root=IF11/"cases"/case
    band=[10.325,10.550]
    probe=10.413333
    means={
      "P0":ROOT/"if06_checkpoint_20261118"/"P2_01"/case/"P0",
      "wide":ROOT/"if08_research_20261119"/"IF08"/case/"resample",
      "IF10":IF10/"cases"/case,
      "Q0":root,
      "Q1":OLD_IF12/"Q1/cases"/case,
      "protected":o/"M8_budget"/case,
    }
    def offered(method, location):
        if method=="IF10":
            return read(location/"query.json")["potential_requests"]
        if method=="Q0":
            return read(location/"query.json")["pool"]
        if method=="Q1":
            return read(location/"Q1.json")["pool"]
        if method=="protected":
            return read(root/"query.json")["pool"]
        return None
    result={}
    for k,p in means.items():
        candidates=offered(k,p)
        q=[c["window"] for c in candidates if c["window"][0]<=probe<=c["window"][1]] if candidates is not None else None
        active=[]
        for trace in sorted(p.glob("**/resample_trace.json")):
            r=read(trace)
            picked=read(trace.parent/"keyframes.json")["frames"]
            frames=read(trace.parent/"frames.json")["frames"]
            times=lambda fs:[{"t":f["t"],"pts":f["pts"],"source_frame":f["source_frame"]}
                             for f in fs if band[0]<=f["t"]<=band[1]]
            active.append({"window":r["effective_window"],"sampled_in_band":times(frames),
                           "returned_in_band":times(picked),"complete_frame_count":len(frames),
                           "returned_count":len(picked),"artifact":str(trace)})
        stage=("generation" if q==[] else "allocation" if not any(x["sampled_in_band"] for x in active)
               else "return_selection" if not any(x["returned_in_band"] for x in active) else "reached_return")
        result[k]={"offer_hits":q,"actual_reads":active,"first_miss":stage,
                   "agent_was_sent_by_this_benchmark":False}
    dump(o/"M8_query_evaluation.json",{"sealed_target":band,"evaluation_only":True,"strategies":result})
    print("query",[(k,v["first_miss"]) for k,v in result.items()],flush=True)


def source_audit(o):
    bank=defaultdict(dict)
    metadata={r["case"]:read(m.pool_path(r["case"]))["source"]|r["source"]
             for r in read(IF11/"run_manifest.json")["cases"]}
    def collect(case,frames):
        for r in frames:
            bank[case][r["native_image"]]=r
    for c in metadata:
        for path in sorted((o/"M8_budget"/c).glob("**/keyframes.json")):
            collect(c,read(path)["frames"])
        m7=o/"connections"/c/"M7_selection.json"
        if m7.exists():
            collect(c,read(m7)["shadow"])
    audit_d=o/"source_pair_audit/native_manifest.json"
    if audit_d.exists():
        collect("F01",read(audit_d)["frames"])
    for base in ("P3","P3_recovered"):
        for result_file in (o/base).glob("*/result.json"):
            r=read(result_file)
            collect(r["case"],r["initial"])
            for p in result_file.parent.glob("**/keyframes.json"):
                collect(r["case"],read(p)["frames"])
    rows,fail,pts=[],[],[]
    for case,p in bank.items():
        src=metadata[case]
        try:
            q=json.loads(command(["ffprobe","-v","error","-show_frames","-select_streams","v:0","-show_entries",
                            "frame=pts,pts_time:stream=time_base","-of","json",src["path"]]))
            a=list(p.values())
            stamps=q["frames"]
            hit=all(f["pts"]==stamps[f["source_frame"]]["pts"] and
                    float(stamps[f["source_frame"]]["pts_time"])==f["t"] and
                    f["time_base"]==q["streams"][0]["time_base"] for f in a)
            pts.append({"case":case,"pts":hit,"image_count":len(a),"source_hash":sha(src["path"])==src["sha256"]})
            for start in range(0,len(a),24):
                rows.extend(audit_images(src,a[start:start+24]))
        except Exception as exc:
            fail.append({"case":case,"error":repr(exc)})
        print("source",case,len(p),flush=True)
    ok=bool(rows) and not fail and all(x["pts"] and x["source_hash"] for x in pts) and all(
        x["exact_source_rgb_equal"] and x["file_hash_equal"] for x in rows)
    dump(o/"native_source_audit.json",{"pass":ok,"source_PTS_checks":pts,"native":rows,"failures":fail})
    print("audit",ok,len(rows),len(pts),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--action",choices=("pairs","query","audit"),required=True)
    a=p.parse_args()
    {"pairs":diagnostic,"query":query_score,"audit":source_audit}[a.action](a.out.resolve())


if __name__=="__main__":
    main()
