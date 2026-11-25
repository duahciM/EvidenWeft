"""Two bounded development sessions using the existing DeepSeek SDK and reader."""
import argparse
import json
from pathlib import Path

from if10_research.p3_live import contact, query_requests
from if11_research.agent import CLI, FIRST, FOLLOW, Harness
from if12_research.agent import INSTRUCTION
from if12_research.contracts import IF10, IF11, read, sha
from if12_research.coverage import packets
from p1.io import dump

RULES=read(Path(__file__).with_name("protocol.json"))["agent"]
ADD="""For a claim that two objects or regions coexist, cite the specific delivered frame(s)
where both can be seen together. If separate frames show different cues, describe them
separately; do not imply coexistence across an interval. For an identity or motion
inference that crosses a gap, give the competing explanation or explicitly leave it
unresolved. Keep that distinction in the final narration."""


def observe(out, case, cli, group):
    old=IF10/"cases"/case
    meta=read(old/"frames.json")
    src=meta["source"]|{"path":next(r["source"]["path"] for r in read(IF11/"run_manifest.json")["cases"] if r["case"]==case)}
    initial=read(IF11/"cases"/case/"selection.json")["arms"][RULES["initial_arm"]]
    folder=out/group/case
    folder.mkdir(parents=True,exist_ok=False)
    assert len(initial)==8
    paths=folder/"initial.png"
    contact(initial,paths,"Initial source observations | development")
    start,end=src["start"],src["end"]
    coverage=packets(initial,[],start,end)
    prompt=FIRST.format(start=start,end=end)+"\n"+INSTRUCTION+"\n"+ADD+"\n"+json.dumps(coverage)
    specs={k:RULES[k] for k in ("provider","model","reasoningEffort","maxTokens")}|{"cwd":str(folder)}
    report={"case":case,"scene_label":None,"independent_eval":False,"source":src,"initial":initial,
            "first_prompt":prompt,"initial_coverage":coverage,"requested_route":specs,"code":sha(__file__)}
    dump(folder/"launch_lock.json",report)
    live=None
    try:
        if sha(src["path"])!=src["sha256"]:
            raise RuntimeError("source hash mismatch")
        live=Harness(folder,cli)
        report["sdk"]=live.result(live.send("initialize",specs))
        report["session_id"]=live.session
        first=live.turn(prompt,[paths],1)
        report["first"]=first
        ans=first["result"]
        if not ans["reply"] or ans["tool_calls"]:
            raise RuntimeError("no usable first reply")
        replies,images=query_requests(ans["reply"],folder,src,meta)
        report["reads"]=replies
        added=[]
        for n,rec in enumerate(replies,1):
            if rec.get("state")!="read_original_video":
                continue
            r=read(Path(rec["trace"]).parent/"keyframes.json")["frames"]
            added.append((f"return_{n}",r,rec["requested"],rec["distinct_candidates"]))
        coverage=packets(initial,added,start,end)
        report["returned_coverage"]=coverage
        message=FOLLOW.format(receipt=json.dumps(replies))+"\n"+INSTRUCTION+"\n"+ADD+"\n"+json.dumps(coverage)
        report["second_prompt"]=message
        second=live.turn(message,images,2)
        report["followup"]=second
        if not second["result"]["reply"] or second["result"]["tool_calls"]:
            raise RuntimeError("no usable second reply")
        report["status"]="completed" if images else "completed_without_return"
    except (RuntimeError,OSError,ValueError,TimeoutError) as exc:
        report.update(status="failed",error=str(exc))
    finally:
        if live:
            live.close()
        dump(folder/"result.json",report)
    print(case,report["status"],report.get("error",""),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--case",choices=RULES["cases"])
    p.add_argument("--cli",type=Path,default=CLI)
    p.add_argument("--group",default="P3")
    a=p.parse_args()
    for case in [a.case] if a.case else RULES["cases"]:
        observe(a.out.resolve(),case,a.cli,a.group)


if __name__=="__main__":
    main()
