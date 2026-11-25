"""Run all locked P3-12 conditions, independently of one another's scores."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from if17_research.catalog import read, ref, sha
from p1.io import dump

from .engine import run_trial
from .protocol import jobs, protocol


def run(out, workers, extra=False):
    lock = read(out / "protocol_lock.json")
    if sha(out / "input/frames.json") != lock["packets"]["sha256"] or lock["study"] != read_policy():
        raise ValueError("Matrix inputs or protocol changed")
    for code in lock["code"]:
        if sha(Path(code["path"])) != code["sha256"]:
            raise ValueError("Locked prediction code changed")
    packets = read(out / "input/frames.json")
    planned = [tuple(row) for row in lock["study"]["randomized_schedule"]]
    if extra:
        planned = [tuple(row) for row in read(out / "additional_repeats.json")["jobs"]]
    entries = {}

    def one(job):
        case, arm, variant, rep = job
        reference, result = run_trial(out, case, arm, variant, rep, packets[case])
        answered = [(job, reference | {"status": result["status"]})]
        # Keep the exact same printed memory for the paired no-tool and tool conditions.
        follow = (case, "C9", "shared_memory_plus_tools", rep)
        if arm == "C7" and follow in planned:
            carry = result.get("memory", {}).get("path")
            if carry is None:
                for attempt in (2, 1):
                    fallback = out / "trials" / case / arm / variant / str(rep) / f"attempt_{attempt}" / "visual" / "memory.json"
                    if fallback.exists() and len(read(fallback)) == len(packets[case]["frames"]):
                        carry = str(fallback)
                        break
            if carry is not None:
                selected, child = run_trial(out, *follow, packets[case], carry=Path(carry))
                answered.append((follow, selected | {"status": child["status"]}))
            else:
                answered.append((follow, {"status": "blocked_by_missing_real_memory"}))
        return answered

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, row) for row in planned if row[1] != "C9"]
        for future in as_completed(futures):
            for job, published in future.result():
                entries["/".join(map(str, job))] = published
            dump(out / ("extra_manifest.json" if extra else "matrix_manifest.json"),
                 {"expected": len(planned), "finished": len(entries), "results": entries,
                  "status_counts": dict(Counter(row["status"] for row in entries.values()))})


def read_policy():
    # JSON round trip normalizes tuple-based randomized schedule.
    import json
    return json.loads(json.dumps(protocol()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--extra", action="store_true")
    args = parser.parse_args()
    run(args.out.resolve(), args.workers, args.extra)
