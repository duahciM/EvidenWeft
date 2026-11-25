"""Resume only a technically truncated text-only terminal step on frozen memory."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from traceback import format_exc

from if17_research.catalog import read, ref, sha
from p1.io import dump

from .engine import integrate, session


def recover(out, key, pointer):
    before = read(pointer["path"])
    if before["arm"] not in {"C7", "C8"} or not before.get("memory") or not before["sessions"]:
        return key, pointer
    if before["sessions"][-1]["role"] != "text_only_master":
        return key, pointer
    memory_ref = before["memory"]
    assert sha(Path(memory_ref["path"])) == memory_ref["sha256"]
    memory = read(memory_ref["path"])
    assert len(memory) == len(before["sessions"]) - 1
    folder = Path(pointer["path"]).parent.parent / "terminal_recovery"
    current = folder / "result.json"
    if current.exists():
        after = read(current)
        return key, ref(current) | {"status": after["status"]}
    after = {k: v for k, v in before.items() if k not in {"status", "error", "final", "sessions"}}
    after["sessions"] = before["sessions"][:-1]
    after["terminal_recovery_from"] = pointer
    started = perf_counter()
    try:
        with session(folder / "master", after["sessions"], "text_only_master") as live:
            final = integrate(live, before["case"], memory)
        original = before["sessions"][-1]["turns"]
        if original:
            assert original[0]["prompt"] == final["prompt"]
        after.update(status="completed", final=final["answer"])
    except Exception:
        after.update(status="failed", error=format_exc())
    after["terminal_recovery_seconds"] = perf_counter() - started
    after["seconds"] += after["terminal_recovery_seconds"]
    dump(current, after)
    return key, ref(current) | {"status": after["status"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = args.out.resolve()
    manifest_path = out / "matrix_manifest.json"
    base = read(manifest_path)
    assert base["finished"] == base["expected"]
    original = base["results"]
    selected = [(key, value) for key, value in original.items() if value["status"] == "failed" and "/terminal_recovery/" not in value.get("path", "").replace("\\", "/")]
    if selected:
        save = out / "pre_terminal_recovery_manifest.json"
        if not save.exists():
            dump(save, base)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(recover, out, key, row) for key, row in selected]
            for job in as_completed(jobs):
                key, result = job.result()
                original[key] = result
                print(key, result["status"], flush=True)
        dump(manifest_path, base | {"results": original,
             "status_counts": dict(Counter(v["status"] for v in original.values()))})
