"""Lock auditable outcome, coverage and observed cost for the completed round."""
import argparse
import ast
from collections import Counter
from pathlib import Path
import re

from if17_research.catalog import read, ref
from p1.io import dump

from .protocol import CONDITIONS, HERE


def attempts(out):
    summary = Counter()
    failures = []
    seen_sessions = set()

    def count_turn(turn):
        values = turn["audit"].get("usage") or {}
        for field in ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens"):
            if type(values.get(field)) is int:
                summary[field] += values[field]
        summary["turns"] += 1
        summary["image_items"] += len(turn["attachments"])
        summary["reasoned_turns"] += int(turn["audit"].get("reasoning_nonempty_blocks", 0) > 0)

    for path in sorted((out / "trials").glob("*/*/*/*/*/result.json")):
        trial = read(path)
        summary["trial_artifacts"] += 1
        summary[f'trial_status_{trial["status"]}'] += 1
        if trial["status"] != "completed":
            failures.append({"case": trial["case"], "arm": trial["arm"], "variant": trial["variant"],
                             "replicate": trial["replicate"], "attempt": trial["attempt"], "result": ref(path),
                             "error": trial.get("error", "").splitlines()[-1]})
        for session in trial["sessions"]:
            session_id = session.get("session_id")
            if not session_id or session_id in seen_sessions:
                continue
            seen_sessions.add(session_id)
            for turn in session["turns"]:
                count_turn(turn)
    trace = out / "extra_c8_terminal_retry.json"
    if trace.exists():
        for item in read(trace)["partially_repeated_turns"]:
            turn = read(item["path"])
            if turn["session"] not in seen_sessions:
                seen_sessions.add(turn["session"])
                count_turn(turn)
                summary["saved_interrupted_turns"] += 1
    summary["sdk_sessions"] = len(seen_sessions)
    return {"totals": dict(summary), "failed_attempts": failures,
                "scope": "All observer attempts with saved turn receipts. Excludes reviewer and separate production transport calls."}


def cell(row):
    measurements = row["target"]
    aliases = ("observe", "historical_initial", "tool_turn")
    scores = [measurements.get(f"{layer}/{label}", 0)
              for layer in aliases
              for label in ("correct", "wrong", "omitted", "uncertain", "unresolved_review")]
    layer = next((scores[i:i + 5] for i in range(0, 15, 5) if any(scores[i:i + 5])), None)
    target = "/".join(map(str, layer)) if layer else "—"
    final = row["final"]
    final_line = f'{final.get("qualified", 0)}/{final["scored"]}' if final.get("scored") else "—"
    return f'{row["complete"]}｜{target}｜{final_line}'


def finish(out):
    report = out / "P3_12_REPORT.md"
    matrix = read(out / "matrix_manifest.json")
    extra = read(out / "extra_manifest.json")
    summary = read(out / "condition_summary.json")
    audit = read(out / "matrix_source_audit.json")
    judgment = read(out / "adjudication.json")
    assert matrix["finished"] == matrix["expected"] == 88
    assert extra["finished"] == extra["expected"]
    assert all(row["status"] == "completed" for row in matrix["results"].values())
    assert read(out / "history_verification.json")["pass"]
    assert read(out / "production_transport/delivery_audit.json")["pass"]
    assert audit["verified_trials"] == sum(row["status"] == "completed"
                                           for m in (matrix, extra) for row in m["results"].values())
    old_log = (out / "regression.log").read_text(encoding="utf-8-sig")
    assert "Ran 136 tests" in old_log and old_log.strip().endswith("OK")
    conditions = summary["conditions"]
    base_table = ["每格为 **有效重复数｜目标帧 正/错/漏/疑/未决｜主复述达标数/已裁定数**。"
                  "C9 的主会话本身没有首次视觉输入，目标分数只在回图碰到该帧时出现。补做计入本条件。",
                  "", "| 条件 | 短 G2 | G3 |", "|---|---|---|"]
    for arm, names in CONDITIONS.items():
        for name in names:
            base_table.append(f'| {arm} · {name} | {cell(conditions[f"short/{arm}/{name}"])} | {cell(conditions[f"g3/{arm}/{name}"])} |')
    metrics = attempts(out)
    dump(out / "technical_attempts.json", metrics)
    qualified = extra["status_counts"].get("completed", 0)
    top = (f"**16 类机制均已实跑，88 条基础会话全部有效，另有 {qualified}/{extra['expected']} 条冲突复跑有效。** "
           "指定目标帧时，原生单图、双图和原生整批都能认出关键关系。"
           "旧拼图综合的错误可以复现，错误也出现在缩小独立图和仅携带旧文字的独立原图中。"
           "这两段开发样本支持局部串线诊断，无法把旧失败单独归给拼接、帧量或顺序。\n\n"
           "**文字记忆的收益依任务而异。** 同一份文字加原图回查，G3 由两条仅报告“文字未提及”变为两条确认实际样本未见。"
           "短 G2 的两条主会话都收到 6.924 和 7.124 秒原图，最终仍省略或改写了目标时点。"
           "组合 C15 的改善集中在 G3，短 G2 未稳定解决。")
    technical = (f"观察侧存档包含 {metrics['totals']['sdk_sessions']} 个 SDK 会话、"
                 f"{metrics['totals']['turns']} 轮和 {metrics['totals']['image_items']} 个实际附件。"
                 f"SDK 报告输出 {metrics['totals'].get('outputTokens', 0):,} token，含推理。"
                 f"保存 {len(metrics['failed_attempts'])} 次技术失败。C8 末轮触顶的恢复保持相同文字包、提示、模型和 token 上限。"
                 f"一条补做的整链技术重试中止并保留 {metrics['totals'].get('saved_interrupted_turns', 0)} 轮，改为只恢复最后的文字综合。"
                 f"逐图裁定中有 {judgment['label_counts'].get('unresolved_review', 0)} 行仍待确认，"
                 "原始评审意见与原答保存。未使用 SDK 数据估算金额。 [成本与全部尝试](technical_attempts.json)")
    doc = report.read_text(encoding="utf-8")
    if "{{LEAD}}" in doc:
        doc = doc.replace("{{LEAD}}", top).replace("{{MATRIX}}", "\n".join(base_table)).replace("{{TECHNICAL}}", technical)
        report.write_text(doc, encoding="utf-8")
    names = [
        "P3_12_REPORT.md", "protocol_lock.json", "input/frames.json", "matrix_manifest.json", "extra_manifest.json",
        "additional_repeats.json", "adjudication.json", "evaluation_protocol.json", "evaluation_only/source_adjudications.json",
        "frame_error_table.csv", "condition_summary.json", "matrix_source_audit.json", "memory_deltas.json",
        "production_transport/delivery_audit.json", "sdk_history_audit.json", "C11_historical_replay_audit.json",
        "history_verification.json", "regression.log", "EXTERNAL_AUDIT.md", "technical_attempts.json",
        "attribution_evidence.json"]
    dump(out / "run_manifest.json", {"phase": "P3-12", "status": "completed", "base_sessions": matrix["status_counts"],
         "extra_sessions": extra["status_counts"], "sources": ["short_G2_11", "g3_10"],
         "review": {"per_frame_labels": judgment["label_counts"], "finals": judgment["narrative_count"],
                    "independent_primary": 2}, "source_audit": "pass", "regression": 136,
         "code": [ref(p) for p in sorted(HERE.glob("*.py"))], "artifacts": {name: ref(out / name) for name in names}})
    for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", doc):
        if not link.startswith("https://"):
            assert (out / link).is_file(), link
    for p in HERE.glob("*.py"):
        ast.parse(p.read_bytes(), filename=str(p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    finish(parser.parse_args().out.resolve())
