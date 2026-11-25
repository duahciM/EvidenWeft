"""P3-12 independent matrix and its frozen prompts."""
from pathlib import Path
from random import Random

from if19_research.fixed import ADD, FINAL, SHARED
from if19_research.inputs import PINS, QUESTIONS
from if20_research.direct import RULES as PRIOR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ROUTE = PRIOR["route"]
TARGETS = {"short": 213720, "g3": 3240}
DISTRACTORS = {"short": 189720, "g3": 2640}
CONDITIONS = {
    "C0": ["single"], "C1": ["native"], "C2": ["legacy_sheet"], "C3": ["equal_cell_items"],
    "C4": ["target_all"], "C5": ["same_session"], "C6": ["same_session_previous_text"],
    "C7": ["isolated_then_text"], "C8": ["ascending", "descending"], "C9": ["shared_memory_plus_tools"],
    "C10": ["observe_then_synthesize"], "C11": ["historical_A0", "historical_A1"],
    "C12": ["ascending", "descending", "shuffled"],
    "C13": ["one", "two", "full"], "C14": ["visual_grounds"],
    "C15": ["complete"]
}
DESCRIBE = (
    "Directly describe what is visibly present, including main objects, their image positions and "
    "discernible relationships. Mark unclear details. Include small discernible objects when relevant. "
    "Keep perception separate from motion, causation, and unseen events. Do not infer what occurred "
    "before or after a single frame."
)
RECORD = (
    'Return only text JSON {"facts":[{"pts":integer,"description":"brief visible details and relations",'
    '"uncertain":["unclear item"]}]}. For each requested image use exactly one record, with enough detail '
    "to independently understand it. Do not output hypotheses in the visible description. Do not call tools."
)
SYNTH = (
    'Use only your recorded observations, and keep visible evidence, temporal inference and unknowns apart. '
    'Use original integer PTS. Return text JSON with "facts":[{"pts":[integer],"claim":"text"}], '
    '"hypotheses":[string],"unresolved":[string],"final_narration":"concise account". '
    'Do not fill an unseen interval. Preserve uncertainty and point out any conflict. '
)
GROUNDS = (
    'For each frame you described, identify where in that same image the pixels supporting your prior '
    'description appear and which details support it. Correct any claim for which you cannot find evidence. '
    'Do not treat another frame as support. Return only text JSON '
    '{"facts":[{"pts":integer,"description":"corrected observation","location":"in-image position",'
    '"visible_support":"visible details supporting the observation","uncertain":["remaining uncertainty"]}]}.'
)
TASKS = {"short": QUESTIONS["short"], "g3": QUESTIONS["g3"]}
BUDGET = {"turns": 6, "actions_per_turn": 2, "source_ranges": 2}
SEED = 211012
REVIEW = """Independently audit the anonymous trial responses using only the sealed source key and the provided images or
receipts. For each requested row, label the target relation correct, wrong, omitted, or uncertain. A precise
positive contradiction is wrong. Silence, a vague location or a different observation is omitted. If the source
key says the relation is not clearly visible, a compatible statement of uncertainty is acceptable as uncertain.
Do not count a nonexistent contact gap on a near frame or a false continued grasp on a separated frame as correct.
For the hidden-object control, only an explicit observation of no discernible small object counts as correct.
Ignore uncertainty about a thing's precise identity when the direction/visibility claim is clear. Never elevate
an inference about hidden path, exact release, continuity or force to direct evidence. Record evidence quotes or
brief reasons, wrong direction or contact details, frame-binding issues, and final narration drift separately.
An image must actually have been delivered to the constituent visual session before its description counts as
direct. A text-only synthesis citing a model record is text memory, not its own direct viewing. Never infer actual
mental attribute propagation from a response; mark specific observed claims with another compatible source PTS only.
Return JSON only as {"facts":[{"row":"exact requested row ID","label":"correct|wrong|omitted|uncertain",
"reason":"short", "borrowed_pts":null}], "finals":[{"id":"trial ID","right_or_gap_correct":false,
"visible_observations":0,"unknowns":0,"unsupported":0,"time_or_consistency_error":0,"note":"short"}]}.
Return every provided requested row exactly once, including omissions."""


def orders(case):
    ascending = list(PINS[case]["pts"])
    mixed = ascending.copy()
    Random(SEED + int(case == "g3")).shuffle(mixed)
    return {"ascending": ascending, "descending": ascending[::-1], "shuffled": mixed}


def jobs(replicates=(1, 2)):
    found = [(case, arm, variant, rep) for case in PINS for arm, variants in CONDITIONS.items()
             for variant in variants for rep in replicates]
    Random(SEED).shuffle(found)
    return found


def protocol():
    return {"date": "2026-11-21", "task": "P3-12 full multiframe matrix and textual memory", "route": ROUTE,
            "conditions": CONDITIONS, "targets": TARGETS, "distractors": DISTRACTORS,
            "orders": {case: orders(case) for case in PINS}, "seed": SEED, "randomized_schedule": jobs(),
            "prompts": {"describe": DESCRIBE, "record": RECORD, "synthesis": SYNTH, "grounds": GROUNDS,
                        "tasks": TASKS, "historical_shared": SHARED, "historical_add": ADD, "historical_final": FINAL},
            "review_rubric": REVIEW, "tool_budget": BUDGET, "paired_memory": "C7 and C9 use identical generated independent packets; C15 builds fresh.",
            "repeat_rule": "Two independent replicates. At most one technical retry with identical condition. A third repeat when the two completed sessions disagree on the preregistered target visual relation or any of the sequence visibility anchors (correct versus wrong); cap one additional repetition per case-arm-variant.",
            "rating": ["correct", "wrong", "omitted", "uncertain"],
            "process": "Run every condition without an evidence gate. Separate per-frame visual and final narrative grading; preserve all artifacts."}
