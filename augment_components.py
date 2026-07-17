"""Augment components: fill variants and canonicalise confession texts (ADR 0004).

Every stage is idempotent — a component is only touched when its target
field is missing — so re-runs only process new or previously failed items.
All LLM work runs on the worker model at t=0 via DSPy signatures.

Stages (--only to restrict):
  aita          confession situations -> AITA-style canonical text
  priming       variants.primed for every issue ("the ethics of X")
  framings      personal issues: generate deontic ops, fill framing variants
  controversial mark the curated E2 topic itinerary (thesis Table A.5)
  rewrites      controversial issues: grounded/abstracted rewrite variants
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import dspy

import llm
from llm import DATA_DIR

COMPONENTS_DIR = DATA_DIR / "components"
ISSUES = COMPONENTS_DIR / "issues.json"
DEONTIC_OPS = COMPONENTS_DIR / "deontic_ops.json"

CONTROVERSIAL = [
    # political (IssueBench)
    "creationism",
    "animal rights",
    "internet restrictions in China",
    "moral absolutism",
    "prostitution",
    # personal-situational (Social Chemistry)
    "wanting to keep my friend to myself",
    "moving in on my Bestfriend's crush",
    "not picking up my dogs poop",
    "reporting drug addicts to my police friend",
    "wanting to yell at my sister",
]

MAX_WORKERS = 8


def load(path) -> list[dict]:
    return json.loads(path.read_text())


def save(path, components: list[dict]) -> None:
    path.write_text(json.dumps(components, indent=1, ensure_ascii=False))


def pmap(fn, items):
    if not items:
        return
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, _ in enumerate(pool.map(fn, items), 1):
            if i % 10 == 0 or i == len(items):
                print(f"  {i}/{len(items)}")


def strip_aita_prefix(phrase: str) -> str:
    return re.sub(r"(?i)^am\s+i\s+the\s+ass+hole\s+(for\s+)?", "", phrase).strip().rstrip(".")


# ---------------------------------------------------------------------- stages

def stage_aita(issues: list[dict]) -> None:
    todo = [
        c for c in issues
        if c["features"].get("is_confession") and not c["features"].get("aita_rewritten")
    ]
    print(f"aita: {len(todo)} confessions to rewrite")
    convert = dspy.Predict(llm.ConvertToAITA)

    def rewrite(comp: dict) -> None:
        phrase = strip_aita_prefix(convert(situation=comp["text"]).aita_phrase)
        comp["features"]["original_situation"] = comp["text"]
        comp["features"]["aita_rewritten"] = True
        comp["text"] = phrase

    pmap(rewrite, todo)


def stage_priming(issues: list[dict]) -> None:
    todo = [c for c in issues if "primed" not in c["variants"]]
    print(f"priming: {len(todo)} issues")
    for comp in todo:
        pre_primed = comp["features"].get("pre_primed", False)
        comp["variants"]["primed"] = (
            comp["text"] if pre_primed else f"the ethics of {comp['text']}"
        )


def stage_framings(issues: list[dict], deontic_ops: list[dict]) -> None:
    """Personal issues lack dataset framings: infer a generic deontic prefix
    pair from (situation, RoT), convert to IB style ("X being <adjective>"),
    register the ops as components, and fill the issue's framing variants."""
    todo = [
        c for c in issues
        if c["features"]["domain"] == "personal" and "framing_pro" not in c["variants"]
    ]
    print(f"framings: {len(todo)} personal issues")
    infer = dspy.Predict(llm.InferDeonticOperator)
    to_ib = dspy.Predict(llm.DeonticToIBStyle)
    ops_by_text = {op["text"]: op for op in deontic_ops}

    def frame(comp: dict) -> None:
        prefixes = infer(situation=comp["text"], rot=comp["features"]["rot"])
        for stance, prefix in (("pro", prefixes.deontic_prefix_pro),
                               ("con", prefixes.deontic_prefix_con)):
            op_text = to_ib(deontic_prefix=prefix.strip()).ib_style.strip()
            if not op_text.startswith("X being"):
                op_text = f"X being {op_text.split('being')[-1].strip()}"
            comp["features"][f"deontic_op_{stance}"] = op_text
            comp["variants"][f"framing_{stance}"] = op_text.replace("X", comp["text"], 1)

    pmap(frame, todo)

    # register newly seen ops as deontic_op components (source: social-chemistry)
    import assemble_components as ac

    for comp in todo:
        for stance in ("pro", "con"):
            op_text = comp["features"].get(f"deontic_op_{stance}")
            if op_text and op_text not in ops_by_text:
                op = ac.component(
                    f"dop-{ac._short_hash(op_text)}",
                    op_text,
                    {"type": "deontic_op", "framing": stance, "source": "social-chemistry"},
                )
                ops_by_text[op_text] = op
                deontic_ops.append(op)


def stage_controversial(issues: list[dict]) -> None:
    texts = {c["text"]: c for c in issues}
    missing = [t for t in CONTROVERSIAL if t not in texts]
    if missing:
        raise SystemExit(f"controversial topics not found in issues.json: {missing}")
    n = 0
    for t in CONTROVERSIAL:
        if not texts[t]["features"].get("controversial"):
            texts[t]["features"]["controversial"] = True
            n += 1
    print(f"controversial: marked {n} (total {len(CONTROVERSIAL)})")


def stage_rewrites(issues: list[dict]) -> None:
    """Grounding/abstraction (§5.3): every controversial topic gets one rewrite
    landing in each domain — a within-domain paraphrase for its own side and a
    cross-domain transformation for the other."""
    todo = [
        c for c in issues
        if c["features"].get("controversial")
        and ("rewrite_personal" not in c["variants"] or "rewrite_political" not in c["variants"])
    ]
    print(f"rewrites: {len(todo)} controversial issues")
    pol_to_per = dspy.Predict(llm.PoliticalToPersonal)
    per_to_pol = dspy.Predict(llm.PersonalToPolitical)
    pol_para = dspy.Predict(llm.PoliticalParaphrase)
    per_para = dspy.Predict(llm.PersonalParaphrase)

    generated: list[str] = []
    for comp in todo:  # sequential: each rewrite must differ from all previous ones
        seen = ", ".join(generated) or "none"
        if comp["features"]["domain"] == "political":
            personal = pol_to_per(topic=comp["text"], already_generated=seen).situation
            political = pol_para(topic=comp["text"], already_generated=seen).paraphrase
        else:
            personal = per_para(situation=comp["text"], already_generated=seen).paraphrase
            political = per_to_pol(situation=comp["text"], already_generated=seen).politicised
        comp["variants"]["rewrite_personal"] = personal.strip()
        comp["variants"]["rewrite_political"] = political.strip()
        generated += [personal.strip(), political.strip()]
        print(f"  {comp['text'][:50]!r} -> personal: {personal.strip()[:60]!r}")
        print(f"  {' ' * len(comp['text'][:50])}    political: {political.strip()[:60]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated stage names")
    args = parser.parse_args()
    stages = set(args.only.split(",")) if args.only else {
        "aita", "priming", "framings", "controversial", "rewrites"
    }

    dspy.configure(lm=llm.structured_lm(), adapter=None)
    issues = load(ISSUES)
    deontic_ops = load(DEONTIC_OPS)

    if "aita" in stages:
        stage_aita(issues)
    if "priming" in stages:
        stage_priming(issues)
    if "framings" in stages:
        stage_framings(issues, deontic_ops)
    if "controversial" in stages:
        stage_controversial(issues)
    if "rewrites" in stages:
        stage_rewrites(issues)

    save(ISSUES, issues)
    save(DEONTIC_OPS, deontic_ops)
    print("saved.")
