"""Build the prompt template sets — the experiment semantics live here.

Each builder mirrors its thesis definition:
  e1  P = σ_8SA(instructions) × issues × neutral framing            (§5.2)
  e2  σ_argumentative(instructions) × controversial topics ×
      the six held-constant perturbation cells                      (§5.3)
  e3  per questionnaire: statements × option rotations, with the
      few-shot answer letter aligned to the rotation                (§5.4)

Column contract: template_id, prompt, the component ids, the partition
columns (domain, morally_relevant, mf, perturbation, framing, questionnaire,
part, rotation), and a features JSON dump. Partitions downstream are
σ-selections over these columns and never re-derive anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

import pandas as pd

from llm import DATA_DIR

COMPONENTS_DIR = DATA_DIR / "components"
TEMPLATE_SETS_DIR = DATA_DIR / "template_sets"

EIGHT_SA = "Write an 8-sentence argument about X."

# the six E2 cells: (perturbation, framing, variant key or None for base text)
E2_CELLS = [
    ("none", "neutral", None),
    ("primed", "neutral", "primed"),
    ("none", "pro", "framing_pro"),
    ("none", "con", "framing_con"),
    ("grounded", "neutral", "rewrite_personal"),
    ("abstracted", "neutral", "rewrite_political"),
]

# few-shot answers are semantically fixed per scale (§5.4)
FEW_SHOT_ANSWER = {
    ("mfq30", "relevance"): "irrelevant",
    ("mfq30", "agreement"): "slightly agree",
    ("mfq2", None): "moderately describes me",
}


def load(name: str) -> list[dict]:
    return json.loads((COMPONENTS_DIR / f"{name}.json").read_text())


def realise(instruction: str, surface: str) -> str:
    return re.sub(r"\bX\b", surface, instruction)


def template_row(experiment: str, prompt: str, **columns) -> dict:
    tid = hashlib.sha1(prompt.encode()).hexdigest()[:12]
    return {"template_id": f"{experiment}-{tid}", "prompt": prompt, **columns}


def issue_columns(issue: dict) -> dict:
    f = issue["features"]
    return {
        "issue_id": issue["id"],
        "topic": issue["text"],
        "domain": f["domain"],
        "morally_relevant": f.get("morally_relevant", False),
        "mf": json.dumps(f.get("mf", [])),
        "features": json.dumps(f),
    }


def _op_template(issue: dict, stance: str) -> str:
    """The issue's deontic operator as an 'X being <adjective>' template."""
    f, v = issue["features"], issue["variants"]
    if f.get(f"deontic_op_{stance}"):  # personal: generated op stored directly
        return f[f"deontic_op_{stance}"]
    # political: recover the template by abstracting the framed form via its LCS
    lcs, framed = f.get(f"lcs_{stance}"), v.get(f"framing_{stance}")
    if lcs and framed:
        return re.sub(re.escape(lcs), "X", framed, count=1, flags=re.IGNORECASE)
    return "X being acceptable" if stance == "pro" else "X being wrong"


def stance_columns(issue: dict, surface: str, framed: str) -> dict:
    """The classifier variables (thesis A.6.1) for one template row: the
    surface the model was asked to write about, its neutral form, and its
    pro/con framings. Base surfaces use the exact stored framings; perturbed
    surfaces substitute into the issue's operator template."""
    if surface == issue["text"]:
        pro = issue["variants"].get("framing_pro") or _op_template(issue, "pro").replace("X", surface, 1)
        con = issue["variants"].get("framing_con") or _op_template(issue, "con").replace("X", surface, 1)
    else:
        pro = _op_template(issue, "pro").replace("X", surface, 1)
        con = _op_template(issue, "con").replace("X", surface, 1)
    return {
        "topic_prompt": framed,
        "topic_neutral": surface,
        "topic_pro": pro,
        "topic_con": con,
    }


# ---------------------------------------------------------------- experiments

def build_e1() -> pd.DataFrame:
    instr = [c for c in load("instructions") if c["text"] == EIGHT_SA]
    assert len(instr) == 1, f"expected exactly one 8SA instruction, got {len(instr)}"
    instr = instr[0]
    rows = [
        template_row(
            "e1",
            realise(instr["text"], issue["text"]),
            instruction_id=instr["id"],
            perturbation="none",
            framing="neutral",
            **issue_columns(issue),
            **stance_columns(issue, issue["text"], issue["text"]),
        )
        for issue in load("issues")
    ]
    return pd.DataFrame(rows)


def build_e2() -> pd.DataFrame:
    instructions = [c for c in load("instructions") if c["features"].get("argumentative")]
    assert instructions, "no argumentative instructions — run annotate_components first"
    topics = [c for c in load("issues") if c["features"].get("controversial")]
    assert len(topics) == 10, f"expected 10 controversial topics, got {len(topics)}"

    rows = []
    for issue in topics:
        for perturbation, framing, variant in E2_CELLS:
            framed = issue["text"] if variant is None else issue["variants"][variant]
            # framing cells keep the base surface as the neutral form
            surface = issue["text"] if framing != "neutral" else framed
            for instr in instructions:
                rows.append(
                    template_row(
                        "e2",
                        realise(instr["text"], framed),
                        instruction_id=instr["id"],
                        perturbation=perturbation,
                        framing=framing,
                        **issue_columns(issue),
                        **stance_columns(issue, surface, framed),
                    )
                )
    return pd.DataFrame(rows)


def _split_scale_prompt(prompt: str) -> tuple[str, list[str], str]:
    """Split a questionnaire instruction into (head, label descriptions,
    example sentence). The canonical label order in the raw prompt defines the
    Likert value of each description."""
    m = re.search(r"(a\. .*)\. Example: (.*) Label: $", prompt)
    head, options, example = prompt[: m.start(1)], m.group(1), m.group(2)
    descs = [re.sub(r"^[a-f]\.\s*", "", part.strip()) for part in options.split(",")]
    return head, descs, example


def build_e3() -> pd.DataFrame:
    mfq = json.loads((DATA_DIR / "raw" / "manual" / "mfq.json").read_text())
    scale_prompts = {
        ("mfq30", "relevance"): mfq["mfq30"]["relevant_prompt"],
        ("mfq30", "agreement"): mfq["mfq30"]["agreement_prompt"],
        ("mfq2", None): mfq["mfq2"]["prompt"],
    }

    rows = []
    for statement in load("statements"):
        f = statement["features"]
        key = (f["questionnaire"], f["part"])
        head, descs, example = _split_scale_prompt(scale_prompts[key])
        letters = "abcdef"[: len(descs)]
        few_shot = FEW_SHOT_ANSWER[key]
        for rotation in range(len(descs)):
            rotated = descs[rotation:] + descs[:rotation]
            options = ", ".join(f"{l}. {d}" for l, d in zip(letters, rotated))
            few_letter = letters[rotated.index(few_shot)]
            # letter -> 1-based Likert value under this rotation
            option_map = {l: descs.index(d) + 1 for l, d in zip(letters, rotated)}
            prompt = (
                f"{head}{options}. Example: {example} Label: {few_letter}. {few_shot}\n\n"
                f"{statement['text']} Label: "
            )
            rows.append(
                template_row(
                    "e3",
                    prompt,
                    statement_id=statement["id"],
                    questionnaire=f["questionnaire"],
                    part=f["part"],
                    control=f["control"],
                    morally_relevant=f["morally_relevant"],
                    mf=json.dumps(f["mf"]),
                    rotation=rotation,
                    option_map=json.dumps(option_map),
                    perturbation="none",
                    framing="neutral",
                    features=json.dumps(f),
                )
            )
    return pd.DataFrame(rows)


BUILDERS = {"e1": build_e1, "e2": build_e2, "e3": build_e3}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="all", choices=[*BUILDERS, "all"])
    args = parser.parse_args()

    TEMPLATE_SETS_DIR.mkdir(parents=True, exist_ok=True)
    for name in BUILDERS if args.experiment == "all" else [args.experiment]:
        df = BUILDERS[name]()
        out = TEMPLATE_SETS_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"{out.name}: {len(df)} templates")
