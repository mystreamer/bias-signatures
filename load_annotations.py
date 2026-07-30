"""Generate the outbound human-annotation workload (task files).

Reads data/components/ (+ the annotation store) and writes one CSV per
task to data/annotation_tasks/ — Label Studio imports these directly (one
row = one task, columns addressable as $column). Completed annotations
return through annotate_components.py --import.

  issues.csv         political topics + 50 blind SC controls (35 moral,
                     7 per foundation; 15 non-moral), shuffled; controls
                     carry no marker — recoverable by id lookup
  issues.xml         Label Studio labeling config for issues.csv (choice
                     values = raw construct names parse_foundation accepts;
                     empty submission = not morally relevant, §3.2.2.3)
  argumentative.csv  up to 50 positive + 50 negative LLM proposals, blind
  framings.csv       LLM-generated pro/con framings of the SC components
  rewrites.csv       cross-domain rewrites of the controversial topics
  aita.csv           AITA-rewritten confessions beside their originals

Deterministic (seed 42): re-running regenerates identical files.
"""

from __future__ import annotations

import json

import pandas as pd

from llm import DATA_DIR

COMPONENTS_DIR = DATA_DIR / "components"
TASKS_DIR = DATA_DIR / "annotation_tasks"
SEED = 42  # = assemble_components.SAMPLING_SEED, so controls stay stable

N_MORAL_PER_FOUNDATION = 7
N_NONMORAL = 15
N_PER_PROPOSAL = 50


def components(name: str) -> list[dict]:
    return json.loads((COMPONENTS_DIR / f"{name}.json").read_text())


def write(name: str, df: pd.DataFrame) -> None:
    df.to_csv(TASKS_DIR / f"{name}.csv", index=False)
    print(f"{name}.csv: {len(df)} rows")


ISSUES_SCHEMA = """\
<View>
  <HyperText name="guidelines" inline="true" clickableLinks="true"
             value="&lt;a href=&quot;https://drive.switch.ch/index.php/s/AVNWmUaSUIwOnzo&quot; target=&quot;_blank&quot;&gt;Annotation guidelines (PDF)&lt;/a&gt;"/>
  <Header value="Is this issue morally relevant? If yes, select all moral foundations it pertains to. If it is not morally relevant, submit without selecting anything."/>
  <Text name="term" value="$term"/>
  <Choices name="mf" toName="term" choice="multiple" showInline="true" required="false">
    <Choice value="care"/>
    <Choice value="equality"/>
    <Choice value="proportionality"/>
    <Choice value="loyalty"/>
    <Choice value="authority"/>
    <Choice value="sanctity"/>
    <Choice value="liberty"/>
  </Choices>
</View>
"""


def issues_task(issues: list[dict]) -> pd.DataFrame:
    political = pd.DataFrame([
        {"id": c["id"], "term": c["text"]}
        for c in issues if c["features"]["domain"] == "political"
    ])
    personal = pd.DataFrame([
        {"id": c["id"], "term": c["text"],
         "moral": c["features"]["morally_relevant"], "mf": c["features"]["mf"][0]}
        for c in issues if c["features"]["domain"] == "personal"
    ])
    controls = pd.concat(
        [g.sample(N_MORAL_PER_FOUNDATION, random_state=SEED)
         for _, g in personal[personal["moral"]].groupby("mf")]
        + [personal[~personal["moral"]].sample(N_NONMORAL, random_state=SEED)]
    )[["id", "term"]]
    combined = pd.concat([political, controls])
    return combined.sample(frac=1, random_state=SEED).reset_index(drop=True)


def argumentative_task(store: list[dict], instructions: list[dict]) -> pd.DataFrame:
    proposals = pd.DataFrame([
        {"id": r["component_id"], "proposal": r["value"]}
        for r in store if r["feature"] == "argumentative"
    ])
    assert not proposals.empty, "no proposals — run annotate_components.py --generate first"
    text = {c["id"]: c["text"] for c in instructions}
    sampled = pd.concat(
        g.sample(min(N_PER_PROPOSAL, len(g)), random_state=SEED)
        for _, g in proposals.groupby("proposal")
    )
    sampled["text"] = sampled["id"].map(text)
    return sampled.sample(frac=1, random_state=SEED)[["id", "text"]].reset_index(drop=True)


def framings_task(issues: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {"id": c["id"], "term": c["text"],
         "framing_pro": c["variants"]["framing_pro"],
         "framing_con": c["variants"]["framing_con"]}
        for c in issues if c["features"]["domain"] == "personal"
    ])


def rewrites_task(issues: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {"id": c["id"], "term": c["text"],
         "direction": key.removeprefix("rewrite_"), "rewrite": rewrite}
        for c in issues if c["features"].get("controversial")
        for key, rewrite in sorted(c["variants"].items()) if key.startswith("rewrite_")
    ])


def aita_task(issues: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {"id": c["id"], "original": c["features"]["original_situation"],
         "rewritten": c["text"]}
        for c in issues if c["features"].get("aita_rewritten")
    ])


if __name__ == "__main__":
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    issues = components("issues")
    store = json.loads((COMPONENTS_DIR / "annotations.json").read_text())
    write("issues", issues_task(issues))
    (TASKS_DIR / "issues.xml").write_text(ISSUES_SCHEMA)
    print("issues.xml: labeling config")
    write("argumentative", argumentative_task(store, components("instructions")))
    write("framings", framings_task(issues))
    write("rewrites", rewrites_task(issues))
    write("aita", aita_task(issues))
