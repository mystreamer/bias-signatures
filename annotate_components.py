"""Judgment-based features via the annotation sidecar (ADR 0007).

  --generate        LLM proposals (argumentativeness of instructions)
  --import FILE     Label Studio export -> sidecar rows (human annotations)
  --evaluate        agreement stats + reify sidecar onto component features

Sidecar rows: {component_id, feature, value, source} with source
"llm:<alias>" or "human:<annotator>". Reification: human annotations win
where present (per-label majority among humans for multi-label features),
otherwise the LLM proposal stands.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import dspy
import pandas as pd

import llm
from constructs import parse_foundation
from llm import DATA_DIR

COMPONENTS_DIR = DATA_DIR / "components"
SIDECAR = COMPONENTS_DIR / "annotations.json"


def load_sidecar() -> list[dict]:
    return json.loads(SIDECAR.read_text()) if SIDECAR.exists() else []


def save_sidecar(rows: list[dict]) -> None:
    SIDECAR.write_text(json.dumps(rows, indent=1, ensure_ascii=False))


def load_components(name: str) -> list[dict]:
    return json.loads((COMPONENTS_DIR / f"{name}.json").read_text())


def save_components(name: str, components: list[dict]) -> None:
    (COMPONENTS_DIR / f"{name}.json").write_text(
        json.dumps(components, indent=1, ensure_ascii=False)
    )


# -------------------------------------------------------------------- generate

def generate() -> None:
    """LLM proposals for instruction argumentativeness (thesis A.6.2)."""
    from concurrent.futures import ThreadPoolExecutor

    rows = load_sidecar()
    done = {r["component_id"] for r in rows if r["feature"] == "argumentative"}
    instructions = [c for c in load_components("instructions") if c["id"] not in done]
    print(f"generate: {len(instructions)} instructions to classify")

    dspy.configure(lm=llm.structured_lm())
    classify = dspy.Predict(llm.ClassifyArgumentativeEssayInstruction)
    source = f"llm:{llm.WORKER_MODEL}"

    def propose(comp: dict) -> dict:
        answer = classify(instruction=comp["text"]).is_argumentative_essay_instruction
        return {
            "component_id": comp["id"],
            "feature": "argumentative",
            "value": answer.strip().lower() == "yes",
            "source": source,
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, row in enumerate(pool.map(propose, instructions), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(instructions):
                print(f"  {i}/{len(instructions)}")
    save_sidecar(rows)


# -------------------------------------------------------------------- import

def import_export(path: str) -> None:
    """Label Studio topic export -> one sidecar row per (topic, annotator).

    Expected columns: term, plus one column per annotator (annotator_*) holding
    comma-separated moral foundation labels; empty = morally irrelevant.
    """
    df = pd.read_csv(path)
    by_text = {c["text"]: c for c in load_components("issues")}
    annotator_cols = [c for c in df.columns if c.startswith("annotator")]
    if not annotator_cols:
        raise SystemExit(f"no annotator_* columns in {path}")

    rows = [r for r in load_sidecar() if not (
        r["feature"] == "mf" and r["source"].startswith("human:")
    )]  # re-import replaces previous human MF rows
    missing = []
    for _, rec in df.iterrows():
        comp = by_text.get(rec["term"])
        if comp is None:
            missing.append(rec["term"])
            continue
        for col in annotator_cols:
            labels = rec[col]
            if pd.isna(labels):  # no labels assigned -> no annotation row
                continue
            rows.append({
                "component_id": comp["id"],
                "feature": "mf",
                "value": sorted(
                    parse_foundation(l).value for l in str(labels).split(",") if l.strip()
                ),
                "source": f"human:{col}",
            })
    if missing:
        print(f"WARNING: {len(missing)} terms not matched to components: {missing[:5]} ...")
    save_sidecar(rows)
    print(f"imported {len(annotator_cols)} annotators x {len(df) - len(missing)} topics")


# -------------------------------------------------------------------- evaluate

def _consolidate(values: list) -> object:
    """Multi-label lists: per-label majority (strict, so 2 annotators means
    intersection). Scalars: mode."""
    if values and isinstance(values[0], list):
        counts = Counter(label for v in values for label in v)
        return sorted(l for l, n in counts.items() if n > len(values) / 2 or len(values) == 1)
    mode = Counter(json.dumps(v, sort_keys=True) for v in values).most_common(1)[0][0]
    return json.loads(mode)


def evaluate() -> None:
    rows = load_sidecar()
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["component_id"], r["feature"])].append(r)

    by_cid: dict[str, dict[str, object]] = defaultdict(dict)
    unvalidated = 0
    disagreements = 0
    for (cid, feature), group in grouped.items():
        human = [r["value"] for r in group if r["source"].startswith("human:")]
        if human:
            if len({json.dumps(v, sort_keys=True) for v in human}) > 1:
                disagreements += 1
            by_cid[cid][feature] = _consolidate(human)
        else:
            by_cid[cid][feature] = _consolidate([r["value"] for r in group])
            unvalidated += 1

    for name in ("instructions", "issues"):
        components = load_components(name)
        touched = 0
        for comp in components:
            for feature, value in by_cid.get(comp["id"], {}).items():
                comp["features"][feature] = value
                touched += 1
            # political moral relevance derives from the MF labels; a topic no
            # annotator assigned foundations to is morally irrelevant (§3.2.2.3)
            if comp["features"].get("domain") == "political":
                comp["features"].setdefault("mf", [])
                if "mf" not in by_cid.get(comp["id"], {}):
                    comp["features"].setdefault("morally_relevant", False)
                else:
                    comp["features"]["morally_relevant"] = bool(comp["features"]["mf"])
        save_components(name, components)
        print(f"{name}: reified {touched} feature values")

    print(f"human disagreements (consolidated by majority): {disagreements}")
    print(f"features resting on unvalidated LLM proposals: {unvalidated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--import", dest="import_path", metavar="FILE")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.generate:
        generate()
    if args.import_path:
        import_export(args.import_path)
    if args.evaluate:
        evaluate()
    if not (args.generate or args.import_path or args.evaluate):
        parser.print_help()
