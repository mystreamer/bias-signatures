"""Assemble components from raw data — mechanically inherited features only.

Reads data/raw/, writes data/components/{instructions,issues,statements,deontic_ops}.json.
Component schema (CONTEXT.md): {id, text, features: {...}, variants: {...}}.

Everything requiring judgment (LLM or human) happens later: augment_components.py
fills variants (and the confession→AITA canonical text), annotate_components.py
adds judgment-based features (argumentativeness, political moral relevance/MF).
Re-running preserves what those stages added.
"""

from __future__ import annotations

import hashlib
import json
import re

import pandas as pd

from constructs import parse_foundation
from llm import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
COMPONENTS_DIR = DATA_DIR / "components"

SAMPLING_SEED = 42  # legacy value; the SC101 stratified subsets depend on it


def component(cid: str, text: str, features: dict, variants: dict | None = None) -> dict:
    return {"id": cid, "text": text, "features": features, "variants": variants or {}}


def _short_hash(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:8]


# ---------------------------------------------------------------- instructions

def assemble_instructions() -> list[dict]:
    df = pd.read_csv(RAW_DIR / "issuebench_templates.csv")
    df = df[df["annot1_template"].notna()]
    return [
        component(
            f"instr-{row.id}",
            row.annot1_template,
            {"type": "instruction", "source": "issuebench", "source_id": row.id},
        )
        for row in df.itertuples()
    ]


# ---------------------------------------------------------------------- issues

def longest_common_substring(s1: str, s2: str) -> str:
    """Longest common substring (case-insensitive), preferring candidates
    without leading/trailing whitespace. Ported verbatim in behaviour from the
    legacy pipeline — the deontic-operator derivation depends on it."""
    if not s1 or not s2:
        return ""
    a, b = s1.lower(), s2.lower()
    m, n = len(a), len(b)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    longest, ends = 0, []
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
                if table[i][j] > longest:
                    longest, ends = table[i][j], [i]
                elif table[i][j] == longest:
                    ends.append(i)
            else:
                table[i][j] = 0
    if longest == 0:
        return ""
    for end in ends:
        candidate = s1[end - longest : end]
        if candidate.strip() == candidate and candidate:
            return candidate
    return s1[ends[0] - longest : ends[0]].strip()


def _political_rows() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "issuebench_issues.csv")
    df = df[df["tag_exclude"].isna()]
    return df.drop_duplicates(subset="topic_neutral")


def assemble_political_issues() -> list[dict]:
    issues = []
    for row in _political_rows().itertuples():
        lcs_pro = longest_common_substring(row.topic_neutral, row.topic_pro or "")
        lcs_con = longest_common_substring(row.topic_neutral, row.topic_con or "")
        issues.append(
            component(
                f"pol-{_short_hash(row.topic_neutral)}",
                row.topic_neutral,
                {
                    "type": "issue",
                    "domain": "political",
                    "source": "issuebench",
                    "pre_primed": row.topic_neutral.lower().startswith("the ethics of"),
                    "lcs_pro": lcs_pro,
                    "lcs_con": lcs_con,
                },
                # per-issue framed surface forms come straight from the dataset
                {"framing_pro": row.topic_pro, "framing_con": row.topic_con},
            )
        )
    return issues


def _load_sc101() -> pd.DataFrame:
    return pd.read_csv(
        RAW_DIR / "social-chem-101" / "social-chem-101.v1.0.tsv", sep="\t", low_memory=False
    )


def _stratify(df: pd.DataFrame) -> pd.DataFrame:
    """Balance univalent rows across foundations, then keep one row per
    situation — the legacy two-step sample (both seeded)."""
    univalent = df[~df["rot-moral-foundations"].str.contains(r"\|", na=False)]
    univalent = univalent[univalent["rot-moral-foundations"].notna()]
    min_count = univalent["rot-moral-foundations"].value_counts().min()
    # plain group loops instead of .apply: pandas 3 drops grouping columns
    # inside .apply, and the per-group .sample calls are identical either way
    stratified = pd.concat(
        g.sample(n=min(len(g), min_count), random_state=SAMPLING_SEED)
        for _, g in univalent.groupby("rot-moral-foundations")
    ).reset_index(drop=True)
    return pd.concat(
        g.sample(n=1, random_state=SAMPLING_SEED)
        for _, g in stratified.groupby("situation-short-id")
    ).reset_index(drop=True)


def assemble_personal_issues() -> list[dict]:
    df = _load_sc101()

    # moral: AITA, 5-worker subset, morality-ethics RoTs; topic text = situation
    moral = df[
        (df["area"] == "amitheasshole")
        & (df["m"] == 5)
        & (df["rot-categorization"] == "morality-ethics")
    ]
    moral = _stratify(moral)

    # non-moral: AITA + confessions, 5-worker subset, excluding any situation
    # with a morality-ethics RoT (checked across all m); topic text = situation
    both = df[df["area"].isin(["amitheasshole", "confessions"])]
    moral_situations = set(
        both[both["rot-categorization"].str.contains("morality-ethics", na=False)][
            "situation-short-id"
        ]
    )
    nonmoral = both[(both["m"] == 5) & ~both["situation-short-id"].isin(moral_situations)]
    nonmoral = _stratify(nonmoral)

    issues = []
    for _, row in moral.iterrows():
        sid = row["situation-short-id"]
        issues.append(
            component(
                f"sc-{sid}",
                row["situation"],
                {
                    "type": "issue",
                    "domain": "personal",
                    "source": "social-chemistry",
                    "area": row["area"],
                    "morally_relevant": True,
                    "mf": [parse_foundation(row["rot-moral-foundations"]).value],
                    "rot": row["rot"],
                    "situation_id": sid,
                },
            )
        )
    for _, row in nonmoral.iterrows():
        sid = row["situation-short-id"]
        features = {
            "type": "issue",
            "domain": "personal",
            "source": "social-chemistry",
            "area": row["area"],
            "morally_relevant": False,
            "mf": [parse_foundation(row["rot-moral-foundations"]).value],
            "rot": row["rot"],
            "situation_id": sid,
        }
        if row["area"] == "confessions":
            features["is_confession"] = True  # augment rewrites text to an AITA phrase
        issues.append(component(f"sc-{sid}", row["situation"], features))
    return issues


# ------------------------------------------------------------------ statements

def assemble_statements() -> list[dict]:
    mfq = json.loads((RAW_DIR / "manual" / "mfq.json").read_text())
    statements = []

    m30 = mfq["mfq30"]
    combined = list(m30["relevance_questions"]) + list(m30["agreement_questions"])
    index_to_mf = {
        i: name for name, cfg in m30["foundations"].items() for i in cfg["items"]
    }
    control_indices = {cfg["index"] for cfg in m30["controls"].values()}
    for i, text in enumerate(combined):
        mf = index_to_mf.get(i)
        statements.append(
            component(
                f"mfq30-{i:02d}",
                text,
                {
                    "type": "statement",
                    "questionnaire": "mfq30",
                    "part": "relevance" if i < len(m30["relevance_questions"]) else "agreement",
                    "control": i in control_indices,
                    "morally_relevant": i not in control_indices,
                    "mf": [parse_foundation(mf).value] if mf else [],
                },
            )
        )

    m2 = mfq["mfq2"]
    index_to_mf2 = {
        i: name for name, cfg in m2["foundations"].items() for i in cfg["items"]
    }
    for i, text in enumerate(m2["questions"]):
        statements.append(
            component(
                f"mfq2-{i:02d}",
                text,
                {
                    "type": "statement",
                    "questionnaire": "mfq2",
                    "part": None,
                    "control": False,
                    "morally_relevant": True,
                    "mf": [parse_foundation(index_to_mf2[i]).value],
                },
            )
        )
    return statements


# ---------------------------------------------------------------- deontic ops

def assemble_deontic_ops() -> list[dict]:
    """IssueBench deontic operators: replace each topic's LCS with X in its
    framed forms, then dedupe. SC operators are LLM-generated in augmentation."""
    ops = set()
    for row in _political_rows().itertuples():
        for framed, stance in ((row.topic_pro, "pro"), (row.topic_con, "con")):
            if not isinstance(framed, str) or not framed.strip():
                continue
            lcs = longest_common_substring(row.topic_neutral, framed)
            if not lcs:
                continue
            op = re.sub(re.escape(lcs), "X", framed, count=1, flags=re.IGNORECASE).strip()
            if op and op != "X":
                ops.add((op, stance))
    return [
        component(
            f"dop-{_short_hash(op)}",
            op,
            {"type": "deontic_op", "framing": stance, "source": "issuebench"},
        )
        for op, stance in sorted(ops)
    ]


# ----------------------------------------------------------------- persistence

def merge_write(path, assembled: list[dict]) -> None:
    """Write components, preserving what augment/annotate added on re-runs:
    existing features/variants survive unless assembly produces the same key,
    and a confession's rewritten canonical text is never reset."""
    existing = {}
    if path.exists():
        existing = {c["id"]: c for c in json.loads(path.read_text())}
    merged = []
    for comp in assembled:
        old = existing.get(comp["id"])
        if old:
            if old["features"].get("aita_rewritten"):
                comp["text"] = old["text"]
            comp["features"] = {**old["features"], **comp["features"]}
            comp["variants"] = {**old["variants"], **comp["variants"]}
        merged.append(comp)
    path.write_text(json.dumps(merged, indent=1, ensure_ascii=False))
    print(f"{path.name}: {len(merged)} components")


if __name__ == "__main__":
    COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)
    merge_write(COMPONENTS_DIR / "instructions.json", assemble_instructions())
    merge_write(
        COMPONENTS_DIR / "issues.json",
        assemble_political_issues() + assemble_personal_issues(),
    )
    merge_write(COMPONENTS_DIR / "statements.json", assemble_statements())
    merge_write(COMPONENTS_DIR / "deontic_ops.json", assemble_deontic_ops())
