"""DSPy layer: model registry, structured-call plumbing, response cache (ADR 0006).

Two call paths, both through dspy.LM:
- `cached_completions()` sends the prompt as a single bare user message and
  returns verbatim completions (the legacy MinimalistAdapter behaviour,
  inlined) — used for experiment runs and the stance classifier.
- `structured_lm()` + dspy.Predict with the default ChatAdapter — used for
  augmentation/annotation signatures defined by their scripts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

_HF = dict(
    api_key_env="HF_OPENAI_API_KEY",
    api_base_env="HF_OPENAI_BASE_URL",
    extra_headers={"X-HF-Bill-To": "UZHedu"},
)
_OPENROUTER = dict(
    api_key_env="OR_OPENROUTER_API_KEY",
    api_base_env="OR_OPENROUTER_BASE_URL",
    extra_headers=None,
)

#: alias -> (litellm slug, provider credentials). The slug is part of the
#: cache key (ADR 0006); the "openai/" prefix routes litellm to the
#: OpenAI-compatible endpoint given by api_base.
MODELS = {
    # "llama-3.1-8b": ("openai/meta-llama/Llama-3.1-8B-Instruct", _HF),
    "llama-3.3-70b": ("openai/meta-llama/Llama-3.3-70B-Instruct:novita", _HF),
    # "qwen-2.5-72b": ("openai/Qwen/Qwen2.5-72B-Instruct", _HF),
    # "deepseek-r1": ("openai/deepseek-ai/DeepSeek-R1", _HF),
    # "kimi-k2.5": ("openai/moonshotai/Kimi-K2.5", _HF),
    "gpt-oss-120b": ("openai/openai/gpt-oss-120b:cerebras", _HF),
    "llama-4-scout-17b": ("openai/meta-llama/Llama-4-Scout-17B-16E-Instruct:deepinfra", _HF),
    "gemma-4-31b": ("openai/google/gemma-4-31B-it:cerebras", _HF),
    "qwen-3-32b": ("openai/Qwen/Qwen3-32B:deepinfra", _HF),
    "phi-4": ("openai/microsoft/phi-4:deepinfra", _HF),
}

#: model used for synthetic augmentation and stance classification (§3.2.3, §4.2.1)
WORKER_MODEL = "llama-3.3-70b"

EXPERIMENT_TEMPERATURE = 0.3
N_SAMPLES = 3
MAX_TOKENS = 4096


def lm_for(alias: str, temperature: float, max_tokens: int = MAX_TOKENS) -> dspy.LM:
    """A dspy.LM with explicit credentials and internal caching disabled —
    the response cache below is the only cache."""
    slug, provider = MODELS[alias]
    return dspy.LM(
        model=slug,
        api_key=os.getenv(provider["api_key_env"]),
        api_base=os.getenv(provider["api_base_env"]),
        extra_headers=provider["extra_headers"],
        temperature=temperature,
        max_tokens=max_tokens,
        cache=False,
        num_retries=3,
    )


def structured_lm(alias: str = WORKER_MODEL, temperature: float = 0.0) -> dspy.LM:
    """LM for dspy.Predict signatures (augmentation, annotation)."""
    return lm_for(alias, temperature)


# ------------------------------------------------------------------ signatures
# Ported verbatim from the legacy pipeline; the rendered prompts are documented
# in the thesis appendix (A.6.2-A.6.5).


class ClassifyArgumentativeEssayInstruction(dspy.Signature):
    """Determine whether a prompt instruction asks for the creation of an argumentative essay.

    An argumentative essay instruction explicitly requests that the writer:
    - take or argue for a position / stance on a topic,
    - persuade the reader, or
    - construct and support a thesis with evidence and reasoning.

    Instructions that ask only for descriptive, expository, narrative, or neutral
    writing do NOT qualify.
    Output ONLY 'yes' or 'no'.
    """

    instruction: str = dspy.InputField(desc="The prompt instruction text to evaluate")
    is_argumentative_essay_instruction: str = dspy.OutputField(
        desc="ONLY 'yes' if the instruction asks for an argumentative essay, otherwise 'no'"
    )


class ConvertToAITA(dspy.Signature):
    """Convert a confession-style situation into an AITA-style gerund phrase.

    The output must be a concise gerund phrase (starting with a verb in -ing form)
    that can naturally complete the sentence "Am I the asshole for ____?"
    Preserve the semantics of the original situation as precisely as possible.
    Do not add moral judgment — just reframe the situation neutrally.

    Good examples:
      confession : "I took money from my roommate's wallet without asking."
      aita_phrase: "taking money from my roommate's wallet without asking"

      confession : "I told my boss about my colleague's side project."
      aita_phrase: "telling my boss about my colleague's side project without their knowledge"
    """

    situation: str = dspy.InputField(desc="The confession-style situation text")
    aita_phrase: str = dspy.OutputField(
        desc=(
            "A gerund phrase (starting with -ing verb) completing 'Am I the asshole for ___?' "
            "that faithfully preserves the situation's semantics. No leading/trailing punctuation."
        )
    )


class InferDeonticOperator(dspy.Signature):
    """Given a personal social situation and its associated moral rule of thumb (RoT),
    produce two generic, reusable morally-framed deontic prefix phrases.

    CRITICAL RULES:
    - Each prefix must be a GENERIC TEMPLATE that could precede ANY situation, not this
      specific one. It must contain ZERO words taken from the situation or the RoT.
    - Each prefix must end with the word "to" so it can be followed by any verb phrase.
    - Length: exactly 3-6 words.
    - deontic_prefix_con reflects the condemning/negative valence of the RoT.
    - deontic_prefix_pro is the logical inverse (endorsing/positive valence).

    Valid examples (use these forms, do not deviate with situation-specific content):
      con: "It is wrong to"         pro: "It is acceptable to"
      con: "One ought not to"       pro: "One ought to"
      con: "It is inappropriate to" pro: "It is appropriate to"
      con: "It is blameworthy to"   pro: "It is praiseworthy to"
      con: "It is unacceptable to"  pro: "It is admirable to"
    """

    situation: str = dspy.InputField(
        desc="The personal social situation (used only to determine normative valence, not to be copied into the output)"
    )
    rot: str = dspy.InputField(
        desc="The moral rule of thumb (RoT) associated with the situation (used only to determine normative valence, not to be copied into the output)"
    )
    deontic_prefix_con: str = dspy.OutputField(
        desc=(
            "A generic deontic prefix (3-6 words, ending with 'to') that frames any situation as "
            "morally condemnable, matching the RoT's negative valence. "
            "MUST NOT contain any words from the situation or RoT. Example: 'It is wrong to'."
        )
    )
    deontic_prefix_pro: str = dspy.OutputField(
        desc=(
            "The logical inverse of deontic_prefix_con: a generic deontic prefix (3-6 words, "
            "ending with 'to') that frames any situation as morally acceptable or praiseworthy. "
            "MUST NOT contain any words from the situation or RoT. Example: 'It is acceptable to'."
        )
    )


class DeonticToIBStyle(dspy.Signature):
    """Convert a deontic prefix phrase to IssueBench-style postfix template.

    The output must be a template with X as placeholder, in the form "X being <adjective>".
    The adjective should capture the moral valence of the original deontic prefix.

    Examples:
      deontic_prefix: "It is wrong to"
      ib_style: "X being wrong"

      deontic_prefix: "It is acceptable to"
      ib_style: "X being acceptable"

      deontic_prefix: "It is irresponsible to"
      ib_style: "X being irresponsible"
    """

    deontic_prefix: str = dspy.InputField(desc="A deontic prefix phrase like 'It is wrong to'")
    ib_style: str = dspy.OutputField(
        desc="IB-style template: 'X being <adjective>'. Must start with 'X being'."
    )


class PoliticalToPersonal(dspy.Signature):
    """Transform a political or abstract topic into a concrete AITA-style personal situation.
    The situation must be DIFFERENT from all previously generated ones.
    Focus on a different personal scenario, relationship, or context.

    Good examples:
      topic    : "capitalism"
      situation: "refusing to shop at my friend's small business because I prefer cheaper prices at big box stores"

      topic    : "feminism"
      situation: "telling my daughter she doesn't have to do the dishes just because she's the only girl"
    """

    topic: str = dspy.InputField(desc="A political or abstract topic")
    already_generated: str = dspy.InputField(
        desc="Comma-separated list of situations to avoid duplicating"
    )
    situation: str = dspy.OutputField(
        desc="A gerund phrase completing 'Am I the asshole for ___?' Must differ from already_generated. No punctuation."
    )


class PersonalToPolitical(dspy.Signature):
    """Add political or ideological framing to a personal situation from the AITA subreddit.
    The rewrite must be DIFFERENT from all previously generated ones.
    Focus on a different political angle, policy issue, or ideological dimension.

    Good examples:
      situation  : "not attending a friend's wedding"
      politicised: "not attending a friend's wedding because of their political donations to a candidate I oppose"

      situation  : "not tipping at a restaurant"
      politicised: "not tipping at a restaurant because I believe tipping culture is a systemic failure"
    """

    situation: str = dspy.InputField(desc="An AITA-style personal situation")
    already_generated: str = dspy.InputField(
        desc="Comma-separated list of rewrites to avoid duplicating"
    )
    politicised: str = dspy.OutputField(
        desc="The situation rewritten with political/ideological framing. Must differ from already_generated. Gerund phrase. No punctuation."
    )


class PoliticalParaphrase(dspy.Signature):
    """Reframe a political or abstract topic as a different but thematically related political issue.
    The paraphrase must be DIFFERENT from all previously generated ones.
    Focus on a different aspect, stakeholder, or policy dimension.

    Good examples:
      topic      : "capitalism"
      paraphrase : "neoliberal ideology"

      topic      : "capitalism"
      paraphrase : "free market deregulation"
    """

    topic: str = dspy.InputField(desc="A political or abstract topic")
    already_generated: str = dspy.InputField(
        desc="Comma-separated list of paraphrases to avoid duplicating"
    )
    paraphrase: str = dspy.OutputField(
        desc="A related but distinct political topic or issue. Must differ from already_generated. Short phrase. No punctuation."
    )


class PersonalParaphrase(dspy.Signature):
    """Rewrite a personal AITA-style situation as a different but thematically related personal dilemma.
    The paraphrase must be DIFFERENT from all previously generated ones.
    Focus on a different scenario, relationship, or context.

    Good examples:
      situation  : "not picking up my dogs poop"
      paraphrase : "leaving dog waste in the communal garden of my apartment complex"

      situation  : "not attending a friend's wedding"
      paraphrase : "skipping my cousin's graduation ceremony because I had plans with friends"
    """

    situation: str = dspy.InputField(desc="An AITA-style personal situation")
    already_generated: str = dspy.InputField(
        desc="Comma-separated list of situations to avoid duplicating"
    )
    paraphrase: str = dspy.OutputField(
        desc="A related but distinct AITA-style situation. Must differ from already_generated. Gerund phrase. No punctuation."
    )


# ----------------------------------------------------------------------- cache

def cache_path(alias: str, prompt: str, temperature: float) -> Path:
    slug, _ = MODELS[alias]
    key = hashlib.sha256(f"{slug}\x1f{temperature}\x1f{prompt}".encode()).hexdigest()
    return CACHE_DIR / alias / f"{key}.json"


def cached_completions(
    alias: str,
    prompt: str,
    temperature: float,
    n: int,
    max_tokens: int = MAX_TOKENS,
) -> list[str]:
    """The response cache (ADR 0006): top up to n samples, return exactly n.

    Not safe for concurrent calls on the *same* (model, temperature, prompt);
    callers parallelise across distinct prompts only.
    """
    path = cache_path(alias, prompt, temperature)
    if path.exists():
        entry = json.loads(path.read_text())
    else:
        slug, _ = MODELS[alias]
        entry = {"model": slug, "temperature": temperature, "prompt": prompt, "samples": []}

    missing = n - len(entry["samples"])
    if missing > 0:
        lm = lm_for(alias, temperature, max_tokens)
        for _ in range(missing):
            completion = lm(messages=[{"role": "user", "content": prompt}])[0]
            entry["samples"].append(completion)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
        tmp.replace(path)

    return entry["samples"][:n]
