# Bias Signatures

Self-contained pipeline for eliciting moral (value) bias signatures from LLMs: assembling typed prompt components from source datasets, building partitioned prompt template sets, running them against models, and analysing the response distributions (JSD, permutation tests, Spearman correlation).

## Language

**Component**:
The smallest semantic unit of prompt construction — one base real-world item (an instruction, issue, deontic operator, or statement) with its features and variants. One row per base item; perturbed surface forms live inside it, not beside it.
_Avoid_: prompt part, fragment

**Feature**:
A typed semantic property of a component (e.g. `domain`, `morally_relevant`, `mf`, `argumentative`), assigned by annotation or inherited from a source dataset. Partitions select on features.

**Variant**:
An alternative surface form of a component produced by a perturbation or framing (e.g. `primed`, `grounded`, `abstracted`, per-issue framing strings). Stored in the component's `variants` map.
_Avoid_: perturbed component, derived component

**Perturbation**:
An operation that yields a variant of a component: moral priming (weak), framing (moderate), grounding/abstraction (strong).

**Deontic Operator**:
A generic framing expression (e.g. "X being wrong") that combines with an issue to produce a pro- or con-framed variant.
_Avoid_: framing prefix, framing suffix (those are surface details)

**Prompt Template Set**:
A tabular artifact whose rows each hold one realised prompt plus the feature/perturbation/framing columns that partitions select on.

**Partition**:
A subset of a prompt template set selected by a σ-condition over its columns (e.g. `framing = neutral ∧ perturbation = none`). The unit between which differential treatment is measured.
_Avoid_: treatment, condition, group

**Annotation**:
A judgment-based claim about a component's feature, recorded in the annotation store with its source (`llm:<model>` or `human:<annotator>`). Distinct from mechanically inherited features, which assembly derives from raw data without judgment.
_Avoid_: label (reserved for elicitor outputs)

**Proposal**:
An LLM-generated annotation awaiting (optional) human validation.

**Reification**:
The evaluate step that consolidates the annotation store into component features: humans win where present (majority vote among humans), otherwise the LLM proposal stands.
_Avoid_: aggregation, merging

**Control**:
A Social Chemistry item with known (inherited) moral-relevance and foundation features, blindly interleaved into the political-issue annotation task to measure how well annotators recover known labels. Controls carry no marker in the task file; they are identified on the return leg by id lookup against the components.
_Avoid_: blind, plant (as artifact names)

**Foundation**:
One of the moral foundations constructs a component can pertain to. Raw source labels (e.g. `sanctity-degradation`, `ingroup`, `purity`) are parsed into Foundation values by `constructs.py`, the single construct abstraction.

**Canonical Foundations**:
The five foundations all consistency comparisons run over: care, fairness, loyalty, authority, sanctity. Normalisation maps proportionality/equality → fairness, ingroup → loyalty, purity → sanctity; liberty-only items are excluded.
_Avoid_: purity as the canonical name. The thesis introduces the foundation as purity in Background §2.2.2, following the original MFT terminology, and uses sanctity as the canonical label in the experiments (Chapter 6).

**Elicitor**:
A map from raw model responses to labels over an outcome space (the thesis's ϵ): the LLM-based stance elicitor (labels 1–5 plus refusal) or the rule-based Likert elicitor (scale points plus refusal/invalid). Refusal and invalid labels are excluded from Likert distributions but refusal is a first-class bin in stance distributions.
_Avoid_: parser, classifier (as module names)

**Bias Signature**:
The per-model table of pairwise JSD values between partitions' response distributions, with permutation-test significance (E1/E3; per-topic and untested in E2).
_Avoid_: using "signature" for DSPy Signature classes — those live in llm.py and are never called signatures in file or artifact names
