"""The single stable abstraction over moral foundations constructs (see CONTEXT.md).

Raw source labels (SC101 rule-of-thumb tags, MFQ scoring keys, author
annotations) are parsed into `Foundation` values; consistency comparisons
normalise onto the canonical five via `canonical()`.
"""

from __future__ import annotations

from enum import Enum


class Foundation(Enum):
    CARE = "care"
    FAIRNESS = "fairness"
    EQUALITY = "equality"
    PROPORTIONALITY = "proportionality"
    LOYALTY = "loyalty"
    AUTHORITY = "authority"
    SANCTITY = "sanctity"
    LIBERTY = "liberty"

    def __str__(self) -> str:
        return self.value


#: The five foundations all consistency comparisons run over.
CANONICAL_FIVE = (
    Foundation.CARE,
    Foundation.FAIRNESS,
    Foundation.LOYALTY,
    Foundation.AUTHORITY,
    Foundation.SANCTITY,
)

_CANONICAL = {
    Foundation.EQUALITY: Foundation.FAIRNESS,
    Foundation.PROPORTIONALITY: Foundation.FAIRNESS,
    Foundation.LIBERTY: None,  # liberty-only items are excluded from comparisons
}

_ALIASES = {
    # Social Chemistry 101 rule-of-thumb tags
    "care-harm": Foundation.CARE,
    "fairness-cheating": Foundation.FAIRNESS,
    "loyalty-betrayal": Foundation.LOYALTY,
    "authority-subversion": Foundation.AUTHORITY,
    "sanctity-degradation": Foundation.SANCTITY,
    # MFQ-30 scoring keys
    "harm": Foundation.CARE,
    "ingroup": Foundation.LOYALTY,
    "purity": Foundation.SANCTITY,
    **{f.value: f for f in Foundation},
}


def parse_foundation(label: str) -> Foundation:
    """Map a raw source label to a Foundation."""
    try:
        return _ALIASES[label.strip().lower().replace("_", "-")]
    except KeyError:
        raise ValueError(f"unknown moral foundation label: {label!r}") from None


def canonical(foundation: Foundation) -> Foundation | None:
    """Normalise onto the canonical five; None means excluded (liberty)."""
    return _CANONICAL.get(foundation, foundation)


# Outcome supports the elicitors label into. Refusal is a first-class bin in
# the stance support (E1/E2); refusal/invalid are excluded from Likert
# distributions (E3) and therefore absent from the MFQ supports.
REFUSAL = "refusal"
INVALID = "invalid"
STANCE_SUPPORT = ("1", "2", "3", "4", "5", REFUSAL)
MFQ30_SUPPORT = ("1", "2", "3", "4", "5", "6")
MFQ2_SUPPORT = ("1", "2", "3", "4", "5")
