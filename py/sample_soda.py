#!/usr/bin/env python3
"""
Stratified sampler: pulls a balanced Wisp training set from SODA pairs.

SODA contains generic social dialogue that spans many conversational types.
This script classifies each Q|R pair into a stratum and samples a target
count from each, producing a curated ~3K pair Wisp training set that covers
the model's natural domain without overwhelming its 3.3M-param capacity.

Usage:
  python3 py/sample_soda.py --input data/soda_100k.txt --output data/wisp_soda.txt
  python3 py/sample_soda.py --input data/soda_full.txt  --output data/wisp_soda.txt
"""

import argparse
import os
import random
import re
import sys
from typing import Optional

# ── Stratum definitions ───────────────────────────────────────────────────────
# Each stratum is a list of regex patterns matched against the UPPERCASE query.
# Order matters: first match wins.

_STRATA_PATTERNS: list[tuple[str, list[str]]] = [
    # Greetings & farewells — the most important stratum for first impressions
    ("greetings", [
        r"^\s*(HELLO|HI|HEY|HOWDY|HIYA|HEYA|GREETINGS|SALUTATIONS|AHOY|SUP|YO)\b",
        r"\bGOOD (MORNING|AFTERNOON|EVENING|DAY)\b",
        r"^\s*(BYE|GOODBYE|SEE YA|SEE YOU|LATER|FAREWELL|TAKE CARE|NIGHT|GOODNIGHT)\b",
        r"\bHOW (ARE YOU|ARE YA|IS IT GOING|HAVE YOU BEEN)\b",
        r"^\s*(WHAT'?S UP|WHAT IS UP|WASSUP|WAZZUP)\b",
        r"\bNICE TO MEET YOU\b",
    ]),

    # Meta / self-referential — critical for "who are you" questions
    ("meta", [
        r"\bWHO ARE YOU\b",
        r"\bWHAT ARE YOU\b",
        r"\bARE YOU (AN AI|A ROBOT|A BOT|HUMAN|REAL)\b",
        r"\bWHAT CAN YOU DO\b",
        r"\bDO YOU HAVE (FEELINGS|EMOTIONS|A NAME)\b",
        r"\bARE YOU (ALIVE|SENTIENT|CONSCIOUS)\b",
        r"\bTELL ME ABOUT YOURSELF\b",
    ]),

    # Jokes & wordplay — template-learnable, Wisp already does this well
    ("jokes", [
        r"\b(TELL ME A|KNOW ANY|HEARD A|GOT A) (JOKE|RIDDLE|PRANK|PUN)\b",
        r"^\s*KNOCK KNOCK\b",
        r"^\s*WHY DID (THE|A|AN)\b",
        r"^\s*WHAT DO YOU CALL\b",
        r"^\s*WHAT DID (THE|A|AN)\b",
        r"\bMake me laugh\b",
        r"^\s*(WANT TO HEAR A|HERE'S A) (JOKE|RIDDLE)\b",
    ]),

    # Emotional acknowledgment — no facts needed, pure empathy patterns
    ("emotional", [
        r"\bI'?M (REALLY |SO |VERY |FEELING )?(SAD|HAPPY|EXCITED|STRESSED|NERVOUS"
        r"|SCARED|PROUD|WORRIED|UPSET|ANGRY|THRILLED|DEVASTATED|OVERWHELMED"
        r"|FRUSTRATED|EXHAUSTED|DEPRESSED|ANXIOUS|LONELY|BORED|ANNOYED)\b",
        r"\bI FEEL (LIKE|SO|REALLY|VERY)\b",
        r"\b(CONGRATS|CONGRATULATIONS)\b",
        r"\bSORRY TO HEAR\b",
        r"\bI JUST (GOT|FOUND OUT|HEARD|LOST|WON|PASSED|FAILED)\b",
        r"\bTHAT'?S (AMAZING|AWFUL|TERRIBLE|WONDERFUL|GREAT NEWS|BAD NEWS)\b",
    ]),

    # Light opinions & preferences — short hedged answers, no recall needed
    ("opinions", [
        r"\bDO YOU (PREFER|LIKE|LOVE|ENJOY|HATE)\b",
        r"\bWHAT'?S YOUR (FAVORITE|FAVOURITE|PREFERRED)\b",
        r"\b(CATS|DOGS) OR (DOGS|CATS)\b",
        r"\b(COFFEE|TEA) OR (TEA|COFFEE)\b",
        r"\bWOULD YOU RATHER\b",
        r"\bWHICH DO YOU\b",
        r"[A-Z]+ OR [A-Z]+\??\s*$",         # short "X or Y?" patterns
        r"\bDO YOU THINK\b",
        r"\bWHAT DO YOU THINK ABOUT\b",
    ]),

    # Social reactions — agreeing, acknowledging, expressing surprise
    ("reactions", [
        r"\bI KNOW(,| )RIGHT\b",
        r"\bISN'?T IT\b",
        r"^\s*(TOTALLY|EXACTLY|ABSOLUTELY|DEFINITELY|OBVIOUSLY)\b",
        r"\bI (AGREE|DISAGREE|TOTALLY AGREE|COULDN'T AGREE MORE)\b",
        r"^\s*(RIGHT\?|FAIR ENOUGH|MAKES SENSE|GOOD POINT|TRUE THAT)\b",
        r"^\s*(WOW|OMG|OH WOW|NO WAY|REALLY\?|SERIOUSLY\?)\b",
        r"\bCAN YOU BELIEVE\b",
    ]),

    # Small talk pivots — keeping conversation flowing
    ("small_talk", [
        r"\b(LONG|ROUGH|HARD|TOUGH|GOOD|GREAT|NICE|BUSY) DAY\b",
        r"^\s*(NOT MUCH|SAME OLD|SAME AS USUAL|JUST CHILLING|JUST HANGING)\b",
        r"\bBEEN UP TO\b",
        r"\bWHAT'?S (NEW|GOING ON|HAPPENING)\b",
        r"\bANYTHING (NEW|INTERESTING|EXCITING|COOL|FUN)\b",
        r"\bHOW WAS (YOUR|THE)\b",
        r"\bHAD A (GOOD|GREAT|BAD|ROUGH|LONG) (DAY|WEEK|NIGHT)\b",
    ]),
]

# Compiled patterns for speed
_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (name, [re.compile(p, re.IGNORECASE) for p in pats])
    for name, pats in _STRATA_PATTERNS
]

# Patterns that indicate character-specific dialogue (should be rejected)
_REJECT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^\s*[A-Z][A-Z]+[,\s]\s*(WHAT|WHERE|WHY|STOP|PLEASE|COME|GO|LOOK|DON'T|I )\b",  # "DIANE, ..." or "DIANE WHAT..."
        r"\bWHAT DID YOU DO\b",          # addressing specific person
        r"\bWHERE WERE YOU\b",
        r"\bYOU PROMISED\b",
        r"\bTOLD YOU\b",
        r"\bYOU SAID\b",
        r"\bYOU ALWAYS\b",
        r"\bYOU NEVER\b",
    ]
]

# ── Name normalisation ───────────────────────────────────────────────────────
# SODA dialogues involve named fictional characters. Replace apparent proper
# names in vocative positions with a generic placeholder so the model learns
# "HELLO HUMAN!" rather than "HELLO CARLATON!".

# Words that appear in vocative position but should NOT be replaced
_NAME_KEEP: frozenset = frozenset([
    "WELL", "YEAH", "YEP", "YES", "NO", "NOPE", "OKAY", "OK",
    "SURE", "RIGHT", "TRUE", "GOOD", "GREAT", "NICE", "COOL",
    "THERE", "HERE", "EVERYONE", "GUYS", "ALL", "BOTH",
    "GOD", "LORD", "MAN", "WOW", "WAIT", "NOW", "THEN",
    "THANKS", "SORRY", "PLEASE", "THANK", "HELLO", "HI", "HEY", "OH",
    "WHAT", "HOW", "WHY", "WHO", "WHERE", "WHEN",
    "NOT", "JUST", "REALLY", "VERY", "SO", "STILL",
])

# Prefix vocative: HEY/OH/HI[,] NAME  →  HEY, HUMAN
_PREFIX_VOC = re.compile(
    r'\b(OH|HEY|HI|HELLO|THANKS|SORRY)([,\s]+)([A-Z]{2,})\b',
    re.IGNORECASE,
)
# Suffix vocative: ", NAME." or ", NAME!" — only before sentence-ending punctuation
_SUFFIX_VOC = re.compile(
    r',\s+([A-Z]{2,})([.!?])',
    re.IGNORECASE,
)


def normalise_names(text: str, placeholder: str = "HUMAN") -> str:
    """Replace apparent character names in vocative positions with placeholder."""
    def _swap(word: str) -> str:
        return word if word.upper() in _NAME_KEEP else placeholder

    def _prefix(m: re.Match) -> str:
        sep = ", " if "," in m.group(2) else " "
        return m.group(1) + sep + _swap(m.group(3))

    def _suffix(m: re.Match) -> str:
        return ", " + _swap(m.group(1)) + m.group(2)

    text = _PREFIX_VOC.sub(_prefix, text)
    text = _SUFFIX_VOC.sub(_suffix, text)
    return text


# Default targets per stratum.
# meta and jokes are excluded from SODA sourcing (0) — SODA's versions are
# character-specific dialogue, not AI self-description or joke setups.
# Supply these via the `supplement` parameter using distilled/generated pairs.
DEFAULT_TARGETS: dict[str, int] = {
    "greetings":  400,
    "meta":         0,   # source from distilled only — see supplement param
    "jokes":        0,   # source from distilled only — SODA has ~18 in 100K
    "emotional":  500,
    "opinions":   400,
    "reactions":  350,
    "small_talk": 350,
}


# ── Core functions (all tested) ───────────────────────────────────────────────

def classify(query: str, response: str) -> Optional[str]:
    """Return the stratum name for a Q|R pair, or None if unclassifiable.

    Returns None for character-specific dialogue (proper-name addressing,
    relationship-specific statements) that wouldn't make sense coming from
    or going to an AI assistant.
    """
    q = query.upper()

    # Reject character-specific dialogue first
    for pat in _REJECT_PATTERNS:
        if pat.search(q):
            return None

    # Match against strata in priority order
    for name, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(q):
                return name

    return None


def sample_strata(
    pairs: list[tuple[str, str]],
    targets: Optional[dict[str, int]] = None,
    supplement: Optional[dict[str, list[tuple[str, str]]]] = None,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """Classify SODA pairs into strata and sample up to `targets[stratum]` each.

    supplement: stratum → list of (q, r) pairs from external sources (e.g.
      distilled data). For strata with target=0, supplement pairs are used
      exclusively. For other strata, supplement pairs fill remaining capacity
      after SODA pairs are exhausted.

    Returns list of (query, response, stratum) tuples, shuffled.
    """
    if targets is None:
        targets = DEFAULT_TARGETS
    if supplement is None:
        supplement = {}

    buckets: dict[str, list[tuple[str, str]]] = {s: [] for s in targets}
    rng = random.Random(seed)

    for q, r in pairs:
        stratum = classify(q, r)
        if stratum and stratum in buckets:
            buckets[stratum].append((normalise_names(q), normalise_names(r)))

    result: list[tuple[str, str, str]] = []
    for stratum, target in targets.items():
        extra = list(supplement.get(stratum, []))
        rng.shuffle(extra)

        if target == 0:
            # This stratum is supplement-only — don't pull from SODA at all
            for q, r in extra:
                result.append((q, r, stratum))
        else:
            pool = buckets[stratum]
            rng.shuffle(pool)
            combined = pool[:target] + extra[:max(0, target - len(pool))]
            for q, r in combined[:target]:
                result.append((q, r, stratum))

    rng.shuffle(result)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample a stratified Wisp training set from SODA pairs"
    )
    parser.add_argument("--input",  "-i", default="data/soda_100k.txt")
    parser.add_argument("--output", "-o", default="data/wisp_soda.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", type=str, default=None,
                        help="JSON overrides e.g. '{\"jokes\": 800}'")
    args = parser.parse_args()

    targets = dict(DEFAULT_TARGETS)
    if args.targets:
        import json
        targets.update(json.loads(args.targets))

    print(f"Loading pairs from {args.input}…")
    pairs: list[tuple[str, str]] = []
    with open(args.input) as f:
        for line in f:
            if "|" in line:
                q, r = line.strip().split("|", 1)
                pairs.append((q.strip(), r.strip()))

    print(f"Loaded {len(pairs):,} pairs. Classifying…")
    result = sample_strata(pairs, targets=targets, seed=args.seed)

    # Count per stratum
    counts: dict[str, int] = {}
    for _, _, s in result:
        counts[s] = counts.get(s, 0) + 1

    print("\nStratum breakdown:")
    for stratum in targets:
        n = counts.get(stratum, 0)
        avail = sum(1 for q, r in pairs if classify(q, r) == stratum)
        print(f"  {stratum:12s}: {n:4d} sampled  ({avail:,} available)")

    total = len(result)
    print(f"\nTotal: {total:,} pairs")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        for q, r, _ in result:
            f.write(f"{q}|{r}\n")
    print(f"Written → {args.output}")


if __name__ == "__main__":
    main()
