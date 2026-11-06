"""Deterministic fact / structure checks.

Shared by the rewrite engine (runtime guard), dataset validation and eval.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from lexcraft.structure import PH_RE

_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+\w")
_URL = re.compile(r"https?://[^\s<>()]+[^\s<>().,;:!?'\"]")
_MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_MONTH_RE = re.compile(
    r"\b(January|February|March|April|June|July|August|September|October|November|December"
    r"|May(?=\s+\d)|(?<=\d\s)May)\b"
)
# Negation / restriction in the source...
_NEG_RE = re.compile(
    r"\b(not|no|never|none|nor|neither|without|unless|except|excluding|only|cannot|"
    r"prohibited|forbidden)\b|n't\b",
    re.I,
)
# ...must be matched by some negation, restriction or exception in the rewrite.
_NEG_OUT_RE = re.compile(
    r"\b(not|no|never|none|nor|neither|without|unless|except|excluding|only|cannot|"
    r"prohibited|forbidden|solely|exclusively|sole|restricted|nothing|refrain|irrespective|"
    r"regardless|notwithstanding|save|other than|absent|fail(?:s|ed|ure|ing)?|limited|"
    r"free (?:of|from)|exclusive|deadlock|default|prohibitions?|disabled?|waives?|waiver|vacant|prevents?|absence|lacks?|non-\w+|un\w+|in\w+able|in\w+ive|disallowed|precluded|barred|bars?|preclude[sd]?)\b",
    re.I,
)
_CONTRACTION = re.compile(r"\b\w+'(?:t|re|ll|ve|d|m)\b|\b(?:it|that|there|what)'s\b", re.I)

_UNITS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_WORD_NUMBERS: dict[str, int] = {w: i for i, w in enumerate(_UNITS)}
_WORD_NUMBERS.update({w: (i + 2) * 10 for i, w in enumerate(_TENS)})
_WORD_NUMBERS.update(
    {
        "hundred": 100,
        "thousand": 1000,
        "million": 1_000_000,
        "dozen": 12,
        "half": 50,
        "week": 7,
        "weeks": 7,
        "fortnight": 14,
        "month": 30,
        "year": 365,
        "twice": 2,
        "once": 1,
        "single": 1,
        "double": 2,
        "first": 1,
        "second": 2,
        "third": 3,
        "quarter": 25,
    }
)

LEGAL_MARKERS = [
    "shall",
    "hereby",
    "herein",
    "hereof",
    "hereto",
    "hereunder",
    "thereof",
    "therein",
    "thereto",
    "pursuant to",
    "notwithstanding",
    "provided that",
    "in accordance with",
    "party",
    "parties",
    "obligation",
    "obligations",
    "agreement",
    "aforementioned",
    "hereinafter",
    "whereas",
    "forthwith",
    "in the event that",
    "deemed",
    "prior to",
    "subject to",
    "set forth",
    "including but not limited to",
    "undersigned",
    "henceforth",
    "shall not",
    "commencing",
    "effective",
    "terminate",
    "liable",
    "liability",
    "entitled",
]
_MARKER_RE = re.compile(r"\b(" + "|".join(re.escape(m) for m in LEGAL_MARKERS) + r")\b", re.I)

# Terms that tie a text to one jurisdiction. Flagged only when added by the rewrite.
REGION_TERMS = [
    r"U\.?S\.?C\.?",
    r"United States Code",
    r"C\.?F\.?R\.?",
    r"Code of Federal Regulations",
    r"Uniform Commercial Code",
    r"\bUCC\b",
    r"Delaware",
    r"California",
    r"New York law",
    r"Companies Act",
    r"Indian Contract Act",
    r"Indian Penal Code",
    r"\bIPC\b",
    r"Bürgerliches Gesetzbuch",
    r"\bBGB\b",
    r"Code civil",
    r"Civil Code of",
    r"England and Wales",
    r"Scots law",
    r"HMRC",
    r"\bIRS\b",
    r"Internal Revenue",
    r"Supreme Court",
    r"High Court",
    r"Federal Court",
    r"Court of Appeal",
    r"\bv\.\s+[A-Z]",
    r"Section \d+[A-Z]? of the",
    r"Article \d+ of the",
    r"Fair Labor Standards Act",
    r"Consumer Rights Act",
    r"Sale of Goods Act",
    r"\bGDPR\b",
    r"\bCCPA\b",
    r"\bHIPAA\b",
    r"Restatement",
    r"Magna Carta",
    r"common law",
    r"civil law jurisdiction",
]
_REGION_RE = re.compile("|".join(REGION_TERMS), re.I)


def _norm_num(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".") or "0"
    return s


def numbers(text: str) -> Counter[str]:
    # Remove placeholders so ⟦0⟧ is not counted as a number.
    return Counter(_norm_num(n) for n in _NUM.findall(PH_RE.sub(" ", text)))


def word_numbers(text: str) -> set[str]:
    found: set[str] = set()
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _WORD_NUMBERS:
            found.add(str(_WORD_NUMBERS[w]))
    return found


@dataclass
class CheckResult:
    ok: bool
    missing_numbers: list[str] = field(default_factory=list)
    added_numbers: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    placeholder_mismatch: bool = False
    negation_lost: bool = False
    added_region_terms: list[str] = field(default_factory=list)
    has_preamble: bool = False

    @property
    def problems(self) -> list[str]:
        out = []
        if self.missing_numbers:
            out.append(f"missing numbers: {', '.join(self.missing_numbers)}")
        if self.missing_entities:
            out.append(f"missing: {', '.join(self.missing_entities)}")
        if self.placeholder_mismatch:
            out.append("placeholders changed")
        if self.negation_lost:
            out.append("negation/condition lost")
        if self.added_region_terms:
            out.append(f"region-specific terms: {', '.join(self.added_region_terms)}")
        if self.has_preamble:
            out.append("contains preamble/commentary")
        return out


_PREAMBLE = re.compile(
    r"^\s*(here is|here's|sure|certainly|below is|the rewritten|rewritten text|"
    r"legal version|as requested)\b",
    re.I,
)


def check(source: str, output: str) -> CheckResult:
    """Compare a rewrite with its source. `ok` is False on hard failures only."""
    src_nums, out_nums = numbers(source), numbers(output)
    missing = sorted((src_nums - out_nums).keys())
    src_words = word_numbers(source)
    added = sorted(k for k in (out_nums - src_nums) if k not in src_words)

    missing_entities = []
    for pattern in (_EMAIL, _URL):
        for ent in set(pattern.findall(source)):
            if ent not in output:
                missing_entities.append(ent)
    out_months = {m.lower() for m in re.findall(r"[A-Za-z]+", output)}
    for month in {m.lower() for m in _MONTH_RE.findall(source)}:
        if month not in out_months:
            missing_entities.append(month.title())

    placeholder_mismatch = Counter(PH_RE.findall(source)) != Counter(PH_RE.findall(output))
    negation_lost = bool(_NEG_RE.search(source)) and not _NEG_OUT_RE.search(output)
    src_region = {m.group(0).lower() for m in _REGION_RE.finditer(source)}
    added_region = sorted(
        {m.group(0) for m in _REGION_RE.finditer(output) if m.group(0).lower() not in src_region}
    )
    has_preamble = bool(_PREAMBLE.match(output)) and not _PREAMBLE.match(source)

    ok = not (missing or missing_entities or placeholder_mismatch or negation_lost or has_preamble)
    return CheckResult(
        ok=ok,
        missing_numbers=missing,
        added_numbers=added,
        missing_entities=missing_entities,
        placeholder_mismatch=placeholder_mismatch,
        negation_lost=negation_lost,
        added_region_terms=added_region,
        has_preamble=has_preamble,
    )


def legal_marker_density(text: str) -> float:
    """Legal markers per 100 words: a cheap register signal."""
    words = max(len(text.split()), 1)
    return 100.0 * len(_MARKER_RE.findall(text)) / words


_EXPAND = [
    (re.compile(r"\bcan't\b", re.I), "cannot"),
    (re.compile(r"\bwon't\b", re.I), "will not"),
    (re.compile(r"n't\b", re.I), " not"),
    (re.compile(r"'ll\b", re.I), " will"),
    (re.compile(r"'re\b", re.I), " are"),
    (re.compile(r"'ve\b", re.I), " have"),
    (re.compile(r"'m\b", re.I), " am"),
]


def _norm_words(text: str) -> list[str]:
    t = text.replace("\u2019", "'")
    for pat, rep in _EXPAND:
        t = pat.sub(rep, t)
    return re.findall(r"[a-z0-9]+", t.lower())


def copy_ratio(source: str, output: str) -> float:
    """Word-level similarity (0-1) of output to source, ignoring case, punctuation and
    contractions. High values mean the output was copied or only lightly edited."""
    from difflib import SequenceMatcher

    a, b = _norm_words(source), _norm_words(output)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


COPY_THRESHOLD = 0.85


def is_legal_register(source: str, output: str) -> bool:
    """Hard register gate. A (near-)copy of the source fails unless the edit adds legal
    wording the source lacked; longer outputs also need a minimum density of legal markers."""
    added = {m.lower() for m in _MARKER_RE.findall(output)} - {
        m.lower() for m in _MARKER_RE.findall(source)
    }
    if copy_ratio(source, output) >= COPY_THRESHOLD and not added:
        return False
    if len(output.split()) < 8:
        return True
    return legal_marker_density(output) >= 3.0


def contraction_count(text: str) -> int:
    return len(_CONTRACTION.findall(text))


def clean_output(text: str) -> str:
    """Strip model artefacts: think blocks, code fences, wrapping quotes."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.replace("<think>", "").replace("</think>", "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").split("\n", 1)[-1].strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text
