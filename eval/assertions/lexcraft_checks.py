"""Promptfoo python assertion: deterministic lexcraft checks.

Wired in by `lexcraft eval run` (promptfoo `type: python`). Uses the same
checks as the runtime engine so eval and product agree on what "correct" means.
"""

from __future__ import annotations

from lexcraft.verify import (
    check,
    contraction_count,
    copy_ratio,
    is_legal_register,
    legal_marker_density,
)


def get_assert(output: str, context: dict) -> dict:
    source = context["vars"]["text"]
    res = check(source, output or "")
    density = legal_marker_density(output or "")
    copied = copy_ratio(source, output or "")
    legal = is_legal_register(source, output or "")
    contractions = contraction_count(output or "")

    components = [
        {
            "pass": not res.missing_numbers and not res.missing_entities,
            "score": 1.0 if not res.missing_numbers and not res.missing_entities else 0.0,
            "reason": "facts preserved"
            if not res.missing_numbers and not res.missing_entities
            else "; ".join(res.problems),
            "assertion": {"type": "python", "metric": "facts_preserved"},
        },
        {
            "pass": not res.placeholder_mismatch,
            "score": 0.0 if res.placeholder_mismatch else 1.0,
            "reason": "placeholders ok" if not res.placeholder_mismatch else "placeholders changed",
            "assertion": {"type": "python", "metric": "placeholders"},
        },
        {
            "pass": not res.negation_lost,
            "score": 0.0 if res.negation_lost else 1.0,
            "reason": "negations kept" if not res.negation_lost else "negation/condition lost",
            "assertion": {"type": "python", "metric": "negation_kept"},
        },
        {
            "pass": not res.added_region_terms,
            "score": 0.0 if res.added_region_terms else 1.0,
            "reason": "region-neutral"
            if not res.added_region_terms
            else f"added: {', '.join(res.added_region_terms)}",
            "assertion": {"type": "python", "metric": "region_neutral"},
        },
        {
            "pass": not res.has_preamble,
            "score": 0.0 if res.has_preamble else 1.0,
            "reason": "no preamble" if not res.has_preamble else "preamble/commentary",
            "assertion": {"type": "python", "metric": "no_preamble"},
        },
        {
            "pass": contractions == 0,
            "score": 1.0 if contractions == 0 else 0.0,
            "reason": f"{contractions} contractions",
            "assertion": {"type": "python", "metric": "no_contractions"},
        },
        {
            "pass": legal,
            "score": min(density / 10.0, 1.0) if legal else 0.0,
            "reason": f"legal marker density {density:.1f}/100 words, copy ratio {copied:.2f}"
            + ("" if legal else " (not legal register)"),
            "assertion": {"type": "python", "metric": "legal_register"},
        },
    ]
    passed = all(c["pass"] for c in components)
    score = sum(c["score"] for c in components) / len(components)
    return {
        "pass": passed,
        "score": score,
        "reason": "all deterministic checks passed"
        if passed
        else "; ".join(c["reason"] for c in components if not c["pass"]),
        "componentResults": components,
    }
