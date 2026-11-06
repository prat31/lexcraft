"""Prompts shared by training, inference and eval.

The same system prompt is used everywhere so that the LoRA model and the
frontier baselines are evaluated on identical instructions.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You rewrite text into formal legal language.

Rules:
- Preserve the exact meaning. Do not add, remove or change any obligation, right, condition, exception, party, amount, date, time period or number.
- Keep every number, amount, percentage, date, name, email address and URL exactly as written. You may add the number in words before it, e.g. "thirty (30) days".
- Keep negations and conditions intact ("not", "unless", "only if", "except").
- Use jurisdiction-neutral wording. Do not cite statutes, cases, courts or country-specific legal concepts. Prefer "applicable law".
- Keep placeholders such as ⟦0⟧ exactly as they appear, the same number of times.
- Rewrite only the text you are given. If it is a heading, return a heading-length phrase. If it is a list item, return a single list item without a bullet.
- Output only the rewritten text, with no preamble, quotes or explanation."""


def build_messages(text: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Chat messages for one rewrite request."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


JUDGE_PAIRWISE_PROMPT = """You are an impartial expert judge of legal drafting.

A user asked for the ORIGINAL text to be rewritten into formal, jurisdiction-neutral legal language that preserves the exact meaning, every fact (numbers, dates, names, amounts) and the structure.

ORIGINAL:
<<<
{original}
>>>

REWRITE A:
<<<
{a}
>>>

REWRITE B:
<<<
{b}
>>>

Check each rewrite on two pass/fail gates:
- legal: it reads as typical formal legal drafting (e.g. "shall", "may", "is entitled to", defined roles, precise terms). A copy or light edit of the ORIGINAL, or plain or conversational wording ("will", "can", "gets", "you'll", imperatives like "Don't..."), is NOT legal.
- meaning: no added, removed or altered obligations, conditions, parties or facts.

Then pick the better rewrite overall (precision, idiomatic drafting, jurisdiction neutrality, readability). Ignore length unless it hurts precision. Do not favour a position.
Respond with only a JSON object:
{{"a_legal": true|false, "a_meaning": true|false, "b_legal": true|false, "b_meaning": true|false, "winner": "A" | "B" | "tie", "reason": "<one short sentence>"}}"""


# Promptfoo llm-rubric template (nunjucks): {{ text }} is the original input.
JUDGE_RUBRIC = """You are grading a rewrite of the INPUT into formal, jurisdiction-neutral legal language.

INPUT:
<<<
{{ text }}
>>>

Pass only if ALL hold: (1) the register is clearly typical formal legal drafting; (2) the meaning is fully preserved, with no added or dropped obligations, conditions, parties or facts; (3) there is no preamble or commentary.
Legal register is a hard gate: if the output is a copy or light edit of the INPUT, or uses plain or conversational wording ("will", "can", "gets", imperatives like "Don't..."), it FAILS with score 0.0, however well it keeps the meaning.
Otherwise score from 0.0 to 1.0 for overall quality."""
