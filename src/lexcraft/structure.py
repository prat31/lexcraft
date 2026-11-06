"""Structure-preserving segmentation.

The model never sees markdown syntax. We split a document into lines, keep the
structural prefix of each line (heading marks, list bullets, quote marks,
indentation) in code, and send only the prose to the model. Inline code, URLs,
markdown links and emails are replaced by placeholders ⟦n⟧ so they survive
verbatim. Bold/italic markers are stripped before rewriting (so the model still
sees the words) and re-applied afterwards where the phrase reappears. Fenced code
blocks, tables, HTML and blank lines are passed through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["heading", "list_item", "paragraph", "quote"]

PH_OPEN, PH_CLOSE = "⟦", "⟧"
PH_RE = re.compile(r"⟦(\d+)⟧")

# Order matters: links before bare URLs.
_PROTECT = re.compile(
    r"(`[^`\n]+`)"  # inline code
    r"|(!?\[[^\]\n]*\]\([^)\s]+\))"  # markdown link / image
    r"|(<https?://[^>\s]+>)"  # autolink
    r"|(https?://[^\s<>()]+[^\s<>().,;:!?'\"])"  # bare URL
    r"|([\w.+-]+@[\w-]+\.[\w.-]+\w)"  # email
)

_HEADING = re.compile(r"^(\s{0,3}#{1,6}\s+)(.*?)(\s+#+\s*)?$")
_LIST = re.compile(r"^(\s*(?:[-*+]|\d{1,9}[.)])\s+(?:\[[ xX]\]\s+)?)(.*)$")
_QUOTE = re.compile(r"^(\s*(?:>\s?)+)(.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE = re.compile(r"^\s*\|.*\|\s*$")
_HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_HTML = re.compile(r"^\s*<[^>]+>")
_TRAILING_BREAK = re.compile(r"( {2,}|\\)$")
_WORD = re.compile(r"[A-Za-z]")
_EMPH = re.compile(
    r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1"  # bold
    r"|(?<![\w*])(\*)(?=[^*\s])([^*\n]+?)(?<=\S)\*(?![\w*])"  # *italic*
    r"|(?<![\w_])(_)(?=[^_\s])([^_\n]+?)(?<=\S)_(?![\w_])"  # _italic_
)
_NUM_IN = re.compile(r"\d+(?:[.,]\d+)*")
_NUM_WORDS = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundred|thousand|million|and|first|second|third|fourth|fifth|"
    r"[a-z]+(?:th|st|nd|rd))"
)


@dataclass
class Segment:
    """One unit sent to the model (or passed through when `text` is None)."""

    prefix: str
    text: str | None
    suffix: str = ""
    kind: Kind = "paragraph"
    placeholders: list[str] = field(default_factory=list)
    raw: str = ""
    emphasis: list[tuple[str, str]] = field(default_factory=list)  # (marker, phrase)
    lost_emphasis: list[str] = field(default_factory=list)

    @property
    def translatable(self) -> bool:
        return self.text is not None


def protect(text: str) -> tuple[str, list[str]]:
    """Replace inline code / links / URLs / emails with ⟦n⟧ placeholders."""
    saved: list[str] = []

    def sub(m: re.Match[str]) -> str:
        saved.append(m.group(0))
        return f"{PH_OPEN}{len(saved) - 1}{PH_CLOSE}"

    return _PROTECT.sub(sub, text), saved


def strip_emphasis(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Remove **bold** / *italic* markers, returning the plain text and what was emphasized."""
    found: list[tuple[str, str]] = []

    def sub(m: re.Match[str]) -> str:
        marker = m.group(1) or m.group(3) or m.group(5)
        inner = m.group(2) or m.group(4) or m.group(6)
        found.append((marker, inner))
        return inner

    return _EMPH.sub(sub, text), found


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Match a phrase loosely: case-insensitive, flexible spacing, and numbers that the
    rewrite may have expanded ("30" also matches "thirty (30)")."""
    out, pos = [], 0
    for m in _NUM_IN.finditer(phrase):
        out.append(re.escape(phrase[pos : m.start()]))
        out.append(rf"(?:{_NUM_WORDS}(?:[\s-]+{_NUM_WORDS})*\s\()?{re.escape(m.group(0))}\)?")
        pos = m.end()
    out.append(re.escape(phrase[pos:]))
    pattern = "".join(out).replace(r"\ ", r"\s+")
    return re.compile(pattern, re.I)


def reapply_emphasis(text: str, emphasis: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Wrap each emphasized phrase again if it appears in the rewrite. Returns lost phrases."""
    lost = []
    for marker, phrase in emphasis:
        m = _phrase_pattern(phrase).search(text)
        if m and marker not in text[max(0, m.start() - len(marker)) : m.start()]:
            text = f"{text[: m.start()]}{marker}{m.group(0)}{marker}{text[m.end() :]}"
        else:
            lost.append(phrase)
    return text, lost


def restore(text: str, saved: list[str]) -> str:
    def sub(m: re.Match[str]) -> str:
        i = int(m.group(1))
        return saved[i] if i < len(saved) else m.group(0)

    return PH_RE.sub(sub, text)


def _segment_line(line: str) -> Segment:
    if not line.strip() or _HR.match(line) or _TABLE.match(line) or _HTML.match(line):
        return Segment(prefix=line, text=None, raw=line)

    suffix = ""
    m_br = _TRAILING_BREAK.search(line)
    if m_br:
        suffix, line = m_br.group(0), line[: m_br.start()]

    kind: Kind = "paragraph"
    prefix, body = "", line
    if m := _HEADING.match(line):
        kind, prefix, body = "heading", m.group(1), m.group(2)
        suffix = (m.group(3) or "") + suffix
    elif m := _QUOTE.match(line):
        kind, prefix, body = "quote", m.group(1), m.group(2)
        if m2 := _LIST.match(body):
            kind, prefix, body = "list_item", prefix + m2.group(1), m2.group(2)
    elif m := _LIST.match(line):
        kind, prefix, body = "list_item", m.group(1), m.group(2)
    else:
        stripped = line.lstrip()
        prefix, body = line[: len(line) - len(stripped)], stripped

    raw = prefix + body + suffix
    protected, saved = protect(body)
    # Nothing worth rewriting (only placeholders / symbols): pass through.
    if not _WORD.search(PH_RE.sub("", protected)):
        return Segment(prefix=raw, text=None, raw=raw)
    plain, emphasis = strip_emphasis(protected)
    return Segment(
        prefix=prefix,
        text=plain,
        suffix=suffix,
        kind=kind,
        placeholders=saved,
        raw=raw,
        emphasis=emphasis,
    )


def _is_plain_paragraph_line(line: str) -> bool:
    if not line.strip():
        return False
    return not any(p.match(line) for p in (_HR, _TABLE, _HTML, _HEADING, _QUOTE, _LIST, _FENCE))


def segment(document: str) -> list[Segment]:
    """Split a document into segments. Joining with '\\n' reproduces the layout.

    Soft-wrapped paragraph lines are merged into one segment so sentences are
    never cut in half; the rewritten paragraph is emitted on a single line
    (identical when rendered as markdown).
    """
    segments: list[Segment] = []
    in_fence = False
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            indent = buffer[0][: len(buffer[0]) - len(buffer[0].lstrip())]
            joined = indent + " ".join(x.strip() for x in buffer[:-1])
            last = buffer[-1].rstrip("\n")
            m_br = _TRAILING_BREAK.search(last)
            tail = last[: m_br.start()] if m_br else last
            joined = (joined + " " if len(buffer) > 1 else indent) + tail.strip()
            segments.append(_segment_line(joined + (m_br.group(0) if m_br else "")))
            if len(buffer) > 1:
                segments[-1].raw = "\n".join(buffer)
            buffer.clear()

    for line in document.split("\n"):
        if _FENCE.match(line):
            flush()
            in_fence = not in_fence
            segments.append(Segment(prefix=line, text=None, raw=line))
            continue
        if in_fence:
            segments.append(Segment(prefix=line, text=None, raw=line))
            continue
        if _is_plain_paragraph_line(line):
            # A hard line break ends the paragraph segment.
            if buffer and _TRAILING_BREAK.search(buffer[-1]):
                flush()
            buffer.append(line)
            continue
        flush()
        segments.append(_segment_line(line))
    flush()
    return segments


def assemble(segments: list[Segment], outputs: dict[int, str]) -> str:
    """Rebuild the document. `outputs` maps segment index -> rewritten protected text."""
    lines = []
    for i, seg in enumerate(segments):
        if not seg.translatable:
            lines.append(seg.prefix)
            continue
        body = outputs.get(i, seg.text or "")
        body = " ".join(body.split())  # one line per segment keeps structure intact
        body, seg.lost_emphasis = reapply_emphasis(body, seg.emphasis)
        lines.append(seg.prefix + restore(body, seg.placeholders) + seg.suffix)
    return "\n".join(lines)
