from lexcraft.structure import assemble, protect, restore, segment


def roundtrip(doc: str) -> str:
    segs = segment(doc)
    return assemble(segs, {})


def test_identity_roundtrip_keeps_layout():
    doc = "# Title\n\n- one item\n- two `code` item\n\n> quote line\n\n```\nraw code\n```\n\n| a | b |\n|---|---|\n"
    assert roundtrip(doc) == doc


def test_kinds_and_prefixes():
    segs = [s for s in segment("## Fees\n1. pay now\n   - nested item\n> note") if s.translatable]
    assert [s.kind for s in segs] == ["heading", "list_item", "list_item", "quote"]
    assert segs[0].prefix == "## "
    assert segs[1].prefix == "1. "
    assert segs[2].prefix == "   - "


def test_code_fence_is_untouched():
    segs = segment("```python\nprint('hi there friend')\n```")
    assert not any(s.translatable for s in segs)


def test_placeholders_protect_urls_code_emails():
    text, saved = protect(
        "Email a@b.co or see https://x.org/p and run `make`. [docs](https://d.io)"
    )
    assert "⟦0⟧" in text and len(saved) == 4
    assert "https://x.org/p" in saved and "`make`" in saved
    assert (
        restore(text, saved)
        == "Email a@b.co or see https://x.org/p and run `make`. [docs](https://d.io)"
    )


def test_soft_wrapped_paragraph_is_one_segment():
    segs = [s for s in segment("first line of a\nparagraph here.\n\nnext para") if s.translatable]
    assert len(segs) == 2
    assert segs[0].text == "first line of a paragraph here."


def test_hard_break_splits_paragraph():
    segs = [s for s in segment("line one  \nline two") if s.translatable]
    assert len(segs) == 2
    assert segs[0].suffix == "  "


def test_assemble_replaces_and_restores():
    segs = segment("- see https://a.com now")
    idx = next(i for i, s in enumerate(segs) if s.translatable)
    out = assemble(segs, {idx: "refer to ⟦0⟧ forthwith;"})
    assert out == "- refer to https://a.com forthwith;"


def test_symbol_only_lines_pass_through():
    segs = segment("---\n***\n- 42")
    assert not any(s.translatable for s in segs)


def test_emphasis_is_stripped_then_reapplied():
    segs = [
        s
        for s in segment("give **30 days' notice** and _sign_ it, not snake_case_name")
        if s.translatable
    ]
    seg = segs[0]
    assert seg.text == "give 30 days' notice and sign it, not snake_case_name"
    assert seg.emphasis == [("**", "30 days' notice"), ("_", "sign")]
    out = assemble(segs, {0: "shall give thirty (30) days' notice and sign the same"})
    assert out == "shall give **thirty (30) days' notice** and _sign_ the same"
    assert segs[0].lost_emphasis == []


def test_lost_emphasis_is_reported():
    segs = [s for s in segment("this is **very important**") if s.translatable]
    out = assemble(segs, {0: "this matter is of the utmost significance"})
    assert "**" not in out and segs[0].lost_emphasis == ["very important"]


def test_emphasis_roundtrip_unchanged():
    doc = "- pay **now** or *later*"
    assert roundtrip(doc) == doc
