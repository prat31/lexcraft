from lexcraft.verify import (
    check,
    clean_output,
    contraction_count,
    copy_ratio,
    is_legal_register,
    legal_marker_density,
)


def test_numbers_must_survive():
    assert check(
        "Pay $1,500 within 30 days.", "Payment of $1500 shall be made within thirty (30) days."
    ).ok
    res = check("Pay within 30 days.", "Payment shall be made promptly.")
    assert not res.ok and res.missing_numbers == ["30"]


def test_added_number_from_word_is_not_flagged():
    res = check("Give two weeks' notice.", "Notice of two (2) weeks shall be given.")
    assert res.ok and res.added_numbers == []


def test_negation_lost():
    assert check("Do not smoke.", "Smoking is prohibited.").ok
    assert not check("Do not smoke.", "Smoking shall be permitted.").ok


def test_placeholder_mismatch():
    assert not check("See ⟦0⟧ and ⟦1⟧.", "Refer to ⟦0⟧.").ok


def test_entities_and_months():
    res = check("Email a@b.io by March 3.", "Notice shall be sent by March 3.")
    assert "a@b.io" in res.missing_entities
    # "may" as a modal verb is not a month
    assert check("You may leave.", "The Employee shall be permitted to leave.").ok


def test_region_terms_flagged_only_when_added():
    res = check("Follow the law.", "The Parties shall comply with the Companies Act.")
    assert res.added_region_terms
    assert not check("GDPR applies.", "The GDPR shall apply.").added_region_terms


def test_preamble_detected():
    assert not check("Pay now.", "Here is the rewritten text: Payment shall be due.").ok


def test_clean_output_strips_think_and_quotes():
    assert clean_output('<think>hmm</think>\n"The Party shall pay."') == "The Party shall pay."
    assert clean_output("```\nThe Party shall pay.\n```") == "The Party shall pay."


def test_register_signals():
    assert legal_marker_density("The Tenant shall pay rent pursuant to the Agreement.") > 10
    assert contraction_count("You can't do that, it's late.") == 2


def test_copy_of_source_is_never_legal_register():
    src = "If you don't respond within 14 days, I'll take the matter further."
    out = "If you do not respond within 14 days, I will take the matter further."
    assert copy_ratio(src, out) == 1.0
    assert not is_legal_register(src, out)
    # short outputs no longer skip the gate
    assert not is_legal_register("no smoking indoors", "No smoking indoors.")
    assert not is_legal_register(
        "The joint venture will last 5 years unless extended.",
        "The joint venture will last five years unless extended.",
    )


def test_real_rewrite_is_legal_register():
    assert is_legal_register(
        "Each apartment gets one vote.", "Each apartment shall be entitled to one vote."
    )
    assert is_legal_register(
        "If the buyer can't get a mortgage, the deposit is returned.",
        "If the Buyer is unable to obtain a mortgage, the deposit shall be returned.",
    )
    # light edit, but it adds legal wording ("shall")
    assert is_legal_register(
        "If the buyer can't get a mortgage, the deposit is returned.",
        "If the buyer cannot obtain a mortgage, the deposit shall be returned.",
    )
