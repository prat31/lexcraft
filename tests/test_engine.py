from lexcraft.backends.base import Backend, ModelDescription
from lexcraft.config import GenerationConfig
from lexcraft.engine import Rewriter


class FakeBackend(Backend):
    def __init__(self, mapping: dict[str, list[str]]):
        self.mapping = mapping
        self.calls: list[str] = []

    def describe(self) -> ModelDescription:
        return ModelDescription("fake", "fake")

    def generate(self, messages, max_tokens, temperature):  # type: ignore[no-untyped-def]
        text = messages[-1]["content"]
        self.calls.append(text)
        outs = self.mapping[text]
        return outs.pop(0) if len(outs) > 1 else outs[0]


def test_rewrite_keeps_structure_and_restores_urls():
    be = FakeBackend(
        {
            "Fees": ["Fees and Charges"],
            "pay in 30 days": ["payment within thirty (30) days;"],
            "see ⟦0⟧": ["as set out at ⟦0⟧."],
        }
    )
    doc = "# Fees\n\n- pay in 30 days\n- see https://x.io/terms\n"
    res = Rewriter(be).rewrite(doc)
    assert (
        res.text
        == "# Fees and Charges\n\n- payment within thirty (30) days;\n- as set out at https://x.io/terms.\n"
    )
    assert not res.warnings


def test_retry_then_fallback_on_lost_facts():
    be = FakeBackend({"Pay $50 now.": ["Payment shall be due.", "Payment shall be due forthwith."]})
    res = Rewriter(be).rewrite("Pay $50 now.")
    assert res.text == "Pay $50 now."  # fell back to the original
    assert len(be.calls) == 2  # one retry
    assert res.blocks[0].fell_back


def test_retry_succeeds():
    be = FakeBackend(
        {"Pay $50 now.": ["Payment shall be due.", "The sum of $50 shall be paid forthwith."]}
    )
    res = Rewriter(be).rewrite("Pay $50 now.")
    assert res.text == "The sum of $50 shall be paid forthwith."


def test_no_verify_returns_raw():
    be = FakeBackend({"Pay $50 now.": ["Payment shall be due."]})
    res = Rewriter(be, GenerationConfig(verify=False)).rewrite("Pay $50 now.")
    assert res.text == "Payment shall be due."
