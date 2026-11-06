# Contributing

Thanks for helping improve lexcraft.

## Setup

```bash
git clone https://github.com/prat31/lexcraft.git
cd lexcraft
uv sync --extra eval
```

## Before opening a PR

```bash
uv run ruff check src tests eval/assertions
uv run ruff format src tests eval/assertions
uv run mypy src
uv run pytest
```

## Adding training data

- Add pairs to a file in `data/seed/` (JSONL: `category`, `format`, `plain`, `legal`).
- Keep the legal side jurisdiction-neutral: no statutes, courts or country-specific terms.
- Every number, date, name, email and URL in `plain` must appear in `legal`.
- Run `uv run lexcraft data build` and fix any pair it reports as failing checks.

## Adding a model to the eval

Edit `eval/models.yaml`. Any OpenAI-compatible endpoint works. Keep the judge in a
different model family from every compared system; the runner enforces this.

## Reporting issues

Include `lexcraft doctor` output, the command you ran and the input text (redact anything private).
