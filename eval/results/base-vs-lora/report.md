# lexcraft eval report

- Run: `20261106-024252`  
- Test items: 20  
- Judge: `deepseek-v4-pro` (deepseek)  
- Primary system: `lexcraft-lora`

## Pairwise (judge, both orders)

| lexcraft-lora vs | n | win | tie | loss | win+tie | 95% CI | at par or better |
|---|---|---|---|---|---|---|---|
| qwen3-1.7b-base | 20 | 8 | 2 | 10 | 50% | 30%–70% | yes |

## Metrics (mean, 0–1)

| system | pass | entail_bwd | entail_fwd | facts_preserved | judge_rubric | legal_register | negation_kept | no_contractions | no_preamble | placeholders | region_neutral | similarity | median latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b-base | 15% | 0.99 | 0.97 | 0.90 | 0.49 | 0.14 | 0.95 | 0.85 | 1.00 | 1.00 | 1.00 | 0.97 | 522 ms |
| lexcraft-lora | 60% | 0.92 | 0.84 | 1.00 | 0.81 | 0.49 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.79 | 874 ms |
