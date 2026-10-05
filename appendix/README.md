# Canvas code appendix

Requirements: [brief/code_appendix_team.txt](../brief/code_appendix_team.txt).

| Item | Path | Status |
|---|---|---|
| Original Dataset | [original_dataset.jsonl](original_dataset.jsonl) | From `corpora/05_all` via merge |
| Development Dataset | [development_dataset/](development_dataset/) | Member + `05_team_dev.jsonl` when built |
| Evaluation Dataset | [evaluation_dataset/](evaluation_dataset/) | Member + `05_team_eval.jsonl` when built |
| Chatlogs LLM Evaluation | [chatlogs_llm_evaluation/](chatlogs_llm_evaluation/) | After `05_team` T4 + `build_chatlogs.py` |
| Evaluation Metrics | [evaluation_metrics/](evaluation_metrics/) | Per-run `metrics.json` copies |
| LLM API Code | [llm_api_code/llm_api_code.zip](llm_api_code/llm_api_code.zip) | From `api/` + `experiments/05_team` |

Rebuild everything:

```bash
python3 appendix/build_appendix.py
```

All slots are filled. If a teammate artifact goes missing, the build writes a `PENDING.md` in its folder instead.
