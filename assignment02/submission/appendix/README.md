# Canvas code appendix (Group 3)

Required uploads from the course appendix brief. Formats: JSON or JSONL for datasets; ZIP for LLM API Python code.

| Item | Path | Status |
|---|---|---|
| Original Dataset | [original_dataset.jsonl](original_dataset.jsonl) | **Ready** — merged Assignment 1 team corpus (854 records) |
| Development Dataset | [development_dataset/](development_dataset/) | Pending Assignment 2 experiments |
| Evaluation Dataset | [evaluation_dataset/](evaluation_dataset/) | Pending Assignment 2 experiments |
| Chatlogs LLM Evaluation Dataset | [chatlogs_llm_evaluation/](chatlogs_llm_evaluation/) | Pending finetune / chatlog evaluation |
| Evaluation Metrics Dataset | [evaluation_metrics/](evaluation_metrics/) | Pending — one file per experiment when jobs run |
| LLM API Code files | [llm_api_code/](llm_api_code/) | Pending — ZIP of modified/custom LLM API Python |

Regenerate the original dataset after changing member corpora:

```bash
python3 assignment01/team/merge_corpus.py
cp assignment01/team/corpus.jsonl assignment02/submission/appendix/original_dataset.jsonl
```

See [code_appendix_team](../../../code_appendix_team) at the repo root for the instructor wording.
