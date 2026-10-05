# 05_team: unified intake experiments

| Path | What |
|---|---|
| `make_split.py` | Builds `data/dev.jsonl` (534) and `data/eval.jsonl` (50), stratified by subtopic, seed 952 |
| `leakage_check.py` | Fails if any EVAL `doc_id` is in the team311 knowledge index |
| `run_team.py` | Runs one experiment (`T0_generate`, `T1_structured_rag`, `T2_tools`, `T3_guarded`, `T4_finetuned`) on Phi-4-mini and scores it |
| `finetune_lora.py` | LoRA-finetunes Phi-4-mini on DEV; adapter goes to `finetune/adapter/` (gitignored) |
| `build_chatlogs.py` | Turns the T4 responses into LLMBox chat logs in `chatlogs/` |
| `run_all.sh` | Runs T0 to T3, finetune, T4, chatlogs, and the appendix build in order |

Outputs:

- `runs/<run>/` holds `responses.jsonl` and `metrics.json` for the 50 EVAL inputs. These feed `appendix/evaluation_metrics/`.
- `runs/dev/<run>/` holds the same for DEV smoke runs (`--split dev`), used for prompt tuning only.
