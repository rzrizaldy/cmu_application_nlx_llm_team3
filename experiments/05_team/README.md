# 05_team: unified intake experiments

| Path | What |
|---|---|
| `make_split.py` | Builds `data/dev.jsonl` (534) and `data/eval.jsonl` (50), stratified by member corpus, seed 952 |
| `leakage_check.py` | Fails if any EVAL item's id or text is in the knowledge base or the DEV neighbors |
| `run_team.py` | Runs one experiment (`T0_generate`, `T1_structured_rag`, `T2_tools`, `T3_guarded`, `T4_finetuned`) through `api/team311` `Router.route_complaint` and scores it |
| `finetune_lora.py` | LoRA-finetunes Phi-4-mini on DEV with the T0 prompt (r 8, alpha 16, attention projections; 80 rows per subtopic, 2 epochs by default). Targets take missing information and the clarification question from the gold issue's knowledge card. The adapter goes to `finetune/adapter/` (gitignored), and `finetune/train_metrics.json` records the run |
| `build_chatlogs.py` | Turns the T4 responses into LLMBox chat logs in `chatlogs/` |
| `run_all.sh` | Runs T0 to T3, finetune, T4, chatlogs, and the appendix build in order |

## Inputs and gold labels

Intake items are Afaq's complaints, Mahika's service requests, Mingchin's example texts, and Rutomo's codebook rows. We clean two kinds of input that state the answer: Mahika's "It was routed to ..." and "Current status: ..." sentences are removed, and a waste codebook row becomes "Resident reports: <issue>." instead of the row's JSON.

Gold labels follow the brief's label contract: `request_type_id`, issue, exact category, and department from the WPRDC codebook, resolved by request type id or issue name through `corpora/05_all/operational_evidence.jsonl`. The member's own label stays in `gold.member_label`. Items we cannot resolve keep the member label and are marked `gold_source: member` (14 of 50 EVAL items). `gold.domain` is the subtopic of the codebook category, which differs from the member's corpus for 37 of 584 items (for example, City Steps is a parks category in the codebook). `data/split_manifest.json` records these counts.

## Metrics

Each `metrics.json` reports issue, department, routed-correctly (category and department), domain, and category accuracy with 95% bootstrap intervals; schema validity against `Ticket311`; completeness; clarification and abstention rates; how often the raw output was valid JSON; how often T2/T3 fell back to the vote's issue; latency and tokens; and accuracy by gold domain and by input origin. T3 also scores the guardrail on the Part D probe sets from Rutomo, Mahika, and Mingchin.

## Results (50 EVAL inputs, Phi-4-mini, greedy, no repetition penalty)

| Run | Issue | Department | Routed correctly (95% CI) | Domain | Schema valid | Clarification | Mean latency |
|---|---|---|---|---|---|---|---|
| T0 prompt only | 0.56 | 0.00 | 0.00 (0.00–0.00) | 0.60 | 1.00 | 0.88 | 11.9 s |
| T1 + retrieved knowledge | 0.56 | 0.70 | 0.38 (0.26–0.52) | 0.68 | 0.98 | 0.94 | 15.2 s |
| T2 + codebook tools and vote | 0.60 | 0.82 | 0.82 (0.70–0.92) | 0.88 | 1.00 | 1.00 | 15.0 s |
| T3 T2 + guardrail | 0.60 | 0.82 | 0.82 (0.70–0.92) | 0.88 | 1.00 | 1.00 | 18.5 s |
| T4 LoRA, T0 prompt | 0.60 | 0.58 | 0.50 (0.36–0.64) | 0.82 | 1.00 | 0.86 | 9.4 s |

T0 never gets the department right because the prompt lists categories but no department names, and nothing grounds the model's guess ("Public Works") to a codebook department. T2 fell back to the retrieval vote's issue for 28% of items. None of the EVAL inputs is adversarial, so T3 matches T2 on EVAL; its guardrail blocks 100%, 86%, and 42% of the adversarial probes from Mahika, Rutomo, and Mingchin, and passes 100%, 94%, and 100% of their benign probes.

T4 is the T0 prompt on a LoRA adapter, with no retrieval or tools at inference. We trained it on 80 DEV rows per subtopic (269 rows, since waste has only 29) for 2 epochs (135 optimizer steps, 30 minutes on the M4). Train loss went from 1.44 to about 0.10, averaging 0.34. Each target carries the gold issue's knowledge-card clarification question and its first two required details; 220 of the 269 rows have a card.

Against T0, the same prompt without the adapter, finetuning takes routed-correctly from 0% to 50% and department from 0% to 58%, and T4 now asks a clarification question 86% of the time. It is also our fastest design (9.4 s) because its prompt has no retrieved context. It still trails T2 by 32 points, and the gap is concentrated where training data is thin or the text is free-form:

| Gold domain | n | T4 routed correctly | T2 routed correctly |
|---|---|---|---|
| Parks | 14 | 0.86 | 0.86 |
| Buildings | 11 | 0.73 | 0.82 |
| Waste | 12 | 0.25 | 1.00 |
| Streets | 13 | 0.15 | 0.62 |

Waste had only 29 training rows, and Afaq's streets complaints are the only free-text resident wording in the set. T4's misses are mostly a plausible but wrong department, such as "DOMI - Permits" for an Allegheny City Electric streetlight. T2 avoids these because it copies the department from a retrieved codebook card.

Our first finetune used 149 rows for 1 epoch, and its targets never had a clarification question. It reached domain 0.72 but 0% routed correctly, never named a real department, and never asked a question. Its metrics are not kept in `runs/`; this README records them.

On our 24 GB M4, training needs about 11 GB of GPU memory, so close other large apps first. Hugging Face `Trainer` stalled before its first step on MPS several times on this machine, while the same forward and backward pass ran in 3 seconds outside it, so `finetune_lora.py` uses a plain PyTorch loop. The settings are AdamW, learning rate 2e-4, 3 warmup steps then linear decay, batch size 1, gradient accumulation 4, and gradient clipping at 1.0.

## Outputs

- `runs/<run>/` holds `responses.jsonl` and `metrics.json` for the 50 EVAL inputs. These feed `appendix/evaluation_metrics/`.
- `runs/dev/<run>/` holds the same for DEV smoke runs (`--split dev`), used for prompt tuning only.
