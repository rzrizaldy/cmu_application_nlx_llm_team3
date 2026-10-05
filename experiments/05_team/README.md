# 05_team: unified intake experiments

| Path | What |
|---|---|
| `make_split.py` | Builds `data/dev.jsonl` (534) and `data/eval.jsonl` (50), stratified by member corpus, seed 952 |
| `leakage_check.py` | Fails if any EVAL item's id or text is in the knowledge base or the DEV neighbors |
| `run_team.py` | Runs one experiment (`T0_generate`, `T1_structured_rag`, `T2_tools`, `T3_guarded`, `T4_finetuned`) through `api/team311` `Router.route_complaint` and scores it |
| `finetune_lora.py` | LoRA-finetunes Phi-4-mini on DEV with the T0 prompt; adapter goes to `finetune/adapter/` (gitignored) |
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
| T4 LoRA, T0 prompt | 0.56 | 0.00 | 0.00 (0.00–0.00) | 0.72 | 0.96 | 0.00 | 9.3 s |

T0 never gets the department right because the prompt lists categories but no department names, and nothing grounds the model's guess ("Public Works") to a codebook department. T2 fell back to the retrieval vote's issue for 28% of items. None of the EVAL inputs is adversarial, so T3 matches T2 on EVAL; its guardrail blocks 100%, 86%, and 42% of the adversarial probes from Mahika, Rutomo, and Mingchin, and passes 100%, 94%, and 100% of their benign probes.

T4 learned the issue and category vocabulary (domain 0.72 against T0's 0.60), but one epoch on 149 DEV rows did not teach it the department strings, and its training targets always had empty `missing_information`, so it never asks a clarification question.

## Outputs

- `runs/<run>/` holds `responses.jsonl` and `metrics.json` for the 50 EVAL inputs. These feed `appendix/evaluation_metrics/`.
- `runs/dev/<run>/` holds the same for DEV smoke runs (`--split dev`), used for prompt tuning only.
