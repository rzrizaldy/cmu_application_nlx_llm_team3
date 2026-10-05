# CMU NL(X) and LLM — Group 3: Pittsburgh 311 Municipal Service Resolution

We are Afaq, Rutomo, Mahika, and Mingchin. Our [team brief](brief/team_corpus_brief.pdf) asks one business question: can an LLM-assisted 311 intake and routing API reduce the time it takes to resolve Pittsburgh service requests? We study the mechanism the brief calls first-time-right intake: classify the request correctly, collect complete details, ask one targeted clarification question, and route it to the right department.

## How we built it

Each of us built one Assignment 1 corpus for one subtopic of the shared WPRDC 311 archive, using the brief's exact category split:

| Lead | Subtopic | Member |
|---|---|---|
| 1 | Streets and Mobility | Afaq |
| 2 | Waste and Neighborhood Cleanliness | Rutomo |
| 3 | Buildings, Construction, and Accessibility | Mahika |
| 4 | Parks, Trees, Animals, and Public Facilities | Mingchin |

For the final project we concatenate the four corpora into one knowledge base (`corpora/05_all`, 854 records). We add one operational evidence file computed for all four subtopics at once from the WPRDC join: volume and median, 75th, and 90th percentile resolution time for 127 issues. The team API retrieves from that knowledge base and turns a resident complaint into a routed 311 ticket.

```mermaid
flowchart LR
  corpora["corpora/01-04"] --> merge["corpora/05_all/merge.py"]
  wprdc["311_data.csv + codebook"] --> ops["operational_evidence.py"]
  merge --> kb["05_all knowledge base"]
  ops --> kb
  kb --> api["api/team311 + llmbox"]
  api --> ticket["Ticket311 JSON"]
  experiments["experiments/05_team"] --> appendix["appendix/ Canvas uploads"]
  api --> experiments
```

The ticket has the fields the brief suggests: domain, category, issue, department, missing information, one clarification question, confidence, an abstention flag, and the historical resolution range.

We run the API on Phi-4-mini locally and compare five designs on the same 50 held-out evaluation inputs: prompt only (T0), plus retrieved knowledge (T1), plus codebook tools (T2), plus a guardrail (T3), and a LoRA-finetuned model (T4).

On the 50 evaluation inputs, our best design is T2. It routes 82% of complaints to the right category and department (95% CI 70–92%), against 0% for the prompt-only baseline, and every ticket it returns is schema-valid. Full results are in [experiments/05_team/README.md](experiments/05_team/README.md#results-50-eval-inputs-phi-4-mini-greedy-no-repetition-penalty).

## Layout

| Path | Contents |
|---|---|
| [corpora/](corpora/) | Our four Assignment 1 corpora and the `05_all` merged knowledge base |
| [experiments/](experiments/) | Each member's Assignment 2 work and our `05_team` runs |
| [api/](api/) | Rutomo's LLMBox fork (M0–M5) and our `team311` routing layer |
| [appendix/](appendix/) | Canvas datasets, metrics, and API ZIP, built by `appendix/build_appendix.py` |
| [memo/](memo/) | Our Assignment 1 and Assignment 2 memos, and [member_summaries.md](memo/member_summaries.md), which summarizes all eight |
| [brief/](brief/) | The team brief and the appendix requirements |

## Status

| Member | Assignment 2 package |
|---|---|
| Afaq | Complete |
| Rutomo | Complete |
| Mahika | Complete |
| Mingchin | Complete |

All four Assignment 2 packages are in.

| Final project piece | State |
|---|---|
| Merged knowledge base, operational evidence, knowledge cards | Done (854 corpus records, 127 issues with resolution times) |
| Team API (`api/team311`, `route_complaint`) | Done |
| DEV/EVAL split with codebook gold labels and leakage check | Done (534 / 50, no overlap) |
| T0–T3 on 50 EVAL inputs | Done; T2 and T3 route 82% correctly |
| T4 LoRA finetune | First run (149 rows, 1 epoch) is committed. It is under-trained: right issue 56% of the time, right department never, and no clarification questions. The retrain is set up but not run yet |
| Member memo summaries | Done ([memo/member_summaries.md](memo/member_summaries.md)) |
| Canvas appendix (datasets, metrics, chatlogs, API ZIP) | Done for the first T4 run; rebuild after the retrain |
| Final memo | Not started |

What we still need to do:

1. Run the T4 retrain. `finetune_lora.py` now defaults to 80 DEV rows per subtopic (269 rows, since waste has only 29) and 2 epochs. Training answers now include the gold issue's knowledge-card clarification question and the first two required details. It needs about 12 GB of GPU memory, so close other large apps first; with them open it stalls at step 0. After training, rerun T4, `build_chatlogs.py`, and `appendix/build_appendix.py` (about 40 minutes in total).
2. Write the final memo. It should report T0–T4 with their confidence intervals, the guardrail probe results, and the historical resolution baselines, within the evaluation boundary below.
3. Look at where T2 fails. On waste items it routes 12 of 12 correctly, but those inputs are codebook rows ("Resident reports: <issue>."), and the row names the issue. On Afaq's free-text resident complaints it routes 8 of 13 correctly (62%), and streets issue accuracy is only 31%. Real complaints are the case that matters, so the memo should report this split. In 28% of items T2 also fell back to the retrieval vote's issue.

## Commands

```bash
# Merged corpus (854 records)
python3 corpora/05_all/merge.py

# Operational evidence for all four subtopics (needs the gitignored 311_data.csv)
python3 corpora/05_all/operational_evidence.py

# Team intake split (534 DEV / 50 EVAL) and leakage check
python3 experiments/05_team/make_split.py
python3 experiments/05_team/leakage_check.py

# Full team experiment pipeline (Phi-4-mini; about 1 hour on an M4)
./experiments/05_team/run_all.sh

# Refresh the Canvas appendix
python3 appendix/build_appendix.py
```

Phi weights: [docs/local_phi_model.md](docs/local_phi_model.md) (`../cmu_application_of_nlx_llm/lab01/models/phi-4-mini-instruct`).

Our AI-use disclosure for integrated code: [docs/ai_use/ai_use_index.md](docs/ai_use/ai_use_index.md).

## Evaluation boundary

Following the brief, we report routing quality (issue and department accuracy, completeness, clarification rate, and abstention quality) and historical time-to-close baselines. Historical data cannot show that our API makes requests close faster; that would need a staff-confirmed pilot.
