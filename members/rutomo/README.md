# Assignment 1: Pittsburgh 311 Waste and Neighborhood Cleanliness

I built this corpus for the group topic **Pittsburgh 311 Municipal Service Resolution**. My individual focus is **Waste and Neighborhood Cleanliness**. The corpus supports structured extraction from official service guidance, historical issue classifications, and administrative summaries.

## Corpus

The final corpus has **180 records**:

- 106 natural-language sections from official City of Pittsburgh service pages
- 41 historical codebook rows
- 33 historical operational summaries
- 74 tabular records in total, or 41.1% of the corpus

I retained the four agreed categories: Garbage and Litter Issues, Neighborhood Issues, Graffiti Issues, and Weeds/Debris. The Neighborhood Issues category remains broad because the historical taxonomy includes property, planning, noise, tree, and related issues.

Each record includes its source, retrieval or inspection time, modality, licensing note, and metadata. [`sources.csv`](sources.csv) records the represented sources and their terms. City guidance is used for private educational analysis because public access does not establish an open license. WPRDC material retains its attribution.

The corpus does not include original resident complaint narratives. Contact and address-bearing blocks were removed from collected guidance, and synthetic complaints were not added. Historical closure time is treated as an administrative field. It does not verify physical resolution or measure the effect of a future API.

## Evaluation results

I finalized a random 25-record reference sample before opening model predictions. The sample uses seed 820 and the same extraction schema throughout the study. The three fields are `service_focus`, `information_type`, and `responsible_department`.

| Run | Exact agreement | Valid output |
|---|---:|---:|
| Phi baseline | 36% | 100% |
| Phi recovery | 40% | 100% |
| GPT-5.6 Luna bonus | 88% | 100% |
| TypeSafe JEV 1.13.0 extension | 72% | 100% |

The recovery prompt addressed Phi's tendency to select `dumping` without explicit evidence. Service-focus agreement improved from 72% to 88%. Two complete records were fixed and one previously correct record regressed. The prompt was tuned and evaluated on the same 25 records, so I treat the result as a local diagnostic.

Cost for the 25-record comparison was approximately $0.00282 for Luna and $0.00079 for JEV at prices published September 21, 2026. JEV input tokens are API-reported; Luna token usage is a reproducible tokenizer estimate because the original run did not retain usage fields. Mean observed latency was 1.15 seconds for Luna and 0.19 seconds for JEV.

The hosted Luna comparison used the same records, baseline prompt, and schema. TypeSafe JEV 1.13.0 evaluated the same records and target definitions through typed Choice and Noul judgments, with deterministic department candidates. JEV reached 72% exact agreement; field agreement was 88% for each field. The interfaces are different, so the result is an additional structured-decision experiment rather than an identical prompt benchmark. [`support/outputs/model_comparison_extended.json`](support/outputs/model_comparison_extended.json) records both comparisons and their limits.

## Reproduce the workflow

The main technical entry point is [`extraction.py`](extraction.py). Commands below assume the repository workspace is the current directory.

```sh
.venv/bin/python assignment1/extraction.py validate
.venv/bin/python assignment1/extraction.py baseline
.venv/bin/python assignment1/extraction.py recovery
.venv/bin/python assignment1/extraction.py clusters
.venv/bin/python assignment1/extraction.py evidence
.venv/bin/python assignment1/extraction.py package
```

Local Phi weights are not redistributed. Set `PHI_MODEL_PATH` when the weights are stored somewhere other than the prepared course location. The optional hosted run reads `OPENAI_API_KEY` and `OPENAI_MODEL` from an ignored `.env` file. The JEV extension reads `TYPESAFE_API_KEY` only from the process environment and uses the official TypeSafe SDK. Credentials never belong in the notebook, source files, outputs, or Git history.

The inference cache separates model, prompt, schema, and input changes. Interrupted runs resume from completed records. Raw responses, validation failures, prompts, run configurations, and metrics are retained under [`support/outputs/`](support/outputs/).

## Directory guide

- `corpus.jsonl`, `sources.csv`, `extraction.py`, `human_labels.jsonl`, `memo_rutomo.pdf`, and this README are the six required submission files. `memo_rutomo.pdf` is added after review.
- `support/docs/` contains the status, final checklist, labeling instructions, memo guide, and memo source.
- `support/labels/` contains the Excel and CSV views of the reference labels.
- `support/learning/` contains the standalone learning lab and its template.
- `support/paper/` contains the two-page scientific research note and self-contained LaTeX source.
- `support/scripts/` contains corpus construction, model, privacy, labeling, and learning-lab helpers.
- `support/data/` contains the small supplied historical tables used for reproducibility.
- `support/outputs/` contains the evidence created by each stage.
- `support/tests/` contains workflow and integrity tests.

The executed notebook and `paper_rutomo.pdf` with its LaTeX source are included as inspectable bonus work. Other support materials, the large raw archive, model weights, environment, caches, reference starter, and credentials are excluded from the submission ZIP.

## Attribution and process notes

The supplied starter attribution and license are retained in `support/attribution/`. Some utilities named in the assignment page were absent from the provided archives, so their required behavior is implemented locally and is not presented as supplied course code.

I used my own judgment to make the final labeling decisions before viewing the Phi predictions. I also reviewed the model disagreements, evaluated the recovery results, interpreted the clusters, and wrote the memo in my own words.

Codex assisted with the Python implementation and suggested a reproducible workflow. I reviewed the resulting code, evidence, and outputs as part of completing the assignment.
