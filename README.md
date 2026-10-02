# CMU NL(X) and LLM — Group 3 (Pittsburgh 311)

One continuous project: **corpus (A1) → LLM API & evaluation (A2+)**. Canvas code uploads follow [code_appendix_team](code_appendix_team) under [appendix/](appendix/).

## Layout

| Path | Role |
|---|---|
| [brief/](brief/) | Team corpus brief and split diagram |
| [members/](members/) | Four subtopic corpora + each author’s scripts |
| [team/](team/) | Merge to 854-record team corpus |
| [sources/](sources/) | Renamed original submission ZIPs |
| [memo/](memo/) | Renamed Assignment 1 memo PDFs |
| [appendix/](appendix/) | Canvas datasets & API ZIP slots (see checklist) |
| [docs/local_phi_model.md](docs/local_phi_model.md) | Phi-4-mini weights in [cmu_application_of_nlx_llm](https://github.com/rzrizaldy/cmu_nlx_llm_lab) |

## Merge team corpus

```bash
python3 team/merge_corpus.py
```

Writes [team/corpus.jsonl](team/corpus.jsonl), [team/sources.csv](team/sources.csv), and refreshes [appendix/original_dataset.jsonl](appendix/original_dataset.jsonl).

## Phi model (local)

Weights live in the other course repo, not here:

```bash
export PHI_MODEL_PATH="/Users/rzrizaldy/CodeFolder/cmu_application_of_nlx_llm/lab01/models/phi-4-mini-instruct"
```

Details: [docs/local_phi_model.md](docs/local_phi_model.md).

## Subtopics

1. Streets and Mobility — `members/afaq`
2. Waste and Neighborhood Cleanliness — `members/rutomo`
3. Buildings, Construction, and Accessibility — `members/mahika`
4. Parks, Trees, Animals, and Public Facilities — `members/mingchin`

Appendix items still pending (dev/eval/chatlogs/metrics/API ZIP) are marked under [appendix/](appendix/).
