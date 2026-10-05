# Assignment 2 experiments (Group 3)

| Folder | Member | Status | Notes |
|---|---|---|---|
| [01_streets_mobility_afaq](01_streets_mobility_afaq/) | Afaq | Partial | Code + `pgh311_complaints.json`; no `results/` or report |
| [02_waste_neighborhood_rutomo](02_waste_neighborhood_rutomo/) | Rutomo | Complete | Full `a2/` runs, labels, Part D/E; memo in [memo/02_waste_neighborhood_rutomo_as02.pdf](../memo/02_waste_neighborhood_rutomo_as02.pdf) |
| [03_buildings_construction_mahika](03_buildings_construction_mahika/) | Mahika | Complete | Parts B–E outputs + report PDF |
| [04_parks_public_spaces_mingchin](04_parks_public_spaces_mingchin/) | Mingchin | Partial | Code, DEV/EVAL splits, probes, 8 metrics JSON, and run responses; no report yet |
| [05_team](05_team/) | Team | In progress | Unified intake API, splits, runs, finetune |

## Ask teammates for

**Afaq:** `results/` from `run_experiments.py`, AS02 report PDF, populated `conf/tool_calling/`, AI-use appendix.

**Mingchin:** AS02 report. His data ZIP (2026-10-05) supplied the splits, probes, `metrics/`, and `outputs/*_responses.jsonl`; upstream LLMBox demo data, Hydra run logs, and non-311 lab chat logs from it were not kept.

Each folder is the member's Assignment 2 ZIP unpacked, without `.venv`, `__pycache__`, or `__MACOSX`. The ZIPs themselves are not kept.
