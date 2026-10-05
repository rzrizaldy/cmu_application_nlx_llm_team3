# Assignment 2 experiments (Group 3)

| Folder | Member | Status | Notes |
|---|---|---|---|
| [01_streets_mobility_afaq](01_streets_mobility_afaq/) | Afaq | Complete | Code, `pgh311_complaints.json`, `results/` (8 runs plus `all_metrics.json`); memo in [memo/01_streets_mobility_afaq_as02.pdf](../memo/01_streets_mobility_afaq_as02.pdf) |
| [02_waste_neighborhood_rutomo](02_waste_neighborhood_rutomo/) | Rutomo | Complete | Full `a2/` runs, labels, Part D/E; memo in [memo/02_waste_neighborhood_rutomo_as02.pdf](../memo/02_waste_neighborhood_rutomo_as02.pdf) |
| [03_buildings_construction_mahika](03_buildings_construction_mahika/) | Mahika | Complete | Parts B–E outputs + report PDF |
| [04_parks_public_spaces_mingchin](04_parks_public_spaces_mingchin/) | Mingchin | Complete | Code, DEV/EVAL splits, probes, 8 metrics JSON, run responses; memo in [memo/04_parks_public_spaces_mingchin_as02.pdf](../memo/04_parks_public_spaces_mingchin_as02.pdf) |
| [05_team](05_team/) | Team | In progress | Unified intake API, splits, runs, finetune |

## Ask teammates for

**Afaq:** nothing outstanding. His combined ZIP (2026-10-05) supplied `results/` and the report; its Assignment 1 files matched `corpora/01_streets_mobility_afaq` exactly. `run_experiments.py` defines its tools inline, so the empty `conf/tool_calling/` folder was dropped.

**Mingchin:** nothing outstanding. Her data ZIP (2026-10-05) supplied the splits, probes, `metrics/`, and `outputs/*_responses.jsonl`; upstream LLMBox demo data, Hydra run logs, and non-311 lab chat logs from it were not kept.

Each folder is the member's Assignment 2 ZIP unpacked, without `.venv`, `__pycache__`, or `__MACOSX`. The ZIPs themselves are not kept.
