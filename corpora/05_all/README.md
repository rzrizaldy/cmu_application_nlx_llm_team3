# 05_all: merged team corpus

The four subtopic corpora (`01`–`04`) concatenated into one knowledge base, as the brief describes for the final project. Everything here except the two `.py` files is generated; rebuild instead of editing by hand:

```bash
python3 corpora/05_all/merge.py
```

| File | Contents |
|---|---|
| `corpus.jsonl` | 854 records; each has `metadata.member` and `metadata.subtopic_key` |
| `human_labels.jsonl` | 100 reference labels (25 per member, 24 for parks), tagged by member |
| `sources.csv` | All member sources with a `subtopic` column |
| `taxonomy.json` | The brief's category groups plus each member's extraction fields |
| `corpus_stats.json` | Record counts, modality, tabular share per subtopic |
| `category_groups.py` | Shared category split and subtopic order from the brief |
| `merge.py` | Validates 01–04 and writes the files above |
| `operational_evidence.py` | Joins `311_data.csv` to the codebook with the brief's cleaning contract and writes the two files below |
| `operational_evidence.jsonl` | 127 records, one per issue in the four subtopics: volume, closed count, median, 75th and 90th percentile resolution days, definition |
| `operational_evidence_summary.json` | Coverage: 815,417 requests, 612,222 in the four subtopics, 142,060 unmapped (blank category), 61,135 in other categories |

`operational_evidence.py` needs the gitignored `corpora/04_parks_public_spaces_mingchin/311_data.csv` ([WPRDC](https://data.wprdc.org/dataset/311-data)):

```bash
python3 corpora/05_all/operational_evidence.py
```

The team311 knowledge index loads `operational_evidence.jsonl` alongside `corpus.jsonl`, and `resolution_range()` fills the ticket's `historical_resolution_range` from it.

The merge also refreshes [appendix/original_dataset.jsonl](../../appendix/original_dataset.jsonl).

Coordinates and address fields are removed from the merged copy per the brief's data rule, and listed in `metadata.redacted_fields`. This currently affects 251 Buildings records (`x`, `y`). The member folders keep their data as submitted.

Human label fields differ by member because each subtopic defined its own extraction schema; see `taxonomy.json`. The waste subtopic's `taxonomy.json` was derived afterwards from its `extraction.py`, which remains that pipeline's source of truth.
