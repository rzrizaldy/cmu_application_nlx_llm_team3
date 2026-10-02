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

The merge also refreshes [appendix/original_dataset.jsonl](../../appendix/original_dataset.jsonl).

Coordinates and address fields are removed from the merged copy per the brief's data rule, and listed in `metadata.redacted_fields`. This currently affects 251 Buildings records (`x`, `y`). The member folders keep their data as submitted.

Human label fields differ by member because each subtopic defined its own extraction schema; see `taxonomy.json`. The waste subtopic's `taxonomy.json` was derived afterwards from its `extraction.py`, which remains that pipeline's source of truth.
