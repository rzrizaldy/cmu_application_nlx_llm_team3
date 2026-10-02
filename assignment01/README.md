# Assignment 1 — Group 3 team corpus

**Group topic:** Pittsburgh 311 municipal service resolution. Each member built an independent subtopic corpus; this repo concatenates them for the final project knowledge base.

| Member folder | Subtopic | Records |
|---|---|---:|
| [members/afaq](members/afaq) | Streets and Mobility | 190 |
| [members/rutomo](members/rutomo) | Waste and Neighborhood Cleanliness | 180 |
| [members/mahika](members/mahika) | Buildings, Construction, and Accessibility | 260 |
| [members/mingchin](members/mingchin) | Parks, Trees, Animals, and Public Facilities | 224 |

The team brief and split diagram live in [brief/](brief/).

## Combined corpus

From the repository root:

```bash
python3 assignment01/team/merge_corpus.py
```

Outputs:

- [team/corpus.jsonl](team/corpus.jsonl) — 854 records with `metadata.member` and `metadata.subtopic_key`
- [team/sources.csv](team/sources.csv) — member sources with a `member` column

Member extraction pipelines stay in each `members/*` folder and use that author’s `taxonomy.json` / prompts. Only the merge step is shared.

## Raw packages

Original submission ZIPs are under [original_zips/](original_zips/). Mingchin’s full WPRDC `311_data.csv` is gitignored (237MB); use the committed `corpus.jsonl` or download the archive from WPRDC locally.
