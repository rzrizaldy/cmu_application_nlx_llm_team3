# Assignment 1 — Pittsburgh 311 Municipal Service Requests

One folder, no packages to install beyond the Python libraries below. Put your
own `corpus.jsonl`, `sources.csv` and `human_labels.jsonl` in here next to the
scripts and run everything from this directory.

| File | What it does |
|---|---|
| `schema.py` | The two contracts: what a `corpus.jsonl` record must look like, and the extraction schema built from `taxonomy.json`. Also validates model output against that schema. |
| `check_corpus.py` | Validates `corpus.jsonl` and `sources.csv` against the brief. Run it before you submit. |
| `corpus_stats.py` | Corpus statistics and k-means clustering, with a plot. |
| `local_model.py` | Loads Phi-4-mini-instruct from local weights and runs it in-process — no server, no network. Greedy decoding, JSON validated against your schema. |
| `llm_utils.py` | Hosted LLM client (Groq, Gemini, Cerebras, OpenRouter, OpenAI) for the optional SLM-vs-LLM bonus. |
| `example_extraction.py` | Example of turning raw files (text, CSV) into corpus records, including one document → several records. |
| `taxonomy.json` | Your group's topic, subtopic and extraction fields. The schema, prompts and label template are all built from this. |
| `mask_sensitive.py` | Masks `<PII>` / `<PHI>` / `<FIN>` / `<CONF>` spans with Phi before data enters the corpus (only needed if your sources contain sensitive information). |
| `make_human_labels.py` | Draws the 20–30 evaluation records and writes a `human_labels.jsonl` template for you to fill in by hand. |
| `extraction.py` | The graded run: load corpus → run Phi → human vs. Phi → recovery prompt → statistics and clusters → optional hosted LLM. |
| `LICENSE`, `ATTRIBUTION.md` | MIT licence, and who wrote what. |

## Setup

Python 3.10–3.12. The Phi-4-mini-instruct weights are assumed to be on disk
already (the `models/phi-4-mini-instruct` folder from lab 1).

**macOS**

```bash
cd path/to/starter_code
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate safetensors scikit-learn matplotlib numpy requests
```

**Windows (PowerShell)**

```powershell
cd path\to\starter_code
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate safetensors scikit-learn matplotlib numpy requests
```

If `Activate.ps1` is blocked, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that window.
For an NVIDIA GPU, use the CUDA wheel from pytorch.org instead of the CPU one.
`requests` is only needed for the optional hosted-LLM bonus.

**Point at the weights.** `local_model.py` searches `models/phi-4-mini-instruct`,
`../models/phi-4-mini-instruct` and the lab-1 folders. Anywhere else, say so once:

```bash
export PHI_MODEL_PATH=/path/to/phi-4-mini-instruct     # macOS
$env:PHI_MODEL_PATH = "C:\path\to\phi-4-mini-instruct" # Windows
```

Check the model loads before anything else — the device should be `mps` on Apple
Silicon, `cuda` on an NVIDIA GPU, `cpu` otherwise:

```bash
python local_model.py --check
```

## Running it

```bash
# 0. fill in taxonomy.json: group topic, subtopic, fields to extract

# 1. build corpus.jsonl from your raw files (your own code, or adapt this example)
python example_extraction.py --text page.txt --csv page_table.csv \
    --source-url https://example.org/page --license-note "Public webpage; see site terms." \
    --id-prefix news --out corpus.jsonl

# 1b. only if your sources contain sensitive information
python mask_sensitive.py --input raw/notes.txt --output raw/notes.masked.txt --report mask_audit.json

# 2. corpus valid? (150–300 records, ≥20% tabular, every domain in sources.csv)
python check_corpus.py corpus.jsonl

# 3. draw the evaluation sample, then fill in human_label by hand
python make_human_labels.py --n 25

# 4. the graded run
python extraction.py

# 4b. optional bonus lane (needs GROQ_API_KEY in a .env file next to llm_utils.py)
python extraction.py --llm groq
```

Step 4 writes to `out/`: `phi_v1.jsonl`, `evaluation.json`, `recovery.json`,
`corpus_stats.json`, `clusters.json`, `clusters.png`. The memo is written from
those numbers.

## Define the task: `taxonomy.json`

Nothing about the extraction task is hard-coded. `extraction.py` reads
`taxonomy.json` and builds the JSON schema, both prompts, the human-label
template and the evaluation from it, so each group defines its own fields:

```json
{
  "group_topic": "news",
  "subtopic": "sports news",
  "fields": {
    "topic":       {"kind": "label", "values": ["match_report", "transfer", "injury"]},
    "league":      {"kind": "label", "values": [], "description": "competition named in the article"},
    "teams":       {"kind": "list",  "description": "team names, copied exactly"},
    "season_year": {"kind": "number", "required": false}
  },
  "prompts": {}
}
```

- **`kind`** is `label` (one string), `list` (array of strings) or `number`.
- **`values`** constrains a label to an enum. Leave it `[]` and the field is
  open — the model answers freely and is scored on exact agreement with your
  label. Enum fields are the ones where Phi's structured-output violations show
  up, so having at least one is worth it.
- **`description`** goes into the prompt and into the labelling instructions.
- **`required: false`** lets Phi omit a field that is often not stated.
- **`prompts`** overrides the generated ones. `v1_original` is the baseline and
  `v2_recovery` the strict rewrite; after reading your v1 errors, put a
  `v2_recovery` here that targets the failure Phi actually repeats on your
  corpus, and keep every version you try — the recovery section is graded on the
  before/after comparison, not on the improvement.

Rename the fields, add or drop them; the run adapts. Different subtopics in one
group can use different fields and still combine, since `corpus.jsonl` keeps the
common top-level fields either way. Keep a copy of `taxonomy.json` with your
submission — it is the record of what you asked the model for.

## Run time

On CPU, Phi takes roughly 20–60 seconds per record, so a 25-record run plus the
recovery rerun is a coffee break. On `mps`/`cuda` it is a few seconds each.

## Project Topic

- Group topic: Pittsburgh 311 Municipal Service Requests
- Individual subtopic: Parks, Trees, Animals, and Public Facilities

## Project Description

This project constructs and evaluates a corpus for classifying Pittsburgh 311 municipal service requests related to parks, trees, animals, and public facilities. The corpus combines resident-style complaint examples with structured operational statistics and uses official Pittsburgh 311 issue, category, request type, and department labels.

## Corpus Summary

This corpus focuses on Pittsburgh 311 requests related to Parks, Trees, Animals, and Public Facilities.

- Total records: 224
- Text records: 168
- Table records: 56
- Tabular share: 25%
- Source domain: WPRDC
- Categories:
  - Parks Issues
  - Tree Issues
  - Animal Issues
  - City Facilities and Infrastructure

The corpus combines short resident-style complaint examples with structured operational statistics derived from Pittsburgh 311 data. Each issue includes six text examples and two tabular records.

The corpus was constructed from the WPRDC 311 Data dataset and the WPRDC 311 Issue and Category Codebook.

## Collection and Preprocessing

The official Pittsburgh 311 request data was joined with the issue and category codebook using request_type_id. Records were filtered to four categories: Parks Issues, Tree Issues, Animal Issues, and City Facilities and Infrastructure.

The codebook was cleaned by removing rows with missing request_type_id or category values and keeping the latest codebook version for each request type. Historical request data was used to calculate request volume and resolution-time statistics.

To avoid exposing sensitive location information, exact street addresses and coordinates were not included in the final corpus.

## Known Limitations
The corpus uses a single WPRDC source domain and covers only four Pittsburgh 311 categories. The resident-style complaint texts are synthetic examples rather than original free-text submissions, so they may not capture the full variation and ambiguity of real 311 reports. Many records are also intentionally short, and tabular records may contain no raw text.