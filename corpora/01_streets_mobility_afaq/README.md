# Assignment 1 Corpus — Streets and Mobility (Pittsburgh 311)

**Author:** Afaq Khan · **Group topic:** Pittsburgh 311 municipal service requests
**Individual subtopic:** Streets and Mobility

## What this corpus is and why it exists

Pittsburgh's 311 system takes ~104,000 requests a year. The City's own 311 Data
User Guide states that **5–10% of web-filed requests are given the wrong
classification by the resident who files them**. Misclassified requests get
misrouted, which delays service. The group project is an LLM intake assistant
that reads a resident's complaint and predicts the correct category, department,
and named locations, so requests land in the right place the first time.

This corpus is the Streets-and-Mobility slice used to test whether a small local
model (Phi-4-mini) can do that routing, and how much a better prompt helps.

## The data constraint (read this first)

The public WPRDC 311 datasets contain **no resident free-text** — only the
already-assigned request type, department, location, status, and timestamps.
There is no column with the resident's own words. This corpus therefore pairs
**real filed requests** (the ground truth) with **authored complaint text
derived from those real requests** (the natural-language input the model reads).
Every complaint is traceable to a real request via `derived_from_request_id`.

## Contents (190 records)


| document_type    | n   | modality | real/authored | description                                                                                                                                                                                                                                |
| ---------------- | --- | -------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `request_row`    | 70  | mixed    | **real**      | a real filed 311 request copied from the 2015–2017 archive (request_id, date, type, origin, status, department, neighborhood), with a one-line factual summary rendered from those fields. The tabular backbone and the answer-key source. |
| `complaint`      | 115 | text     | authored      | resident complaints derived from real requests, in two registers: **web-form** (69, informal resident text) and **operator-note** (46, neutral third-person). Ground-truth labels come from the parent request.                            |
| `knowledge_card` | 5   | text     | authored      | routing rules per category, grounded in the real department mapping.                                                                                                                                                                       |


Tabular share **36.8%** (>=20% required). Near-duplicate rate **0%**.
Complaints span all five categories; the tricky/confusable request types are
over-sampled on purpose so model errors are visible and analyzable.

## Extraction task (`taxonomy.json`)

From each complaint's text, the model predicts:

- **category** — one of the five Streets categories
- **department** — the exact city unit (the hard field; the recovery experiment targets it)
- **entities** — neighborhoods/streets named in the text



## Sources (`sources.csv`)

- **WPRDC 311 Data Archive (2015–2017 request log)** — `data.wprdc.org`, Creative
Commons Attribution (CC-BY). Real filed requests + the complaints derived from them.
- **City of Pittsburgh 311** — `pittsburghpa.gov`. Authority for the routing rules
in the knowledge cards.



## Key decisions

1. **Real requests as ground truth.** Every label is what the City actually did
  with a real request, not an opinion. Every synthetic complaint names the
   `request_id` it was built from.
2. **Two complaint registers** (web-form vs operator-note) so the analysis can
  compare model performance on messy resident text vs. clean intake text.
3. **Pre-2025 taxonomy on purpose.** The five categories exist in the historical
  (2015–2025) system; Pittsburgh replaced its 311 system in Feb 2025 and
   restructured categories. Using the historical taxonomy keeps a stable frame.
4. **Confusables over-sampled** (streetlight repair vs request; sign repair vs
  new sign; illegal parking vs abandoned vehicle; car in bike lane).



## Known limitations (addressed in the memo)

1. Complaints are authored (no resident free-text exists publicly), so results
  are a proof-of-concept, not a measurement on true resident language.
2. Source concentration is ~97% one domain (WPRDC), by design.
3. Documents are short (median ~92 chars); civic complaints are terse.
4. The department field reflects the DPW->DOMI renaming of this era, so the same
  work appears under two department names — a real routing ambiguity.
5. Real filed requests were pooled from the archive and stratified for category
  coverage, not sampled to be population-representative.



## Reproducing

`build_corpus.py` regenerates everything from `2015_2017_311_records.xlsx` (place it in the same folder). I downloaded the dataset directly from 311 website. Since this was not part of the assignment, I didn't upload the dataset. The generated `corpus.jsonl` is included. Validate with:
`python check_corpus.py corpus.jsonl --sources sources.csv`