# Buildings, Construction & Accessibility — Pittsburgh 311 Corpus

**Course:** NL(X) and LLM · **Group 3** · Assignment 1
**Author:** Mahika Gunjkar

## Group topic and subtopic

- **Group topic:** Pittsburgh 311 service requests. The four group members each
  take a slice of the same WPRDC 311 dataset so the corpora recombine into one.
- **My subtopic:** Buildings, Construction & Accessibility, i.e. the 311
  categories **Building Maintenance**, **Construction Issues**, **Permits**, and
  **Accessibility**.

## What the corpus contains

Each record is one 311 service request, stored as `modality: "mixed"`:

- `raw_text` — a short natural-language description composed from the request's
  fields (issue, neighborhood, origin, date, department, status), so the corpus
  reads as language rather than a raw table.
- `table_json` — the administrative fields for the same request (identifiers,
  timestamp, geographic zones, coordinates). This is what makes every record
  count toward the tabular requirement.

The four common fields the group agreed on are extracted per record via
`taxonomy.json`: `issue_category` (a four-value enum), `request_type`,
`responsible_department`, `neighborhood`, and `entities`.

## Source and licensing


|---|---|
| Source | Pittsburgh 311 Data — archive (2015–Feb 2025), WPRDC |
| Data resource | https://data.wprdc.org/dataset/311-data/resource/29462525-62a6-45bf-9b5e-ad2e1c06348d |
| Category mapping | Project-authored `category_map.csv` (request type → one of the four buckets) |
| License | Creative Commons Attribution 4.0 (CC-BY), used under the WPRDC Data User Agreement |

The archive table is used (not the post-February-2025 live feed). The published
archive has no category column and the request-type taxonomy has changed over
time, so the four subtopic buckets are assigned with a small project-authored
mapping (`category_map.csv`) rather than an official codebook. Full provenance for
every record is in `sources.csv`.

## Records collected

- **Total records:** 260 (balanced across the four categories)
- **By category:** Building Maintenance 65 · Construction Issues 65 · Permits 65 · Accessibility 65
- **Modality:** all `mixed` (text + table)
- **Tabular share:** 100% (requirement is ≥20%)

## Collection and preprocessing decisions

1. **Category is assigned by a project mapping.** The archive stores a specific
   `request_type_name` but no category. Rather than an official codebook (the
   published taxonomy has deprecated entries), a curated `category_map.csv` maps
   each relevant request type to one of the four buckets. Deprecated
   "DO NOT USE" types and types belonging to teammates' subtopics (parking, trees,
   graffiti, snow removal) were deliberately excluded. The mapping is editable and
   is included with the submission for transparency.
2. **Balanced sampling.** Categories are very uneven in the raw data (Building
   Maintenance and Construction dominate, Accessibility is rare), so up to
   `--per-category` records are sampled per category with a fixed seed, keeping
   the sample reproducible and the classes comparable for evaluation.
3. **Text composed from fields.** The public 311 data has no free-text complaint
   description, so `raw_text` is a templated natural-language rendering of the
   structured fields.
4. **The label is hidden from the model.** The derived category is stored only
   in `metadata.codebook_category` as a hand-labelling reference; it is never
   placed in `raw_text` or `table_json`, so extracting `issue_category` remains a
   real classification task.
5. **Sensitive information.** 311 records are public and administrative; personal
   detail is already withheld upstream, so no masking pass was required. (If a
   record surfaced anything sensitive, `mask_sensitive.py` would be run first.)

## Known limitations

- **Single source and one document type.** Every record is a 311 request from one
  dataset, so the corpus cannot answer questions that need cross-source context
  (building codes, permit outcomes, inspection results).
- **Templated text.** Because `raw_text` is generated from fields, phrasing is
  regular; the corpus does not represent how residents actually describe problems.
- **Balanced, not representative.** Sampling equal numbers per category distorts
  the true frequency of each issue type in Pittsburgh.
- **Category noise.** A single request type can be borderline between buckets
  (a permit complaint about construction), so both the mapping and the human
  labels carry some ambiguity.
- **Thin, blurred "Construction" bucket.** In this data almost all construction
  complaints are filed as permit issues, so "Construction Issues" is small and
  overlaps heavily with "Permits". Balancing to 65 per category oversamples it.
- **No resolution/outcome signal.** Status is a coarse open/closed flag with no
  resolution time, so questions about how requests are resolved are out of scope.
