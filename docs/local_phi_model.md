# Local Phi-4-mini-instruct weights

This team repo does **not** ship model weights. Use the copy already set up in the course lab repo on the same machine:

| Location | Path |
|---|---|
| Sibling checkout (default search) | `../cmu_application_of_nlx_llm/lab01/models/phi-4-mini-instruct` |
| GitHub | [rzrizaldy/cmu_nlx_llm_lab](https://github.com/rzrizaldy/cmu_nlx_llm_lab) — clone and run lab01 setup |

## One-time setup

```bash
export PHI_MODEL_PATH="/Users/rzrizaldy/CodeFolder/cmu_application_of_nlx_llm/lab01/models/phi-4-mini-instruct"
```

Copy [.env.example](../.env.example) to `.env` and set `PHI_MODEL_PATH` there for local runs (`.env` is gitignored).

Member scripts that load Phi:

- [members/mingchin/local_model.py](../members/mingchin/local_model.py) — searches the sibling path first
- [members/rutomo/extraction.py](../members/rutomo/extraction.py) — defaults to the sibling path when `PHI_MODEL_PATH` is unset

Verify:

```bash
cd members/mingchin && python local_model.py --check
```
