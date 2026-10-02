# Local Phi-4-mini-instruct

This repo does not ship model weights. Use the copy from the course lab repo ([rzrizaldy/cmu_nlx_llm_lab](https://github.com/rzrizaldy/cmu_nlx_llm_lab)), checked out next to this one:

```
CodeFolder/
  cmu_application_of_nlx_llm/lab01/models/phi-4-mini-instruct/   <- weights
  cmu_application_nlx_llm_team3/                                  <- this repo
```

Scripts that load Phi look there by default:

- [corpora/04_parks_public_spaces_mingchin/local_model.py](../corpora/04_parks_public_spaces_mingchin/local_model.py) searches the sibling path first.
- [corpora/02_waste_neighborhood_rutomo/extraction.py](../corpora/02_waste_neighborhood_rutomo/extraction.py) uses it when `PHI_MODEL_PATH` is unset.

If the weights are elsewhere, copy [.env.example](../.env.example) to `.env` (gitignored) or export the path:

```bash
export PHI_MODEL_PATH=/path/to/phi-4-mini-instruct
```

Check that the model loads:

```bash
cd corpora/04_parks_public_spaces_mingchin && python local_model.py --check
```
