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
- [api/team311/model.py](../api/team311/model.py) (`PhiRunner`) runs every team experiment, T0 to T4. It uses MPS and bf16, greedy decoding, no repetition penalty, and at most 320 new tokens. T4 loads the LoRA adapter from `experiments/05_team/finetune/adapter` on top of the same weights.

Inference fits comfortably on a 24 GB machine. LoRA training needs about 11 GB of GPU memory on top of everything else that is open, so close browsers and other large apps before running `experiments/05_team/finetune_lora.py`. It takes about 30 minutes for 269 rows and 2 epochs on an M4. The script loads the weights straight onto MPS and trains with a plain PyTorch loop, because Hugging Face `Trainer` stalled before its first step on MPS several times on our machine.

A repetition penalty breaks this task. It also penalizes prompt tokens, and the ticket has to copy category and issue names from the prompt verbatim. In our first full run with a penalty of 1.15, the finetuned model returned an empty category and issue for every item.

If the weights are elsewhere, copy [.env.example](../.env.example) to `.env` (gitignored) or export the path:

```bash
export PHI_MODEL_PATH=/path/to/phi-4-mini-instruct
```

Check that the model loads:

```bash
cd corpora/04_parks_public_spaces_mingchin && python local_model.py --check
```
