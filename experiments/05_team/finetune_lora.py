#!/usr/bin/env python3
"""LoRA finetune Phi-4-mini on DEV intake chat logs.

Training conversations use the same system and user prompt as the T0 run, so
T4 (adapter, T0 prompt) is a direct comparison against T0. Only DEV rows are
used; loss is computed on the assistant turn only. The target's
missing_information and clarification_question come from the gold issue's
knowledge card, which the team wrote per category, so they are not WPRDC labels.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "api"))
from team311.knowledge import find_card, load_knowledge  # noqa: E402
from team311.pipeline import SYSTEM, t0_prompt  # noqa: E402

DATA = HERE / "data"
OUT = HERE / "finetune"
PHI = Path(os.environ.get("PHI_MODEL_PATH", REPO.parent / "cmu_application_of_nlx_llm/lab01/models/phi-4-mini-instruct"))
SEED = 952


def sample_dev(per_subtopic: int) -> list[dict]:
    rows = [json.loads(l) for l in (DATA / "dev.jsonl").read_text().splitlines() if l.strip()]
    rng = random.Random(SEED)
    by_sub: dict[str, list[dict]] = {}
    for r in rows:
        by_sub.setdefault(r["subtopic_key"], []).append(r)
    picked = []
    for sk in sorted(by_sub):
        group = by_sub[sk][:]
        rng.shuffle(group)
        picked.extend(group[:per_subtopic])
    rng.shuffle(picked)
    return picked


def build_training_jsonl(per_subtopic: int) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = OUT / "train_conversations.jsonl"
    index = load_knowledge()
    with train_path.open("w") as f:
        for r in sample_dev(per_subtopic):
            card = find_card(index, r["gold"]["issue"]) or {}
            assistant = json.dumps({
                "domain": r["gold"]["domain"],
                "category": r["gold"]["category"],
                "issue": r["gold"]["issue"],
                "department": r["gold"]["department"],
                "missing_information": (card.get("required_information") or [])[:2],
                "clarification_question": card.get("clarification_question"),
                "confidence": 0.9,
                "abstain": False,
                "historical_resolution_range": None,
            }, ensure_ascii=False)
            f.write(json.dumps({
                "doc_id": r["doc_id"],
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": t0_prompt(r["input"])},
                    {"role": "assistant", "content": assistant},
                ],
            }, ensure_ascii=False) + "\n")
    return train_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subtopic", type=int, default=80)
    ap.add_argument("--epochs", type=float, default=2.0)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset as TorchDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    torch.manual_seed(SEED)
    train_path = build_training_jsonl(args.per_subtopic)
    records = [json.loads(l) for l in train_path.read_text().splitlines() if l.strip()]
    tokenizer = AutoTokenizer.from_pretrained(PHI, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(PHI, dtype=torch.bfloat16, local_files_only=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class ChatDS(TorchDataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, idx):
            msgs = records[idx]["messages"]
            prompt = tokenizer.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(msgs, tokenize=False)
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=768)["input_ids"]
            labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
            return {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "labels": torch.tensor(labels),
            }

    adapter_dir = OUT / "adapter"
    targs = TrainingArguments(
        output_dir=str(OUT / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=3,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=SEED,
    )
    class FreeMPSCache(TrainerCallback):
        # Variable-length batches fragment the MPS cache; without this, 24 GB runs out mid-epoch.
        def on_step_end(self, args, state, control, **kwargs):
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    trainer = Trainer(model=model, args=targs, train_dataset=ChatDS(), callbacks=[FreeMPSCache()])
    result = trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (OUT / "train_metrics.json").write_text(json.dumps({
        "rows": len(records),
        "per_subtopic": args.per_subtopic,
        "epochs": args.epochs,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "train_loss": result.training_loss,
        "log_history": trainer.state.log_history,
        "adapter": str(adapter_dir.relative_to(REPO)),
    }, indent=2) + "\n")
    print("saved", adapter_dir)


if __name__ == "__main__":
    main()
