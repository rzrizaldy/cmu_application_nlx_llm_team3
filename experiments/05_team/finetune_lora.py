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
import math
import os
import random
import sys
import time
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    train_path = build_training_jsonl(args.per_subtopic)
    records = [json.loads(l) for l in train_path.read_text().splitlines() if l.strip()]
    tokenizer = AutoTokenizer.from_pretrained(PHI, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        PHI, dtype=torch.bfloat16, local_files_only=True, device_map={"": device}, low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def encode(msgs: list[dict]) -> tuple:
        prompt = tokenizer.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(msgs, tokenize=False)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=768)["input_ids"]
        labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
        return torch.tensor([ids], device=device), torch.tensor([labels], device=device)

    # A plain loop: on MPS, HF Trainer stalled before its first step on this machine.
    batches = [encode(r["messages"]) for r in records]
    accum, log_every, warmup, lr = 4, 5, 3, 2e-4
    total_steps = math.ceil(len(batches) * args.epochs / accum)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)),
    )
    rng = random.Random(SEED)
    order = []
    for _ in range(math.ceil(args.epochs)):
        epoch = list(range(len(batches)))
        rng.shuffle(epoch)
        order.extend(epoch)
    order = order[: int(len(batches) * args.epochs)]
    model.train()
    log_history, window, all_losses, step, t0 = [], [], [], 0, time.time()
    for i, idx in enumerate(order, 1):
        ids, labels = batches[idx]
        loss = model(input_ids=ids, labels=labels).loss
        (loss / accum).backward()
        window.append(loss.item())
        if i % accum == 0 or i == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0 or step == total_steps:
                entry = {"loss": sum(window) / len(window), "learning_rate": sched.get_last_lr()[0],
                         "epoch": i / len(batches), "step": step}
                log_history.append(entry)
                print(f"step {step}/{total_steps} loss {entry['loss']:.4f} {time.time() - t0:.0f}s", flush=True)
                all_losses.extend(window)
                window = []
    all_losses.extend(window)
    log_history.append({"train_runtime": time.time() - t0, "train_loss": sum(all_losses) / len(all_losses),
                        "epoch": args.epochs, "step": step})

    adapter_dir = OUT / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (OUT / "train_metrics.json").write_text(json.dumps({
        "rows": len(records),
        "per_subtopic": args.per_subtopic,
        "epochs": args.epochs,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "optimizer": {"name": "AdamW", "lr": lr, "warmup_steps": warmup, "schedule": "linear",
                      "gradient_accumulation": accum, "batch_size": 1, "grad_clip": 1.0},
        "train_loss": log_history[-1]["train_loss"],
        "log_history": log_history,
        "adapter": str(adapter_dir.relative_to(REPO)),
    }, indent=2) + "\n")
    print("saved", adapter_dir)


if __name__ == "__main__":
    main()
