#!/usr/bin/env python3
"""Populate Canvas appendix folders from corpora, experiments, and api/."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "appendix"
EXP = REPO / "experiments"
CORPUS = REPO / "corpora" / "05_all" / "corpus.jsonl"


def copy_corpus_original() -> None:
    shutil.copy2(CORPUS, APP / "original_dataset.jsonl")


def dev_eval_datasets() -> None:
    dev_dir = APP / "development_dataset"
    eval_dir = APP / "evaluation_dataset"
    dev_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    team_dev = EXP / "05_team" / "data" / "dev.jsonl"
    team_eval = EXP / "05_team" / "data" / "eval.jsonl"
    if team_dev.exists():
        shutil.copy2(team_dev, dev_dir / "05_team_dev.jsonl")
    if team_eval.exists():
        shutil.copy2(team_eval, eval_dir / "05_team_eval.jsonl")
    mahika_dev = EXP / "03_buildings_construction_mahika" / "data" / "dev.jsonl"
    mahika_eval = EXP / "03_buildings_construction_mahika" / "data" / "eval.jsonl"
    if mahika_dev.exists():
        shutil.copy2(mahika_dev, dev_dir / "03_buildings_construction_mahika_dev.jsonl")
    if mahika_eval.exists():
        shutil.copy2(mahika_eval, eval_dir / "03_buildings_construction_mahika_eval.jsonl")
    rutomo_dev = EXP / "02_waste_neighborhood_rutomo" / "data" / "inputs" / "dev50_task.jsonl"
    rutomo_eval = EXP / "02_waste_neighborhood_rutomo" / "data" / "inputs" / "eval50_task.jsonl"
    if rutomo_dev.exists():
        shutil.copy2(rutomo_dev, dev_dir / "02_waste_neighborhood_rutomo_dev.jsonl")
    if rutomo_eval.exists():
        shutil.copy2(rutomo_eval, eval_dir / "02_waste_neighborhood_rutomo_eval.jsonl")
    afaq_data = EXP / "01_streets_mobility_afaq" / "data" / "pgh311_complaints.json"
    if afaq_data.exists():
        data = json.loads(afaq_data.read_text())
        complaints = data.get("complaints") or []
        dev = data.get("dev") or complaints[:50]
        ev = data.get("eval") or complaints[50:100]
        (dev_dir / "01_streets_mobility_afaq_dev.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in dev)
        )
        (eval_dir / "01_streets_mobility_afaq_eval.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ev)
        )
    pending_dev = dev_dir / "04_parks_public_spaces_mingchin_PENDING.md"
    pending_eval = eval_dir / "04_parks_public_spaces_mingchin_PENDING.md"
    if not (dev_dir / "04_parks_public_spaces_mingchin_dev.jsonl").exists():
        pending_dev.write_text("Awaiting Mingchin development_dataset.jsonl from teammate.\n")
    if not (eval_dir / "04_parks_public_spaces_mingchin_eval.jsonl").exists():
        pending_eval.write_text("Awaiting Mingchin evaluation_dataset.jsonl from teammate.\n")


def metrics_exports() -> None:
    out = APP / "evaluation_metrics"
    out.mkdir(parents=True, exist_ok=True)
    team_runs = EXP / "05_team" / "runs"
    if team_runs.exists():
        for run_dir in sorted(team_runs.iterdir()):
            m = run_dir / "metrics.json"
            if m.exists():
                shutil.copy2(m, out / f"05_team_{run_dir.name}.json")
    rutomo_runs = EXP / "02_waste_neighborhood_rutomo" / "runs"
    if rutomo_runs.exists():
        for run_dir in sorted(rutomo_runs.iterdir())[:8]:
            m = run_dir / "metrics.json"
            if m.exists():
                shutil.copy2(m, out / f"02_rutomo_{run_dir.name}.json")
    mahika_out = EXP / "03_buildings_construction_mahika" / "out"
    if mahika_out.exists():
        for p in mahika_out.glob("*metrics*.json"):
            shutil.copy2(p, out / f"03_mahika_{p.name}")
    if not (out / "01_afaq_PENDING.md").exists() and not any(out.glob("01_*")):
        (out / "01_afaq_PENDING.md").write_text("Awaiting Afaq experiment metrics JSON from teammate.\n")
    if not any(out.glob("04_*")):
        (out / "04_mingchin_PENDING.md").write_text("Awaiting Mingchin evaluation metrics from teammate.\n")


def chatlogs_export() -> None:
    out = APP / "chatlogs_llm_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    src = EXP / "05_team" / "chatlogs" / "finetuned_eval_sessions.jsonl"
    if src.exists():
        shutil.copy2(src, out / "05_team_T4_finetuned_chatlogs.jsonl")
    else:
        pending = out / "PENDING.md"
        if not pending.exists():
            pending.write_text("Run finetune + T4_finetuned, then build_chatlogs.py.\n")


def api_zip() -> None:
    out_dir = APP / "llm_api_code"
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / "llm_api_code.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for base in [REPO / "api", REPO / "experiments" / "05_team"]:
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_dir():
                    continue
                if any(p in path.parts for p in (".venv", "__pycache__", "adapter", "runs")):
                    continue
                if path.suffix in {".pyc"}:
                    continue
                zf.write(path, path.relative_to(REPO))


def main() -> None:
    copy_corpus_original()
    dev_eval_datasets()
    metrics_exports()
    chatlogs_export()
    api_zip()
    print("appendix build complete")


if __name__ == "__main__":
    main()
