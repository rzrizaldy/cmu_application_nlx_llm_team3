"""One front door for Assignment 2.  Run from assignment02/:

    .venv/bin/python a2/cli.py status                 what is done, what is next
    .venv/bin/python a2/cli.py <command> [args]       see the table below

Every command calls a function that lives in a2/scripts/. The scripts still
work on their own; this file only gives them one place and one name each.

  status                          checklist of the whole pipeline, with the next command to run
  labels export                   blank labeling workbook  -> a2/labels/labeling_a2.xlsx
  labels import                   workbook -> a2/labels/human_labels_a2.jsonl + labeling_timing.json
  labels import-lab FILE          label-studio export (JSON from the 311 NLX Lab) -> same two files
  run EXP [hydra overrides]       one LLMBox batch run (B1 B3 C1 C2 C3 C4 D_base D_guard E_gemma ...; see a2/run.sh)
  score RUN... | --all [--draft]  metrics.json + scored.jsonl for each run (--draft scores against the AI draft)
  pool [--draft] | pool --audit RUN   build the DEV-only few-shot pool for C4 / audit a C4 run for EVAL leakage
  probes                          a2/data/probes/probes_seed.jsonl -> a2/data/inputs/partd_probes.jsonl
  partd score | sheet | agree     Part D: baseline vs guarded, blind adjudication sheet, human-vs-guard agreement
  nonllm [--draft]                TF-IDF + logistic regression alternative (Part E)
  cost review-sheet RUN | cost model RUN...   Part E review timing sheet / cost model
  tables [--draft]                results/tables/*.md + memo/evidence_packet.md
  vm-patch                        print the LLMBox patch (A2 changes on top of the course release) for the VM
  test                            run all A2 unit tests
"""
import json
import subprocess
import sys
from pathlib import Path

A2 = Path(__file__).resolve().parent
ROOT = A2.parent
sys.path.insert(0, str(A2 / "scripts"))

# Runs that make up the submission, in the order the memo reports them.
OFFICIAL_RUNS = ["B1_generate_defaults", "B3_generate_tuned", "C1_structured", "C2_tools", "C3_guarded", "C4_fewshot",
                 "C1abl_dev_dept_optional", "C1abl_dev_dept_required", "C2abl_dev_tools_auto", "C2abl_dev_tools_required",
                 "E_gemma270m_structured", "E_nonllm_tfidf_lr"]
RUN_OF = {"B1": "B1_generate_defaults", "B3": "B3_generate_tuned", "C1": "C1_structured", "C2": "C2_tools",
          "C3": "C3_guarded", "C4": "C4_fewshot", "E_gemma": "E_gemma270m_structured"}


def _rows(path):
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()) if Path(path).exists() else 0


def status():
    from a2common import DATA, LABELS, RESULTS, RUNS
    mark = lambda ok: "✅" if ok else "⬜"
    lines, nxt = [], []

    labels = _rows(LABELS / "human_labels_a2.jsonl")
    timing = (LABELS / "labeling_timing.json").exists()
    lines.append(f"{mark(labels == 100)} Labels: {labels}/100 imported{' + timing' if timing else ''}")
    if labels < 100:
        nxt.append("Label the 100 records in the 311 NLX Lab (Label studio), save the export into a2/labels/, then:\n"
                   "     .venv/bin/python a2/cli.py labels import-lab a2/labels/lab_labels_export.json")

    for run in OFFICIAL_RUNS:
        d = RUNS / run
        n = _rows(d / "responses.jsonl")
        official = (d / "metrics.json").exists()
        state = "scored" if official else ("not scored" if n else "not run")
        lines.append(f"{mark(n == 50 and official)} {run:28s} rows {n:>2}/50  {state}")
        if not n:
            exp = next((k for k, v in RUN_OF.items() if v == run), None)
            if run == "C4_fewshot":
                nxt.append("C4 needs your DEV labels:  a2/cli.py pool  &&  a2/cli.py run C4")
            elif run == "E_nonllm_tfidf_lr":
                nxt.append("a2/cli.py nonllm")
            elif exp:
                nxt.append(f"a2/cli.py run {exp}" + ("   (pass your chosen generation.* settings)" if exp == "B3" else ""))
        elif labels == 100 and not official:
            nxt.append(f"a2/cli.py score {run}")

    seeds = _rows(DATA / "probes" / "probes_seed.jsonl")
    d_ok = _rows(RUNS / "D_baseline" / "responses.jsonl") and _rows(RUNS / "D_guarded" / "responses.jsonl")
    lines.append(f"{mark(seeds and d_ok)} Part D: {seeds} probe seeds; baseline/guarded runs {'present' if d_ok else 'missing'}")
    adj = LABELS / "adjudication_partd.csv"
    if adj.exists():
        import csv
        with adj.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        filled = sum(r.get("should_block (yes/no)", "").strip().lower() in {"yes", "no"} for r in rows)
        lines.append(f"{mark(rows and filled == len(rows))} Part D adjudication: {filled}/{len(rows)} rows judged")
        if filled < len(rows):
            nxt.append("Fill should_block yes/no in a2/labels/adjudication_partd.csv (blind), then: a2/cli.py partd agree")

    cost = json.loads((DATA / "cost_assumptions.json").read_text()) if (DATA / "cost_assumptions.json").exists() else {}
    missing = [k for k, v in cost.items() if v in (None, "")]
    review = (LABELS / "review_timing.json").exists()
    lines.append(f"{mark(cost and not missing and review)} Part E: cost assumptions {'complete' if cost and not missing else f'{len(missing)} empty'}; review timing {'yes' if review else 'no'}")
    if missing:
        nxt.append("Fill a2/data/cost_assumptions.json with values AND sources")
    scored = any((RUNS / r / "metrics.json").exists() for r in OFFICIAL_RUNS)
    lines.append(f"{mark(scored and (RESULTS / 'tables' / 'partC.md').exists())} Official tables: a2/results/tables/ "
                 f"({'rebuild with a2/cli.py tables' if scored else 'after official scoring'})")

    print("Assignment 2 status\n" + "\n".join("  " + x for x in lines))
    if nxt:
        print("\nNext:\n" + "\n".join(f"  {i}. {x}" for i, x in enumerate(nxt[:6], 1)))
    else:
        print("\nEverything measurable is done. Write the memo; then tag and merge (see README).")


def main(argv):
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return
    cmd, args = argv[0], argv[1:]
    draft = "--draft" in args
    args = [a for a in args if a != "--draft"]

    if cmd == "status":
        status()
    elif cmd == "labels":
        import labeling
        {"export": labeling.export, "import": labeling.import_,
         "import-lab": lambda: labeling.import_lab(args[1])}[args[0]]()
    elif cmd == "run":
        sys.exit(subprocess.call(["bash", str(A2 / "run.sh"), *args]))
    elif cmd == "score":
        import score
        runs = [r for r in OFFICIAL_RUNS if (A2 / "runs" / r / "responses.jsonl").exists()] if args == ["--all"] else args
        for run in runs:
            m = score.score(run, draft=draft)
            print(f"{run:28s} n={m['n']}  valid={m['valid_output_rate']:.0%}  exact={m['exact_match']:.0%}")
    elif cmd == "pool":
        import build_fewshot_pool as pool
        pool.audit(args[1]) if args[:1] == ["--audit"] else pool.build(draft=draft)
    elif cmd == "probes":
        import build_probes
        build_probes.main(20)
    elif cmd == "partd":
        import score_partd as sp
        {"score": lambda: sp.score("D_baseline", "D_guarded"),
         "sheet": lambda: sp.sheet("D_baseline", "D_guarded", 40),
         "agree": lambda: sp.agree("D_guarded")}[args[0]]()
    elif cmd == "nonllm":
        import baseline_nonllm
        baseline_nonllm.main(draft=draft)
    elif cmd == "cost":
        import cost
        cost.review_sheet(args[1]) if args[0] == "review-sheet" else cost.model(args[1:])
    elif cmd == "tables":
        import make_draft_tables
        import make_tables
        (make_draft_tables.main if draft else make_tables.main)()
    elif cmd == "vm-patch":
        sys.stdout.write(subprocess.run(["git", "diff", "--relative=assignment02/llmbox", "dadd8b9", "HEAD", "--", "."],
                                        cwd=ROOT / "llmbox", capture_output=True, text=True, check=True).stdout)
    elif cmd == "test":
        py = sys.executable
        rc = subprocess.call([py, "-m", "pytest", "-q", str(A2 / "tests")])
        rc |= subprocess.call([py, "-m", "pytest", "-q", *sorted(str(p) for p in (ROOT / "llmbox" / "tests").glob("test_a2_*.py"))],
                              cwd=ROOT / "llmbox")
        sys.exit(rc)
    else:
        sys.exit(f"unknown command '{cmd}'. Run: a2/cli.py help")


if __name__ == "__main__":
    main(sys.argv[1:])
