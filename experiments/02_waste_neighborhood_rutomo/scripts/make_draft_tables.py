"""Separate provisional tables scored against AI-assisted, unreviewed labels."""
from a2common import RESULTS, RUNS
from make_tables import run_table


def main():
    target = RESULTS / "draft_tables"
    target.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.name for p in RUNS.iterdir() if (p / "metrics_draft.json").exists())
    groups = {
        "partB.md": [r for r in runs if r.startswith("B")],
        "partC.md": [r for r in runs if r.startswith("C") and "abl" not in r],
        "ablations.md": [r for r in runs if "abl" in r],
        "alternatives.md": [r for r in runs if r.startswith("E_")],
    }
    for name, group in groups.items():
        table = run_table(group, "PROVISIONAL: AI-assisted draft reference, student review pending", "metrics_draft.json")
        (target / name).write_text(table, encoding="utf-8")
    print("draft tables:", sorted(groups))


if __name__ == "__main__":
    main()
