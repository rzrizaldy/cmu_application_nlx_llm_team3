"""Memo figures, drawn only from scored runs and the cost model.

  uv run --with matplotlib python a2/scripts/make_memo_figures.py
      -> memo/figures/fig_quality.png, memo/figures/fig_cost.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from a2common import A2, RESULTS, RUNS  # noqa: E402

OUT = A2.parent / "memo" / "figures"
QUALITY_RUNS = [("B1 default", "B1_generate_defaults", "DEV"), ("B3 tuned", "B3_generate_tuned", "DEV"),
                ("C1 structured", "C1_structured", "EVAL"), ("C2 tools", "C2_tools", "EVAL"),
                ("C3 guarded", "C3_guarded", "EVAL"), ("C4 few-shot", "C4_fewshot", "EVAL"),
                ("TF-IDF", "E_nonllm_tfidf_lr", "EVAL"), ("Gemma 270M", "E_gemma270m_structured", "EVAL")]
COLORS = {"DEV": "#9e9e9e", "EVAL": "#3b6ea5"}


def metrics(run):
    return json.loads((RUNS / run / "metrics.json").read_text())


def quality():
    names, exact, lo, hi, valid, lat, colors = [], [], [], [], [], [], []
    for name, run, split in QUALITY_RUNS:
        m = metrics(run)
        names.append(name)
        exact.append(100 * m["exact_match"])
        lo.append(100 * (m["exact_match"] - m["exact_match_ci95"][0]))
        hi.append(100 * (m["exact_match_ci95"][1] - m["exact_match"]))
        valid.append(100 * m["valid_output_rate"])
        lat.append(m.get("latency_s", {}).get("mean", 0.0))
        colors.append(COLORS[split])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    y = range(len(names))[::-1]
    a.barh(list(y), exact, xerr=[lo, hi], color=colors, capsize=3, height=0.6)
    a.scatter(valid, list(y), marker="|", s=180, color="black", zorder=3, label="valid JSON rate")
    for yi, e in zip(y, exact):
        a.text(e + 2, yi + 0.18, f"{e:.0f}%", fontsize=8)
    a.set_yticks(list(y), names)
    a.set_xlim(0, 105)
    a.set_xlabel("exact match on all three tags (%), bars show 95% CI")
    a.set_title("(a) Output quality, 50 records per run", fontsize=10, loc="left")
    a.legend(loc="upper right", fontsize=8, frameon=False)
    b.barh(list(y), lat, color=colors, height=0.6)
    for yi, s in zip(y, lat):
        b.text(s + 0.3, yi - 0.15, f"{s:.1f}s", fontsize=8)
    b.set_yticks(list(y), [])
    b.set_xlim(0, max(lat) * 1.25)
    b.set_xlabel("mean seconds per request")
    b.set_title("(b) Speed", fontsize=10, loc="left")
    fig.text(0.01, 0.01, "Grey = DEV-50 (baseline tuning). Blue = EVAL-50 (held out). TF-IDF uses no model call.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig_quality.png", dpi=220)


def cost():
    c = json.loads((RESULTS / "cost_model.json").read_text())
    a = c["assumptions"]
    staff = a["staff_cost_per_hour_usd"] / 3600
    manual_s = c["manual_seconds_per_record"]
    review_s = c["review"]["seconds_per_output"]
    setup = a["setup_hours"] * a["staff_cost_per_hour_usd"]

    order = [("C3 guarded", "C3_guarded"), ("C4 few-shot", "C4_fewshot"), ("C1 structured", "C1_structured"),
             ("C2 tools", "C2_tools")]
    fig, (p, q) = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = ["Manual only"] + [n for n, _ in order]
    rev = [0.0] + [review_s * staff for _ in order]
    redo = [manual_s * staff] + [(1 - c["runs"][r]["usable_rate"]) * manual_s * staff for _, r in order]
    comp = [0.0] + [c["runs"][r]["compute_seconds_per_record"] * a["compute_cost_per_hour_usd"] / 3600 for _, r in order]
    x = range(len(labels))
    p.bar(x, rev, color="#3b6ea5", label="review every output")
    p.bar(x, redo, bottom=rev, color="#e3a33b", label="redo by hand (manual or unusable)")
    p.bar(x, comp, bottom=[r + d for r, d in zip(rev, redo)], color="black", label="compute (tiny)")
    manual = manual_s * staff
    p.axhline(manual, ls="--", color="#b23b3b", lw=1, label=f"manual line ${manual:.2f}")
    for xi, (r, d, k) in enumerate(zip(rev, redo, comp)):
        p.text(xi, r + d + k + 0.006, f"${r + d + k:.3f}", ha="center", fontsize=8)
    p.set_xticks(list(x), labels, fontsize=8)
    p.set_ylabel("staff + compute cost per record (USD)")
    p.set_ylim(0, 0.34)
    p.set_title("(a) Cost per record", fontsize=10, loc="left")
    p.legend(fontsize=7.5, frameon=False, loc="upper left")

    usable = c["runs"]["C3_guarded"]["usable_rate"]
    scenarios = [(f"measured: review {review_s:.0f}s, {usable:.0%} usable", review_s, usable, "#3b6ea5", "-"),
                 (f"what if review {review_s / 2:.0f}s", review_s / 2, usable, "#3b6ea5", "--"),
                 ("what if 90% usable", review_s, 0.90, "#6a9f58", "--")]
    vols = [1000 * 1.08 ** i for i in range(90)]
    for name, rs, u, col, ls in scenarios:
        saving = manual - (rs + (1 - u) * manual_s) * staff
        q.plot(vols, [(saving * v - setup) / 1000 for v in vols], color=col, ls=ls, label=name)
        if saving > 0:
            q.scatter([setup / saving], [0], color=col, s=18, zorder=3)
            q.annotate(f"{setup / saving / 1000:.0f}k", (setup / saving, 0), textcoords="offset points",
                       xytext=(4, 6), fontsize=8, color=col)
    q.axhline(0, color="black", lw=0.8)
    q.axvline(a["annual_record_volume"], color="#b23b3b", ls=":", lw=1)
    q.text(a["annual_record_volume"] * 1.1, 6, f"my estimate\n{a['annual_record_volume']:,} tags/year",
           fontsize=7.5, color="#b23b3b")
    q.set_xscale("log")
    q.set_xlabel("records tagged (log scale)")
    q.set_ylabel("net saving after setup (USD thousands)")
    q.set_title(f"(b) C3 break-even, setup {a['setup_hours']:.0f}h = ${setup:,.0f}", fontsize=10, loc="left")
    q.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost.png", dpi=220)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    quality()
    cost()
    print(f"wrote {OUT}/fig_quality.png, fig_cost.png")
