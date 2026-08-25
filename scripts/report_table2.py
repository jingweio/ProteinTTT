"""Assemble the Table-2-style reproduction report from per-run score directories.

Reads every `<model>__baseline` and `<model>__ttt__seed<k>` directory under the
scores root, aggregates each with ProteinGym's official recipe, and prints the
reproduction next to the published numbers.

Also runs two consistency checks that the layout makes free:
  * every TTT seed's `score_pre_ttt` column must aggregate to the baseline value
    (otherwise `ttt_reset()` is leaking state between assays);
  * the paper reports mean +- std over 5 seeds, so std is reported the same way.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from aggregate_proteingym import CATEGORIES, DEPTHS, official_aggregate, per_assay_spearman

# Table 2 / Table A4 of "One protein is all you need" (ICLR 2026).
PAPER = {
    "esm2_t12_35M_UR50D": {
        "label": "ESM2 (35M)",
        "published_column": "ESM2 (35M)",
        "baseline": dict(avg=0.3211, Activity=0.3137, Binding=0.2907, Expression=0.3435,
                         OrganismalFitness=0.2184, Stability=0.4392,
                         Low=0.2394, Medium=0.2707, High=0.4510),
        "ttt": dict(avg=0.3407, Activity=0.3407, Binding=0.2942, Expression=0.3550,
                    OrganismalFitness=0.2403, Stability=0.4733,
                    Low=0.2445, Medium=0.3144, High=0.4598, std=0.00014),
    },
    "esm2_t33_650M_UR50D": {
        "label": "ESM2 (650M)",
        "published_column": "ESM2 (650M)",
        "baseline": dict(avg=0.4139, Activity=0.4254, Binding=0.3366, Expression=0.4151,
                         OrganismalFitness=0.3691, Stability=0.5233,
                         Low=0.3346, Medium=0.4063, High=0.5153),
        "ttt": dict(avg=0.4153, Activity=0.4323, Binding=0.3376, Expression=0.4168,
                    OrganismalFitness=0.3702, Stability=0.5195,
                    Low=0.3363, Medium=0.4126, High=0.5075, std=0.00003),
    },
}


def fmt(x):
    return "  n/a " if x is None else f"{x:.4f}"


def row(name, agg, paper=None, std=None):
    cells = [agg["avg_spearman"]] + [agg["by_function"][c] for c in CATEGORIES]
    s = f"  {name:<34}" + (f"{cells[0]:.4f}" + (f" ±{std:.5f}" if std is not None else "        "))
    s += "  " + "  ".join(f"{c:.4f}" for c in cells[1:])
    if paper is not None:
        s += f"   | paper {paper['avg']:.4f}  Δ={cells[0] - paper['avg']:+.4f}"
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores_root", required=True, type=Path)
    p.add_argument("--dms_reference", required=True, type=Path)
    p.add_argument("--published", type=Path, default=None)
    p.add_argument("--out_md", type=Path, default=None)
    args = p.parse_args()

    ref = pd.read_csv(args.dms_reference)
    pub = (
        pd.read_csv(args.published).rename(columns={"DMS ID": "DMS_id"})
        if args.published
        else None
    )

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    header = (
        f"  {'run':<34}{'Avg.Spearman':<16}"
        + "  ".join(f"{c[:6]:<6}" for c in CATEGORIES)
    )

    for model, meta in PAPER.items():
        base_dir = args.scores_root / f"{model}__baseline"
        seed_dirs = sorted(
            (
                d
                for d in args.scores_root.glob(f"{model}__ttt__seed*")
                if d.is_dir() and re.search(r"seed\d+$", d.name)
            ),
            key=lambda p: int(re.search(r"seed(\d+)$", p.name).group(1)),
        )
        if not base_dir.exists() and not seed_dirs:
            continue

        emit()
        emit(f"=== {meta['label']} ===")
        emit(header)

        base_agg = None
        if base_dir.exists():
            pa = per_assay_spearman(base_dir, "score_pre_ttt")
            base_agg = official_aggregate(pa, ref)
            emit(row(f"baseline (n={base_agg['n_assays']})", base_agg, meta["baseline"]))
            if pub is not None:
                pub_agg = official_aggregate(
                    pub[["DMS_id", meta["published_column"]]].rename(
                        columns={meta["published_column"]: "Spearman"}
                    ),
                    ref,
                )
                cmp = pa.merge(
                    pub[["DMS_id", meta["published_column"]]], on="DMS_id", how="inner"
                )
                d = (cmp["Spearman"] - cmp[meta["published_column"]]).abs()
                emit(
                    f"    validation vs ProteinGym leaderboard: max|Δ|={d.max():.4f} "
                    f"mean|Δ|={d.mean():.4f} n>{0.01}={int((d > 0.01).sum())} "
                    f"| leaderboard avg {pub_agg['avg_spearman']:.4f}"
                )

        seed_aggs = []
        for sd in seed_dirs:
            k = int(re.search(r"seed(\d+)$", sd.name).group(1))
            pa_pre = per_assay_spearman(sd, "score_pre_ttt")
            pa_ttt = per_assay_spearman(sd, "score_ttt")
            if pa_ttt.empty:
                continue
            a_ttt = official_aggregate(pa_ttt, ref)
            complete = base_agg is None or a_ttt["n_assays"] == base_agg["n_assays"]
            if not complete:
                emit(
                    f"  +ProteinTTT seed{k}: PARTIAL, {a_ttt['n_assays']}/"
                    f"{base_agg['n_assays']} assays -- not aggregated"
                )
                continue
            a_pre = official_aggregate(pa_pre, ref)
            seed_aggs.append(a_ttt)
            emit(row(f"+ProteinTTT seed{k} (n={a_ttt['n_assays']})", a_ttt, meta["ttt"]))
            if a_pre["n_assays"] == a_ttt["n_assays"]:
                drift = abs(a_pre["avg_spearman"] - base_agg["avg_spearman"])
                flag = "OK" if drift < 1e-9 else f"*** RESET LEAK {drift:.2e} ***"
                emit(f"    reset check (score_pre_ttt vs baseline): {flag}")

        if len(seed_aggs) > 1:
            avg = pd.Series([a["avg_spearman"] for a in seed_aggs])
            emit(
                f"    {len(seed_aggs)} seeds: mean {avg.mean():.4f} ± {avg.std(ddof=0):.5f}"
                f"   | paper {meta['ttt']['avg']:.4f} ± {meta['ttt']['std']:.5f}"
            )
        if base_agg is not None and seed_aggs:
            gain = pd.Series([a["avg_spearman"] for a in seed_aggs]).mean() - base_agg["avg_spearman"]
            paper_gain = meta["ttt"]["avg"] - meta["baseline"]["avg"]
            emit(
                f"    TTT gain: {gain:+.4f}   | paper {paper_gain:+.4f}"
                f"   ({gain / paper_gain * 100:.0f}% of reported)"
                if paper_gain
                else ""
            )

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("```\n" + "\n".join(lines) + "\n```\n")
        print(f"\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
