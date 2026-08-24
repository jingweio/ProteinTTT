"""Aggregate per-assay ProteinGym scores into Table-2-style numbers.

Implements ProteinGym's official aggregation, transcribed from
`ProteinGym/proteingym/performance_DMS_benchmarks.py`:

    per-assay Spearman
      -> groupby (UniProt_ID, coarse_selection_type).mean()   # assays -> protein
      -> groupby coarse_selection_type.mean()                 # proteins -> function
      -> mean over the 5 function categories                  # "Avg. Spearman"

This is deliberately *not* an arithmetic mean over the 217 assays; ProteinGym
corrects for the number of assays per protein and per function group.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

CATEGORIES = [
    "Activity",
    "Binding",
    "Expression",
    "OrganismalFitness",
    "Stability",
]
DEPTHS = ["Low", "Medium", "High"]


def per_assay_spearman(score_dir, column):
    rows = []
    for csv in sorted(Path(score_dir).glob("*.csv")):
        df = pd.read_csv(csv)
        if column not in df.columns:
            continue
        rho = spearmanr(df["DMS_score"], df[column])[0]
        rows.append({"DMS_id": csv.stem, "Spearman": rho})
    return pd.DataFrame(rows)


def official_aggregate(per_assay, ref):
    m = per_assay.merge(
        ref[["DMS_id", "UniProt_ID", "coarse_selection_type", "MSA_Neff_L_category"]],
        on="DMS_id",
        how="left",
    )
    assert m["UniProt_ID"].notna().all(), "assay missing from reference file"
    by_function = (
        m.groupby(["UniProt_ID", "coarse_selection_type"])["Spearman"]
        .mean()
        .groupby("coarse_selection_type")
        .mean()
    )
    by_depth = (
        m.groupby(["UniProt_ID", "MSA_Neff_L_category"])["Spearman"]
        .mean()
        .groupby("MSA_Neff_L_category")
        .mean()
    )
    return {
        "n_assays": int(len(m)),
        "avg_spearman": float(by_function.mean()),
        "by_function": {k: float(by_function.get(k, float("nan"))) for k in CATEGORIES},
        "by_msa_depth": {k: float(by_depth.get(k, float("nan"))) for k in DEPTHS},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--score_dir", required=True, type=Path)
    p.add_argument("--column", default="score_pre_ttt")
    p.add_argument("--dms_reference", required=True, type=Path)
    p.add_argument(
        "--published",
        type=Path,
        default=None,
        help="DMS_substitutions_Spearman_DMS_level.csv, to validate the harness",
    )
    p.add_argument(
        "--published_column",
        default=None,
        help='e.g. "ESM2 (650M)" -- the ProteinGym leaderboard column to check against',
    )
    p.add_argument("--out_json", type=Path, default=None)
    args = p.parse_args()

    ref = pd.read_csv(args.dms_reference)
    per_assay = per_assay_spearman(args.score_dir, args.column)
    if per_assay.empty:
        raise SystemExit(f"no assay CSVs with column '{args.column}' in {args.score_dir}")
    result = official_aggregate(per_assay, ref)
    result["score_dir"] = str(args.score_dir)
    result["column"] = args.column

    print(f"{args.score_dir}  column={args.column}")
    print(f"  n_assays      : {result['n_assays']}")
    print(f"  Avg. Spearman : {result['avg_spearman']:.4f}")
    print(
        "  by function   : "
        + "  ".join(f"{k}={result['by_function'][k]:.4f}" for k in CATEGORIES)
    )
    print(
        "  by MSA depth  : "
        + "  ".join(f"{k}={result['by_msa_depth'][k]:.4f}" for k in DEPTHS)
    )

    if args.published and args.published_column:
        pub = pd.read_csv(args.published).rename(columns={"DMS ID": "DMS_id"})
        cmp = per_assay.merge(
            pub[["DMS_id", args.published_column]], on="DMS_id", how="inner"
        )
        cmp["delta"] = cmp["Spearman"] - cmp[args.published_column]
        pub_agg = official_aggregate(
            pub[["DMS_id", args.published_column]].rename(
                columns={args.published_column: "Spearman"}
            ),
            ref,
        )
        result["validation"] = {
            "published_column": args.published_column,
            "n_compared": int(len(cmp)),
            "max_abs_delta": float(cmp["delta"].abs().max()),
            "mean_abs_delta": float(cmp["delta"].abs().mean()),
            "n_over_0.01": int((cmp["delta"].abs() > 0.01).sum()),
            "published_avg_spearman": pub_agg["avg_spearman"],
            "reproduced_avg_spearman": result["avg_spearman"],
        }
        v = result["validation"]
        print(f"\n  validation vs published '{args.published_column}':")
        print(f"    assays compared : {v['n_compared']}")
        print(f"    max |delta|     : {v['max_abs_delta']:.4f}")
        print(f"    mean |delta|    : {v['mean_abs_delta']:.4f}")
        print(f"    assays |d|>0.01 : {v['n_over_0.01']}")
        print(
            f"    Avg. Spearman   : ours {v['reproduced_avg_spearman']:.4f}"
            f"  vs published {v['published_avg_spearman']:.4f}"
        )
        worst = cmp.reindex(cmp["delta"].abs().sort_values(ascending=False).index).head(5)
        print("    worst 5 assays:")
        for _, r in worst.iterrows():
            print(
                f"      {r['DMS_id']:<45} ours={r['Spearman']:.4f} "
                f"pub={r[args.published_column]:.4f} d={r['delta']:+.4f}"
            )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2))
        per_assay.to_csv(args.out_json.with_suffix(".per_assay.csv"), index=False)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
