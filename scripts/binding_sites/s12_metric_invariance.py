"""S12: are BindingGYM's metrics invariant to a monotone transform of the prediction?

If they are, no per-assay scalar -- the wild-type DMS_score included -- can move any
reported number, because a scalar can only shift/scale the prediction vector.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef, ndcg_score, average_precision_score

def calc(df, pred_col, label_col="DMS_score"):
    """Verbatim from BindingGYM's calc_metric.py::calc_zero_shot_metric (top_test=False)."""
    label_bin = (df[label_col] > np.percentile(df[label_col].values, 90)) + 0
    pred_bin  = (df[pred_col]  > np.percentile(df[pred_col].values,  90)) + 0
    return {"Spearman": df[label_col].rank().corr(df[pred_col].rank()),
            "AUC": roc_auc_score(label_bin, df[pred_col]),
            "MCC": matthews_corrcoef(label_bin, pred_bin),
            "NDCG": ndcg_score(df[label_col].rank().values.reshape(1, -1),
                               df[pred_col].values.reshape(1, -1), k=df.shape[0] // 10),
            "AP": average_precision_score(label_bin, df[pred_col])}

DMS = "/home/guoj0f/share/BindingGYM/input/Binding_substitutions_DMS/KRAS_RALGDS-RBD_norfitness_1LFD.csv"
WT = -0.0004                                    # this assay's wild-type anchor (see data/wt_reference.csv)
rng = np.random.default_rng(0)
d = pd.read_csv(DMS)
pred = 0.6 * d.DMS_score.values + rng.normal(0, 1.0, len(d))   # a prediction with real signal
cases = {"identity": pred, "minus WT DMS_score": pred - WT,
         "affine to WT (a*x+b)": 2.7 * pred + WT,
         "tanh": np.tanh(pred / 3), "exp": np.exp(pred / 5)}
keys = list(calc(d.assign(pred=pred), "pred"))
print(f"{'transform':24s}" + "".join(f"{k:>11s}" for k in keys))
ref = None
for name, p in cases.items():
    m = calc(d.assign(pred=p), "pred")
    print(f"{name:24s}" + "".join(f"{m[k]:>11.6f}" for k in keys))
    ref = ref or m
    assert all(abs(m[k] - ref[k]) < 1e-12 for k in keys), f"NOT invariant under {name}"
print("\nAll metrics identical => every BindingGYM metric is invariant to a strictly "
      "increasing transform of the prediction. A per-assay scalar cannot change any of them.")
