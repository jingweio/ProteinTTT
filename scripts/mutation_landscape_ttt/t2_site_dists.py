"""T2: per-variant, per-mutated-site distances to the other entity.

variant_labels.parquet only stores min over the mutated sites, which fixes the aggregation
to `min` and makes experiment 1 (feature tuning) impossible. This dumps every site's
distance so any aggregation can be tested.

Built by joining the already-computed per-residue interface table rather than re-running the
structure pipeline. Correctness is pinned by asserting that re-aggregating with `min`
reproduces min_dist_to_partner row for row.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "binding_sites"))
from bg_common import DMS_DIR, OUT_DIR, load_index, parse_mut_dict, muts_of

res = pd.read_csv(f"{OUT_DIR}/interface_residues_all_chains.csv")
D = {(r.DMS_id, r.chain, int(r.seq_pos)): r.min_dist_to_other_entity for r in res.itertuples()}

ref = pd.read_parquet(f"{OUT_DIR}/variant_labels.parquet")
rows, miss = [], 0
for _, r in load_index().iterrows():
    dms = r["DMS_id"]
    df = pd.read_csv(os.path.join(DMS_DIR, r["DMS_filename"]))
    vi = 0                                   # index into the kept (n_mut>0) variants of this assay
    for s in df["mutant"]:
        ds = []
        for ch, v in parse_mut_dict(s).items():
            if not v.strip(): continue
            for _, ps, _ in muts_of(v):
                d = D.get((dms, ch, int(ps)))
                if d is None: miss += 1; continue
                ds.append((ch, int(ps), d))
        if not ds: continue                  # matches s3's `keep = n_mut > 0`
        for ch, p, d in ds:
            rows.append((dms, vi, ch, p, d))
        vi += 1

sd = pd.DataFrame(rows, columns=["DMS_id", "vi", "chain", "pos", "dist"])
assert miss == 0, f"{miss} mutated sites absent from the interface table"

# hard check: re-aggregating with min must reproduce the stored column, row for row.
# The interface table stores distances rounded to 3 decimals, so the tolerance is 1e-3 A --
# far below any scale that matters for exp(-d/tau).
chk = sd.groupby(["DMS_id", "vi"], sort=False).dist.min().reset_index()
worst = 0.0
for dms, g in chk.groupby("DMS_id", sort=False):
    want = ref[ref.DMS_id == dms].min_dist_to_partner.to_numpy()
    got = g.sort_values("vi").dist.to_numpy()
    assert len(want) == len(got), f"{dms}: {len(got)} vs {len(want)} variants"
    worst = max(worst, np.abs(want - got).max())
    assert np.allclose(want, got, atol=1e-3), f"{dms}: min mismatch"

sd.to_parquet(f"{OUT_DIR}/variant_site_dists.parquet", index=False)
print(f"OK  {len(sd):,} (variant, site) pairs over {sd.DMS_id.nunique()} assays, "
      f"{chk.shape[0]:,} variants; min-agg reproduces min_dist_to_partner, max dev {worst:.2e} A")
print(sd.groupby('DMS_id').size().describe().to_string())
