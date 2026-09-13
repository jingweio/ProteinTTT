"""S-0: does moving the encoder's interface axis actually move the zero-shot ranking?

Section 3 of the structure-TTT proposal establishes only that assays whose frozen encoder
separates interface residues better tend to have better zero-shot Spearman (+0.515, p=0.060).
That is a correlation across assays and three confounders could produce it. This intervenes
instead: no training, just edit the frozen h_V and rescore.

    h~' = h~ + alpha * (u . h~) u          u = unit logistic-probe weight vector

which multiplies each residue's OWN component along u by (1 + alpha). A uniform translation
would leave the residues' relative interface-ness untouched and test nothing.

Two controls, because any large enough edit to h_V degrades the model and without a matched
comparison "moving along the interface axis" cannot be separated from "moving h_V at all":
a random unit direction (several seeds) and u with its components shuffled. Controls are
matched on the induced Frobenius norm rather than on alpha -- the true u points along a
direction carrying real variance, so the same alpha would be a much larger edit for it.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bgmpnn import load_model, AssayContext
from ga2_encoder_probe import pdb_chain_slots, CUT

BG = "/data/guoj0f/share/BindingGYM"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = f"{ROOT}/workstation-records/mutation-landscape-TTT/data"


def labels_in_packing_order(dms, ctx, pdb, ent):
    order = ctx.designed + [c for c in ctx.all_chains if c not in ctx.designed]
    e1 = set(ent.loc[dms, "entity1"])
    slots, ent_of = [], []
    for c in order:
        sl = pdb_chain_slots(pdb, c)
        assert len(sl) == ctx.chain_lengths[c], f"{dms} {c}"
        slots += sl; ent_of += [0 if c in e1 else 1] * len(sl)
    ent_of = np.asarray(ent_of)
    ok = np.array([x is not None for x in slots]) & (ctx.mask[0].cpu().numpy() > 0)
    A = np.vstack([slots[i] for i in np.flatnonzero(ok & (ent_of == 0))])
    B = np.vstack([slots[i] for i in np.flatnonzero(ok & (ent_of == 1))])
    d = np.full(len(slots), np.nan)
    for i in np.flatnonzero(ok):
        other = B if ent_of[i] == 0 else A
        d[i] = np.sqrt(((slots[i][:, None, :] - other[None, :, :]) ** 2).sum(-1)).min()
    return ok, d


def score_all(ctx, S, bs):
    out = torch.empty(len(S))
    with torch.no_grad():
        for i in range(0, len(S), bs):
            out[i:i + bs] = ctx.score(S[i:i + bs]).float().cpu()
    return out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="+", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[-1.5, -1.0, -0.5, 0.5, 1.0, 2.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--tag", default="s0")
    a = ap.parse_args()
    dev = torch.device("cuda:0"); assert "A100" in torch.cuda.get_device_name(0)
    model = load_model(f"{BG}/training/cache/v_48_020.pt", dev)
    ent = pd.read_csv(f"{ROOT}/local-records/binding-sites-analysis/data/entity_partition.csv").set_index("DMS_id")
    idx = pd.read_csv(f"{BG}/input/BindingGYM.csv")
    lab = pd.read_parquet(f"{ROOT}/local-records/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet")

    rows = []
    for dms in a.assays:
        r = idx[idx.DMS_id == dms].iloc[0]
        df = pd.read_csv(f"{BG}/input/Binding_substitutions_DMS/{r.DMS_filename}")
        df["chain_id"] = df["chain_id"].fillna("")
        pdb = f"{BG}/input/structures/{r.pdb_file}"
        ctx = AssayContext(model, pdb, df["chain_id"].values[0], a.M, dev)
        bs = max(8, min(128, int(3.0e8 / (ctx.L * 48 * 256))))

        S = torch.stack([ctx.seq_to_S(eval(x)) for x in df["mutated_sequence"]])
        nm = df["mutant"].map(lambda x: sum(len(v.split(":")) if v.strip() else 0
                                            for v in eval(x).values())).to_numpy()
        keep = np.flatnonzero(nm > 0)
        S_v = S[keep]
        y = lab[lab.DMS_id == dms].DMS_score.to_numpy(float)
        assert len(y) == len(keep), f"{dms}: {len(keep)} vs {len(y)}"

        ok, d = labels_in_packing_order(dms, ctx, pdb, ent)
        iface = (d <= CUT).astype(int)
        h0 = ctx.h_V.clone()                                    # (1, L, 128) frozen original
        h = h0[0].float().cpu().numpy()
        mu, sd = h[ok].mean(0), h[ok].std(0) + 1e-8
        ht = (h - mu) / sd                                      # standardised, all rows
        clf = LogisticRegression(max_iter=3000).fit(ht[ok], iface[ok])
        u = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
        proj = ht[ok] @ u                                       # per-residue component along u
        base_rho = stats.spearmanr(score_all(ctx, S_v, bs), y).statistic

        rng = np.random.default_rng(0)
        dirs = {"iface": u}
        for s_ in a.seeds:
            g = np.random.default_rng(100 + s_).normal(size=len(u)); dirs[f"rand{s_}"] = g / np.linalg.norm(g)
        dirs["shuffled"] = u[rng.permutation(len(u))]

        for alpha in a.alphas:
            # target edit size, set by the interface arm; controls are matched to it
            T = abs(alpha) * np.linalg.norm(proj)
            for name, v in dirs.items():
                pv = ht[ok] @ v
                av = alpha if name == "iface" else np.sign(alpha) * T / (np.linalg.norm(pv) + 1e-12)
                ht2 = ht.copy(); ht2[ok] = ht[ok] + av * pv[:, None] * v[None, :]
                if name == "iface":                              # the identity that defines the op
                    ratio = np.median((ht2[ok] @ u) / (proj + 1e-12))
                    assert abs(ratio - (1 + alpha)) < 1e-4, (ratio, 1 + alpha)
                h2 = ht2 * sd + mu
                ctx.set_h_V(torch.tensor(h2, dtype=h0.dtype, device=dev)[None])
                sc = score_all(ctx, S_v, bs)
                p2 = clf.decision_function(ht2[ok])
                rows.append(dict(DMS_id=dms, arm=name, alpha=alpha, alpha_eff=float(av),
                                 eps=float(np.linalg.norm(ht2[ok] - ht[ok]) / np.linalg.norm(ht[ok])),
                                 rho=stats.spearmanr(sc, y).statistic, rho_base=base_rho,
                                 auc=roc_auc_score(iface[ok], p2),
                                 ap=average_precision_score(iface[ok], p2)))
                ctx.set_h_V(h0)
                print(f"{dms:34s} {name:9s} a={alpha:+.1f} a_eff={av:+.3f} eps={rows[-1]['eps']:.3f} "
                      f"rho={rows[-1]['rho']:.4f} (base {base_rho:.4f}, {rows[-1]['rho']-base_rho:+.4f})",
                      flush=True)
            pd.DataFrame(rows).to_csv(f"{OUT}/{a.tag}_intervention.csv", index=False)

    t = pd.DataFrame(rows); t["delta"] = t.rho - t.rho_base
    t.to_csv(f"{OUT}/{a.tag}_intervention.csv", index=False)
    json.dump(vars(a), open(f"{OUT}/{a.tag}_config.json", "w"), indent=1)
    print("\n=== per-arm mean delta rho, by alpha ===")
    print(t.pivot_table(index="alpha", columns="arm", values="delta", aggfunc="mean")
          .to_string(float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
