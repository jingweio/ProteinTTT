"""E-1: does the divergence+anchor objective have an operating window?

Both ends of the lambda curve are already known: lambda -> infinity is the pretrained model
(0.3903 on the 14), lambda -> 0 collapses the score onto the interface feature (0.2564).
Only the middle is unknown, and that is the whole question -- if there is no peak, the
objective has no operating window and the approach is refuted, exactly as the WT-likelihood
objective was.

    L = pearson(standardize(w), standardize(s_theta))  +  lambda * ||s_theta - s_frozen||^2 / var(s_frozen)

The M = 5 decoding orders are the official seed-1 ones, fixed for the whole run. Training
samples one of them per step (the plan's M = 1) and anchors against the frozen score AT THAT
SAME ORDER, so the anchor cannot spend its budget penalising decoding-order noise.
Evaluation averages all 5, which is the baseline's protocol.
"""
import argparse, json, os, random, sys, time
import numpy as np, pandas as pd, torch
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bgmpnn import load_model, AssayContext, set_trainable

BG = "/data/guoj0f/share/BindingGYM"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = f"{ROOT}/workstation-records/mutation-landscape-TTT/data"


def build(dms, model, M, seed, dev):
    """Frozen context + variant tensors for one assay. Single (POI, chain_id) group only."""
    r = pd.read_csv(f"{BG}/input/BindingGYM.csv").query("DMS_id == @dms").iloc[0]
    df = pd.read_csv(f"{BG}/input/Binding_substitutions_DMS/{r.DMS_filename}")
    df["chain_id"] = df["chain_id"].fillna("")
    groups = df.groupby(["POI", "chain_id"])
    assert len(groups) == 1, f"{dms}: {len(groups)} (POI, chain_id) groups, not handled here"
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    ctx = AssayContext(model, f"{BG}/input/structures/{r.pdb_file}", df['chain_id'].values[0], M, dev)
    S = torch.stack([ctx.seq_to_S(eval(s)) for s in df["mutated_sequence"]])
    return ctx, S, df


def frozen_scores(ctx, S, batch):
    """s_frozen[v, m] for every variant and every decoding order."""
    out = torch.empty(len(S), ctx.M)
    with torch.no_grad():
        for m in range(ctx.M):
            for i in range(0, len(S), batch):
                out[i:i + batch, m] = ctx.score(S[i:i + batch], m_idx=m).float().cpu()
    return out


def spearman(a, b):
    return stats.spearmanr(a, b).statistic


def run_one(ctx, S, w, y, s_frozen, lam, lr, steps, bs, mode, dev, seed, log=None):
    """One TTT run. Returns per-variant scores at M=5 after training."""
    sd0 = ctx.model.state_dict()
    state = {k: v.detach().clone() for k, v in sd0.items()}
    ps = set_trainable(ctx.model, mode)
    opt = torch.optim.Adam(ps, lr=lr)
    var_f = float(s_frozen.mean(1).var())
    g = torch.Generator().manual_seed(seed)

    # stratified minibatches: a batch whose w has no spread gives L_div no gradient
    bins = np.digitize(w, np.quantile(w, [.2, .4, .6, .8]))
    by_bin = [np.where(bins == b)[0] for b in range(5)]
    by_bin = [b for b in by_bin if len(b)]
    per = max(1, bs // len(by_bin))
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    sf = s_frozen.to(dev)

    ctx.model.train()
    for step in range(steps):
        ix = np.concatenate([b[torch.randint(len(b), (per,), generator=g).numpy()] for b in by_bin])
        m = int(torch.randint(ctx.M, (1,), generator=g))
        s = ctx.score(S[ix], m_idx=m)
        ws = (wt[ix] - wt[ix].mean()) / (wt[ix].std() + 1e-8)
        ss = (s - s.mean()) / (s.std() + 1e-8)
        l_div = (ws * ss).mean()
        l_anc = ((s - sf[ix, m]) ** 2).mean() / var_f
        loss = l_div + lam * l_anc
        opt.zero_grad(); loss.backward(); opt.step()
        if log is not None and (step % 50 == 0 or step == steps - 1):
            log.append(dict(lam=lam, step=step, l_div=float(l_div), l_anchor=float(l_anc)))
    ctx.model.eval()

    with torch.no_grad():
        out = torch.empty(len(S))
        for i in range(0, len(S), 64):
            out[i:i + 64] = ctx.score(S[i:i + 64]).float().cpu()
    ctx.model.load_state_dict(state)          # restore: every lambda starts from pretrained
    return out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="+", required=True)
    ap.add_argument("--lams", nargs="+", type=float,
                    default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--agg", default="max")
    ap.add_argument("--mode", default="decoder")
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--permute_d", action="store_true", help="E-3 null: shuffle d within assay")
    ap.add_argument("--tag", default="e1")
    a = ap.parse_args()

    dev = torch.device("cuda:0"); assert "A100" in torch.cuda.get_device_name(0)
    model = load_model(f"{BG}/training/cache/v_48_020.pt", dev)
    sdt = pd.read_parquet(f"{ROOT}/local-records/binding-sites-analysis/data/variant_site_dists.parquet")
    lab = pd.read_parquet(f"{ROOT}/local-records/binding-sites-analysis-pred/data/variant_labels_with_mpnn.parquet")

    rows, curves, preds = [], [], {}
    for dms in a.assays:
        t0 = time.time()
        ctx, S, df = build(dms, model, a.M, a.seed, dev)
        bs_eff = max(8, min(a.bs, int(3.0e8 / (ctx.L * 48 * 256))))
        sf = frozen_scores(ctx, S, bs_eff)

        # variant rows drop the WT row the same way variant_labels does (n_mut > 0)
        g = sdt[sdt.DMS_id == dms]
        vi = g.vi.to_numpy()
        assert (np.diff(vi) >= 0).all(), f"{dms}: site table not grouped by variant"
        f = np.exp(-g.dist.to_numpy(float) / a.tau)
        st = np.r_[0, np.flatnonzero(np.diff(vi)) + 1]
        if a.agg == "max":     w_var = np.fmax.reduceat(f, st)
        elif a.agg == "sum":   w_var = np.add.reduceat(f, st)
        elif a.agg == "mean":  w_var = np.add.reduceat(f, st) / np.diff(np.r_[st, len(f)])
        else: raise SystemExit(f"agg {a.agg} not implemented")
        y = lab[lab.DMS_id == dms].DMS_score.to_numpy(float)
        # variant_labels keeps only n_mut > 0, which drops the WT row; reproduce that filter
        nm = df["mutant"].map(lambda s: sum(len(v.split(":")) if v.strip() else 0
                                            for v in eval(s).values())).to_numpy()
        keep = np.flatnonzero(nm > 0)
        assert len(keep) == len(y) == len(w_var), \
            f"{dms}: keep {len(keep)} y {len(y)} w {len(w_var)}"
        S_v, sf_v = S[keep], sf[keep]
        base = float(np.mean([spearman(sf_v[:, m].numpy(), y) for m in range(a.M)]))
        base5 = spearman(sf_v.mean(1).numpy(), y)

        if a.permute_d:
            w_var = w_var[np.random.default_rng(a.seed).permutation(len(w_var))]

        log = []
        for lam in a.lams:
            sc = run_one(ctx, S_v, w_var, y, sf_v, lam, a.lr, a.steps, bs_eff,
                         a.mode, dev, a.seed, log)
            rho = spearman(sc, y)
            corr_w = float(np.corrcoef(w_var, sc)[0, 1])
            rows.append(dict(DMS_id=dms, lam=lam, rho=rho, rho_base=base5,
                             gain=rho - base5, corr_w_after=corr_w,
                             corr_w_before=float(np.corrcoef(w_var, sf_v.mean(1).numpy())[0, 1]),
                             ds_rms=float(np.sqrt(((sc - sf_v.mean(1).numpy()) ** 2).mean()))))
            preds[f"{dms}|{lam}"] = sc
            print(f"{dms:38s} lam={lam:<6g} rho={rho:.4f} (base {base5:.4f}, "
                  f"{rho-base5:+.4f})  corr(w,s) {rows[-1]['corr_w_before']:+.3f}->{corr_w:+.3f}")
        curves += log
        print(f"  [{dms}] L={ctx.L} n={len(keep)} bs={bs_eff} M={a.M} "
              f"base(M=1 mean) {base:.4f} base(M=5) {base5:.4f}  {time.time()-t0:.0f}s")

    t = pd.DataFrame(rows); suf = "_null" if a.permute_d else ""
    t.to_csv(f"{OUT}/{a.tag}_lambda_sweep{suf}.csv", index=False)
    pd.DataFrame(curves).to_csv(f"{OUT}/{a.tag}_loss_curves{suf}.csv", index=False)
    np.savez_compressed(f"{OUT}/{a.tag}_preds{suf}.npz", **preds)
    print("\n=== lambda sweep ===")
    print(t.pivot_table(index="lam", columns="DMS_id", values="gain")
          .to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nper-assay best lambda:")
    for dms, g_ in t.groupby("DMS_id"):
        b = g_.loc[g_.rho.idxmax()]
        print(f"  {dms:40s} lam*={b.lam:<6g} rho {b.rho:.4f} vs base {b.rho_base:.4f} "
              f"({b.gain:+.4f})")
    json.dump(vars(a), open(f"{OUT}/{a.tag}_config{suf}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
