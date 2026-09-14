"""E-1: encoder-side TTT, anchored on the SCORE rather than on the representation.

The encoder is adapted at test time using only the structure: the per-residue label is
1[d_heavy(r, other entity) <= 5 A], derived from the PDB and the metadata entity partition,
so it is label-free with respect to DMS. The decoder is frozen, so any change in the
reported Spearman is attributable to the encoder.

    L(theta,psi) = L_probe + lambda * ||s_theta - s_frozen||^2 / var(s_frozen)
    L_probe = -(1/L') sum_r [ w_pos*y_r*log p_r + (1-y_r)*log(1-p_r) ],  w_pos = (1-pi)/pi

The anchor is on the model's OUTPUT, following the form already validated on the decoder
side. Anchoring h_V (and h_E) instead was the previous design and is retired: a Frobenius
penalty constrains how far the representation moves, not whether that movement reaches the
decoder, and it is the score we actually care about. var(s_frozen) is taken over the whole
assay, so lambda is comparable across assays whose score scale differs by ~3.3x.

Consequence for the compute: the anchor requires s_theta, so each training step now runs the
DECODER over a batch of variants and backpropagates through it into the encoder. The old
design only ever ran the encoder.

Three things that fail silently if got wrong, all handled here:
  * gap-padded slots. parse_PDB pads crystallographic gaps with an 'X' whose coordinates
    become (0,0,0); they occupy rows in h_V but are not residues. Everything is masked to
    `ok` = (real residue) AND (tied_featurize's own mask).
  * a stale encoder cache. AssayContext caches h_V AND h_E and builds h_EXV_encoder/h_EXV_fw
    from both. Retraining changes both, so the context is REBUILT from the trained weights,
    reusing the same randn, rather than patched through set_h_V.
  * decoding-order noise. The two arms must share randn, else the paired difference is
    swamped (per-assay seed sigma median 0.0184 vs paired gain sd 0.0037). Training keeps
    model.eval() so dropout neither randomises the objective nor advances the CUDA RNG.
"""
import argparse, copy, json, os, random, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bgmpnn import load_model, AssayContext                      # noqa: E402
from bgpdb import pdb_chain_slots                                # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
from protein_mpnn_utils import gather_nodes, cat_neighbors_nodes  # noqa: E402

BG = os.environ.get("BG_ROOT", "/ibex/user/guoj0f/share/BindingGYM")
HERE = os.path.dirname(os.path.abspath(__file__))
REFS = f"{HERE}/refs"
CUT = 5.0


# ----------------------------------------------------------------------------- encoder
def encoder_forward(model, ctx, n=1):
    """Differentiable rerun of AssayContext._build_encoder_cache's encoder half.

    The M clones are identical, so only the first is needed for the objective; taking n=1
    cuts both memory and compute by M with no change to the result.
    """
    m = model
    X, mask = ctx.X[:n], ctx.mask[:n]
    E, E_idx = m.features(X, mask, ctx.residue_idx[:n], ctx.chain_enc[:n])
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
    h_E = m.W_e(E)
    ma = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    ma = mask.unsqueeze(-1) * ma
    for layer in m.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, ma)
    return h_V, h_E


def trainable_encoder_params(model, train_features):
    """features maps coordinates to RBF terms -- the basis the structure is represented in.
    Training it would change that basis; the interface encoding is formed by the message
    passing layers, so those plus W_e are the default set."""
    for p in model.parameters():
        p.requires_grad_(False)
    mods = [model.encoder_layers, model.W_e]
    if train_features:
        mods.append(model.features)
    ps = []
    for mod in mods:
        for p in mod.parameters():
            p.requires_grad_(True); ps.append(p)
    return ps


def score_with(ctx, model, h_V, h_E, S_batch, m):
    """ctx.score() but reading CALLER-SUPPLIED h_V/h_E, so gradients reach the encoder.

    h_EXV_encoder and h_EXV_fw derive from BOTH h_V and h_E, so both are rebuilt here; using
    the cached ones would silently score the pretrained representation. Everything else --
    the summed NLL, no WT term, no length normalisation -- matches the official read-out.
    """
    B = S_batch.shape[0]
    E_idx = ctx.E_idx.expand(B, -1, -1)
    hE_b = h_E.expand(B, -1, -1, -1)
    mask = ctx.mask[:1].expand(B, -1)
    h_EX = cat_neighbors_nodes(torch.zeros_like(model.W_s(ctx.S_wt[:1])), h_E, ctx.E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX, ctx.E_idx)
    fw = (ctx.mask_fw[m:m + 1] * h_EXV_encoder)
    bw = ctx.mask_bw[m:m + 1]
    h_S = model.W_s(S_batch)
    h_ES = cat_neighbors_nodes(h_S, hE_b, E_idx)
    hv = h_V.expand(B, -1, -1)
    for layer in model.decoder_layers:
        h_ESV = bw * cat_neighbors_nodes(hv, h_ES, E_idx) + fw
        hv = layer(hv, h_ESV, mask)
    lp = F.log_softmax(model.W_out(hv), dim=-1)
    nll = F.nll_loss(lp.reshape(-1, lp.size(-1)), S_batch.reshape(-1),
                     reduction="none").view(B, -1)
    return -(nll * mask).sum(-1)


# ----------------------------------------------------------------------------- labels
def interface_labels(ctx, pdb_path, entity1, cut=CUT):
    """Per-slot interface label in tied_featurize packing order. No WT alignment needed:
    h_V is indexed by PDB residue, so labels are built directly in PDB space."""
    order = ctx.designed + [c for c in ctx.all_chains if c not in ctx.designed]
    e1 = set(entity1)
    slots, ent_of = [], []
    for c in order:
        s = pdb_chain_slots(pdb_path, c)
        assert len(s) == ctx.chain_lengths[c], f"chain {c}: {len(s)} vs {ctx.chain_lengths[c]}"
        slots += s
        ent_of += [0 if c in e1 else 1] * len(s)
    ent_of = np.asarray(ent_of)
    valid = ctx.mask[0].detach().cpu().numpy() > 0
    assert len(slots) == len(valid), f"{len(slots)} slots vs mask {len(valid)}"
    ok = np.array([s is not None for s in slots]) & valid
    A = np.vstack([slots[i] for i in np.flatnonzero(ok & (ent_of == 0))])
    B = np.vstack([slots[i] for i in np.flatnonzero(ok & (ent_of == 1))])
    d = np.full(len(slots), np.nan)
    for i in np.flatnonzero(ok):
        other = B if ent_of[i] == 0 else A
        d[i] = np.sqrt(((slots[i][:, None, :] - other[None, :, :]) ** 2).sum(-1)).min()
    y = np.zeros(len(slots), dtype=np.float32)
    y[ok] = (d[ok] <= cut).astype(np.float32)
    return y, ok, d


# ----------------------------------------------------------------------------- TTT
def weighted_bce(logit, y, w_pos):
    """Class-weighted so the two classes carry equal total weight. The interface is the
    minority class (pi 0.038-0.339 -> w_pos 25.3-1.95), and the headroom is in AP_norm,
    which every false positive costs directly; unweighted BCE optimises the majority."""
    w = torch.where(y > 0.5, torch.full_like(y, w_pos), torch.ones_like(y))
    return (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum() / w.sum()


def run_ttt(model, ctx, y, ok, S_all, sf_per_m, sf_mean, args, log):
    """sf_per_m: (M, N) frozen scores per decoding order; sf_mean: (N,) their mean.

    var() for the normaliser comes from sf_mean -- the quantity the benchmark reports -- so
    lambda stays comparable across assays whose score scale differs ~3.3x.
    """
    dev = ctx.device
    okt = torch.as_tensor(ok, device=dev)
    yt = torch.as_tensor(y[ok], device=dev)
    pi = float(y[ok].mean())
    w_pos = (1.0 - pi) / max(pi, 1e-9)

    sf_m = torch.as_tensor(sf_per_m, device=dev, dtype=torch.float32)     # (M, N)
    var_sf = float(np.var(sf_mean))
    with torch.no_grad():
        hV0, _ = encoder_forward(model, ctx)
    hV0 = hV0.detach()

    head = nn.Linear(hV0.shape[-1], 1).to(dev)
    # (i) fit the head on the FROZEN h_V first: a randomly initialised head would pour
    # meaningless gradients into the encoder for the first steps.
    opt_h = torch.optim.AdamW(head.parameters(), lr=args.head_lr,
                              weight_decay=args.weight_decay)
    for _ in range(args.head_steps):
        loss = weighted_bce(head(hV0[0][okt]).squeeze(-1), yt, w_pos)
        opt_h.zero_grad(); loss.backward(); opt_h.step()
    log["head_loss0"] = float(loss.detach())

    # (ii) joint. model stays in eval(): dropout would both randomise the objective and
    # advance the CUDA RNG, which desynchronises the two arms' decoding orders.
    model.eval()
    ps = trainable_encoder_params(model, args.train_features)
    log["n_trainable"] = int(sum(p.numel() for p in ps))
    # 🔴 weight_decay must be passed explicitly: AdamW defaults to 0.01, and decay alone --
    # with no probe loss and a near-zero anchor gradient -- moved rho by +0.0115 (job 51897820)
    opt = torch.optim.AdamW(list(ps) + list(head.parameters()), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = np.random.RandomState(args.seed)       # numpy, so the CUDA RNG is left untouched
    n = len(S_all)
    hist = []
    for step in range(args.steps):
        hV, hE = encoder_forward(model, ctx)
        l_probe = weighted_bce(head(hV[0][okt]).squeeze(-1), yt, w_pos)
        # anchor on the OUTPUT: sample a batch of this assay's variants and score them
        # through the decoder with the current encoder. Transductive -- sequences only,
        # never DMS_score.
        sel = rng.choice(n, size=min(args.anchor_batch, n), replace=False)
        ms = [int(rng.randint(ctx.M))] if args.anchor_M == 1 else list(range(ctx.M))
        st = 0.0
        for mm in ms:
            st = st + score_with(ctx, model, hV, hE, S_all[sel], mm)
        st = st / len(ms)
        # compare against the SAME decoding order(s), so the anchor is exactly 0 at step 0
        ref = sf_m[ms][:, sel].mean(0)
        l_anchor = ((st - ref) ** 2).mean() / var_sf
        loss = args.probe_weight * l_probe + args.lam * l_anchor
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            hist.append(dict(step=step, probe=float(l_probe), anchor=float(l_anchor),
                             total=float(loss)))
    log["pi"] = pi; log["w_pos"] = w_pos; log["n_ok"] = int(ok.sum()); log["L"] = int(len(y))
    log["var_s_frozen"] = var_sf
    log["hist"] = hist
    return head


# ----------------------------------------------------------------------------- probe
def probe_apnorm(h, y, ok, seed=0):
    """AP_norm from an INDEPENDENT 5-fold logistic probe on h_V -- the sanity check that the
    training did what it claims. It must not reuse the head trained alongside the encoder:
    that head co-adapted with h_V and would report its own fit, not the representation's.

    AP_norm = (AP - pi)/(1 - pi): AP's random baseline is the prevalence pi, which varies
    3.8%-33.9% across assays, so raw AP is not comparable between them.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score, roc_auc_score
    X, yy = h[ok], y[ok]
    if yy.min() == yy.max():
        return dict(ap_norm=float("nan"), auc=float("nan"), pi=float(yy.mean()))
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    pa = np.zeros(len(yy))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, yy):
        pa[te] = LogisticRegression(max_iter=2000).fit(Xs[tr], yy[tr]).predict_proba(Xs[te])[:, 1]
    pi = float(yy.mean())
    ap = average_precision_score(yy, pa)
    return dict(ap_norm=float((ap - pi) / (1 - pi)), auc=float(roc_auc_score(yy, pa)), pi=pi)


# ----------------------------------------------------------------------------- scoring
@torch.no_grad()
def score_all(ctx, S, batch):
    out = []
    for i in range(0, len(S), batch):
        out.append(ctx.score(S[i:i + batch]).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def score_all_per_m(ctx, S, batch):
    """(M, N) frozen scores, one row per decoding order.

    The anchor must compare like with like. The reported score averages M orders, but a
    training step samples ONE order, and a single order differs from the M-average by far
    more than training ever moves it -- anchoring the single order against the average makes
    the model chase 'imitate the M-average' instead of 'stay near the pretrained score'.
    Measured at step 0, where the anchor must be exactly 0: it was 0.845.
    """
    rows = []
    for m in range(ctx.M):
        vals = []
        for i in range(0, len(S), batch):
            vals.append(ctx.score(S[i:i + batch], m_idx=m).float().cpu())
        rows.append(torch.cat(vals))
    return torch.stack(rows).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="*", default=None)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--anchor_batch", type=int, default=32,
                    help="variants per step used for the score anchor")
    ap.add_argument("--weight_decay", type=float, default=0.0,
                    help="AdamW's default is 0.01, NOT 0. Leaving it unset silently applied "
                         "decay for every run up to 2026-09-15 and produced gains on its own")
    ap.add_argument("--probe_weight", type=float, default=1.0,
                    help="0 = anchor only. The permutation null still bought 88%% of the gain, "
                         "so this asks whether ANY probe signal is needed at all")
    ap.add_argument("--anchor_M", type=int, default=1,
                    help="decoding orders per training step (1 = sample one, else all M)")
    ap.add_argument("--head_lr", type=float, default=1e-2)
    ap.add_argument("--head_steps", type=int, default=300)
    ap.add_argument("--train_features", action="store_true")
    ap.add_argument("--permute_labels", type=int, default=-1,
                    help="permutation null: shuffle y among valid slots with this seed")
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--gpu_assert", default="A100",
                    help="substring the GPU name must contain; local smoke runs pass A4500. "
                         "ibex-usage: local-GPU output is a sanity check only, never reported")
    ap.add_argument("--limit", type=int, default=0, help="smoke only: cap variants per assay")
    ap.add_argument("--probe_qa", action="store_true",
                    help="measure AP_norm with an independent probe before and after TTT")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}", flush=True)
    assert a.gpu_assert in name, f"expected {a.gpu_assert}, got {name}"
    # cu117 wheels on a newer driver: prove a real kernel launch works before trusting anything
    _t = torch.randn(4096, 4096, device=dev); print("cuda check:", float((_t @ _t).sum()), flush=True)

    model = load_model(a.ckpt or f"{BG}/training/cache/v_48_020.pt", dev)
    state0 = copy.deepcopy(model.state_dict())
    ent = pd.read_csv(f"{REFS}/entity_partition.csv").set_index("DMS_id")
    ref14 = pd.read_csv(f"{REFS}/g1e_canonical14.csv").set_index("DMS_id")
    idx = pd.read_csv(f"{BG}/input/BindingGYM.csv")
    assays = a.assays or list(ref14.index)

    out_dir = a.out or f"{os.path.normpath(os.path.join(HERE,'..','..'))}/ibex-records/structure-encoder-TTT/data"
    os.makedirs(out_dir, exist_ok=True)
    rows, t0 = [], time.time()
    for dms in assays:
        model.load_state_dict(state0); model.eval()       # every assay starts from pretrained
        torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
        r = idx[idx.DMS_id == dms].iloc[0]
        df = pd.read_csv(f"{BG}/input/Binding_substitutions_DMS/{r.DMS_filename}")
        df["chain_id"] = df["chain_id"].fillna("")
        assert df.groupby(["POI", "chain_id"]).ngroups == 1, f"{dms}: expected one POI group"
        pdb = f"{BG}/input/structures/{df['pdb_file'].values[0]}"
        ctx = AssayContext(model, pdb, df["chain_id"].values[0], a.M, dev)
        randn = ctx.randn                                  # shared by both arms -> paired

        if a.limit:                                    # smoke only -- never for reported runs
            df = df.head(a.limit).copy()
        S = torch.stack([ctx.seq_to_S(eval(s)) for s in df["mutated_sequence"]])
        sf_per_m = score_all_per_m(ctx, S, a.batch)      # (M, N)
        base = sf_per_m.mean(0)                          # == ctx.score() averaged over M
        keep = df["DMS_score"].notna().to_numpy()
        rho_base = stats.spearmanr(base[keep], df["DMS_score"].to_numpy()[keep]).correlation

        y, ok, d = interface_labels(ctx, pdb, ent.loc[dms, "entity1"])
        if a.permute_labels >= 0:
            rs = np.random.RandomState(a.permute_labels)
            v = y[ok].copy(); rs.shuffle(v); y = y.copy(); y[ok] = v

        log = {}
        if a.probe_qa:
            with torch.no_grad():
                hV_before, _ = encoder_forward(model, ctx)
            q_before = probe_apnorm(hV_before[0].float().cpu().numpy(), y, ok)
        run_ttt(model, ctx, y, ok, S, sf_per_m, base, a, log)
        # REBUILD the whole cache from the trained weights: h_E changed too, and set_h_V()
        # would silently keep the stale one.
        ctx2 = AssayContext(model, pdb, df["chain_id"].values[0], a.M, dev, randn=randn)
        if a.probe_qa:
            q_after = probe_apnorm(ctx2.h_V[0].float().cpu().numpy(), y, ok)
            log["q_before"], log["q_after"] = q_before, q_after
        ttt = score_all(ctx2, S, a.batch)
        rho_ttt = stats.spearmanr(ttt[keep], df["DMS_score"].to_numpy()[keep]).correlation

        same = float(np.abs(ttt - base).max())
        row = dict(DMS_id=dms, n=int(keep.sum()), L=log["L"], n_ok=log["n_ok"], pi=log["pi"],
                   w_pos=log["w_pos"], n_trainable=log["n_trainable"],
                   rho_base=rho_base, rho_ttt=rho_ttt, delta=rho_ttt - rho_base,
                   rho_ref=float(ref14.loc[dms, "rho_base"]) if dms in ref14.index else np.nan,
                   max_abs_score_change=same, lr=a.lr, steps=a.steps, lam=a.lam,
                   permute=a.permute_labels, M=a.M, seed=a.seed,
                   weight_decay=a.weight_decay, probe_weight=a.probe_weight)
        if a.probe_qa:
            row.update(apn_before=q_before["ap_norm"], apn_after=q_after["ap_norm"],
                       apn_delta=q_after["ap_norm"] - q_before["ap_norm"],
                       auc_before=q_before["auc"], auc_after=q_after["auc"])
        rows.append(row)
        print(f"{dms:44s} rho {rho_base:.4f} -> {rho_ttt:.4f}  ({rho_ttt-rho_base:+.4f})   "
              f"ref {row['rho_ref']:.4f}  pi={log['pi']:.3f} w+={log['w_pos']:.1f} "
              f"|dscore|max={same:.3e}  [{time.time()-t0:.0f}s]", flush=True)
        if a.probe_qa:
            print(f"{'':44s}   AP_norm {q_before['ap_norm']:.4f} -> {q_after['ap_norm']:.4f} "
                  f"({q_after['ap_norm']-q_before['ap_norm']:+.4f})   "
                  f"AUC {q_before['auc']:.4f} -> {q_after['auc']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(f"{out_dir}/{a.tag}_per_assay.csv", index=False)
        json.dump(log["hist"], open(f"{out_dir}/{a.tag}_{dms}_loss.json", "w"), indent=1)

    t = pd.DataFrame(rows)
    print(f"\n=== {a.tag} | n={len(t)} assay | lr={a.lr} steps={a.steps} lam={a.lam} "
          f"permute={a.permute_labels} ===")
    print(f"mean rho_base {t.rho_base.mean():.4f}   mean rho_ttt {t.rho_ttt.mean():.4f}   "
          f"mean delta {t.delta.mean():+.4f}   wins {int((t.delta>0).sum())}/{len(t)}")
    if "apn_delta" in t.columns:
        print(f"mean AP_norm {t.apn_before.mean():.4f} -> {t.apn_after.mean():.4f}   "
              f"mean delta {t.apn_delta.mean():+.4f}   up in {int((t.apn_delta>0).sum())}/{len(t)}")
    if t.rho_ref.notna().all():
        print(f"baseline vs recorded rho_base: max|d| = {np.abs(t.rho_base-t.rho_ref).max():.4f}")
    if a.steps == 0:
        ok_gate = bool(t.max_abs_score_change.max() < 1e-6)
        print(f"NO-OP GATE {'PASS' if ok_gate else 'FAIL'}: "
              f"max|delta score| = {t.max_abs_score_change.max():.3e} (tolerance 1e-6)")
        sys.exit(0 if ok_gate else 1)


if __name__ == "__main__":
    main()
