"""E-1: encoder-side TTT on a class-weighted interface objective, then rescore.

The encoder is adapted at test time using ONLY the structure: the per-residue label is
1[d_heavy(r, other entity) <= 5 A], derived from the PDB and the metadata entity partition,
so it is label-free with respect to DMS. The decoder is frozen throughout, so any change in
the reported Spearman is attributable to the encoder's representation.

    L = L_probe + lambda * L_anchor
    L_probe  = -(1/L') sum_r [ w_pos*y_r*log p_r + (1-y_r)*log(1-p_r) ],  w_pos = (1-pi)/pi
    L_anchor = ||h_V - h_V^frozen||_F^2 / (L'*128) + ||h_E - h_E^frozen||_F^2 / (L'*K*128)

Both encoder outputs are anchored. h_E is NOT visible to the probe yet reaches the decoder
along two paths (h_ES into the backward term, h_EX_encoder into the forward term) and holds
K=48 times the elements h_V does, so anchoring h_V alone would let it drift unwatched.

Three things that fail silently if got wrong, all handled here:
  * gap-padded slots. parse_PDB pads crystallographic gaps with an 'X' whose coordinates
    become (0,0,0); they occupy rows in h_V but are not residues. Everything is masked to
    `ok` = (real residue) AND (tied_featurize's own mask).
  * a stale encoder cache. AssayContext caches h_V AND h_E and builds h_EXV_encoder/h_EXV_fw
    from both; set_h_V() rebuilds only the h_V-derived half. Retraining changes both, so the
    context is REBUILT from the trained weights, reusing the same randn.
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
from protein_mpnn_utils import gather_nodes                      # noqa: E402

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


def run_ttt(model, ctx, y, ok, args, log):
    dev = ctx.device
    okt = torch.as_tensor(ok, device=dev)
    yt = torch.as_tensor(y[ok], device=dev)
    pi = float(y[ok].mean())
    w_pos = (1.0 - pi) / max(pi, 1e-9)
    nok = int(ok.sum())

    with torch.no_grad():
        hV0, hE0 = encoder_forward(model, ctx)
    hV0, hE0 = hV0.detach(), hE0.detach()
    K = hE0.shape[2]

    head = nn.Linear(hV0.shape[-1], 1).to(dev)
    # (i) fit the head on the FROZEN h_V first: a randomly initialised head would pour
    # meaningless gradients into the encoder for the first steps.
    opt_h = torch.optim.AdamW(head.parameters(), lr=args.head_lr)
    for _ in range(args.head_steps):
        loss = weighted_bce(head(hV0[0][okt]).squeeze(-1), yt, w_pos)
        opt_h.zero_grad(); loss.backward(); opt_h.step()
    log["head_loss0"] = float(loss.detach())

    # (ii) joint. model stays in eval(): dropout would both randomise the objective and
    # advance the CUDA RNG, which desynchronises the two arms' decoding orders.
    model.eval()
    ps = trainable_encoder_params(model, args.train_features)
    log["n_trainable"] = int(sum(p.numel() for p in ps))
    opt = torch.optim.AdamW(list(ps) + list(head.parameters()), lr=args.lr)
    hist = []
    for step in range(args.steps):
        hV, hE = encoder_forward(model, ctx)
        l_probe = weighted_bce(head(hV[0][okt]).squeeze(-1), yt, w_pos)
        a_V = ((hV[0][okt] - hV0[0][okt]) ** 2).sum() / (nok * hV0.shape[-1])
        a_E = ((hE[0][okt] - hE0[0][okt]) ** 2).sum() / (nok * K * hE0.shape[-1])
        loss = l_probe + args.lam * (a_V + a_E)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            hist.append(dict(step=step, probe=float(l_probe), anchor_V=float(a_V),
                             anchor_E=float(a_E), total=float(loss)))
    log["pi"] = pi; log["w_pos"] = w_pos; log["n_ok"] = nok; log["L"] = int(len(y))
    log["hist"] = hist
    return head


# ----------------------------------------------------------------------------- scoring
@torch.no_grad()
def score_all(ctx, S, batch):
    out = []
    for i in range(0, len(S), batch):
        out.append(ctx.score(S[i:i + batch]).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="*", default=None)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lam", type=float, default=1.0)
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
        base = score_all(ctx, S, a.batch)
        keep = df["DMS_score"].notna().to_numpy()
        rho_base = stats.spearmanr(base[keep], df["DMS_score"].to_numpy()[keep]).correlation

        y, ok, d = interface_labels(ctx, pdb, ent.loc[dms, "entity1"])
        if a.permute_labels >= 0:
            rs = np.random.RandomState(a.permute_labels)
            v = y[ok].copy(); rs.shuffle(v); y = y.copy(); y[ok] = v

        log = {}
        run_ttt(model, ctx, y, ok, a, log)
        # REBUILD the whole cache from the trained weights: h_E changed too, and set_h_V()
        # would silently keep the stale one.
        ctx2 = AssayContext(model, pdb, df["chain_id"].values[0], a.M, dev, randn=randn)
        ttt = score_all(ctx2, S, a.batch)
        rho_ttt = stats.spearmanr(ttt[keep], df["DMS_score"].to_numpy()[keep]).correlation

        same = float(np.abs(ttt - base).max())
        row = dict(DMS_id=dms, n=int(keep.sum()), L=log["L"], n_ok=log["n_ok"], pi=log["pi"],
                   w_pos=log["w_pos"], n_trainable=log["n_trainable"],
                   rho_base=rho_base, rho_ttt=rho_ttt, delta=rho_ttt - rho_base,
                   rho_ref=float(ref14.loc[dms, "rho_base"]) if dms in ref14.index else np.nan,
                   max_abs_score_change=same, lr=a.lr, steps=a.steps, lam=a.lam,
                   permute=a.permute_labels, M=a.M, seed=a.seed)
        rows.append(row)
        print(f"{dms:44s} rho {rho_base:.4f} -> {rho_ttt:.4f}  ({rho_ttt-rho_base:+.4f})   "
              f"ref {row['rho_ref']:.4f}  pi={log['pi']:.3f} w+={log['w_pos']:.1f} "
              f"|dscore|max={same:.3e}  [{time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(f"{out_dir}/{a.tag}_per_assay.csv", index=False)
        json.dump(log["hist"], open(f"{out_dir}/{a.tag}_{dms}_loss.json", "w"), indent=1)

    t = pd.DataFrame(rows)
    print(f"\n=== {a.tag} | n={len(t)} assay | lr={a.lr} steps={a.steps} lam={a.lam} "
          f"permute={a.permute_labels} ===")
    print(f"mean rho_base {t.rho_base.mean():.4f}   mean rho_ttt {t.rho_ttt.mean():.4f}   "
          f"mean delta {t.delta.mean():+.4f}   wins {int((t.delta>0).sum())}/{len(t)}")
    if t.rho_ref.notna().all():
        print(f"baseline vs recorded rho_base: max|d| = {np.abs(t.rho_base-t.rho_ref).max():.4f}")
    if a.steps == 0:
        ok_gate = bool(t.max_abs_score_change.max() < 1e-6)
        print(f"NO-OP GATE {'PASS' if ok_gate else 'FAIL'}: "
              f"max|delta score| = {t.max_abs_score_change.max():.3e} (tolerance 1e-6)")
        sys.exit(0 if ok_gate else 1)


if __name__ == "__main__":
    main()
