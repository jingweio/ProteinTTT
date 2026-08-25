"""Score ProteinGym DMS substitution assays with ESM2 and ESM2 + ProteinTTT.

The reproduction target is Table 2 of "One protein is all you need" (ICLR 2026),
restricted to the two rows the `proteinttt` package ships configs for:

    ESM2 (35M)  0.3211 -> 0.3407 + ProteinTTT
    ESM2 (650M) 0.4139 -> 0.4153 + ProteinTTT

Scoring is the ProteinGym `masked-marginals` log-odds ratio (Eq. 6 in the paper),
re-implemented from
`ProteinGym/proteingym/baselines/esm/compute_fitness.py` so that masked forward
passes can be batched.  The re-implementation is *validated*, not assumed: the
`--mode baseline` output must reproduce ProteinGym's published per-assay Spearman
(see `aggregate_proteingym.py --check_baseline`).
"""

import argparse
import copy
import json
import os
import time
import zlib
from pathlib import Path

import pandas as pd
import torch

import esm
from proteinttt.models.esm2 import (
    ESM2TTT,
    DEFAULT_ESM2_35M_TTT_CFG,
    DEFAULT_ESM2_650M_TTT_CFG,
)
from proteinttt.utils.torch import get_optimal_window

MODELS = {
    "esm2_t12_35M_UR50D": DEFAULT_ESM2_35M_TTT_CFG,
    "esm2_t33_650M_UR50D": DEFAULT_ESM2_650M_TTT_CFG,
}

# ProteinGym resolves `mutant_col` from the reference file; for
# reference_files/DMS_substitutions.csv `DMS_mutant_column` is absent, so this
# is what it falls back to.
MUTANT_COL = "mutant"

# ESM2 / ProteinGym context window.  Matches TTTConfig.crop_size.
MODEL_WINDOW = 1024


def compute_token_probs(model, alphabet, seq, device, batch_size):
    """Masked-marginals log-probabilities, one masked forward pass per token.

    Returns a [L + 2, vocab] float32 CPU tensor indexed like ProteinGym's
    `token_probs[0]`, i.e. including the BOS/EOS positions.

    Identical arithmetic to ProteinGym's loop; the only change is that positions
    sharing a context window are evaluated in one batch.
    """
    _, _, tokens = alphabet.get_batch_converter()([(None, seq)])
    tokens = tokens.to(device)
    n_tok = tokens.size(1)  # len(seq) + 2

    # Group positions by the window ProteinGym would crop for them.
    windows = {}
    for i in range(n_tok):
        if n_tok > MODEL_WINDOW:
            start, end = get_optimal_window(i, n_tok, MODEL_WINDOW)
        else:
            start, end = 0, n_tok
        windows.setdefault((start, end), []).append(i)

    out = torch.empty(n_tok, len(alphabet), dtype=torch.float32)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for (start, end), positions in windows.items():
            window = tokens[:, start:end]
            for c in range(0, len(positions), batch_size):
                chunk = positions[c : c + batch_size]
                batch = window.repeat(len(chunk), 1).clone()
                for row, i in enumerate(chunk):
                    batch[row, i - start] = alphabet.mask_idx
                logits = model(batch)["logits"]
                log_probs = torch.log_softmax(logits, dim=-1)
                for row, i in enumerate(chunk):
                    out[i] = log_probs[row, i - start].float().cpu()
    model.train(was_training)
    return out


def target_region(row):
    """The sequence ProteinGym actually scores, plus its 1-based offset.

    compute_fitness.py truncates `target_seq` to the MSA-covered region whenever
    that region is not the whole sequence.  Only 5 of the 217 substitution assays
    are affected (all with seq_len > 1024), but for those it is the difference
    between reproducing the published per-assay Spearman and missing it by up to
    0.23 -- the model simply sees a different context for the mutated positions.
    """
    seq = row["target_seq"].upper()
    msa_start, msa_end = int(row["MSA_start"]), int(row["MSA_end"])
    if msa_start != 1 or msa_end != len(seq):
        return seq[msa_start - 1 : msa_end], msa_start
    return seq, 1


def score_mutants(mutants, seq, token_probs, alphabet, offset_idx):
    """ProteinGym `label_row`: summed log-odds ratio over mutated positions."""
    scores = []
    for mutant in mutants:
        total = 0.0
        for sub in str(mutant).split(":"):
            wt, idx, mt = sub[0], int(sub[1:-1]) - offset_idx, sub[-1]
            assert seq[idx] == wt, (
                f"listed wildtype {wt} does not match sequence at {idx}: {seq[idx]}"
            )
            total += (
                token_probs[1 + idx, alphabet.get_idx(mt)]
                - token_probs[1 + idx, alphabet.get_idx(wt)]
            ).item()
        scores.append(total)
    return scores


def assay_seed(seed, dms_id):
    """Per-(seed, assay) RNG seed.

    Derived rather than constant so that a single assay can be re-run in
    isolation and reproduce bit-for-bit, while still being independent across
    assays of equal length.
    """
    return (seed * 1_000_003 + zlib.crc32(dms_id.encode())) % (2**31 - 1)


def param_fingerprint(model):
    """Sum of squares over every parameter -- an O(1)-cost reset guard.

    `ttt_reset()` restores modules by deep copy, so this must come back
    bit-identical.  Cheaper than re-scoring the whole assay, and strictly
    stronger: it sees any weight change, not just ones that move Spearman.
    """
    with torch.no_grad():
        return sum(
            float((p.detach().double() ** 2).sum()) for p in model.parameters()
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--mode", required=True, choices=["baseline", "ttt"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dms_reference", required=True, type=Path)
    p.add_argument("--dms_dir", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--score_batch_size", type=int, default=16)
    p.add_argument(
        "--assays",
        default=None,
        help="comma-separated DMS_id or 0-based index; default = all 217",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--steps", type=int, default=None, help="override TTT steps")
    p.add_argument(
        "--pre_score",
        default="every",
        choices=["every", "first", "never"],
        help="in ttt mode, how often to also score the un-customized model. "
        "'every' is a per-assay sanity column; 'first' halves the scoring cost "
        "for large models (the reset guard is the weight fingerprint, not this).",
    )
    p.add_argument(
        "--no_tie_restore",
        action="store_true",
        help="reproduce the shipped behaviour instead: let ttt_reset() untie "
        "lm_head.weight from embed_tokens.weight, so every assay after the first "
        "additionally trains the output projection. Measures what the tying fix costs.",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}", flush=True)

    ref = pd.read_csv(args.dms_reference)
    if args.assays:
        wanted = args.assays.split(",")
        keep = []
        for w in wanted:
            keep.append(ref["DMS_id"].iloc[int(w)] if w.isdigit() else w)
        ref = ref[ref["DMS_id"].isin(keep)].reset_index(drop=True)
    if args.limit:
        ref = ref.iloc[: args.limit].reset_index(drop=True)

    tag = (
        f"{args.model}__{args.mode}"
        + (f"__seed{args.seed}" if args.mode == "ttt" else "")
        + ("__untied" if args.no_tie_restore else "")
    )
    out_dir = args.out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{tag}.jsonl"
    print(f"tag: {tag}\nout: {out_dir}\nassays: {len(ref)}", flush=True)

    base_model, alphabet = getattr(esm.pretrained, args.model)()
    base_model = base_model.eval().to(device)

    if args.mode == "ttt":
        cfg = copy.deepcopy(MODELS[args.model])
        cfg.seed = args.seed
        if args.steps is not None:
            cfg.steps = args.steps
        cfg.logger_level = "WARNING"
        print(
            f"TTTConfig: lr={cfg.lr} batch_size={cfg.batch_size} ags={cfg.ags} "
            f"steps={cfg.steps} loss_kind={cfg.loss_kind} optimizer={cfg.optimizer} "
            f"mask_ratio={cfg.mask_ratio} crop_size={cfg.crop_size}",
            flush=True,
        )
        model = ESM2TTT.ttt_from_pretrained(base_model, ttt_cfg=cfg)
        assert model._ttt_initial_state, "initial state not captured"
        if args.no_tie_restore:
            model._ttt_restore_tied_parameters = lambda tied: None
            print(
                "WARNING: tie restoration disabled -- reproducing the pre-fix "
                "behaviour where assays 2..N also train lm_head.weight",
                flush=True,
            )
        fingerprint_0 = param_fingerprint(model)
        print(f"param fingerprint (pre-TTT): {fingerprint_0!r}", flush=True)
    else:
        model = base_model
        fingerprint_0 = None

    for n, row in ref.iterrows():
        dms_id = row["DMS_id"]
        out_csv = out_dir / f"{dms_id}.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"[{n + 1}/{len(ref)}] {dms_id}: exists, skip", flush=True)
            continue

        seq, offset_idx = target_region(row)
        df = pd.read_csv(args.dms_dir / row["DMS_filename"])
        t0 = time.time()

        do_pre = args.mode == "baseline" or args.pre_score == "every" or (
            args.pre_score == "first" and n == 0
        )
        rec = dict(
            dms_id=dms_id,
            seq_len=len(seq),
            offset_idx=offset_idx,
            n_variants=int(len(df)),
            mode=args.mode,
            seed=args.seed if args.mode == "ttt" else None,
        )
        if do_pre:
            pre = compute_token_probs(
                model, alphabet, seq, device, args.score_batch_size
            )
            df["score_pre_ttt"] = score_mutants(
                df[MUTANT_COL], seq, pre, alphabet, offset_idx
            )
            rec["t_score_pre"] = round(time.time() - t0, 2)
            rec["spearman_pre_ttt"] = float(
                df["DMS_score"].corr(df["score_pre_ttt"], method="spearman")
            )

        if args.mode == "ttt":
            t1 = time.time()
            model.ttt_generator.manual_seed(assay_seed(args.seed, dms_id))
            ttt_out = model.ttt(seq)
            rec["t_ttt"] = round(time.time() - t1, 2)
            rec["final_loss"] = float(ttt_out["df"]["loss"].iloc[-1])

            t2 = time.time()
            post = compute_token_probs(
                model, alphabet, seq, device, args.score_batch_size
            )
            df["score_ttt"] = score_mutants(
                df[MUTANT_COL], seq, post, alphabet, offset_idx
            )
            rec["t_score_post"] = round(time.time() - t2, 2)
            rec["spearman_ttt"] = float(
                df["DMS_score"].corr(df["score_ttt"], method="spearman")
            )
            model.ttt_reset()
            fingerprint = param_fingerprint(model)
            rec["reset_ok"] = fingerprint == fingerprint_0
            if not rec["reset_ok"] and not args.no_tie_restore:
                rec["fingerprint_delta"] = fingerprint - fingerprint_0
                raise SystemExit(
                    f"ttt_reset() did not restore weights on {dms_id}: "
                    f"fingerprint {fingerprint!r} != {fingerprint_0!r}"
                )

        rec["t_total"] = round(time.time() - t0, 2)
        rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        torch.cuda.reset_peak_memory_stats()
        cols = (
            [MUTANT_COL, "DMS_score"]
            + (["score_pre_ttt"] if do_pre else [])
            + (["score_ttt"] if args.mode == "ttt" else [])
        )
        df[cols].to_csv(out_csv, index=False)
        with open(summary_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{n + 1}/{len(ref)}] {json.dumps(rec)}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
