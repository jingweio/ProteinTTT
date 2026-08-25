"""Score ProteinGym DMS substitution assays with ProSST (K=2048) and + ProteinTTT.

Reproduction target, Table 2 of "One protein is all you need" (ICLR 2026):

    ProSST (K=2048)                 0.5068
    ProSST (K=2048) + ProteinTTT    0.5087 +- 0.00004

ProSST scores differently from ESM2, so this cannot reuse eval_proteingym.py:

  * one forward pass over the *unmasked* sequence, not one masked pass per
    position -- the log-odds ratio conditions on the full sequence x rather than
    x\\i (Appendix F.2.3 of the ProteinTTT paper).  Transcribed from
    ProteinGym/proteingym/baselines/prosst/compute_fitness.py, which is what
    produced the published leaderboard column.
  * the model additionally consumes a quantized *structure* token per residue.

Structure tokens come from the archive both ProSST and ProteinGym point at for
the leaderboard run (Google Drive 1lSckfPlx7FhzK1FX7EtmmXUOrdiMRerY); they are
not recomputed here, because re-quantizing would risk tokens that differ from
the published ones and there would be no way to tell that apart from a real
effect.
"""

import argparse
import copy
import json
import time
import zlib
from pathlib import Path

import pandas as pd
import torch

from proteinttt.models.prosst import ProSSTTTT, DEFAULT_PROSST_TTT_CFG

MUTANT_COL = "mutant"

# The structure archive predates ProteinGym v1.3, which corrected three UniProt
# identifiers. Sequences are byte-identical across the rename (verified).
PROSST_NAME = {
    "F7YBW8_MESOW_Ding_2023": "F7YBW7_MESOW_Ding_2023",
    "PSAE_PICP2_Tsuboyama_2023_1PSE": "PSAE_SYNP2_Tsuboyama_2023_1PSE",
    "Q6WV12_9MAXI_Somermeyer_2022": "Q6WV13_9MAXI_Somermeyer_2022",
}


def read_fasta(path):
    return "".join(
        line.strip() for line in path.read_text().split("\n")[1:] if line.strip()
    )


def tokenize_structure_sequence(structure_sequence):
    """ProteinGym's tokenize_structure_sequence: shift by 3, wrap in BOS/EOS."""
    shifted = [i + 3 for i in structure_sequence]
    return torch.tensor([[1, *shifted, 2]], dtype=torch.long)


def score_prosst(model, tokenizer, seq, ss_input_ids, mutants, device):
    """ProteinGym's score_protein: a single unmasked forward, then log-odds."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        tokenized = tokenizer([seq], return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ss_input_ids=ss_input_ids,
            labels=input_ids,
        )
        logits = torch.log_softmax(outputs.logits[:, 1:-1, :], dim=-1)[0].float().cpu()
    model.train(was_training)

    vocab = tokenizer.get_vocab()
    scores = []
    for mutant in mutants:
        total = 0.0
        for sub in str(mutant).split(":"):
            wt, idx, mt = sub[0], int(sub[1:-1]) - 1, sub[-1]
            assert seq[idx] == wt, f"wildtype {wt} != {seq[idx]} at {idx}"
            total += (logits[idx, vocab[mt]] - logits[idx, vocab[wt]]).item()
        scores.append(total)
    return scores


def assay_seed(seed, dms_id):
    return (seed * 1_000_003 + zlib.crc32(dms_id.encode())) % (2**31 - 1)


def param_fingerprint(model):
    with torch.no_grad():
        return sum(float((p.detach().double() ** 2).sum()) for p in model.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["baseline", "ttt"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dms_reference", required=True, type=Path)
    p.add_argument("--dms_dir", required=True, type=Path)
    p.add_argument("--prosst_dir", required=True, type=Path,
                   help="unpacked proteingym_benchmark.zip (residue_sequence/, structure_sequence/)")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--assays", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    ref = pd.read_csv(args.dms_reference)
    if args.assays:
        keep = [ref["DMS_id"].iloc[int(w)] if w.isdigit() else w
                for w in args.assays.split(",")]
        ref = ref[ref["DMS_id"].isin(keep)].reset_index(drop=True)
    if args.limit:
        ref = ref.iloc[: args.limit].reset_index(drop=True)

    tag = "ProSST-2048__" + args.mode + (f"__seed{args.seed}" if args.mode == "ttt" else "")
    out_dir = args.out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{tag}.jsonl"
    print(f"tag: {tag}\nout: {out_dir}\nassays: {len(ref)}", flush=True)

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    base_model = AutoModelForMaskedLM.from_pretrained(
        "AI4Protein/ProSST-2048", trust_remote_code=True
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "AI4Protein/ProSST-2048", trust_remote_code=True
    )

    if args.mode == "ttt":
        cfg = copy.deepcopy(DEFAULT_PROSST_TTT_CFG)
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
        # ProSSTForMaskedLM.__init__ requires `config`; ttt_from_pretrained forwards
        # **kwargs to the model class constructor.
        model = ProSSTTTT.ttt_from_pretrained(
            base_model, ttt_cfg=cfg, config=base_model.config
        )
        assert model._ttt_initial_state, "initial state not captured"
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

        stem = PROSST_NAME.get(dms_id, dms_id)
        seq = read_fasta(args.prosst_dir / "residue_sequence" / f"{stem}.fasta")
        assert seq == row["target_seq"].upper(), f"{dms_id}: sequence mismatch"
        sst = [int(i) for i in
               read_fasta(args.prosst_dir / "structure_sequence" / "2048" / f"{stem}.fasta").split(",")]
        assert len(sst) == len(seq), f"{dms_id}: {len(sst)} structure tokens for {len(seq)} residues"
        ss_input_ids = tokenize_structure_sequence(sst).to(device)

        df = pd.read_csv(args.dms_dir / row["DMS_filename"])
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        rec = dict(dms_id=dms_id, seq_len=len(seq), n_variants=int(len(df)),
                   mode=args.mode, seed=args.seed if args.mode == "ttt" else None)

        df["score_pre_ttt"] = score_prosst(
            model, tokenizer, seq, ss_input_ids, df[MUTANT_COL], device
        )
        rec["t_score_pre"] = round(time.time() - t0, 2)
        rec["spearman_pre_ttt"] = float(
            df["DMS_score"].corr(df["score_pre_ttt"], method="spearman")
        )

        if args.mode == "ttt":
            t1 = time.time()
            model.ttt_generator.manual_seed(assay_seed(args.seed, dms_id))
            ttt_out = model.ttt(seq, ss_input_ids=ss_input_ids)
            rec["t_ttt"] = round(time.time() - t1, 2)
            rec["final_loss"] = float(ttt_out["df"]["loss"].iloc[-1])

            t2 = time.time()
            df["score_ttt"] = score_prosst(
                model, tokenizer, seq, ss_input_ids, df[MUTANT_COL], device
            )
            rec["t_score_post"] = round(time.time() - t2, 2)
            rec["spearman_ttt"] = float(
                df["DMS_score"].corr(df["score_ttt"], method="spearman")
            )
            model.ttt_reset()
            fingerprint = param_fingerprint(model)
            rec["reset_ok"] = fingerprint == fingerprint_0
            if not rec["reset_ok"]:
                raise SystemExit(
                    f"ttt_reset() did not restore weights on {dms_id}: "
                    f"{fingerprint!r} != {fingerprint_0!r}"
                )

        rec["t_total"] = round(time.time() - t0, 2)
        rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        cols = [MUTANT_COL, "DMS_score", "score_pre_ttt"] + (
            ["score_ttt"] if args.mode == "ttt" else []
        )
        df[cols].to_csv(out_csv, index=False)
        with open(summary_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{n + 1}/{len(ref)}] {json.dumps(rec)}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
