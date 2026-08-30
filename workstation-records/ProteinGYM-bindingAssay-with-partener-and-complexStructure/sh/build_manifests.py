#!/usr/bin/env python
"""Build the two ESMFold2 manifests from refs/partner_sequences.fasta.

  monomer.csv : id,chains,seq_len,seqs        one chain  = ProteinGym target_seq
  complex.csv : id,chains,seq_len,seqs        chain A = target, chain B = full partner

Read-only w.r.t. ProteinGym / BindingGYM. Writes only under --out.
"""
import argparse, csv, os

ap = argparse.ArgumentParser()
ap.add_argument("--fasta", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

seqs = {}
lines = open(a.fasta).read().strip().split("\n")
for i in range(0, len(lines), 2):
    h = lines[i][1:].split("|")
    seqs[(h[0], h[1])] = lines[i + 1]
assays = sorted({k[0] for k in seqs})

with open(os.path.join(a.out, "monomer.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "chains", "seq_len", "seqs"])
    for x in assays:
        t = seqs[(x, "target")]
        w.writerow([x, "A", len(t), t])

with open(os.path.join(a.out, "complex.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "chains", "seq_len", "seqs"])
    for x in assays:
        t, p = seqs[(x, "target")], seqs[(x, "partner")]
        w.writerow([x, "A:B", len(t) + len(p), f"{t}:{p}"])

for n in ("monomer.csv", "complex.csv"):
    rows = list(csv.DictReader(open(os.path.join(a.out, n))))
    print(f"{n}: {len(rows)} rows, L {min(int(r['seq_len']) for r in rows)}"
          f"..{max(int(r['seq_len']) for r in rows)}")
