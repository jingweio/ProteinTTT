#!/usr/bin/env python
"""正确性 gate:本 harness 的逐 variant 分数 vs anchor 那轮的逐 variant 分数。

判据:rho(mine, anchor) > 0.95 —— 两者只差解码序随机(randn 的 RNG 流不同),
     不应期望逐位相同;若低于 0.95 说明 harness 有实质偏差。
不通过 ⇒ exit 1,让 job 真的失败,而不是伪装成 COMPLETED。
"""
import sys, os, glob
import pandas as pd
from scipy.stats import spearmanr

OUT, ANCHOR, THRESH = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
files = sorted(glob.glob(os.path.join(OUT, "*.csv")))
if not files:
    print("FAIL: 没有任何输出 csv"); sys.exit(1)
print(f"{'assay':<38}{'n':>6}{'rho(mine,anchor)':>19}{'rho_DMS_mine':>14}{'rho_DMS_anchor':>16}{'':>4}")
bad = 0
for f in files:
    b = os.path.basename(f); af = os.path.join(ANCHOR, b)
    if not os.path.exists(af):
        print(f"{b[:-4]:<38}  (无 anchor 对照,跳过)"); continue
    m, a = pd.read_csv(f), pd.read_csv(af)
    if len(m) != len(a):
        print(f"{b[:-4]:<38}  FAIL 行数 {len(m)} vs {len(a)}"); bad += 1; continue
    rp = spearmanr(m.design_score, a.design_score).correlation
    rm = spearmanr(m.DMS_score, m.design_score).correlation
    ra = spearmanr(a.DMS_score, a.design_score).correlation
    ok = "OK" if rp > THRESH else "FAIL"
    if rp <= THRESH: bad += 1
    print(f"{b[:-4]:<38}{len(m):>6}{rp:>19.4f}{rm:>14.4f}{ra:>16.4f}{ok:>6}")
print(f"\n判据 rho(mine,anchor) > {THRESH}")
sys.exit(1 if bad else 0)
