import json, sys, statistics as st
from collections import defaultdict

path = sys.argv[1]
d = json.load(open(path))
buckets = [int(b) for b in d["config"]["buckets"].split(",")]
runs = [r for r in d["runs"] if r["phase"] == "test"]
agg_rows = defaultdict(lambda: defaultdict(list))
for r in runs:
    for b in buckets:
        for k in ("auc", "pr_auc", "neg_p50", "neg_p99", "pos_p50"):
            agg_rows[r["agg"]][(b, k)].append(r["buckets"][str(b)][k])

def m(a, b, k):
    v = agg_rows[a][(b, k)]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

have_split = "buckets_split" in runs[0]
if have_split:
    split_rows = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for b in buckets:
            split_rows[r["agg"]][b].append(r["buckets_split"][str(b)]["pr_auc"])

for metric, name in (("auc", "ROC-AUC"), ("pr_auc", "PR-AUC")):
    print(f"\n### {name}  (mean +- sd over seeds, 30:1)")
    print(f"{'aggregator':<13}" + "".join(f"{('len='+str(b)):>16}" for b in buckets)
          + f"{'drop 48->long':>16}")
    for a in ("mean_pool", "global_topk", "kadane_tau"):
        cells = ""
        for b in buckets:
            mu, sd = m(a, b, metric)
            cells += f"{mu:>11.3f}±{sd:.3f}"
        d0 = m(a, buckets[0], metric)[0] - m(a, buckets[-1], metric)[0]
        print(f"{a:<13}{cells}{d0:>16.3f}")

print("\n### 负样本分数随长度的漂移 (p99 of negative scores, seed mean)")
print(f"{'aggregator':<13}" + "".join(f"{('len='+str(b)):>12}" for b in buckets)
      + f"{'p99 漂移':>14}{'正负间隔(p50)':>16}")
for a in ("mean_pool", "global_topk", "kadane_tau"):
    cells = "".join(f"{m(a,b,'neg_p99')[0]:>12.2f}" for b in buckets)
    drift = m(a, buckets[-1], "neg_p99")[0] - m(a, buckets[0], "neg_p99")[0]
    gap_s = m(a, buckets[0], "pos_p50")[0] - m(a, buckets[0], "neg_p50")[0]
    gap_l = m(a, buckets[-1], "pos_p50")[0] - m(a, buckets[-1], "neg_p50")[0]
    print(f"{a:<13}{cells}{drift:>14.2f}   {gap_s:>6.2f} -> {gap_l:.2f}")

if have_split:
    print("\n### 分散注入压力测试  (训练用连续 payload，测试把 payload 拆成 3 段) PR-AUC")
    print(f"{'aggregator':<13}" + "".join(f"{('len='+str(b)):>16}" for b in buckets)
          + f"{'相对连续的损失':>18}")
    for a in ("mean_pool", "global_topk", "kadane_tau"):
        cells, losses = "", []
        for b in buckets:
            v = split_rows[a][b]
            mu, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
            cells += f"{mu:>11.3f}±{sd:.3f}"
            losses.append(m(a, b, "pr_auc")[0] - mu)
        print(f"{a:<13}{cells}{st.mean(losses):>18.3f}")
