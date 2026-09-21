"""
Does the max-subarray + fixed-tau scheme actually fix signal dilution?

A controlled simulation of the CC_dPID setting. It is NOT a claim about the real
data -- it tests one mechanism question that a simulation can answer: as the
benign carrier grows, does the sequence score of the aggregator keep separating
the classes?

Setup mirrors the agreed plan:
  - token head is the only scoring path
  - loss = ASL(seq_logit, label) + lam * (0.5 * pos_region_BCE + 0.5 * neg_region_BCE)
  - region terms are top-k over logits inside / outside the true span
  - benign:malicious = 30:1, ASL gamma_pos=1 gamma_neg=2 clip=0.01

Three aggregators share one encoder architecture, one data set, one budget:
  mean_pool    sentence head on mean-pooled states  (the old approach)
  global_topk  mean of the top-k token logits       (permutation invariant)
  kadane_tau   max over contiguous segments of sum(z - tau)
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

NEG_INF = -1e4


# ---------------------------------------------------------------- data

class Synth:
    """Weak contiguous evidence buried in a benign carrier.

    A single suspicious token is only weakly separable (mu is small relative to
    the unit-variance background), so no per-token rule can decide the sequence.
    Benign carriers carry isolated suspicious tokens at a fixed *rate*, so a
    longer benign sequence contains proportionally more of them -- this is the
    regime where an order-statistic aggregator is most at risk.
    """

    def __init__(self, dim=16, mu_norm=0.9, bg_rate=0.03,
                 payload=(30, 40), seed=0):
        self.dim, self.bg_rate, self.payload = dim, bg_rate, payload
        rng = np.random.default_rng(seed)
        mu = rng.normal(size=dim)
        self.mu = (mu / np.linalg.norm(mu) * mu_norm).astype(np.float32)

    def make(self, n, length, malicious, rng, split=1):
        x = rng.normal(size=(n, length, self.dim)).astype(np.float32)
        span = np.zeros((n, length), dtype=bool)
        # background: isolated suspicious tokens, same rate in both classes
        iso = rng.random((n, length)) < self.bg_rate
        x[iso] += self.mu
        if malicious:
            lo, hi = self.payload
            plen = np.minimum(rng.integers(lo, hi + 1, size=n), length)
            for i in range(n):
                # cut the payload into `split` pieces, place them disjointly
                total = int(plen[i])
                k = max(1, min(split, total))
                cuts = np.sort(rng.choice(np.arange(1, total), size=k - 1,
                                          replace=False)) if k > 1 else np.array([], int)
                sizes = np.diff(np.concatenate([[0], cuts, [total]]))
                free = length - total
                if free < 0:
                    sizes = np.array([length]); k = 1; free = 0
                # random gaps between pieces
                gaps = rng.multinomial(free, np.ones(k + 1) / (k + 1))
                pos = gaps[0]
                for j in range(k):
                    sz = int(sizes[j])
                    x[i, pos:pos + sz] = rng.normal(size=(sz, self.dim)) + self.mu
                    span[i, pos:pos + sz] = True
                    pos += sz + int(gaps[j + 1])
        return x, span


def build_split(synth, n_pos_per_bucket, ratio, buckets, seed, split=1):
    """One example per row; length is fixed within a bucket."""
    rng = np.random.default_rng(seed)
    rows = []
    for L in buckets:
        xp, sp = synth.make(n_pos_per_bucket, L, True, rng, split=split)
        xn, sn = synth.make(n_pos_per_bucket * ratio, L, False, rng)
        for i in range(len(xp)):
            rows.append((xp[i], sp[i], 1, L))
        for i in range(len(xn)):
            rows.append((xn[i], sn[i], 0, L))
    rng.shuffle(rows)
    return rows


def collate(rows, dim):
    n = len(rows)
    L = max(r[0].shape[0] for r in rows)
    x = np.zeros((n, L, dim), dtype=np.float32)
    span = np.zeros((n, L), dtype=bool)
    valid = np.zeros((n, L), dtype=bool)
    y = np.zeros(n, dtype=np.float32)
    ln = np.zeros(n, dtype=np.int64)
    for i, (xi, si, yi, Li) in enumerate(rows):
        x[i, :Li] = xi
        span[i, :Li] = si
        valid[i, :Li] = True
        y[i], ln[i] = yi, Li
    return (torch.from_numpy(x), torch.from_numpy(span),
            torch.from_numpy(valid), torch.from_numpy(y), torch.from_numpy(ln))


# ---------------------------------------------------------------- model

class Encoder(nn.Module):
    """Dilated conv stack: a *local* contextual scorer (receptive field ~29),
    so a token can tell it sits inside a run but cannot see the whole sequence.
    """

    def __init__(self, dim, d=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, d, 5, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(d, d, 5, padding=8, dilation=4), nn.GELU(),
        )
        self.d = d

    def forward(self, x):                      # x [B, L, dim]
        return self.net(x.transpose(1, 2)).transpose(1, 2)   # [B, L, d]


def kadane(v):
    """max over non-empty contiguous segments of sum(v), differentiable, O(L)."""
    z = torch.zeros(v.shape[0], 1, dtype=v.dtype, device=v.device)
    P = torch.cat([z, v.cumsum(1)], 1)                    # [B, L+1]
    pref_min = torch.cummin(P[:, :-1], dim=1).values      # min over P[0..j]
    return (P[:, 1:] - pref_min).max(dim=1).values


class Model(nn.Module):
    def __init__(self, dim, agg, d=24, seq_top_k=3, tau=1.0):
        super().__init__()
        self.enc = Encoder(dim, d)
        self.agg, self.seq_top_k, self.tau = agg, seq_top_k, tau
        self.token_head = nn.Linear(d, 1)
        self.sent_head = nn.Linear(d, 1)

    def token_logits(self, x, valid):
        z = self.token_head(self.enc(x)).squeeze(-1)       # [B, L]
        return z.masked_fill(~valid, NEG_INF)

    def forward(self, x, valid):
        if self.agg == "mean_pool":
            h = self.enc(x)
            m = valid.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return self.sent_head(pooled).squeeze(-1), None
        z = self.token_logits(x, valid)
        if self.agg == "global_topk":
            k = min(self.seq_top_k, int(valid.sum(1).min().item()))
            seq = z.topk(max(k, 1), dim=1).values.mean(1)
        elif self.agg == "kadane_tau":
            v = torch.where(valid, z - self.tau, torch.full_like(z, NEG_INF))
            seq = kadane(v)
        else:
            raise ValueError(self.agg)
        return seq, z


# ---------------------------------------------------------------- losses

def asl(logit, y, gamma_pos=1.0, gamma_neg=2.0, clip=0.01):
    p = torch.sigmoid(logit)
    pos = (1 - p).pow(gamma_pos) * torch.log(p.clamp(min=1e-8))
    pm = (p - clip).clamp(min=0)
    neg = pm.pow(gamma_neg) * torch.log((1 - pm).clamp(min=1e-8))
    return -(y * pos + (1 - y) * neg).mean()


def topk_mean_masked(z, mask, k):
    """mean of the top-k logits inside mask; rows with empty mask -> None flag."""
    has = mask.any(1)
    zz = z.masked_fill(~mask, NEG_INF)
    n = mask.sum(1)
    kk = int(min(k, max(int(n[has].min().item()), 1))) if has.any() else 1
    return zz.topk(kk, dim=1).values.mean(1), has


def region_loss(z, span, valid, y, mal_k, ben_k):
    """0.5 * positive-region BCE + 0.5 * negative-region BCE, per sample.

    Missing term contributes 0 and the other is NOT doubled, so a bare-payload
    sample does not receive twice the localisation weight of a long one.
    """
    mal = y > 0.5
    pos_mask = span & valid & mal.unsqueeze(1)
    neg_mask = valid & ~span                      # carrier of malicious + all of benign
    s_pos, has_pos = topk_mean_masked(z, pos_mask, mal_k)
    s_neg, has_neg = topk_mean_masked(z, neg_mask, ben_k)
    lp = F.binary_cross_entropy_with_logits(
        s_pos, torch.ones_like(s_pos), reduction="none") * has_pos
    ln = F.binary_cross_entropy_with_logits(
        s_neg, torch.zeros_like(s_neg), reduction="none") * has_neg
    return (0.5 * lp + 0.5 * ln).mean()


# ---------------------------------------------------------------- train / eval

def run_epochs(model, train, dim, epochs, bs, lr, lam, mal_k, ben_k, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(train) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train))
        for i in range(0, len(order), bs):
            rows = [train[j] for j in order[i:i + bs]]
            x, span, valid, y, _ = collate(rows, dim)
            seq, z = model(x, valid)
            loss = asl(seq, y)
            if z is not None and lam > 0:
                loss = loss + lam * region_loss(z, span, valid, y, mal_k, ben_k)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def score_all(model, rows, dim, bs=256):
    model.eval()
    out, ys, lens = [], [], []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        x, span, valid, y, ln = collate(chunk, dim)
        seq, _ = model(x, valid)
        out.append(seq.numpy()); ys.append(y.numpy()); lens.append(ln.numpy())
    return np.concatenate(out), np.concatenate(ys), np.concatenate(lens)


def bucket_report(scores, ys, lens):
    rep = {}
    for L in sorted(set(lens.tolist())):
        m = lens == L
        s, y = scores[m], ys[m]
        rep[int(L)] = {
            "n": int(m.sum()),
            "n_pos": int(y.sum()),
            "auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
            "pr_auc": float(average_precision_score(y, s)) if y.sum() else float("nan"),
            "neg_p50": float(np.percentile(s[y == 0], 50)),
            "neg_p99": float(np.percentile(s[y == 0], 99)),
            "pos_p50": float(np.percentile(s[y == 1], 50)) if y.sum() else float("nan"),
        }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["time", "sweep", "full"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--mal_k", type=int, default=3)
    ap.add_argument("--ben_k", type=int, default=8)
    ap.add_argument("--seq_top_k", type=int, default=3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--ratio", type=int, default=30)
    ap.add_argument("--train_pos", type=int, default=70)   # per bucket
    ap.add_argument("--test_pos", type=int, default=100)   # per bucket
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--buckets", default="48,96,192,320")
    ap.add_argument("--mu", type=float, default=0.9)
    ap.add_argument("--bg_rate", type=float, default=0.03)
    ap.add_argument("--eval_split", type=int, default=3)
    ap.add_argument("--out", default="experiments/results.json")
    a = ap.parse_args()

    buckets = [int(b) for b in a.buckets.split(",")]
    synth = Synth(mu_norm=a.mu, bg_rate=a.bg_rate, seed=0)
    dim = synth.dim
    train = build_split(synth, a.train_pos, a.ratio, buckets, seed=1)
    val = build_split(synth, 40, a.ratio, buckets, seed=2)
    test = build_split(synth, a.test_pos, a.ratio, buckets, seed=3)
    test_split = build_split(synth, a.test_pos, a.ratio, buckets, seed=4,
                             split=a.eval_split)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  "
          f"pos_rate {sum(r[2] for r in train)/len(train):.4f}", flush=True)

    def one(agg, tau, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        m = Model(dim, agg, seq_top_k=a.seq_top_k, tau=tau)
        t0 = time.time()
        run_epochs(m, train, dim, a.epochs, a.bs, a.lr, a.lam,
                   a.mal_k, a.ben_k, seed)
        return m, time.time() - t0

    results = {"config": vars(a), "runs": []}

    if a.mode == "time":
        m, dt = one("kadane_tau", a.tau, 0)
        s, y, L = score_all(m, val, dim)
        print(f"one run {dt:.1f}s  val overall AUC {roc_auc_score(y, s):.4f}")
        return

    if a.mode in ("sweep", "full"):
        print("\n=== tau sweep (kadane_tau, validation) ===", flush=True)
        best_tau, best_v = None, -1
        for tau in [0.0, 0.5, 1.0, 2.0, 4.0]:
            m, dt = one("kadane_tau", tau, 0)
            s, y, L = score_all(m, val, dim)
            rep = bucket_report(s, y, L)
            long_pr = rep[buckets[-1]]["pr_auc"]
            overall = roc_auc_score(y, s)
            print(f"  tau={tau:<4} {dt:5.1f}s  val AUC {overall:.4f}  "
                  f"long-bucket PR-AUC {long_pr:.4f}", flush=True)
            results["runs"].append(
                {"phase": "sweep", "agg": "kadane_tau", "tau": tau,
                 "val_auc": float(overall), "buckets": rep})
            if long_pr > best_v:
                best_v, best_tau = long_pr, tau
        print(f"  -> selected tau={best_tau} (long-bucket PR-AUC)", flush=True)
        results["selected_tau"] = best_tau

    if a.mode == "full":
        print("\n=== test, per length bucket, 3 seeds ===", flush=True)
        for agg in ["mean_pool", "global_topk", "kadane_tau"]:
            for seed in range(a.seeds):
                tau = results["selected_tau"] if agg == "kadane_tau" else 0.0
                m, dt = one(agg, tau, seed)
                s, y, L = score_all(m, test, dim)
                rep = bucket_report(s, y, L)
                s2, y2, L2 = score_all(m, test_split, dim)
                rep2 = bucket_report(s2, y2, L2)
                results["runs"].append(
                    {"phase": "test", "agg": agg, "tau": tau, "seed": seed,
                     "secs": dt, "buckets": rep, "buckets_split": rep2})
                prs = " ".join(f"{rep[b]['pr_auc']:.3f}" for b in buckets)
                pr2 = " ".join(f"{rep2[b]['pr_auc']:.3f}" for b in buckets)
                print(f"  {agg:<12} seed{seed} {dt:5.1f}s  PR-AUC: {prs}   "
                      f"| split-payload: {pr2}", flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
