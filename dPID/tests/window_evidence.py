"""Does the changed loss actually widen the decoded window? Run and see.

Optimizes each loss over free token logits and then decodes with maximum_subarray,
which this change does not touch at all, reporting the window it selects.

    python tests/window_evidence.py

What this is evidence of, and what it is not. The token logits are free
parameters rather than an encoder's output, so this isolates one question --
what shape does each loss WANT the token scores to take -- and answers it
exactly. That is a necessary condition for the fix to work and not a sufficient
one: an encoder has to produce that shape from real text, and nothing here says
it can. Read it as ruling out the failure where the loss itself prefers the
degenerate window, which is the failure the deployed model exhibits.
"""
import sys, torch
from torch.nn import functional as F
sys.path.insert(0, "/tmp/shim"); sys.path.insert(0, ".")
from segment_scoring import (maximum_subarray, region_losses, _region_half,
                             stable_sequence_asl)

L, SPAN_A, SPAN_B, N_BENIGN = 300, 100, 135, 30
TAU, LAM, STEPS = 2.0, 0.1, 400
torch.manual_seed(0)

valid = torch.ones(1 + N_BENIGN, L, dtype=torch.bool)
malicious = torch.zeros_like(valid); malicious[0, SPAN_A:SPAN_B] = True
benign_mask = valid & ~malicious
labels = torch.zeros(1 + N_BENIGN); labels[0] = 1.0


def old_region(z, mal, ben, k_mal, k_ben):
    """region_loss_per_sample as it stands today: per-sample, then batch mean."""
    def region(mask, k, positive):
        mask = mask.bool()
        count = mask.sum(dim=1).clamp(max=int(k))
        selected = z.float().masked_fill(~mask, -torch.inf).topk(min(int(k), z.shape[1]), dim=1).values
        ranks = torch.arange(selected.shape[1])[None, :]
        pooled = selected.masked_fill(ranks >= count[:, None], 0).sum(dim=1) / count.clamp(min=1)
        losses = F.softplus(-pooled if positive else pooled)
        return torch.where(count > 0, losses, torch.zeros_like(losses))
    return (0.5 * region(mal, k_mal, True) + 0.5 * region(ben, k_ben, False)).mean()


def run(name, use_coverage, coverage=0.5, k_mal=3, gamma_pos=1.0):
    z = (torch.randn(1 + N_BENIGN, L) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(STEPS):
        scores, _, _ = maximum_subarray(z, valid, TAU)
        loss = stable_sequence_asl(scores, labels, gamma_pos)
        if use_coverage:
            p, ph, n, nh = region_losses(z, malicious, benign_mask, coverage, 32)
            loss = loss + LAM * (0.5 * _region_half(p, ph) + 0.5 * _region_half(n, nh))
        else:
            loss = loss + LAM * old_region(z, malicious, benign_mask, k_mal, 32)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        scores, start, end = maximum_subarray(z, valid, TAU)
    s, e = int(start[0]), int(end[0])
    hit = len(set(range(s, e)) & set(range(SPAN_A, SPAN_B)))
    union = len(set(range(s, e)) | set(range(SPAN_A, SPAN_B)))
    return {
        "name": name, "width": e - s, "span": SPAN_B - SPAN_A,
        "iou": hit / union, "cover": hit / (SPAN_B - SPAN_A),
        "leak": (e - s - hit),
        "margin_mal": float(scores[0]), "margin_ben_max": float(scores[1:].max()),
        "gap": float(scores[0] - scores[1:].max()),
    }


print("=" * 88)
print(f"  真实文本 {L} token,payload 在 [{SPAN_A}, {SPAN_B}) 共 {SPAN_B-SPAN_A} 个;batch = 1 恶意 + {N_BENIGN} 良性")
print(f"  优化 {STEPS} 步后,用未改动的 maximum_subarray 解码")
print("=" * 88)
rows = [
    run("旧: k=3 常数 + batch-mean", False),
    run("只修归一化: k=3 + 正确权重", True, coverage=3/35),
    run("改目标: coverage=0.25", True, coverage=0.25),
    run("改目标: coverage=0.5", True, coverage=0.5),
    run("改目标: coverage=0.75", True, coverage=0.75),
    run("coverage=0.5 + gamma_pos=0", True, coverage=0.5, gamma_pos=0.0),
]
print(f"{'配置':<30}{'窗口宽':>8}{'IoU':>8}{'覆盖率':>9}{'越界':>7}"
      f"{'恶意分':>10}{'良性最高':>10}{'间隔':>9}")
print("-" * 88)
for r in rows:
    print(f"  {r['name']:<28}{r['width']:>8}{r['iou']:>8.3f}{r['cover']:>9.1%}{r['leak']:>7}"
          f"{r['margin_mal']:>10.2f}{r['margin_ben_max']:>10.2f}{r['gap']:>9.2f}")
print(f"\n  payload 真实长度 = {SPAN_B - SPAN_A}。IoU=1.0 表示窗口与 payload 完全重合。")
