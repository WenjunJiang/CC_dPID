# 旧 vs 新目标函数：真实训练栈 A/B

`run.sh` → `dPID/tests/ab_segment_objective.py`。两臂都跑生产代码（email collator、
make_segment_model、QAT→LoRA、SegmentTrainer、transformers.Trainer），只换代码版本
（d843fb1 vs 当前）和各自独有的超参。共享超参固定为上次 HPO 冠军：
τ=2, λ=0.1, benign_top_k=32, gamma_pos=1。新臂：coverage=0.5, leak=0.1, malicious_per_batch=4。

替身（离线不可得）：随机初始化 2 层 ModernBERT、word-level tokenizer、合成 payload/邮件。
训练 30:1（40 恶意），测试 50:1（80 恶意 / 4000 良性），PR-AUC 按 500:1 重加权。
消融：在解码窗口内反复把证据最强的 token 换成填充词，阈值固定在干净良性分数的 1% FPR。

## 结果（3 seeds 均值）

| 指标 | 旧 | 新 |
|---|---|---|
| PR-AUC（500:1） | 0.421（0.552 / 0.703 / **0.010**） | **0.888**（0.890 / 0.874 / 0.900） |
| 长样本 PR-AUC | 0.236 | **0.828** |
| Recall@1%FPR，删 0/1/2/3 个 token | 0.55 / 0.52 / 0.48 / 0.45 | **0.95 / 0.92 / 0.87 / 0.82** |
| 长样本 Recall，删 0/1/2/3 | 0.42 / 0.38 / 0.32 / 0.31 | **0.92 / 0.86 / 0.79 / 0.71** |
| 恶意中位 margin − 阈值 | 0.30 | **5.75** |
| 恶意窗口宽（token） | 4.8 | 5.6 |
| 窗口与 payload IoU | 0.16 | 0.24 |
| 窗口越界比例 | 0.36 | **0.00** |

## 结论

- 成立：检测质量、长样本、跨 seed 稳定性（旧的 seed 2 完全塌掉）、裕度、窗口不越界。
- **未成立：窗口没有显著变宽**（4.8 → 5.6，IoU 0.24）。自由 logit 的玩具实验里
  coverage=0.5 给出 30/35 的宽度，但接上编码器后没复现。鲁棒性提升主要来自裕度，
  不是宽度。原因待查：pooled mean 在 softplus 下可被少数高分 token 满足；τ=2 对这个
  规模的 logit 可能偏高（HPO 会搜 τ）。
- 本合成数据里旧模型对单 token 删除也不算脆弱（相对降 5%），所以这里测不出
  `ignore→Ignore` 那种 100% 翻转；需要在真实 mmBERT + 真实数据上用 perturb_test 复测。
