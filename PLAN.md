# 修复计划：为什么模型的证据窗口塌成 2 个 token

## 观察到的症状

- 选中窗口 2 个 token（payload 有 30–40 个）
- 标准注入 `ignore all previous instructions` 的 margin 6.447，决策边界 6.055 —— 裕度 6%
- `ignore` → `Ignore` 翻转；`all_upper` 下 margin 6.447 → −1.852（全线崩）
- HPO 前三名 val_f1 差 0.00067（在选噪声）

## 根因：loss 里没有任何一项的取值依赖于证据区域的宽度

实测三种表示同一条 malicious 样本的方式（τ=2, malicious_top_k=3, λ=0.1, micro_batch=32）：

| 模式 | 窗口 | margin | region 正 | 总 loss | 排名 |
|---|---|---|---|---|---|
| 2 个 token @ z=5.2 | 2 | 6.4 | 0.0426 | 6.94e-05 | 2 |
| 8 个 token @ z=3.5 | 8 | 12.0 | 0.0298 | **4.65e-05** | **1** |
| 35 个 token @ z=2.5 | 35 | 17.5 | 0.0789 | 1.23e-04 | 3 |

排序完全由 **top-3 均值**决定，与宽度无关。margin 最大的反而 loss 最高。
优化器收敛到窄窗口不是 bug，是在忠实最小化这个 loss。

三个相互独立的成因：

1. **`malicious_top_k` 是常数。** 只看 span 里最高的 3 个，第 4–35 个 token 抬不抬
   一分不给。把 3 改成 16 仍然是常数，payload 长度一变又失效。
2. **region 的两半在 batch 层面权重不等。** 逐样本是 0.5/0.5，但 `.mean()` 之后
   benign 样本的正区域为空贡献 0；30:1 下只有 1/31 的样本有正区域。
   实测聚合比 1:117，有效正例系数 = 0.1 × 0.5 / 32 = 0.00156。
3. **ASL 在 margin≈6 已经饱和。** 决策边界处的梯度是 margin=1 时的 0.011%，
   对 margin 6.4 和 17.5 的区分能力（2.8e-06）比 region 项的差异还小一个量级。
   主损失在实际工作点上已经退出，是被缩小 320 倍的辅助项在主导排序。

## 改动

### 目标函数（必须一起做，单独做任何一条都不翻转排序）

- **A. 正区域的 k 随 span 长度走**：`k = ceil(coverage × |span|)`，`coverage` 进 HPO。
  语义从"有 3 个高的就行"变成"span 里过半 token 都要是证据"。
  负区域的 `benign_top_k` 保持常数 —— 那边是 hard negative mining，要的就是最高的几个。
- **B. 两半各按"拥有该区域的样本数"归一化**，而不是按 batch size。
  让 A 的信号真的有分量。

改完之后，上表的排序翻转为 35-token 模式最优（见 tests/test_segment_objective.py）。

### 目标函数（可选加强）

- **C. 窗口外泄漏惩罚**：`relu(sum over (预测窗口 ∖ span) of (z − τ))`。
  只罚窗口伸出 payload，不要求窗口等于 span，因此不依赖精确攻击边界。
  A 从内部推宽，C 从外部封口。默认权重 0，opt-in。

### 选择指标（不改的话前面全白做）

- **D. HPO 不再用 50:1 的 val_f1。** 它已饱和，且看不见 `long_segment_length`。
  换成按长度分桶、用部署先验重加权的 PR-AUC，取各桶最小值。

### 采样（不增加显存）

- **E. batch 内分层采样 + 重要性权重**，`max_micro_batch_size` 保持 32。
  30:1 下 micro_batch=32 有 35% 的批次零正样本；长样本恶意例只占 1.08%，
  约 71% 的批次里没有任何"长上下文中的注入"。
  固定每批 N 条 malicious 并按 `w_pos = p_true / p_batch` 重加权，
  期望不变、显存不变、空正例批次归零。

### 超参（做完上面再谈，优先级最低）

- `gamma_pos` 目前写死 1.0，加进搜索空间可把饱和点从 margin≈6 推到 ≈12
- `weights.malicious` 提高 email 形式占比，缓解长样本稀缺

## 不在本次范围

数据增强 + consistency loss（治大小写/同义词的词表依赖）。
**先修 loss 再做**：现在正区域有效系数 0.0016，喂再多变体模型也学不动；
而且当前的扰动 profile 是在"2 个 token 承载全部分数"的退化状态下测的，
`casing` 100% 翻转里有相当一部分只是宽度问题的表现，修完再测才知道哪些是真的词表依赖。

宽度修复会吸收一部分伤害：扰动单个 token 的 margin 损失约为 `1/窗口宽度`，
从 2 个 token 的 ~50% 降到 17 个的 ~6%。但 `all_upper`、同义词这类
**全局**扰动改变了每个 token 的身份，宽度完全帮不上，仍需增强。

## 文件

| 文件 | 改动 |
|---|---|
| `segment_scoring.py` | A、B、C |
| `segment_training.py` | D |
| `configs/training/peft_benign_exposure_mmbert2_email_segment.yaml` | 搜索空间、选择指标、weights |
| `train_benign_exposure_mmbert2_dilute_email.py` | 三处小改，转发验证比例给 D |
| `tests/test_segment_objective.py` | 新增，把排序翻转写成验收判据 |
| `email_augmentation/`（**未拿到**） | E 需要这个文件才能改 |
