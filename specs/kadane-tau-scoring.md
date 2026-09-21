# Spec: 用可学阈值 τ + 最大子段解码替换全局 top-k 聚合

基于已定的方案（token head 为唯一主打分路径，训练与推理共用同一聚合）做一处升级。
其余部分（ASL、region loss、三个 region top-k、数据、encoder、保存恢复）不动。

## 动机

全局 top-k 是**集合**操作，对位置置换不变：3 个高分 token 连成一句，和散落在第
12、200、400 位，分数完全相同。但注入是**一段连续文本**，这个先验现在没被用上。

改成"最可疑的连续区间"的得分，并让区间的起点、终点、长度由模型自己定。

## 主评分路径

```
z          = token_head(encoder(x))          # [L] token logits
z          = mask 掉 padding 和特殊 token
seq_logit  = max over 所有连续区间 (i,j) of  Σ_{t=i..j} (z_t − τ)
```

这个最大化就是最大子段和，**Kadane 算法 O(L)**。τ 是一个**可学的标量参数**：
token 分数高于 τ 就让区间延伸，低于 τ 就收缩，所以区间长度是涌现的，不是超参。

训练和推理走同一段代码、同一个 τ。

## 关键改动一：τ 在 `L_seq` 里必须 detach

```python
seq_logit = kadane(z - tau.detach())      # z 照常回传，只切断 τ
```

理由（不要跳过）：`seq_logit = Σz_t − |区间|·τ`，所以 `∂seq_logit/∂τ = −|区间|`，
τ 的梯度正比于区间长度。malicious 样本区间长（~payload 长度），benign 样本区间常
退化成单个 token，两边力量严重不对等；而且 τ 下降 → 区间变长 → 推 τ 的力更大，是
正反馈，区间会无界外扩。

**只 detach τ，不要 detach z。** z 从 `L_seq` 拿的梯度是主要学习信号。

## 关键改动二：新增 `L_loc`，用 span 监督区间边界

τ detach 之后只剩这一个梯度来源。用真实 span 做结构化 margin
（loss-augmented inference，标准结构化预测写法）：

```
s_t      = +1 若 t ∉ span，−1 若 t ∈ span
L_loc    = max(0,  kadane(z − τ + δ·s)  −  Σ_{t∈span}(z_t − τ))
```

语义：真实 span 必须是得分最高的区间，且留有 margin。它是自校正的——

- τ 偏低 → 区间吸进 span 边界外的 carrier token，外扩区间得分更高 → 推高 τ
- τ 偏高 → 区间在 span 内部就被截断，某个子区间得分更高 → 压低 τ

`∂L_loc/∂τ = −|预测区间| + |真实span|`，是长度的**差**，有界、良态。

**benign 样本 `L_loc = 0`**，不要这一项。它是边界监督项，benign 没有 span 就没有
边界可教；把它当负例压制用会变成一条过强的硬约束（要求每个 token 都低于 τ 至少
δ），压垮召回。负例压制交给 `L_seq` 和 region 负例 loss。

## 总 loss

```
loss = L_seq(seq_logit, y) + λ₁ · L_region + λ₂ · L_loc
```

`L_region` 完全不变，仍用真实 span 和现有的三个 top-k，与 τ、与区间无关。

Kadane 次数：malicious 样本 2 次（预测 + loss-augmented），benign 1 次。都是 O(L)，
相对 encoder 可忽略。推理只跑第 1 次。

## 聚合器要可插拔

```
aggregator = "global_topk" | "kadane_tau"
```

`global_topk` 是现方案，不假设连续性。两者都保留，由 config 选。
这既是对照组，也是后路——如果真实注入不连续，`kadane_tau` 会退化而 `global_topk` 不会。

## 需要你根据现有代码决定的事

- 上面这些怎么接进现有的 model / loss / train loop，尽量小改动、不重构。
- τ 的初始化、要不要单独设学习率。
- `δ` 和 `λ₂` 的取值与 HPO 候选（δ 同时充当 margin，不需要再设一个 m）。
- Kadane 用循环还是向量化实现；batch 内变长怎么处理。
- 两种 aggregator 怎么共享代码。

先读代码，再决定方案，**动手前把打算怎么改说一遍。**

## 注意

- 聚合必须排除 padding 和特殊 token（`offset_mapping == (0,0)`）。实现后抽样核对
  有效 token 数。
- **τ 要跟模型一起保存和恢复**，它是推理路径的一部分，丢了模型就废了。
- **训练时把 τ 的值记进日志。** detach 后 τ 只有一个梯度来源，如果单调漂移不收敛，
  说明 δ 或 λ₂ 要调。这条曲线几乎零成本，但能第一时间发现问题。
- 评估**按输入长度分桶**报 AUC / PR-AUC，重点看长桶；另外画**负样本分数 vs 长度**
  的分位曲线，确认没有随长度上漂。整体平均会把问题盖掉。
- HPO 的选择指标不能用整体指标（短样本占多数会主导），用长桶 PR-AUC 或者
  「短 regime 与长 regime 指标的最小值」。
- 改完长桶指标仍不动，如实报告，不要调参去凑。
