# Spec: 让 token head 参与推理打分（方案 B）

## 背景

模型是 BERT 类 encoder + 两个 head：
- sentence head：pooled representation → 2-way 分类（当前唯一的推理路径）
- token head：逐 token 二分类，训练时以 `loss = CE_sentence + λ * BCE_token` 的形式存在

问题：长上下文下召回显著下降。假设原因是 pooling 把注入片段（30–40 token）
在长序列（数百 token）里稀释掉了；token head 已学到局部可分性，但推理时完全没用上。

本任务：把 token head 接入推理打分，并评估它在长文本上是否优于 sentence head。

**不需要重新训练**，直接加载现有 checkpoint。

## 要实现的内容

### 1. token 聚合打分函数

对每条样本，从 token head 的 logits 得到一个序列级分数：

```
probs      = sigmoid(token_logits)              # [B, L]
valid_mask = attention_mask & (offset_start != offset_end)   # 排除 [CLS]/[SEP]/padding
k          = max(1, min(K, valid_mask.sum(dim=1)))           # 逐样本自适应
score_tok  = 每条样本在 valid 位置上 top-k probs 的均值
```

- `K` 做成参数，默认 8；需要支持在 {1, 4, 8, 16, 32} 上扫。K=1 等价于 max。
- **必须逐样本处理**，不同样本 valid 长度不同；不要在 padding 上取 top-k。

### 2. ensemble 分数

```
score_ens = w * p_sentence + (1 - w) * score_tok
```

`w` 在 valid 集上扫 {0, 0.1, ..., 1.0}，只在 valid 上选，不碰 test。

### 3. 评估脚本

新增一个脚本（不要改训练代码），输入：checkpoint、数据集、K、bucket 边界。

对三种打分方式各算一遍指标：`sentence`（baseline）/ `token_topk` / `ensemble`。

**按输入总长度分桶**，桶边界默认 `<64 / 64–128 / 128–256 / 256–512`，每桶分别报：

| 指标 | 说明 |
|---|---|
| ROC-AUC | 主要看这个，与阈值无关 |
| PR-AUC | 正类稀疏时更敏感 |
| F1 / Precision / Recall | 阈值**只在 valid 上选**，固定后用于 test |
| 样本数 | 每桶 n，桶太小的要标注出来 |

输出一张 markdown 表 + 一个 CSV，保存到 `results/`。

## 验收标准

- 脚本可复现跑通，不改动任何训练代码、不重新训练。
- 产出上面那张「3 种打分 × 4 个长度桶」的表。
- 阈值和 `w` 的选择过程可见（记录在输出里），且明确只用了 valid 集。

## 注意事项

1. **mask 是最容易出错的地方**。特殊 token 的 `offset_mapping` 是 `(0,0)`，padding 也是。
   如果没排干净，top-k 会挑到这些位置，分数完全失真。实现后先打印几条样本的
   valid token 数和原文 token 数核对。
2. 先看 AUC，不要先看 F1。阈值没重新调过，F1 会给出误导性的结论。
3. token head 是当初以辅助 loss 训出来的，可能校准很差 —— 这不影响 AUC，
   也正是要单独扫阈值的原因。
4. 如果 `token_topk` 在长桶上的 AUC 仍随长度衰减，如实报告，不要调参去凑。
   那个结果说明问题不在聚合方式，而在 encoder 表征，属于另一个方向。
