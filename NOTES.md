# CC_dPID — 讨论记录

## 任务
二分类：一段英文文本中是否含有 direct prompt injection（benign / malicious）。
小模型（BERT 类）。已有在自造短文本（30–40 token）上训练的 checkpoint。

## 已确定的设计

**数据构造**：一条 payload → 一条样本，长度 ≤512。
- 短样本：裸 payload。
- 长样本：task template（summarize / classification / sentiment）+ carrier（如 email）+ 插入的 payload。
- benign 与 malicious 走**完全相同**的合成与插入流程，标签只由 payload 内容决定。
- 超长时：payload-centered 裁剪（保留包含 payload 的 512 窗口）。
  - 已知取舍：与推理时的盲截断不完全一致。决定保留此方案，因为一对一映射才能精确控制类别比例。
  - 后续若需支持超长输入：在**推理端**加滑窗 + max 聚合，训练侧不动。记入 README limitations。

**span**：插入位置的字符起止偏移已记录并保留。

**比例**：两个正交的轴
- 轴 1：malicious : benign（已在控）
- 轴 2：短 : 长，两类内部取同一值。当前约 2:1，待 ablation 决定。
- 长度分布：benign 与 malicious 已基本一致，非泄漏源。

## 核心问题（当前焦点）

**稀释（dilution）**：payload 约 30–40 token，嵌入 email 等上下文后总长涨到数百 token，
注入信号占比从 100% 降到个位数百分比。模型在短文本上可用，套上长上下文后检测不出来。

待讨论。

## 待议
- 模板设计（任务种类、避免 template ↔ label 相关）
- 攻击家族分类与覆盖
- 训练配置（从旧 checkpoint 继续 vs 重训）
- 评估方案与指标
