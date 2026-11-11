---
kind: upgrade-guide
description: "Cobot 原始升降坐标改为从最低位置起算的位移，双臂安装关系采用 USD 数据。"
---

# Cobot 升降坐标与双臂安装关系

[English](guide.md) | 中文

## 变更

新生成的 `cobot-magic` 模型采用厂家 USD 的双臂和升降运动学。原始 `joints.lift` 以及 Cobot `lift` 技能的 `height` 参数，从底盘顶面以上的升降台高度 `[0.35, 1.05]` 米改为从最低位置起算的位移 `[0, 0.60]` 米。初始值从 `0.70` 改为 `0`。Ego 平面机械臂的虚拟 `lift` 仍表示 TCP 高度。

## 迁移

1. 重新创建已保存的 Cobot 实体；内嵌模型不会自动更新。World 与 Ego Runtime 必须使用同一份模型。
2. 将原始升降指令更新到 `[0, 0.60]`。目标臂根高度为 `H` 时，在范围内使用 `H - 0.9732298` 米。这个离地偏移由轮组几何推定，未经实机标定。虚拟 TCP 高度指令保持不变。
3. 复用轨迹前检查新的关节范围和安装关系。[本体配置说明](../../../../steward/embodiments/cobot-magic.yaml)记录了映射和假设。
