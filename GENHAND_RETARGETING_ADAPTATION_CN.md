# GenHand Retargeting Adaptation

## 目标

把 `Generalised_Human_Grasp_Kinematic_Retargeting` 的核心思路迁到当前 `FR5 + RH56E2 + MuJoCo` 项目里，重点不是复刻其感知部分，而是复刻它的重定向算法结构：

1. 以物体坐标系为中心表达抓取参考
2. 先构造人手到机器人可执行的接触锚点目标
3. 再做机器人运动学优化
4. 最后用物理仿真做验证，而不是让物理搜索主导最终姿态

## 与原项目的对应关系

原 GenHand 方法的主线是：

1. 从人手与物体的接触区域中提取候选接触
2. 将接触区域聚类成适合机器人手型的接触锚点
3. 通过稳定抓取和碰撞相关损失优化接触锚点
4. 再通过机器人运动学把这些接触锚点实现出来

当前项目的本地化适配如下：

- 不使用论文里的图像重建、SDF 网络和完整 MANO mesh contact 提取
- 直接复用 D-Grasp 已有的：
  - `final_ee_rel`
  - `final_contacts`
  - `final_obj_pos`
- 把 D-Grasp 的 link/contact 语义改写成当前 RH56E2 的 `12` 个接触组
- 用 MuJoCo 里的 contact proxy 代替论文里针对特定机器人定义的接触点集

## 当前实现结构

当前 `pose_driven_data.py` 中已经加入两条并行的重定向路径：

- `legacy_projection`
  - 原有启发式投影路径
- `genhand_teacher`
  - 新增的 GenHand 风格 teacher 路径

最终流程是：

1. 两条路径都生成候选机器人抓取姿态
2. 都经过相同的 MuJoCo settle / hold 验证
3. 按统一的物理评分规则选出更好的结果

这样做的原因是：

- 现阶段 D-Grasp 提供的是稀疏 keypoint/contact，而不是完整接触面
- `genhand_teacher` 已经开始在部分样本上优于旧方法
- 但还没有稳定到可以完全替代旧路径
- 所以现在采用“teacher proposal + legacy fallback”的稳妥集成方式

## 当前 teacher 路径

### 1. Object-centric grasp representation

全部关键目标都以物体坐标系表达：

- `human contact anchors`
- `human finger directions`
- `wrist pose object`
- `robot contact proxy points object`

### 2. Human contact anchor construction

当前使用 D-Grasp 21 个关键点，在物体坐标系中构造 `12` 个人手接触锚点：

- `palm`
- `thumb proximal / middle / distal`
- `index / middle / ring / little proximal / distal`

其中：

- palm 采用语义 palm center
- 各段手指 contact anchor 使用相邻关键点的 segment center
- 四指 distal 采用中远两段中心的平均

这样比直接拿单个 keypoint 更接近真实“连杆接触”。

### 3. Family-wise contact assignment

没有把 `12` 个接触组 rigidly 逐项绑定，而是在家族内做匹配：

- palm -> palm
- thumb -> thumb 的 3 个接触组
- index -> index 的 2 个接触组
- middle -> middle 的 2 个接触组
- ring -> ring 的 2 个接触组
- little -> little 的 2 个接触组

每次优化评估时：

1. 取该家族激活的人手锚点
2. 取该家族机器人 contact proxy 点
3. 用 Hungarian assignment 做最小距离匹配

这对应了 GenHand 里“contact abstraction + matching”的思想，只是这里用的是 link-level contact proxy，而不是 dense mesh contact set。

### 4. Teacher optimization objective

当前 teacher 路径优化变量为：

- `wrist local translation delta`
- `wrist local rotation delta`
- `hand_qpos_6`

主残差包括：

- anchor position residual
- active-contact distance residual
- finger direction residual
- palm approach residual
- palm / wrist position residual
- penetration residual
- table collision residual
- wrist / hand regularization

### 5. Physics validation

teacher 路径优化完成后，不直接写入数据，而是继续做：

- MuJoCo settle
- hold test

并和 legacy 路径统一比较：

- penetration
- contact match
- hold drop
- hold contact groups
- thumb opposition

## 当前落地文件

核心实现位于：

- `fr5_rh56e2_dgrasp_rl/pose_driven_data.py`

关键输出字段新增：

- `projected_anchor_rmse_m`
- `projected_anchor_max_error_m`
- `projected_anchor_pairs`
- `retarget_method`
- `teacher_cost`

## 当前限制

当前 teacher 路径仍然有两个主要限制：

1. D-Grasp 输入是稀疏 keypoint/contact，不是 dense contact region  
这会让 GenHand 的 contact clustering / force-closure 思路只能做“结构化近似”，不能原样复刻。

2. 机器人是 `FR5 + RH56E2`  
因此在实现阶段必须经过 arm IK，导致 teacher 优化不仅要考虑手-物关系，还要考虑机械臂可达性。这和原论文中的 free-floating hand 假设不同。

## 当前推荐使用方式

当前更适合作为：

- 高质量重定向 teacher 的第一版
- 给后续更强 teacher/student 重定向器打基础

不建议现在就把 legacy 路径完全删掉。

## 后续建议

下一步如果继续增强 teacher 路径，优先级建议是：

1. 为盒状物体加入更强的 face-normal / antipodal 约束
2. 把 `teacher_cost` 和最终 physics score 联合用于 shortlist 排序
3. 只在高质量样本上训练 student retargeter
