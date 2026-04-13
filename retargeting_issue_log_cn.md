# 当前重定向问题整理

这份文件用于汇总前面排查过程中提到的重点问题，方便后续继续改重定向时快速回看。

内容分成两类：

1. 已确认的实现现状
2. 当前真实存在的问题和风险

---

## 1. 当前主线到底是什么

### 问题

当前项目到底走的是哪条重定向主线？还是不是旧的 `ConvertedGoal`？

### 结论

当前真正生效的主线是 `PoseDrivenSample`，入口在：

- `fr5_rh56e2_dgrasp_rl/pose_driven_data.py`
- 核心函数：`prepare_pose_driven_samples`

旧链路：

- `label_conversion.py`
- `converted_goal.py`

现在属于遗留代码，不是当前训练主线。

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py)
- [label_conversion.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/label_conversion.py)
- [converted_goal.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/converted_goal.py)

---

## 2. 当前保存的是动态结果还是静态目标

### 问题

当前重定向是不是把动态仿真后的物体姿态直接保存了？查看回放是不是也在跑动态物理？

### 结论

当前已经改成：

- 候选筛选阶段可以用动态仿真
- 最终保存默认是“固定物体下的静态目标”
- 查看脚本默认也是静态展示，不默认跑动态物理

当前默认保存策略：

- `candidate_screening_mode = hold`
- `saved_target_mode = settled_anchored`

`settled_anchored` 的含义是：

- 借用动态 settle 后更真实的手型和接触几何
- 但再锚定回原始固定物体姿态保存

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L474)
- [task_config.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/task_config.py#L34)
- [default_task.json](/E:/A_mujoco/UR5_INSPARE/config/default_task.json#L94)
- [view_pose_driven_sample.py](/E:/A_mujoco/UR5_INSPARE/view_pose_driven_sample.py)

---

## 3. 当前接触标签到底是不是 12 维

### 问题

之前不是说不再用 12 维接触标签了吗？现在到底用什么？

### 结论

当前主线仍然在用 `12D contact_mask_12`。

更准确地说，不是“弃用 12D”，而是：

- 不再把 “12/12 完全一致” 当唯一标准
- 但 12D 仍然承担目标表示、奖励、观测和接触统计

当前 12 维接触是从原始 16 维接触压出来的。

### 代码位置

- [semantics.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/semantics.py#L17)
- [semantics.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/semantics.py#L108)
- [env.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/env.py#L235)

### 当前风险

- 当前 `12D` 接触在重定向候选搜索里不是主导项
- 它更像后评估统计和 RL 目标

---

## 4. 七个语义点是什么

### 问题

当前重定向真正对齐的是哪些几何目标？

### 结论

当前只保留 7 个语义点：

1. `wrist_mount`
2. `palm_center`
3. `thumb_tip`
4. `index_tip`
5. `middle_tip`
6. `ring_tip`
7. `little_tip`

系统真正对齐的是：

- 手腕位置和朝向
- 掌心位置
- 五个指尖位置

不是人手全部关节角。

### 代码位置

- [semantics.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/semantics.py#L8)
- [semantics.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/semantics.py#L84)

---

## 5. GenHand 到底有没有接进去

### 问题

当前项目是不是只是“借用了 GenHand 的名字”，还是说真的用了 GenHand 的方法？

### 结论

GenHand 确实接进来了，但接入位置不是“最终标签格式”，而是“接触锚点生成和候选优化层”。

当前真正接入的东西有：

1. `ManoLayer`
2. `FCLoss`
3. `ICP`

但没有完整接入：

1. GenHand 原仓训练主线
2. 原仓网络 checkpoint 推理
3. 原仓完整 pipeline

所以现在更准确地说是：

“GenHand 风格的直接重定向”

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L859)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L899)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L914)
- [loss.py](/E:/A_mujoco/Generalised_Human_Grasp_Kinematic_Retargeting/optimisation/loss.py)
- [icp.py](/E:/A_mujoco/Generalised_Human_Grasp_Kinematic_Retargeting/optimisation/icp.py)

---

## 6. MANO 锚点、机器人 pad、12D 接触三层关系

### 问题

现在到底有几套“接触系统”？它们分别干什么？

### 结论

当前有三层接触相关表示：

1. `MANO dense candidate / human anchor`
   人手侧的接触先验，用于构造人手目标锚点

2. `GenHand 5 通道 robot pad`
   机器人手侧的局部表面候选，用于去追人手锚点

3. `12D MuJoCo physical contact`
   真实物理接触统计，用于安全性、接触诊断和 RL 目标

三层不是同一个东西。

### 当前风险

- 这三层很容易被误认为是一层
- 调参时容易把“几何匹配”与“真实物理接触”混在一起

---

## 7. 锚点到底是一点还是一片

### 问题

当前接触锚点到底是一个点，还是一圈点、一片点？

### 结论

最终优化后的接触锚点是单个点，不是一圈点。

但它前面经历了三层变化：

1. 人手侧先有很多 dense candidate 点
2. 聚类后压成少量目标点
3. FC 优化后得到最终 anchor 点

最终 anchor 通常是 `5` 个单点，每个点配一个法向。

机器人侧 pad 才是一片点，不是最终 anchor 本身。

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L1126)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L1486)
- [robot_model.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/robot_model.py#L338)

---

## 8. FC / 力闭合优化到底优化的是谁

### 问题

FC 优化到底是在优化 MANO 还是在优化我的灵巧手？

### 结论

FC 优化不是在优化你的灵巧手，而是在优化“人手侧投影到物体表面的接触锚点”。

也就是：

- 先从 MANO 手表面抽接触候选
- 再在物体表面上优化这些 anchor 点的位置和摩擦锥权重

后面的灵巧手只是去追这些锚点。

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L1379)
- [loss.py](/E:/A_mujoco/Generalised_Human_Grasp_Kinematic_Retargeting/optimisation/loss.py#L8)

---

## 9. 力闭合 loss 是怎么计算的

### 问题

当前的力闭合 / 平衡 loss 是不是只靠感觉写的？需不需要梯度求解？

### 结论

当前不是手写解析梯度，而是 `PyTorch autograd` 自动求导。

优化变量是：

- `x`: 接触点位置
- `w`: 每个接触点 4 条摩擦锥边的权重

loss 主要由这些部分组成：

1. `sdf`
   点要贴着物体表面

2. `lin_ind`
   grasp matrix 不要退化

3. `net_wrench`
   合力和合力矩尽量平衡

4. `int_fc`
   权重不能乱，不能出现明显非法值

5. `e_dist`
   优化后别离初始 anchor 太远

6. `inter_dist`
   各个 anchor 之间别挤得太近

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L1395)
- [loss.py](/E:/A_mujoco/Generalised_Human_Grasp_Kinematic_Retargeting/optimisation/loss.py#L36)
- [loss.py](/E:/A_mujoco/Generalised_Human_Grasp_Kinematic_Retargeting/optimisation/loss.py#L48)
- [loss.py](/E:/A_mujoco/Generalised_Human_Grasp_Kinematic_Retargeting/optimisation/loss.py#L61)
- [loss.py](/E:/A_mujoco/Generalised_Human_Grasp_Kinematic_Retargeting/optimisation/loss.py#L70)
- [loss.py](/E:/A_mujoco/Generalised_Human_Grasp_Kinematic_Retargeting/optimisation/loss.py#L85)

### 当前风险

- 这套 loss 对“人手锚点质量”有帮助
- 但不能当成“机器人最终真实可抓”的充分证明
- 外部 `FCLoss` 里摩擦系数写死成 `0.1`
- 本地 fallback 指标用的是 `0.7`

---

## 10. 锚点聚类到底怎么做

### 问题

当前 anchor 聚类是怎么做的，输入输出是什么？

### 结论

输入：

- `candidate_points_obj`: 一堆候选接触点
- `candidate_normals_obj`: 每个点对应法向
- `robot_contact_count`: 希望最后保留多少个 anchor，当前通常是 `5`

输出：

- 一个压缩后的 anchor 点集，通常是 `5 x 3`

做法：

1. 如果点太少，直接走 fallback，不做 HDBSCAN
2. 正常情况对“法向”和“位置”各做一次 HDBSCAN
3. 只保留双重聚类都不是噪声的点
4. 再在法向簇内部看位置簇，算中心
5. 最后按启发式挑出若干代表点

所以它是一个“位置+法向”的启发式压缩器，不是端到端学习模块。

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L1288)

### 当前风险

- 聚类后 family 标签不再被强约束保留
- 最终 `5` 个 anchor 不一定严格是一指一个

---

## 11. 优化后 anchor 离初始点有多大限制

### 问题

如果 MANO 和我的灵巧手尺寸不一致，优化后的 anchor 能不能离原始点远一点？

### 结论

当前没有硬上限，但有明显的软约束。

主要有两层限制：

1. 候选预筛阶段：
   MANO 候选点投影到物体表面后，距离最好不超过 `3cm`

2. FC 优化阶段：
   每个 anchor 若偏离初始点超过 `1cm`，就开始被罚

所以：

- 不是完全不能偏
- 但系统明显偏向“别离原始 human anchor 太远”

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L81)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L1161)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L1422)

### 当前风险

- `1cm` 的软约束对“MANO 和机器人手尺寸不匹配”的场景可能偏紧
- 当前 anchor 层更偏 human-prior，而不是 robot-aware

---

## 12. robot pad 是怎么参与锚点对齐的

### 问题

自己的灵巧手 contact pad 具体怎么参与锚点匹配？

### 结论

当前你的 pad 不是直接拿来判物理接触，而是拿来做几何匹配。

流程是：

1. 每根手指 family 下面有一片局部采样点和法向
2. 在当前候选姿态下，把这些点变到世界系
3. 再变到物体系
4. 用每个 family 的中心和法向去对齐 human anchor
5. family 对齐后，再在这片 pad 内找最合适的局部点

所以 pad 的作用是：

- 表达“机器人哪些表面点可以去追人手 anchor”
- 不是直接替代真实物理接触

### 代码位置

- [contact_rh56e2.json](/E:/A_mujoco/UR5_INSPARE/assets/base_scene/contact_rh56e2.json)
- [robot_model.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/robot_model.py#L326)
- [robot_model.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/robot_model.py#L338)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L1617)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L1682)

---

## 13. seed 搜索是不是全局最优

### 问题

当前是不是只是在一堆 seed 里挑“最不差”的？是不是很可能错过更优解？

### 结论

是的。当前是：

- 多个 seed
- 每个 seed 做局部 least-squares 优化
- 再从这些局部解里选最好的

所以当前不是全局最优搜索。

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L1792)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L1853)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L2526)

### 当前风险

- seed 覆盖范围有限
- 优化窗口有限
- 很容易只得到局部最优

---

## 14. 当前 12D 接触在优化阶段到底怎么用

### 问题

既然现在还保留 12D 接触，那它在优化阶段是怎么参与的？

### 结论

当前 12D 接触分成两种：

1. `source_contact_mask_12`
   来自人手标签，是目标接触

2. `runtime.get_contact_diagnostics_12()`
   来自 MuJoCo，是机器人当前真实或近似接触

但要注意：

- `source_contact_mask_12` 在 GenHand seed 优化目标里基本没直接驱动优化
- 它主要在候选评估阶段拿来算 `miss / extra / matched`
- 当前排序 key 对这些量也不是最敏感

所以现在 12D 接触更像：

- 统计量
- 安全量
- RL 目标

不太像当前候选搜索的主导量。

### 代码位置

- [robot_model.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/robot_model.py#L554)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L2270)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L2058)

### 当前风险

- 当前 box 主线里，12D 接触不是候选搜索的最强主导项
- 可能导致“几何还行，但目标接触不够对”的候选也能被选中

---

## 15. 当前动态评估的三个阶段

### 问题

当前动态评估阶段到底分几步？每步在做什么？

### 结论

当前动态评估分三层：

1. `commanded`
   刚把姿态摆进去时的状态

2. `settled`
   在固定控制下跑一段物理，让系统自己松弛到近似静平衡

3. `hold`
   shortlist 再做“撤桌后还能不能撑住”的测试

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L2270)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L408)
- [robot_model.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/robot_model.py#L454)

---

## 16. `settle` 是什么意思，为什么会收敛

### 问题

`settle` 具体在做什么，为什么系统还能自己“收敛”下来？

### 结论

`settle` 的本质是：

- 固定住机器人控制目标
- 让带接触、阻尼和 position control 的 MuJoCo 系统自己松弛到一个近似静平衡状态

它不是求解析解，而是跑若干步仿真。

当前之所以通常能收敛，是因为：

1. actuator 一直把系统往目标关节拉
2. 初始速度被清零
3. 有 damping
4. MuJoCo 求解器和接触解算会慢慢把小抖动松掉

### 代码位置

- [robot_model.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/robot_model.py#L454)
- [scene_builder.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/scene_builder.py#L347)

---

## 17. hold 阶段是怎么做的

### 问题

当前 hold test 具体怎么做？

### 结论

hold 流程是：

1. 从 `settled` 状态出发
2. 打开 arm hold mode，机械臂更硬
3. 给 wrist 一个向上的支撑力
4. preload 若干步
5. 直接把桌子降下去
6. 再跑若干步

它在测的是：

- 撤掉桌面支撑后，东西还能不能被当前抓取结构维持住

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L408)
- [robot_model.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/robot_model.py#L198)
- [robot_model.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/robot_model.py#L204)

### 当前风险

- hold 非常苛刻
- 桌子是瞬间降下去的，不是缓慢撤掉
- 这会把不少“姿态已可用于 RL 静态目标”的候选也判得很差

---

## 18. 如果所有 hold 都不好，会发生什么

### 问题

如果 shortlist 里所有 hold test 都不好，会不会没有结果？

### 结论

不会。系统仍然会从这些“都不太好”的候选里选一个最不差的。

也就是说：

- 不会自动回退到 settle best
- 不会自动放弃保存
- 但大概率 `valid_execution=False`

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L2526)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L2713)

### 当前风险

- 当前训练环境不会过滤 `valid_execution=False` 的样本
- 所以“hold 很差的目标”仍然可能进入训练

---

## 19. 当前 settle / hold 阶段的主导指标

### 问题

这些阶段到底看哪些指标，哪些是真正主导的？

### 结论

settle 阶段主要看：

1. 安全性
   - `table_contact`
   - `total_penetration_m`
   - `max_penetration_m`
   - `object_translation_drift_m`
   - `object_rotation_drift_deg`

2. reachability
   - `reach_object_facing_cos`
   - `reach_base_facing_cos`
   - `reach_downward_component`
   - `reach_min_arm_table_clearance_m`

3. GenHand 锚点质量
   - `genhand_fc_*`
   - `anchor_rmse_m`
   - `anchor_assignment_norm_cost`

4. 几何拟合
   - `source_site_rmse_m`
   - `source_semantic_frame_error_deg`
   - `teacher_cost`

5. 基本支撑结构
   - `settled_contact_group_count`
   - `settled_hard_contact_group_count`
   - `settled_has_thumb_opposition`

hold 阶段主要看：

1. `hold_object_drop_m`
2. `hold_object_translation_m`
3. `hold_object_rotation_deg`
4. `hold_table_contact`
5. `hold_hybrid_contact_group_count`
6. `hold_hard_contact_group_count`
7. `hold_has_thumb_opposition`

### 代码位置

- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L2058)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L2213)
- [pose_driven_data.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/pose_driven_data.py#L2270)

### 当前风险

- 当前 `12D target contact match` 不是 settle 排序里的强主导项
- hold 对最终 `valid_execution` 影响过重

---

## 20. 当前训练和当前重定向筛选并不一致

### 问题

为什么有些姿态 hold 很差，但看起来仍然适合做 RL 训练目标？

### 结论

因为当前重定向筛选和训练环境标准并不一致：

- 重定向默认用 `hold` 做候选筛选
- 训练时默认不 drop table
- RL 训练更像“先学到位姿和接触”
- 重定向筛选更像“预先要求接近独立 hold”

所以两边标准是错位的。

### 代码位置

- [train_loop.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/train_loop.py#L240)
- [evaluate.py](/E:/A_mujoco/UR5_INSPARE/fr5_rh56e2_dgrasp_rl/evaluate.py#L34)

### 当前风险

- 现在很容易出现“静态可用，但被 hold 判太差”的样本

---

## 21. 当前最值得后续继续改的点

### 待改问题清单

1. `seed + 局部优化` 覆盖范围有限，当前不是全局最优搜索
2. `12D target contact` 在候选搜索里权重偏弱
3. `anchor` 的 human-prior 偏强，对机器人尺寸差异适应不够
4. `GENHAND_CONTACT_TARGET_TOL_M = 1cm` 可能偏紧
5. `hold` 过于苛刻，不适合直接作为“静态 RL 目标是否可用”的一票否决项
6. 训练环境没有过滤 `valid_execution=False`
7. `required_target_contact_count()` 当前写死成 `2`
8. 配置里仍有旧链路遗留项，容易误导调参

### 建议的后续方向

1. 把 `valid_execution` 拆成：
   - 静态可用
   - 动态可抓

2. 让 anchor 层更 robot-aware

3. 让 12D 接触匹配真正进入候选主排序

4. 让 hold test 更温和，或只作为附加标签

5. 决定训练时是否过滤低质量样本

