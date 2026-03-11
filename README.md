# FR5 + RH56E2 D-Grasp RL

这个仓库是 FR5 + RH56E2 的 MuJoCo 抓取强化学习项目，当前聚焦于 `004_sugar_box`，并复用 D-Grasp 的低层功能性抓取思路。

## 当前状态

- 训练数据已经重映射为 FR5 + RH56E2 可执行格式
- 当前训练直接使用 `18` 条 pose-driven 样本
- 支持本地 Windows 可视化查看
- 已整理为单项目可搬运结构，可独立上传 GitHub
- 已补 Linux 无头训练脚本，后续可以在服务器训练、本地只看结果

## 独立仓库说明

当前仓库已经内置最小必需资产:

- `assets/base_scene`
  - FR5 + RH56E2 基础场景
  - 机器人 mesh
- `assets/dgrasp_bundle`
  - sugar box mesh
  - sugar box 训练标签子集

因此后续只需要这个项目目录，不再依赖同级的其他项目目录。

## 常用入口

### Windows 本地

```powershell
E:\A_mujoco\fr5_rh56e2_dgrasp_rl\run_in_env.ps1 E:\A_mujoco\fr5_rh56e2_dgrasp_rl\view_training_scene.py
E:\A_mujoco\fr5_rh56e2_dgrasp_rl\run_in_env.ps1 E:\A_mujoco\fr5_rh56e2_dgrasp_rl\view_pose_driven_sample.py --best-fit
E:\A_mujoco\fr5_rh56e2_dgrasp_rl\run_in_env.ps1 E:\A_mujoco\fr5_rh56e2_dgrasp_rl\view_eval_policy.py --checkpoint E:\A_mujoco\fr5_rh56e2_dgrasp_rl\checkpoints\pose_driven_stage1_u50_e8\checkpoint_0049.pt
```

### Linux 无头

```bash
./train_headless.sh --run-name pose_driven_stage2
./eval_headless.sh --checkpoint ./checkpoints/pose_driven_stage2/checkpoint_0049.pt --run-name pose_driven_stage2_eval
```

## 目录

- `config/default_task.json`: task、reward、PPO、eval 配置
- `data/`: 当前训练数据
- `assets/`: 当前仓库内置的最小运行资产
- `fr5_rh56e2_dgrasp_rl/`: 核心代码
- `GITHUB_LINUX_HEADLESS_SETUP.md`: GitHub 上传与 Linux 无头训练说明
- `WORKLOG_2026-03-10_CN.md`: 当前阶段工作总结

## 训练接口

当前训练环境使用 pose-driven 目标:

- goal:
  - `wrist_pose_goal_object`
  - `hand_qpos_6`
  - `semantic_sites_goal_object_21`
  - `contact_mask_12`
- action:
  - `3D` wrist 局部平移增量
  - `3D` wrist 局部旋转增量
  - `6D` RH56E2 手指关节增量
- reward:
  - 语义点位姿对齐
  - wrist 位姿对齐
  - 手部关节对齐
  - `12D` 接触匹配
  - 接触力奖励
  - 滑落惩罚
  - 物体/手掌运动惩罚
  - 穿透惩罚

## 依赖

基础依赖见 `requirements.txt`。

`torch` 需要按目标机器单独安装:

- 本地 CPU 可以装 CPU 版
- Linux GPU 服务器建议按 CUDA 版本安装官方对应版

## 说明

运行过程中产生的:

- `build/`
- `logs/`
- `checkpoints/`
- `evals/`
- `replays/`

都属于运行产物，不属于版本化源码。
