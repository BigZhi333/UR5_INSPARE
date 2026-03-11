# GitHub And Linux Headless Setup

## 1. 当前仓库状态

这个仓库现在已经整理成“单项目可搬运”结构，后续只上传 `fr5_rh56e2_dgrasp_rl` 这一份即可。

已内置的关键资产:

- `assets/base_scene`
  - FR5 + RH56E2 基础机器人场景
  - 基础场景 metadata
  - 所需 robot mesh
- `assets/dgrasp_bundle`
  - `004_sugar_box` 物体 mesh
  - `dexycb_train_labels.pkl` 的 sugar box 子集

因此后续在新的机器上，不再依赖同级的:

- `mujoco_fr5_rh56e2`
- `dgrasp - 副本`

## 2. GitHub 建议保留内容

建议提交:

- `fr5_rh56e2_dgrasp_rl/`
  - `assets/`
  - `config/`
  - `data/`
  - `fr5_rh56e2_dgrasp_rl/`
  - 根目录脚本和文档

建议不提交:

- `build/`
- `logs/`
- `checkpoints/`
- `evals/`
- `.tmp/`
- `.pip_cache/`
- `.torch/`

这些目录已经在 `.gitignore` 里排除了。

## 3. Linux 无头服务器安装

以下命令假设仓库已经 clone 到:

```bash
/data/fr5_rh56e2_dgrasp_rl
```

### 3.1 创建环境

```bash
conda create -n mujoco_rl python=3.10 -y
conda activate mujoco_rl
```

### 3.2 安装 PyTorch

先按服务器的 CUDA 情况单独安装 PyTorch。

如果服务器有 NVIDIA GPU，按官方页面选对应 CUDA 版本安装:

- https://pytorch.org/get-started/locally/

如果只跑 CPU，也可以安装 CPU 版。

### 3.3 安装其余依赖

```bash
cd /data/fr5_rh56e2_dgrasp_rl
pip install -r requirements.txt
```

## 4. Linux 无头训练

MuJoCo 无头训练建议使用 EGL:

```bash
cd /data/fr5_rh56e2_dgrasp_rl
PYTHON_BIN=$(which python) ./train_headless.sh --run-name pose_driven_stage2
```

如果只想先跑短 smoke:

```bash
cd /data/fr5_rh56e2_dgrasp_rl
PYTHON_BIN=$(which python) ./train_headless.sh --run-name smoke_linux --updates 10 --num-envs 8
```

训练输出会写到仓库内部:

- `logs/`
- `checkpoints/`

## 5. Linux 无头评测

```bash
cd /data/fr5_rh56e2_dgrasp_rl
PYTHON_BIN=$(which python) ./eval_headless.sh --checkpoint /data/fr5_rh56e2_dgrasp_rl/checkpoints/pose_driven_stage2/checkpoint_0049.pt --run-name pose_driven_stage2_eval
```

评测输出会写到:

- `evals/<run-name>/eval_summary.json`
- `evals/<run-name>/trajectories.npz`

## 6. 结果传回本地查看

建议从服务器拉回:

- `checkpoints/<run_name>/`
- `evals/<run_name>/`
- `logs/<run_name>/`

示例:

```bash
scp -r user@server:/data/fr5_rh56e2_dgrasp_rl/checkpoints/pose_driven_stage2 E:/A_mujoco/fr5_rh56e2_dgrasp_rl/checkpoints/
scp -r user@server:/data/fr5_rh56e2_dgrasp_rl/evals/pose_driven_stage2_eval E:/A_mujoco/fr5_rh56e2_dgrasp_rl/evals/
scp -r user@server:/data/fr5_rh56e2_dgrasp_rl/logs/pose_driven_stage2 E:/A_mujoco/fr5_rh56e2_dgrasp_rl/logs/
```

## 7. 本地查看结果

Windows 本地可继续使用:

```powershell
E:\A_mujoco\fr5_rh56e2_dgrasp_rl\run_in_env.ps1 E:\A_mujoco\fr5_rh56e2_dgrasp_rl\view_eval_policy.py --checkpoint E:\A_mujoco\fr5_rh56e2_dgrasp_rl\checkpoints\pose_driven_stage2\checkpoint_0049.pt
```

或离线回放:

```powershell
E:\A_mujoco\fr5_rh56e2_dgrasp_rl\run_in_env.ps1 E:\A_mujoco\fr5_rh56e2_dgrasp_rl\replay_episode.py --trajectory-file E:\A_mujoco\fr5_rh56e2_dgrasp_rl\evals\pose_driven_stage2_eval\trajectories.npz
```

## 8. 当前限制

- 这个仓库已经能独立训练和评测 sugar box 任务。
- 如果后面要扩展到其他 D-Grasp 对象，还需要继续把对应 object mesh 和标签子集内置进 `assets/dgrasp_bundle`。
- 当前 `requirements.txt` 没有锁定 PyTorch，因为服务器可能是 CPU 或不同 CUDA 版本，这部分必须按机器环境单独安装。

