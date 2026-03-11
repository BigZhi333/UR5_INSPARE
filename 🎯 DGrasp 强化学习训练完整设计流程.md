# 🎯 DGrasp 强化学习训练完整设计流程

## 📋 目录

1. 训练数据准备

1. 环境初始化

1. 分阶段训练机制

1. 噪声注入策略

1. 奖励函数设计

1. 训练循环流程

1. 策略优化

1. 代码位置索引

------

## 1. 训练数据准备

### 📂 数据来源

位置: runner.py 第104-166行

*# 加载DexYCB数据集的抓取示范*

*`if* not args.test:`

  `dict_labels = joblib.load("raisimGymTorch/data/dexycb_train_labels.pkl")`

`*else*:`

  `dict_labels = joblib.load("raisimGymTorch/data/dexycb_test_labels.pkl")`

### 🎯 每个样本包含的信息

dict_labels[obj_id] = {

  'final_qpos':    [N, 51]  *# 目标手部配置（位置+关节角度）*

  'final_obj_pos':  [N, 7]  *# 目标物体位姿（位置+四元数）*

  'final_pose':    [N, 48]  *# 目标手部姿态（关节角度）*

  'final_ee':     [N, 63]  *# 目标关键点位置（21点×3D）*

  'final_ee_rel':   [N, 63]  *# 相对关键点位置*

  'final_contact_pos':[N, 48]  *# 目标接触点位置（16点×3D）*

  'final_contacts':  [N, 16]  *# 目标接触点掩码（0/1）*

  

  *# 初始状态*

  'qpos_reset':    [N, 51]  *# 初始手部配置*

  'obj_pose_reset':  [N, 7]  *# 初始物体位姿*

  'obj_w_stacked':  [N]    *# 物体重量*

  'obj_dim_stacked': [N, 3]  *# 物体尺寸*

  'obj_type_stacked': [N]    *# 物体类型（0-3）*

  'obj_idx_stacked': [N]    *# 物体索引（0-20）*

}

### 🔄 数据增强

位置: runner.py 第131-144行

*# 使用 num_repeats 重复每个样本（默认10次）*

final_qpos = np.repeat(final_qpos, num_repeats, 0)

final_obj_pos = np.repeat(final_obj_pos, num_repeats, 0)

*# ... 所有数据都重复 num_repeats 次*

目的: 增加训练数据多样性，每个基础抓取样本会在不同的随机噪声下训练

------

## 2. 环境初始化

### 🎮 环境创建

位置: runner.py 第169-181行

num_envs = cfg['environment']['num_envs'] *# 默认21个并行环境*

cfg['environment']['num_envs'] = num_envs

*# 创建向量化环境*

env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", yaml_config), cfg)

env.load_object(obj_idx_stacked, obj_w_stacked, obj_dim_stacked, obj_type_stacked)

### 🎯 设置目标

位置: runner.py 第268行, Environment.hpp 第358-400行

*# Python层*

env.set_goals(final_obj_pos, final_ee, final_pose, final_contact_pos, final_contacts)

*// C++层 (Environment.hpp:358-400)*

void set_goals(

  const Eigen::Ref<EigenVec>& *goal_obj_pos*,   *// 目标物体位姿*

  const Eigen::Ref<EigenVec>& *ee_goal_pos*,    *// 目标关键点位置*

  const Eigen::Ref<EigenVec>& *goal_pose*,     *// 目标关节角度*

  const Eigen::Ref<EigenVec>& *contact_goal_pos*, *// 目标接触点位置*

  const Eigen::Ref<EigenVec>& *goal_contacts*   *// 目标接触掩码*

) {

  *// 将目标转换到物体坐标系*

  *// 计算相对姿态*

  *// 存储到 final_\* 变量中*

}

### 🔧 PD控制器参数

位置: Environment.hpp 第80-87行

*// PD增益设置*

Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);

jointPgain.head(3).setConstant(50);    *// 手腕位置P增益*

jointDgain.head(3).setConstant(0.1);    *// 手腕位置D增益*

jointPgain.tail(nJoints_).setConstant(50.0); *// 关节P增益*

jointDgain.tail(nJoints_).setConstant(0.2);  *// 关节D增益*

mano_->setPdGains(jointPgain, jointDgain);

------

## 3. 分阶段训练机制 ⭐⭐⭐

DGrasp使用三阶段训练策略，这是项目的核心设计！

### 📍 阶段配置

位置: cfg.yaml 第26-27行

pre_grasp_steps: 60  *# 阶段1：接近阶段（2秒）*

trail_steps: 135    *# 阶段2+3：抓取+提升（4.5秒）*

### 🎯 阶段1: 接近阶段 (Pre-grasp Phase)

时长: 60步 × 0.03秒 = 1.8秒

位置: Environment.hpp 第474-508行

*// 激活腕部引导（root_guided = true）*

if (root_guided) {

  *// 计算物体相对于手腕的位置*

  obj_pos_raisim = final_obj_pos - Obj_Position;

  raisim::matvecmul(init_or_, obj_pos_raisim, act_or_pose);

  

  *// 逐步引导手腕位置*

  actionMean_.head(3) = act_or_pose * min(1.0, 0.001 * root_guiding_counter_);

  actionMean_.head(3) += gc_.head(3);

  

  *// 逐步引导手腕姿态*

  actionMean_.segment(3,3) = rel_obj_pose_ * min(1.0, 0.0005 * root_guiding_counter_);

  actionMean_.segment(3,3) += gc_.segment(3,3);

  

  root_guiding_counter_ += 1;

}

特点:

- 🎯 腕部引导: 自动引导手腕移动到物体附近

- 📈 渐进式: 引导强度随步数线性增加

- 🔒 

  小动作标准差

  :

  

   actionStd_.head(3).setConstant(0.001);   *// 位置std很小*

   actionStd_.segment(3,3).setConstant(0.01); *// 姿态std很小*

目的: 降低学习难度，让策略专注于学习手指动作

### 🎯 阶段2: 抓取阶段 (Grasping Phase)

时长: ~100步

位置: runner.py 第315-338行, Environment.hpp 第417-508行

*# 训练循环*

*for* step *in* range(n_steps): *# n_steps = pre_grasp_steps + trail_steps*

  obs = env.observe()

  action = ppo.act(obs) *# 策略输出动作*

  reward, dones = env.step(action)

  

  *# 奖励包含所有组件*

  reward.clip(*min*=reward_clip) *# -2.0*

  ppo.step(*value_obs*=obs, *rews*=reward, *dones*=dones)

特点:

- 🤖 完全自主: 策略完全控制51个自由度

- 🎁 全奖励: 位置+姿态+接触+力+正则化

- 🔓 

  正常动作标准差

  :

   actionStd_.setConstant(0.015); *// 手指动作*

### 🎯 阶段3: 提升阶段 (Motion Synthesis / Lifting Phase)

时长: ~35步

位置: Environment.hpp 第731-754行, runner_motion.py 第350-355行

*// 在测试/可视化时激活*

void set_root_control() {

  motion_synthesis = true;

  

  *// 大幅增加腕部PD增益*

  jointPgain.head(3).setConstant(500); *// 10倍增益！*

  jointDgain.head(3).setConstant(0.1);

  mano_->setPdGains(jointPgain, jointDgain);

  

  *// 施加向上的力抵消物体重量*

  raisim::matvecmul(init_or_, {0, 0, obj_weight_*10}, up_gen_vec);

  gen_force_.head(3) = up_gen_vec.e();

  mano_->setGeneralizedForce(gen_force_);

}

*# runner_motion.py 第350-355行*

*if* step > grasp_steps:

  *if* not set_guide:

​    env.set_root_control() *# 激活motion synthesis*

​    set_guide = True

特点:

- 🚀 强制提升: 高增益PD控制器拉起手腕

- ⚖️ 重力补偿: 施加向上的力补偿物体重量

- 🎯 保持手指: 手指姿态保持抓取配置

目的: 测试抓取是否成功（物体是否跟随手移动）

### 📊 阶段时间线

|<---- 60步 (1.8s) ---->|<-------- 100步 (3.0s) -------->|<-- 35步 (1.05s) -->|

|            |                 |          |

| 阶段1: 接近     | 阶段2: 抓取           | 阶段3: 提升    |

| (Root Guided)    | (Autonomous)          | (Motion Synthesis)|

|            |                 |          |

| • 腕部自动引导    | • 策略完全控制          | • 高增益提升    |

| • 小动作标准差    | • 学习手指运动          | • 重力补偿     |

| • 接近物体      | • 建立接触            | • 测试抓取     |

|            | • 形成抓取            |          |

------

## 4. 噪声注入策略

### 🎲 初始化噪声

位置: runner.py 第306-314行

*# 每个rollout开始时添加噪声*

random_noise_pos = np.random.uniform(

  [-0.02, -0.02, 0.01], *# 最小值 [x, y, z] 米*

  [0.02, 0.02, 0.01],  *# 最大值*

  (num_envs, 3)

)

random_noise_qpos = np.random.uniform(

  -0.05, 0.05,      *# ±2.86度*

  (num_envs, 48)     *# 48个关节*

)

qpos_noisy_reset = qpos_reset.copy()

qpos_noisy_reset[:, :3] += random_noise_pos[:, :3]  *# 手腕位置噪声*

qpos_noisy_reset[:, 3:] += random_noise_qpos[:, :]  *# 关节角度噪声*

env.reset_state(qpos_noisy_reset, np.zeros((num_envs,51)), obj_pose_reset)

### 🎯 噪声目的

1. 防止过拟合: 避免策略只学会从精确的初始状态抓取

1. 增强鲁棒性: 训练策略适应各种初始条件

1. 模拟现实: 真实机器人无法精确定位

### 📊 噪声范围

| 参数     | 噪声范围  | 物理意义             |
| :------- | :-------- | :------------------- |
| 手腕 x   | ±2cm      | 左右偏移             |
| 手腕 y   | ±2cm      | 前后偏移             |
| 手腕 z   | 0~1cm     | 只向上（不穿透桌面） |
| 关节角度 | ±0.05 rad | ±2.86度              |

------

## 5. 奖励函数设计

### 📊 完整奖励组件

位置: cfg.yaml 第29-57行, Environment.hpp 第529-563行

reward:

 *# 主要奖励*

 pos_reward:     coeff: 2.0  *# 手指位置*

 impulse_reward:   coeff: 2.0  *# 接触力*

 contact_reward:   coeff: 1.0  *# 接触匹配*

 falling_reward:   coeff: 1.0  *# 防掉落*

 pose_reward:    coeff: 0.1  *# 手部姿态*

 

 *# 正则化惩罚*

 rel_obj_reward_:  coeff: -1.0  *# 物体速度*

 body_vel_reward_:  coeff: -0.5  *# 手部线速度*

 body_qvel_reward_: coeff: -0.5  *# 手部角速度*

 torque:       coeff: -0.0  *# 力矩（未启用）*

### 🎁 奖励裁剪

位置: runner.py 第333行, cfg.yaml 第28行

reward_clip = cfg['environment']['reward_clip'] *# -2.0*

reward.clip(*min*=reward_clip)

目的: 防止极端负奖励破坏训练稳定性

------

## 6. 训练循环流程

### 🔄 完整训练循环

位置: runner.py 第278-493行

*for* update *in* range(args.num_iterations): *# 默认3001次迭代*

  

  *# ===== 1. 模型保存 =====*

  *if* update % cfg['environment']['eval_every_n'] == 0: *# 每200次*

​    torch.save({

​      'actor_architecture_state_dict': actor.architecture.state_dict(),

​      'actor_distribution_state_dict': actor.distribution.state_dict(),

​      'critic_architecture_state_dict': critic.architecture.state_dict(),

​      'optimizer_state_dict': ppo.optimizer.state_dict(),

​    }, saver.data_dir + f"/full_{update}.pt")

​    env.save_scaling(saver.data_dir, str(update))

  

  *# ===== 2. 添加初始化噪声 =====*

  random_noise_pos = np.random.uniform([-0.02, -0.02, 0.01], [0.02, 0.02, 0.01], (num_envs, 3))

  random_noise_qpos = np.random.uniform(-0.05, 0.05, (num_envs, 48))

  qpos_noisy_reset = qpos_reset.copy()

  qpos_noisy_reset[:, :3] += random_noise_pos

  qpos_noisy_reset[:, 3:] += random_noise_qpos

  

  *# ===== 3. 重置环境 =====*

  env.reset_state(qpos_noisy_reset, np.zeros((num_envs, 51)), obj_pose_reset)

  

  *# ===== 4. Rollout收集 =====*

  *for* step *in* range(n_steps): *# n_steps = pre_grasp + trail = 195*

​    obs = env.observe().astype('float32')

​    action = ppo.act(obs) *# 采样动作*

​    reward, dones = env.step(action.astype('float32'))

​    

​    *# 收集奖励组件分解*

​    reward_info_list = env.get_reward_info()

​    *for* env_id, info *in* enumerate(reward_info_list):

​      *for* name *in* reward_component_names:

​        reward_components_sum[name] += info.get(name, 0.0)

​    

​    *# 裁剪并存储*

​    reward.clip(*min*=reward_clip)

​    ppo.step(*value_obs*=obs, *rews*=reward, *dones*=dones)

​    

​    reward_ll_sum += np.sum(reward)

​    done_sum += np.sum(dones)

  

  *# ===== 5. 策略更新 =====*

  obs = env.observe().astype('float32')

  infos = ppo.update(

​    *actor_obs*=obs,

​    *value_obs*=obs,

​    *log_this_iteration*=update % 10 == 0,

​    *update*=update

  )

  

  *# ===== 6. 强制最小探索 =====*

  actor.distribution.enforce_minimum_std(

​    (torch.ones(act_dim) * 0.2).to(device)

  )

  

  *# ===== 7. TensorBoard日志 =====*

  *# 记录所有统计信息...*

  runner_writer.add_scalar('Reward/average', average_ll_performance, update)

  *for* name *in* reward_component_names:

​    avg_comp = reward_components_sum[name] / total_steps

​    runner_writer.add_scalar(f'RewardComponent/{name}', avg_comp, update)

  *# ... 更多指标*

### 📊 训练统计

位置: runner.py 第377-438行

记录的关键指标：

*# 总奖励统计*

\- Reward/average, best, std, min, max

*# 每个奖励组件统计*

\- RewardComponent/{pos_reward, pose_reward, contact_reward, ...}

*# Rollout统计*

\- Rollout/reward_mean, std, min, max

\- RolloutComponent/{component}_mean, std

*# Episode统计*

\- Episode/dones, done_count

*# 性能统计*

\- Performance/fps, iter_time_sec

*# 策略统计*

\- Policy/mean_std, std_min, std_max, learning_rate

*# PPO统计（每10次）*

\- PPO/mean_value_loss, mean_surrogate_loss, mean_advantage, ...

------

## 7. 策略优化

### 🧠 网络架构

位置: cfg.yaml 第58-61行, runner.py 第200-203行

architecture:

 policy_net: [128, 128]  *# 2层MLP，每层128维*

 value_net: [128, 128]

*# Actor网络*

actor = ppo_module.Actor(

  ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim, act_dim),

  ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler),

  device

)

*# Critic网络*

critic = ppo_module.Critic(

  ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim, 1),

  device

)

### 🎯 PPO超参数

位置: runner.py 第205-216行

ppo = PPO.PPO(

  *actor*=actor,

  *critic*=critic,

  *num_envs*=21,

  *num_transitions_per_env*=195, *# n_steps*

  *num_learning_epochs*=4,

  *gamma*=0.996,

  *lam*=0.95,

  *num_mini_batches*=4,

  *device*=device,

  *log_dir*=saver.data_dir,

  *shuffle_batch*=False

)

### 📉 学习率调度

位置: PPO实现内部（隐式）

- 初始学习率通常为 3e-4

- 可能使用线性衰减或自适应调整

### 🎲 探索策略

位置: runner.py 第352行

*# 强制最小标准差，保证持续探索*

actor.distribution.enforce_minimum_std(

  (torch.ones(act_dim) * 0.2).to(device)

)

------

## 8. 代码位置索引 📂

### 核心文件结构

dgrasp/raisimGymTorch/

├── raisimGymTorch/

│  ├── env/

│  │  └── envs/

│  │    └── dgrasp/

│  │      ├── Environment.hpp    # C++环境实现 ⭐⭐⭐

│  │      │  ├── 第44-118行: 初始化

│  │      │  ├── 第158-275行: reset()

│  │      │  ├── 第280-356行: reset_state()

│  │      │  ├── 第358-400行: set_goals()

│  │      │  ├── 第417-508行: step() - 三阶段逻辑

│  │      │  ├── 第529-563行: 奖励计算

│  │      │  ├── 第566-722行: updateObservation()

│  │      │  └── 第731-754行: set_root_control()

│  │      │

│  │      ├── cfgs/cfg.yaml     # 配置文件 ⭐⭐

│  │      │  ├── 第4-28行: 环境参数

│  │      │  ├── 第29-57行: 奖励系数

│  │      │  └── 第58-61行: 网络架构

│  │      │

│  │      ├── runner.py       # 训练主脚本 ⭐⭐⭐

│  │      │  ├── 第43-62行:  命令行参数

│  │      │  ├── 第104-166行: 数据加载

│  │      │  ├── 第169-181行: 环境创建

│  │      │  ├── 第200-216行: PPO初始化

│  │      │  ├── 第268行:   设置目标

│  │      │  ├── 第278-493行: 训练循环

│  │      │  ├── 第306-314行: 噪声注入

│  │      │  ├── 第315-338行: Rollout

│  │      │  ├── 第346-352行: PPO更新

│  │      │  └── 第377-438行: 日志记录

│  │      │

│  │      └── runner_motion.py    # 可视化脚本

│  │        └── 第350-355行: 激活motion synthesis

│  │

│  ├── data/

│  │  ├── dexycb_train_labels.pkl    # 训练数据 ⭐

│  │  └── dexycb_test_labels.pkl    # 测试数据

│  │

│  └── algo/ppo/

│    ├── ppo.py             # PPO算法

│    └── module.py           # 网络模块

│

└── test_run/

  └── {exp_name}/

​    └── {timestamp}/

​      ├── full_0.pt, full_200.pt, ...  # 保存的模型

​      ├── mean0.csv, var0.csv, ...   # 归一化参数

​      └── tensorboard/          # TensorBoard日志

### 关键函数索引

| 功能       | 文件            | 行号    | 说明                         |
| :--------- | :-------------- | :------ | :--------------------------- |
| 环境初始化 | Environment.hpp | 44-118  | 设置PD增益、权重、动作标准差 |
| 重置环境   | Environment.hpp | 280-356 | 重置手和物体状态             |
| 设置目标   | Environment.hpp | 358-400 | 设置抓取目标                 |
| 环境步进   | Environment.hpp | 417-508 | 三阶段控制逻辑               |
| 奖励计算   | Environment.hpp | 529-563 | 11个奖励组件                 |
| 观测更新   | Environment.hpp | 566-722 | 计算相对特征                 |
| 提升模式   | Environment.hpp | 731-754 | 高增益+重力补偿              |
| 数据加载   | runner.py       | 104-166 | DexYCB标签                   |
| 噪声注入   | runner.py       | 306-314 | 初始化随机化                 |
| 训练循环   | runner.py       | 278-493 | 完整训练流程                 |
| PPO更新    | runner.py       | 346-352 | 策略和价值优化               |

------

## 🎯 设计亮点总结

### 1. 三阶段训练 ⭐⭐⭐

- 接近阶段: 腕部自动引导，降低学习难度

- 抓取阶段: 策略自主学习手指控制

- 提升阶段: 验证抓取成功性

### 2. 渐进式课程学习

- 从简单（腕部引导）到困难（完全自主）

- 引导强度随训练步数线性增加

- 动作标准差分阶段调整

### 3. 多组件奖励

- 11个奖励组件，精细调控行为

- 位置和力奖励并重（系数2.0）

- 正则化项防止不自然动作

### 4. 数据增强

- 每个样本重复10次

- 每次rollout添加随机噪声

- 提高策略鲁棒性

### 5. 并行训练

- 21个环境并行采样

- 提高数据效率

- 加速训练

### 6. 详细监控

- TensorBoard记录所有指标

- 每个奖励组件单独追踪

- 便于调试和分析

这套设计使DGrasp能够学习到稳定、自然、鲁棒的抓取策略！🤖✨