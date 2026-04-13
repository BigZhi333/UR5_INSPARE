import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"E:\\A_mujoco\\UR5_INSPARE")))
import numpy as np
from fr5_rh56e2_dgrasp_rl.task_config import TaskConfig
from fr5_rh56e2_dgrasp_rl.scene_builder import build_training_scene
from fr5_rh56e2_dgrasp_rl.robot_model import RobotSceneModel
from fr5_rh56e2_dgrasp_rl.pose_driven_data import PROJECTION_SETTLE_STEPS, PROJECTION_SETTLE_ARM_KP_SCALE, PROJECTION_SETTLE_HAND_KP_SCALE

config = TaskConfig.from_json(Path(r"E:\\A_mujoco\\UR5_INSPARE\\config\\default_task.json"))
scene_xml, metadata_path = build_training_scene(config)
runtime = RobotSceneModel(config, scene_xml=scene_xml, metadata_path=metadata_path)
object_pose = np.asarray(config.default_object_pose, dtype=np.float64)
runtime.reset()
runtime.set_object_pose(object_pose)
runtime.settle_actuated_pose(runtime.home_actuated.copy(), PROJECTION_SETTLE_STEPS, fixed_object_pose=object_pose, arm_kp_scale=PROJECTION_SETTLE_ARM_KP_SCALE, hand_kp_scale=PROJECTION_SETTLE_HAND_KP_SCALE)
final_pose = runtime.get_object_pose()
linear, angular = runtime.get_object_velocity()
print(f"object_pose_error={np.linalg.norm(final_pose - object_pose):.10f}")
print(f"object_linear_speed={np.linalg.norm(linear):.10f}")
print(f"object_angular_speed={np.linalg.norm(angular):.10f}")
print(f"actuated_qvel_norm={np.linalg.norm(runtime.get_actuated_qvel()):.10f}")
