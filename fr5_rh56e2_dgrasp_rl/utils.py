from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def quat_wxyz_to_matrix(quat: Iterable[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    w /= n
    x /= n
    y /= n
    z /= n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        diag = np.diag(matrix)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return normalize(quat)


def quat_slerp_wxyz(start: Iterable[float], end: Iterable[float], alpha: float) -> np.ndarray:
    q0 = normalize(np.asarray(list(start), dtype=np.float64))
    q1 = normalize(np.asarray(list(end), dtype=np.float64))
    t = float(np.clip(alpha, 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize((1.0 - t) * q0 + t * q1)
    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return normalize(s0 * q0 + s1 * q1)


def pose7_to_matrix(pose: Iterable[float]) -> np.ndarray:
    pose_arr = np.asarray(list(pose), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_wxyz_to_matrix(pose_arr[3:])
    matrix[:3, 3] = pose_arr[:3]
    return matrix


def matrix_to_pose7(matrix: np.ndarray) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float64)
    pose[:3] = matrix[:3, 3]
    pose[3:] = matrix_to_quat_wxyz(matrix[:3, :3])
    return pose


def rotvec_to_matrix(rotvec: Iterable[float]) -> np.ndarray:
    rx, ry, rz = [float(v) for v in rotvec]
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = np.array([rx, ry, rz], dtype=np.float64) / theta
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    return np.eye(3, dtype=np.float64) + sin_theta * skew + (1.0 - cos_theta) * (skew @ skew)


def apply_local_pose_delta(
    pose: Iterable[float],
    translation_local: Iterable[float],
    rotvec_local: Iterable[float],
) -> np.ndarray:
    pose_arr = np.asarray(list(pose), dtype=np.float64)
    rotation = quat_wxyz_to_matrix(pose_arr[3:])
    updated = pose_arr.copy()
    updated[:3] = updated[:3] + rotation @ np.asarray(list(translation_local), dtype=np.float64)
    updated_rotation = rotation @ rotvec_to_matrix(rotvec_local)
    updated[3:] = matrix_to_quat_wxyz(updated_rotation)
    return updated


def transform_points(points: np.ndarray, pose: Iterable[float]) -> np.ndarray:
    matrix = pose7_to_matrix(pose)
    homogeneous = np.ones((points.shape[0], 4), dtype=np.float64)
    homogeneous[:, :3] = points
    return (matrix @ homogeneous.T).T[:, :3]


def inverse_pose7(pose: Iterable[float]) -> np.ndarray:
    matrix = pose7_to_matrix(pose)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ matrix[:3, 3]
    return matrix_to_pose7(inverse)


def compose_pose7(first: Iterable[float], second: Iterable[float]) -> np.ndarray:
    return matrix_to_pose7(pose7_to_matrix(first) @ pose7_to_matrix(second))


def interpolate_pose7(start_pose: Iterable[float], end_pose: Iterable[float], alpha: float) -> np.ndarray:
    start = np.asarray(list(start_pose), dtype=np.float64)
    end = np.asarray(list(end_pose), dtype=np.float64)
    t = float(np.clip(alpha, 0.0, 1.0))
    pose = np.zeros(7, dtype=np.float64)
    pose[:3] = (1.0 - t) * start[:3] + t * end[:3]
    pose[3:] = quat_slerp_wxyz(start[3:], end[3:], t)
    return pose


def rotation_6d_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix[:, :2].reshape(-1)


def clamp_array(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(values, lower), upper)


def rotation_angle_deg(current: np.ndarray, target: np.ndarray) -> float:
    delta = current.T @ target
    trace = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(trace)))


def rotation_error_world(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta_local = current.T @ target
    cos_theta = np.clip((np.trace(delta_local) - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(float(cos_theta))
    skew_part = np.array(
        [
            delta_local[2, 1] - delta_local[1, 2],
            delta_local[0, 2] - delta_local[2, 0],
            delta_local[1, 0] - delta_local[0, 1],
        ],
        dtype=np.float64,
    )
    if theta < 1e-9:
        rotvec_local = 0.5 * skew_part
    else:
        rotvec_local = (theta / (2.0 * math.sin(theta))) * skew_part
    return current @ rotvec_local


def damped_least_squares(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    jt = jacobian.T
    lhs = jacobian @ jt + damping * np.eye(jacobian.shape[0], dtype=np.float64)
    return jt @ np.linalg.solve(lhs, error)
