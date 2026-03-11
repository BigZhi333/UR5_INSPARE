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


def rotation_6d_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix[:, :2].reshape(-1)


def clamp_array(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(values, lower), upper)


def rotation_angle_deg(current: np.ndarray, target: np.ndarray) -> float:
    delta = current.T @ target
    trace = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(trace)))


def damped_least_squares(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    jt = jacobian.T
    lhs = jacobian @ jt + damping * np.eye(jacobian.shape[0], dtype=np.float64)
    return jt @ np.linalg.solve(lhs, error)
