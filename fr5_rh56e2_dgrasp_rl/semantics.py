from __future__ import annotations

import numpy as np

from .utils import normalize


SEMANTIC_SITE_NAMES = (
    "wrist_mount",
    "palm_center",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "little_tip",
)
SEMANTIC_CONTACT_NAMES = (
    "palm",
    "thumb_proximal",
    "thumb_middle",
    "thumb_distal",
    "index_proximal",
    "index_distal",
    "middle_proximal",
    "middle_distal",
    "ring_proximal",
    "ring_distal",
    "little_proximal",
    "little_distal",
)
SEMANTIC_CONTACT_DISTANCE_THRESHOLDS_M = (
    0.0055,  # palm
    0.0050,  # thumb proximal
    0.0045,  # thumb middle
    0.0045,  # thumb distal
    0.0050,  # index proximal
    0.0045,  # index distal
    0.0050,  # middle proximal
    0.0045,  # middle distal
    0.0050,  # ring proximal
    0.0045,  # ring distal
    0.0050,  # little proximal
    0.0045,  # little distal
)

WRIST_INDEX = 0
# D-Grasp stores the 21 joints in wrist, index, middle, ring, little, thumb order.
INDEX_TIP_INDEX = 4
MIDDLE_TIP_INDEX = 8
RING_TIP_INDEX = 12
LITTLE_TIP_INDEX = 16
THUMB_TIP_INDEX = 20
# Use the four non-thumb proximal joints to approximate the palm center.
PALM_CENTER_SOURCE_INDICES = (1, 5, 9, 13)
ROBOT_WRIST_TO_PALM_OFFSET_M = 0.12


def flatten_sites(sites: np.ndarray) -> list[float]:
    return [float(v) for v in sites.reshape(-1)]


def unflatten_sites(values: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(len(SEMANTIC_SITE_NAMES), 3)


def mano_semantic_sites_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    # The input keypoints follow the reordered D-Grasp layout, not the raw MANO order.
    sites = np.zeros((len(SEMANTIC_SITE_NAMES), 3), dtype=np.float64)
    palm_center = np.mean(keypoints[list(PALM_CENTER_SOURCE_INDICES)], axis=0)
    mano_wrist = keypoints[WRIST_INDEX]
    wrist_to_palm = normalize(palm_center - mano_wrist)
    sites[1] = palm_center
    sites[0] = palm_center - wrist_to_palm * ROBOT_WRIST_TO_PALM_OFFSET_M
    sites[2] = keypoints[THUMB_TIP_INDEX]
    sites[3] = keypoints[INDEX_TIP_INDEX]
    sites[4] = keypoints[MIDDLE_TIP_INDEX]
    sites[5] = keypoints[RING_TIP_INDEX]
    sites[6] = keypoints[LITTLE_TIP_INDEX]
    return sites


def semantic_frame_from_sites(sites: np.ndarray) -> np.ndarray:
    wrist = sites[0]
    palm = sites[1]
    index_tip = sites[3]
    little_tip = sites[6]
    approach = normalize(palm - wrist)
    across = normalize(index_tip - little_tip)
    normal = normalize(np.cross(across, approach))
    if np.linalg.norm(normal) < 1e-8:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    across = normalize(np.cross(approach, normal))
    return np.column_stack((across, normal, approach))


def contact_mask_16_to_12(mask_16: np.ndarray) -> np.ndarray:
    mask_16 = np.asarray(mask_16, dtype=np.float64).reshape(16)
    return np.array(
        [
            float(mask_16[0] > 0.5),
            float(mask_16[13] > 0.5),
            float(mask_16[14] > 0.5),
            float(mask_16[15] > 0.5),
            float(mask_16[1] > 0.5),
            float(mask_16[1:4].max() > 0.5),
            float(mask_16[4] > 0.5),
            float(mask_16[4:7].max() > 0.5),
            float(mask_16[7] > 0.5),
            float(mask_16[7:10].max() > 0.5),
            float(mask_16[10] > 0.5),
            float(mask_16[10:13].max() > 0.5),
        ],
        dtype=np.float64,
    )


def contact_mask_16_to_6(mask_16: np.ndarray) -> np.ndarray:
    mask_16 = np.asarray(mask_16, dtype=np.float64).reshape(16)
    return np.array(
        [
            float(mask_16[0] > 0.5),
            float(mask_16[13:16].max() > 0.5),
            float(mask_16[1:4].max() > 0.5),
            float(mask_16[4:7].max() > 0.5),
            float(mask_16[7:10].max() > 0.5),
            float(mask_16[10:13].max() > 0.5),
        ],
        dtype=np.float64,
    )


def site_rmse(current_sites: np.ndarray, target_sites: np.ndarray, include_wrist: bool = False) -> float:
    start = 0 if include_wrist else 1
    diff = current_sites[start:] - target_sites[start:]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
