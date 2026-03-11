from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
BUNDLED_ASSETS_DIR = PROJECT_DIR / "assets"
BUNDLED_BASE_SCENE_DIR = BUNDLED_ASSETS_DIR / "base_scene"
BUNDLED_DGRASP_DIR = BUNDLED_ASSETS_DIR / "dgrasp_bundle"
BASE_SCENE_DIR = WORKSPACE_DIR / "mujoco_fr5_rh56e2"
BASE_SCENE_BUILD_DIR = BASE_SCENE_DIR / "build"


def add_workspace_to_path() -> None:
    workspace_text = str(WORKSPACE_DIR)
    if workspace_text not in sys.path:
        sys.path.insert(0, workspace_text)


def runtime_dirs() -> dict[str, Path]:
    return {
        "build": PROJECT_DIR / "build",
        "data": PROJECT_DIR / "data",
        "logs": PROJECT_DIR / "logs",
        "checkpoints": PROJECT_DIR / "checkpoints",
        "evals": PROJECT_DIR / "evals",
        "replays": PROJECT_DIR / "replays",
        "tmp": PROJECT_DIR / ".tmp",
        "pip_cache": PROJECT_DIR / ".pip_cache",
        "torch_home": PROJECT_DIR / ".torch",
    }


def ensure_runtime_dirs() -> dict[str, Path]:
    dirs = runtime_dirs()
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def configure_local_runtime_env() -> dict[str, Path]:
    dirs = ensure_runtime_dirs()
    os.environ["TMP"] = str(dirs["tmp"])
    os.environ["TEMP"] = str(dirs["tmp"])
    os.environ["PIP_CACHE_DIR"] = str(dirs["pip_cache"])
    os.environ["TORCH_HOME"] = str(dirs["torch_home"])
    return dirs


def find_first_file(file_name: str) -> Path:
    matches = list(WORKSPACE_DIR.rglob(file_name))
    if not matches:
        raise FileNotFoundError(file_name)
    return matches[0]


def locate_dgrasp_root() -> Path:
    if BUNDLED_DGRASP_DIR.exists():
        return BUNDLED_DGRASP_DIR
    summary_path = find_first_file("DGRASP_REPRO_SUMMARY_CN.md")
    return summary_path.parent


def locate_dgrasp_data_file(file_name: str) -> Path:
    if BUNDLED_DGRASP_DIR.exists():
        matches = list(BUNDLED_DGRASP_DIR.rglob(file_name))
        if matches:
            return matches[0]

    if BUNDLED_DGRASP_DIR.exists():
        summary_path = find_first_file("DGRASP_REPRO_SUMMARY_CN.md")
        dgrasp_root = summary_path.parent
    else:
        dgrasp_root = locate_dgrasp_root()
    matches = list(dgrasp_root.rglob(file_name))
    if not matches:
        raise FileNotFoundError(file_name)
    return matches[0]


def locate_fr5_description_dir() -> Path:
    urdf_path = find_first_file("fr5v6.urdf")
    return urdf_path.parents[1]


def locate_rh56e2_dir() -> Path:
    urdf_path = find_first_file("RH56E2_R_2025_9_11.urdf")
    return urdf_path.parents[1]
