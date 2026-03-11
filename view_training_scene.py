from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer as mujoco_viewer
import numpy as np

from fr5_rh56e2_dgrasp_rl.paths import configure_local_runtime_env
from fr5_rh56e2_dgrasp_rl.scene_builder import build_training_scene
from fr5_rh56e2_dgrasp_rl.task_config import TaskConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View or render the FR5+RH56E2 training scene.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default_task.json",
    )
    parser.add_argument("--preview", action="store_true", help="Render an offscreen preview image instead of opening the viewer.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "build" / "training_scene_preview.png",
    )
    return parser


def configure_camera(camera: mujoco.MjvCamera, lookat: np.ndarray | None = None) -> None:
    camera.lookat[:] = np.array([0.42, 0.02, 0.63], dtype=np.float64) if lookat is None else lookat
    camera.distance = 1.15
    camera.azimuth = 138.0
    camera.elevation = -22.0


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def render_preview(model: mujoco.MjModel, data: mujoco.MjData, width: int, height: int, output: Path) -> Path:
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    configure_camera(camera)
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()

    try:
        from PIL import Image

        Image.fromarray(pixels).save(output)
    except ImportError:
        ppm_path = output.with_suffix(".ppm")
        ppm_path.parent.mkdir(parents=True, exist_ok=True)
        with ppm_path.open("wb") as handle:
            handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            handle.write(pixels[:, :, :3].tobytes())
        return ppm_path
    return output


def launch_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    with mujoco_viewer.launch_passive(model, data) as viewer:
        camera = viewer.cam
        configure_camera(camera)
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


def main() -> None:
    configure_local_runtime_env()
    args = build_arg_parser().parse_args()
    config = TaskConfig.from_json(args.config)
    scene_xml, _ = build_training_scene(config)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    reset_home(model, data)

    if args.preview:
        output = render_preview(model, data, args.width, args.height, args.output)
        print(f"Preview written to: {output}")
        return

    launch_viewer(model, data)


if __name__ == "__main__":
    main()
