from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from .paths import (
    BUNDLED_BASE_SCENE_DIR,
    BASE_SCENE_BUILD_DIR,
    add_workspace_to_path,
    ensure_runtime_dirs,
    locate_dgrasp_root,
    locate_fr5_description_dir,
    locate_rh56e2_dir,
)
from .task_config import TaskConfig


DEFAULT_SCENE_XML_NAME = "fr5_rh56e2_sugar_box_scene.xml"
DEFAULT_SCENE_METADATA_NAME = "fr5_rh56e2_sugar_box_metadata.json"


def _import_base_builder():
    add_workspace_to_path()
    from mujoco_fr5_rh56e2.build_scene import (  # pylint: disable=import-outside-toplevel
        FINAL_MJCF_NAME,
        METADATA_NAME,
        build_scene,
    )

    return build_scene, FINAL_MJCF_NAME, METADATA_NAME


def _ensure_base_scene() -> tuple[Path, Path]:
    bundled_scene_xml = BUNDLED_BASE_SCENE_DIR / "fr5_rh56e2_scene.xml"
    bundled_metadata = BUNDLED_BASE_SCENE_DIR / "scene_metadata.json"
    if bundled_scene_xml.exists() and bundled_metadata.exists():
        return bundled_scene_xml, bundled_metadata

    build_scene, final_name, metadata_name = _import_base_builder()
    scene_xml = BASE_SCENE_BUILD_DIR / final_name
    metadata_path = BASE_SCENE_BUILD_DIR / metadata_name
    if scene_xml.exists() and metadata_path.exists():
        return scene_xml, metadata_path

    build_scene(
        build_dir=BASE_SCENE_BUILD_DIR,
        source_dir=BASE_SCENE_BUILD_DIR / "source_scene",
        fr5_dir=locate_fr5_description_dir(),
        rh56e2_dir=locate_rh56e2_dir(),
    )
    return scene_xml, metadata_path


def _find_named(parent: ET.Element, tag: str, name: str) -> ET.Element | None:
    for child in parent.findall(tag):
        if child.attrib.get("name") == name:
            return child
    return None


def _ensure_named(parent: ET.Element, tag: str, name: str) -> ET.Element:
    element = _find_named(parent, tag, name)
    if element is not None:
        return element
    return ET.SubElement(parent, tag, {"name": name})


def _remove_named(parent: ET.Element, tag: str, name: str) -> None:
    element = _find_named(parent, tag, name)
    if element is not None:
        parent.remove(element)


def _set_mesh_paths_absolute(root: ET.Element, base_scene_xml: Path) -> None:
    asset = root.find("asset")
    if asset is None:
        return
    for mesh in asset.findall("mesh"):
        file_attr = mesh.attrib.get("file")
        if not file_attr:
            continue
        mesh_path = Path(file_attr)
        if not mesh_path.is_absolute():
            mesh.attrib["file"] = str((base_scene_xml.parent / mesh_path).resolve())


def _add_semantic_site(body: ET.Element, name: str, pos: str, rgba: str = "0.2 0.9 0.2 1") -> None:
    site = _ensure_named(body, "site", name)
    site.attrib.update({"pos": pos, "size": "0.006", "rgba": rgba, "type": "sphere"})


def _copy_obj_without_materials(source_obj: Path, target_obj: Path) -> None:
    lines = source_obj.read_text(encoding="utf-8", errors="ignore").splitlines()
    filtered = [line for line in lines if not line.startswith("mtllib ") and not line.startswith("usemtl ")]
    target_obj.write_text("\n".join(filtered) + "\n", encoding="utf-8")


def build_training_scene(config: TaskConfig, force_rebuild: bool = False) -> tuple[Path, Path]:
    dirs = ensure_runtime_dirs()
    scene_xml_path = dirs["build"] / DEFAULT_SCENE_XML_NAME
    metadata_path = dirs["build"] / DEFAULT_SCENE_METADATA_NAME

    if not force_rebuild and scene_xml_path.exists() and metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        matches_config = (
            existing_metadata.get("table_center") == config.table_center
            and existing_metadata.get("table_size") == config.table_size
            and existing_metadata.get("default_object_pose") == config.default_object_pose
        )
        if matches_config:
            return scene_xml_path, metadata_path

    base_scene_xml, base_metadata_path = _ensure_base_scene()
    base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    dgrasp_root = locate_dgrasp_root()
    object_mesh_source = dgrasp_root / "rsc" / "meshes_simplified" / config.object_name / "textured_meshlab.obj"
    object_mesh_copy = dirs["build"] / f"{config.object_name}_visual.obj"
    if force_rebuild or not object_mesh_copy.exists():
        _copy_obj_without_materials(object_mesh_source, object_mesh_copy)

    tree = ET.parse(base_scene_xml)
    root = tree.getroot()
    _set_mesh_paths_absolute(root, base_scene_xml)

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib["timestep"] = f"{config.timestep:.10g}"
    option.attrib["iterations"] = "50"
    option.attrib["integrator"] = "implicitfast"

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Base scene is missing worldbody.")

    _remove_named(worldbody, "geom", "floor")
    _remove_named(worldbody, "geom", "robot_pedestal")

    table_geom = _ensure_named(worldbody, "geom", "grasp_table")
    table_geom.attrib.update(
        {
            "type": "box",
            "pos": " ".join(str(v) for v in config.table_center),
            "size": " ".join(str(v) for v in config.table_size),
            "rgba": "0.65 0.62 0.57 1",
            "condim": "3",
            "friction": " ".join(str(v) for v in config.table_friction),
        }
    )

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    sugar_mesh = _ensure_named(asset, "mesh", "sugar_box_visual_mesh")
    sugar_mesh.attrib.update(
        {
            "file": str(object_mesh_copy.resolve())
        }
    )

    object_body = _ensure_named(worldbody, "body", "sugar_box")
    object_body.attrib.update(
        {
            "pos": " ".join(str(v) for v in config.default_object_pose[:3]),
            "quat": " ".join(str(v) for v in config.default_object_pose[3:]),
        }
    )
    for child in list(object_body):
        if child.tag == "freejoint" and child.attrib.get("name") != "sugar_box_freejoint":
            object_body.remove(child)
    freejoint = _ensure_named(object_body, "freejoint", "sugar_box_freejoint")
    freejoint.attrib.update({"name": "sugar_box_freejoint"})
    collision = _ensure_named(object_body, "geom", "sugar_box_collision")
    half_extents = [0.5 * value for value in config.object_dims_m]
    collision.attrib.update(
        {
            "type": "box",
            "size": " ".join(str(v) for v in half_extents),
            "mass": f"{config.object_mass_kg:.8g}",
            "rgba": "0 0 0 0",
            "friction": "1.1 0.02 0.001",
            "condim": "6",
        }
    )
    visual = _ensure_named(object_body, "geom", "sugar_box_visual")
    visual.attrib.update(
        {
            "type": "mesh",
            "mesh": "sugar_box_visual_mesh",
            "contype": "0",
            "conaffinity": "0",
            "rgba": "0.9 0.9 0.9 1",
        }
    )
    object_site = _ensure_named(object_body, "site", "sugar_box_center")
    object_site.attrib.update({"type": "sphere", "size": "0.008", "rgba": "0.9 0.2 0.2 1"})

    j6_body = None
    for body in worldbody.iter("body"):
        if body.attrib.get("name") == "j6_Link":
            j6_body = body
            break
    if j6_body is None:
        raise ValueError("Could not find j6_Link in base scene.")

    _add_semantic_site(j6_body, "wrist_mount", "0 0 0.109013")
    _add_semantic_site(j6_body, "palm_center", "-0.00036753 0.012296 0.228614")

    site_specs = {
        "rh56e2_right_thumb_4": ("thumb_tip", "-0.01883 0.027914 0.0068326"),
        "rh56e2_right_index_2": ("index_tip", "0.0086237 0.052572 0.0060954"),
        "rh56e2_right_middle_2": ("middle_tip", "0.0098294 0.056051 0.0061006"),
        "rh56e2_right_ring_2": ("ring_tip", "0.0086237 0.052572 0.0060954"),
        "rh56e2_right_little_2": ("little_tip", "0.0063972 0.042707 0.0061154"),
    }
    for body in worldbody.iter("body"):
        body_name = body.attrib.get("name")
        if body_name in site_specs:
            site_name, pos = site_specs[body_name]
            _add_semantic_site(body, site_name, pos)

    keyframe = root.find("keyframe")
    if keyframe is not None:
        for key in keyframe.findall("key"):
            if key.attrib.get("name") == "home":
                base_home = [float(v) for v in base_metadata["home_qpos"]]
                home_qpos = base_home + list(config.default_object_pose)
                key.attrib["qpos"] = " ".join(f"{value:.8g}" for value in home_qpos)

    ET.indent(root, space="  ")
    tree.write(scene_xml_path, encoding="utf-8", xml_declaration=True)

    metadata = {
        **base_metadata,
        "scene_xml": str(scene_xml_path),
        "scene_metadata": str(metadata_path),
        "semantic_sites": [
            "wrist_mount",
            "palm_center",
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "little_tip",
        ],
        "object_body_name": "sugar_box",
        "object_joint_name": "sugar_box_freejoint",
        "object_center_site": "sugar_box_center",
        "table_geom_name": "grasp_table",
        "table_center": config.table_center,
        "table_size": config.table_size,
        "object_dims_m": config.object_dims_m,
        "default_object_pose": config.default_object_pose,
        "contact_group_bodies": {
            "palm": ["j6_Link"],
            "thumb": [
                "rh56e2_right_thumb_1",
                "rh56e2_right_thumb_2",
                "rh56e2_right_thumb_3",
                "rh56e2_right_thumb_4"
            ],
            "index": ["rh56e2_right_index_1", "rh56e2_right_index_2"],
            "middle": ["rh56e2_right_middle_1", "rh56e2_right_middle_2"],
            "ring": ["rh56e2_right_ring_1", "rh56e2_right_ring_2"],
            "little": ["rh56e2_right_little_1", "rh56e2_right_little_2"]
        }
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return scene_xml_path, metadata_path
