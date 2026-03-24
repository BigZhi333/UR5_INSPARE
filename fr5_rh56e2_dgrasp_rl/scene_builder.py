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


SCENE_BUILD_VERSION = 7
OBJECT_BODY_NAME = "task_object"
OBJECT_JOINT_NAME = "task_object_freejoint"
OBJECT_COLLISION_NAME = "task_object_collision"
OBJECT_VISUAL_NAME = "task_object_visual"
OBJECT_VISUAL_MESH_NAME = "task_object_visual_mesh"
OBJECT_CENTER_SITE_NAME = "task_object_center"

HAND_COLLISION_PROXY_ATTRS = {
    "contype": "1",
    "conaffinity": "1",
    "condim": "6",
    "friction": "1.1 0.02 0.002",
    "margin": "0.0015",
    "rgba": "0.1 0.7 0.2 0.08",
}

HAND_COLLISION_PROXIES: dict[str, list[dict[str, str]]] = {
    "rh56e2_right_thumb_1": [],
    "j6_Link": [
        {
            "name": "proxy_palm_main",
            "type": "box",
            "pos": "0.0000 0.0045 0.184",
            "quat": "-0.707107 -2.61376e-06 -2.61376e-06 -0.707107",
            "size": "0.024 0.028 0.016",
        },
        {
            "name": "proxy_palm_upper",
            "type": "box",
            "pos": "-0.00036753 0.012296 0.221",
            "quat": "3.90296e-06 -1.32455e-06 -0.707108 -0.707105",
            "size": "0.018 0.02 0.008",
        },
    ],
    "rh56e2_right_thumb_2": [
        {
            "name": "proxy_thumb_proximal",
            "type": "capsule",
            "fromto": "-0.008 0.012 0.003  -0.047 0.031 -0.006",
            "size": "0.009",
        },
    ],
    "rh56e2_right_thumb_3": [
        {
            "name": "proxy_thumb_middle",
            "type": "capsule",
            "fromto": "-0.002 0.006 0.004  -0.017 0.017 0.001",
            "size": "0.007",
        },
    ],
    "rh56e2_right_thumb_4": [
        {
            "name": "proxy_thumb_distal",
            "type": "capsule",
            "fromto": "-0.003 0.007 0.004  -0.017 0.025 0.007",
            "size": "0.006",
        },
    ],
    "rh56e2_right_index_1": [
        {
            "name": "proxy_index_proximal",
            "type": "capsule",
            "fromto": "0.004 0.006 0.006  0.008 0.028 0.006",
            "size": "0.007",
        },
    ],
    "rh56e2_right_index_2": [
        {
            "name": "proxy_index_distal",
            "type": "capsule",
            "fromto": "0.006 0.008 0.006  0.0085 0.05 0.006",
            "size": "0.006",
        },
    ],
    "rh56e2_right_middle_1": [
        {
            "name": "proxy_middle_proximal",
            "type": "capsule",
            "fromto": "0.0045 0.006 0.006  0.009 0.029 0.006",
            "size": "0.0075",
        },
    ],
    "rh56e2_right_middle_2": [
        {
            "name": "proxy_middle_distal",
            "type": "capsule",
            "fromto": "0.007 0.008 0.006  0.0095 0.053 0.006",
            "size": "0.0065",
        },
    ],
    "rh56e2_right_ring_1": [
        {
            "name": "proxy_ring_proximal",
            "type": "capsule",
            "fromto": "0.004 0.006 0.006  0.008 0.028 0.006",
            "size": "0.007",
        },
    ],
    "rh56e2_right_ring_2": [
        {
            "name": "proxy_ring_distal",
            "type": "capsule",
            "fromto": "0.006 0.008 0.006  0.0085 0.05 0.006",
            "size": "0.006",
        },
    ],
    "rh56e2_right_little_1": [
        {
            "name": "proxy_little_proximal",
            "type": "capsule",
            "fromto": "0.0035 0.006 0.006  0.0068 0.026 0.006",
            "size": "0.0065",
        },
    ],
    "rh56e2_right_little_2": [
        {
            "name": "proxy_little_distal",
            "type": "capsule",
            "fromto": "0.0045 0.007 0.006  0.0065 0.041 0.006",
            "size": "0.0055",
        },
    ],
}

HAND_JOINT_FORCE_LIMITS = {
    "rh56e2_right_thumb_1_joint": 8.0,
    "rh56e2_right_thumb_2_joint": 6.0,
    "rh56e2_right_thumb_3_joint": 5.0,
    "rh56e2_right_thumb_4_joint": 4.0,
    "rh56e2_right_index_1_joint": 6.0,
    "rh56e2_right_index_2_joint": 4.0,
    "rh56e2_right_middle_1_joint": 6.0,
    "rh56e2_right_middle_2_joint": 4.0,
    "rh56e2_right_ring_1_joint": 6.0,
    "rh56e2_right_ring_2_joint": 4.0,
    "rh56e2_right_little_1_joint": 5.0,
    "rh56e2_right_little_2_joint": 3.5,
}

HAND_ACTUATOR_KP = {
    "rh56e2_right_thumb_1_joint_act": 42.0,
    "rh56e2_right_thumb_2_joint_act": 38.0,
    "rh56e2_right_index_1_joint_act": 34.0,
    "rh56e2_right_middle_1_joint_act": 34.0,
    "rh56e2_right_ring_1_joint_act": 34.0,
    "rh56e2_right_little_1_joint_act": 30.0,
}

HAND_JOINT_DAMPING = {
    "rh56e2_right_thumb_1_joint": 0.25,
    "rh56e2_right_thumb_2_joint": 0.18,
    "rh56e2_right_thumb_3_joint": 0.12,
    "rh56e2_right_thumb_4_joint": 0.10,
    "rh56e2_right_index_1_joint": 0.16,
    "rh56e2_right_index_2_joint": 0.10,
    "rh56e2_right_middle_1_joint": 0.16,
    "rh56e2_right_middle_2_joint": 0.10,
    "rh56e2_right_ring_1_joint": 0.16,
    "rh56e2_right_ring_2_joint": 0.10,
    "rh56e2_right_little_1_joint": 0.14,
    "rh56e2_right_little_2_joint": 0.09,
}


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


def _find_body(worldbody: ET.Element, name: str) -> ET.Element | None:
    for body in worldbody.iter("body"):
        if body.attrib.get("name") == name:
            return body
    return None


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


def _configure_hand_collision_proxies(worldbody: ET.Element) -> None:
    for body in worldbody.iter("body"):
        body_name = body.attrib.get("name", "")
        if body_name == "j6_Link" or body_name.startswith("rh56e2_"):
            for geom in body.findall("geom"):
                if geom.attrib.get("type") == "mesh":
                    geom.attrib["contype"] = "0"
                    geom.attrib["conaffinity"] = "0"

    for body_name, proxy_defs in HAND_COLLISION_PROXIES.items():
        body = _find_body(worldbody, body_name)
        if body is None:
            continue
        for geom in body.findall("geom"):
            if geom.attrib.get("name", "").startswith("proxy_"):
                geom.attrib.update(HAND_COLLISION_PROXY_ATTRS)
        for proxy_def in proxy_defs:
            proxy = _ensure_named(body, "geom", proxy_def["name"])
            proxy.attrib.update(HAND_COLLISION_PROXY_ATTRS)
            proxy.attrib.update(proxy_def)


def _copy_obj_without_materials(source_obj: Path, target_obj: Path) -> None:
    lines = source_obj.read_text(encoding="utf-8", errors="ignore").splitlines()
    filtered = [line for line in lines if not line.startswith("mtllib ") and not line.startswith("usemtl ")]
    target_obj.write_text("\n".join(filtered) + "\n", encoding="utf-8")


def _configure_hand_actuation(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    for body in worldbody.iter("body"):
        for joint in body.findall("joint"):
            joint_name = joint.attrib.get("name")
            if joint_name in HAND_JOINT_FORCE_LIMITS:
                limit = HAND_JOINT_FORCE_LIMITS[joint_name]
                joint.attrib["actuatorfrcrange"] = f"{-limit:.8g} {limit:.8g}"
            if joint_name in HAND_JOINT_DAMPING:
                joint.attrib["damping"] = f"{HAND_JOINT_DAMPING[joint_name]:.8g}"

    actuator = root.find("actuator")
    if actuator is None:
        return
    for position in actuator.findall("position"):
        actuator_name = position.attrib.get("name")
        if actuator_name in HAND_ACTUATOR_KP:
            position.attrib["kp"] = f"{HAND_ACTUATOR_KP[actuator_name]:.8g}"


def build_training_scene(config: TaskConfig, force_rebuild: bool = False) -> tuple[Path, Path]:
    dirs = ensure_runtime_dirs()
    scene_xml_path = config.default_scene_xml
    metadata_path = config.default_scene_metadata

    if not force_rebuild and scene_xml_path.exists() and metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        matches_config = (
            existing_metadata.get("scene_build_version") == SCENE_BUILD_VERSION
            and
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
    object_visual_mesh = _ensure_named(asset, "mesh", OBJECT_VISUAL_MESH_NAME)
    object_visual_mesh.attrib.update(
        {
            "file": str(object_mesh_copy.resolve())
        }
    )

    object_body = _ensure_named(worldbody, "body", OBJECT_BODY_NAME)
    object_body.attrib.update(
        {
            "pos": " ".join(str(v) for v in config.default_object_pose[:3]),
            "quat": " ".join(str(v) for v in config.default_object_pose[3:]),
        }
    )
    for child in list(object_body):
        if child.tag == "freejoint" and child.attrib.get("name") != OBJECT_JOINT_NAME:
            object_body.remove(child)
    freejoint = _ensure_named(object_body, "freejoint", OBJECT_JOINT_NAME)
    freejoint.attrib.update({"name": OBJECT_JOINT_NAME})
    collision = _ensure_named(object_body, "geom", OBJECT_COLLISION_NAME)
    collision_attrs = {
        "mass": f"{config.object_mass_kg:.8g}",
        "rgba": "0 0 0 0",
        "friction": "1.1 0.02 0.001",
        "condim": "6",
        "margin": "0.0015",
    }
    if config.object_geom_type == "cylinder":
        radius = float(config.object_dims_m[0])
        height = float(config.object_dims_m[1])
        collision_attrs.update(
            {
                "type": "cylinder",
                "size": f"{radius:.8g} {0.5 * height:.8g}",
            }
        )
    else:
        half_extents = [0.5 * value for value in config.object_dims_m]
        collision_attrs.update(
            {
                "type": "box",
                "size": " ".join(str(v) for v in half_extents),
            }
        )
    collision.attrib.update(collision_attrs)
    visual = _ensure_named(object_body, "geom", OBJECT_VISUAL_NAME)
    visual.attrib.update(
        {
            "type": "mesh",
            "mesh": OBJECT_VISUAL_MESH_NAME,
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "mass": "0",
            "rgba": "0.9 0.9 0.9 1",
        }
    )
    object_site = _ensure_named(object_body, "site", OBJECT_CENTER_SITE_NAME)
    object_site.attrib.update({"type": "sphere", "size": "0.008", "rgba": "0.9 0.2 0.2 1"})

    j6_body = None
    for body in worldbody.iter("body"):
        if body.attrib.get("name") == "j6_Link":
            j6_body = body
            break
    if j6_body is None:
        raise ValueError("Could not find j6_Link in base scene.")

    _configure_hand_collision_proxies(worldbody)
    _configure_hand_actuation(root)

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
        "scene_build_version": SCENE_BUILD_VERSION,
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
        "object_body_name": OBJECT_BODY_NAME,
        "object_joint_name": OBJECT_JOINT_NAME,
        "object_center_site": OBJECT_CENTER_SITE_NAME,
        "table_geom_name": "grasp_table",
        "table_center": config.table_center,
        "table_size": config.table_size,
        "object_dims_m": config.object_dims_m,
        "object_geom_type": config.object_geom_type,
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
