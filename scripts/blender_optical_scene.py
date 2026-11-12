# FILE PURPOSE: Render actual 3D crystal/vessel scenes in Blender's bundled Python.
# RGB, unoccluded geometry and refracted emission proxy passes are saved separately.
# This script implements illustrative optics; no physical defect response is claimed.
"""Blender-only Cycles renderer called by generate_vision_images.py."""

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def material(name, color, *, ior=None, roughness=0.05, emission=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nodes = m.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    if emission:
        node = nodes.new("ShaderNodeEmission")
        node.inputs["Color"].default_value = (*color, 1)
    else:
        node = nodes.new("ShaderNodeBsdfPrincipled")
        node.inputs["Base Color"].default_value = (*color, 1)
        node.inputs["Roughness"].default_value = roughness
        if ior is not None:
            node.inputs["Transmission Weight"].default_value = 1
            node.inputs["IOR"].default_value = ior
    m.node_tree.links.new(node.outputs[0], output.inputs["Surface"])
    return m


def cube(name, location, scale, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    o = bpy.context.object
    o.name = name
    o.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    return o


def sphere(name, position, radius, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=16, ring_count=8, radius=radius, location=position
    )
    o = bpy.context.object
    o.name = name
    o.data.materials.append(mat)
    for polygon in o.data.polygons:
        polygon.use_smooth = True
    return o


def crystal(name, frame, position, rotation, scale, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(frame["vertices"], [], frame["faces"])
    mesh.update()
    o = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o)
    o.location = position
    o.rotation_euler = rotation
    o.scale = (scale, scale, scale)
    o.data.materials.append(mat)
    return o


def render_frame(case, frame, protocol, folder):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = protocol["samples"]
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 10
    scene.cycles.transparent_max_bounces = 10
    scene.cycles.seed = case["seed"] + frame["index"]
    scene.render.threads_mode = "FIXED"
    scene.render.threads = protocol["threads"]
    scene.render.resolution_x = scene.render.resolution_y = protocol["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "AgX"
    scene.world.use_nodes = True
    world = scene.world.node_tree.nodes.get("Background")
    world.inputs["Color"].default_value = (0.75, 0.75, 0.75, 1)
    world.inputs["Strength"].default_value = 0.5
    glass = material("vessel_glass", (1, 1, 1), ior=case["glass_ior"])
    liquid = material("liquid", (0.98, 0.99, 1), ior=case["liquid_ior"], roughness=0)
    solid = material(
        "crystal",
        (0.93, 0.97, 1),
        ior=case["crystal_ior"] / case["liquid_ior"],
        roughness=case["roughness"],
    )
    bubble = material("bubble", (1, 1, 1), ior=1 / case["liquid_ior"], roughness=0)
    dark = material("inclusion", (0.02, 0.025, 0.03), roughness=0.7)
    background = material("background", (0.72, 0.70, 0.62), roughness=0.8)
    opaque = [cube("background", (0, 0, -0.15), (10, 10, 0.1), background)]
    media = [
        cube("bottom", (0, 0, -0.04), (6.0, 6.0, 0.06), glass),
        cube("liquid", (0, 0, 0.85), (5.8, 5.8, 1.7), liquid),
    ]
    for axis in (0, 1):
        for sign in (-1, 1):
            location = [0, 0, 0.9]
            location[axis] = sign * 2.95
            dims = [6.0, 6.0, 1.9]
            dims[axis] = 0.08
            media.append(cube(f"wall_{axis}_{sign}", location, dims, glass))
    for i, item in enumerate(case["bubbles"]):
        if (
            "present_object_ids" in frame
            and f"bubble_{i}" not in frame["present_object_ids"]
        ):
            continue
        media.append(sphere(f"bubble_{i}", item["position"], item["radius"], bubble))
    targets = [(crystal("seed", frame, (0, 0, 0.65), case["rotation"], 1, solid), 0)]
    if frame["secondary_visible"]:
        for i, item in enumerate(case["secondaries"]):
            if (
                "present_object_ids" in frame
                and f"secondary_{i}" not in frame["present_object_ids"]
            ):
                continue
            targets.append(
                (
                    crystal(
                        f"secondary_{i}",
                        frame,
                        item["position"],
                        (0, 0, i + 0.3),
                        item["scale"],
                        solid,
                    ),
                    1,
                )
            )
    for i, item in enumerate(case["inclusions"]):
        if (
            "present_object_ids" in frame
            and f"inclusion_{i}" not in frame["present_object_ids"]
        ):
            continue
        targets.append(
            (sphere(f"inclusion_{i}", item["position"], item["radius"], dark), 2)
        )
    bpy.ops.object.camera_add(location=case["camera"]["position"])
    camera = bpy.context.object
    camera.rotation_euler = (
        (Vector((0, 0, 0.65)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = case["camera"]["ortho_scale"]
    scene.camera = camera
    bpy.ops.object.light_add(type="AREA", location=(-1, -0.5, 4))
    light = bpy.context.object
    light.data.energy = case["light_energy"]
    light.data.shape = "DISK"
    light.data.size = 3
    folder.mkdir(parents=True, exist_ok=False)

    def render(name):
        scene.render.filepath = str(folder / name)
        bpy.ops.render.render(write_still=True)

    render("rgb_raw.png")
    scene.cycles.use_denoising = False
    scene.view_settings.view_transform = "Standard"
    world.inputs["Color"].default_value = (0, 0, 0, 1)
    world.inputs["Strength"].default_value = 0
    light.hide_render = True
    white = material("label_white", (1, 1, 1), emission=True)
    black = material("label_black", (0, 0, 0), emission=True)
    for o in opaque:
        o.data.materials[0] = black
    records = []
    for target, class_id in targets:
        original = target.data.materials[0]
        target.data.materials[0] = white
        proxy = f"{target.name}_optical.png"
        geometric = f"{target.name}_geometric.png"
        render(proxy)
        hidden = []
        for o in scene.objects:
            if o.type == "MESH" and o != target:
                hidden.append(o)
                o.hide_render = True
        render(geometric)
        for o in hidden:
            o.hide_render = False
        target.data.materials[0] = original
        records.append(
            dict(
                id=target.name,
                class_id=class_id,
                class_name=protocol["classes"][class_id],
                optical_proxy=proxy,
                geometric_mask=geometric,
            )
        )
    (folder / "render.json").write_text(
        json.dumps(
            dict(
                targets=records,
                constructed_object_ids=frame.get("present_object_ids"),
                blender=bpy.app.version_string,
                optics="illustrative_nested_dielectrics_not_calibrated",
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    d = json.loads(a.input.read_text())
    for case in d["scenes"]:
        for frame in case["frames"]:
            render_frame(
                case,
                frame,
                d["protocol"],
                a.output / case["id"] / f"frame_{frame['index']:04d}",
            )
    print("CRYSTALAB_OPTICAL_RENDER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
