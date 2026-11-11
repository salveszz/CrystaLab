# FILE PURPOSE: Prepare grouped faceted scenes, run Blender and audit optical-image artifacts.
# Geometry comes from the existing face-plane solid. Camera/material/noise ranges
# and temporal appearances are illustrative design assumptions. Geometric masks
# and refracted emission proxies remain separate, never measured real labels.
"""Reproducible Blender optical-image preparation and artifact auditing."""

from collections import Counter
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

import cv2
import numpy as np

from simulation.crystal import Crystal
from vision.video import file_sha256

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["seed", "secondary_crystal", "inclusion_candidate"]


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def select_proxy_mask(proxy, geometric, threshold):
    """Select the proxy component anchored by direct geometry; retain reflection counts."""
    foreground = (proxy >= threshold).astype(np.uint8) * 255
    count, components = cv2.connectedComponents(foreground)
    overlaps = [
        int(np.count_nonzero((components == i) & (geometric > 127)))
        for i in range(1, count)
    ]
    mask = np.zeros_like(foreground)
    if overlaps and max(overlaps) > 0:
        mask[components == int(np.argmax(overlaps)) + 1] = 255
    return mask, int(np.count_nonzero(foreground)) - int(np.count_nonzero(mask))


def prepare_scenes(protocol):
    if (
        protocol.get("schema") != "crystalab_optical_images_v1"
        or protocol.get("classes") != CLASSES
    ):
        raise ValueError("Unsupported optical scene/class contract")
    for key in (
        "resolution",
        "samples",
        "threads",
        "frames_per_scene",
        "minimum_label_pixels",
    ):
        if type(protocol[key]) is not int or protocol[key] < 1:
            raise ValueError(f"Invalid {key}")
    if protocol["resolution"] < 64 or not 0 < protocol["optical_mask_threshold"] < 255:
        raise ValueError("Invalid image resolution/mask threshold")
    settings = json.loads((ROOT / protocol["geometry_config"]).read_text())[
        "simulation"
    ]
    families, identifiers, seeds, scenes = {}, set(), set(), []
    for case in protocol["scenes"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]) or case["id"] in identifiers:
            raise ValueError("Unsafe/duplicate scene ID")
        identifiers.add(case["id"])
        if case["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown scene split")
        if families.setdefault(case["family_id"], case["split"]) != case["split"]:
            raise ValueError("Scene family cannot cross splits")
        if type(case["seed"]) is not int or case["seed"] < 0 or case["seed"] in seeds:
            raise ValueError("Scenes need distinct nonnegative integer seeds")
        seeds.add(case["seed"])
        for key in ("secondary_count", "inclusion_count", "bubble_count"):
            if type(case[key]) is not int or not 0 <= case[key] <= 3:
                raise ValueError(f"Invalid {key}")
        rng = np.random.default_rng(case["seed"])
        scales = rng.uniform(0.8, 1.2, 3)
        normals = np.asarray(settings["seed_normals"], float) / scales
        offsets = np.asarray(settings["seed_offsets"], float)
        scene = dict(
            case,
            camera=dict(
                ortho_scale=float(rng.uniform(3.4, 4.1)),
                position=[
                    float(rng.uniform(-0.5, 0.5)),
                    float(rng.uniform(-0.4, 0.4)),
                    6.5,
                ],
            ),
            rotation=[
                float(rng.uniform(-0.2, 0.2)),
                float(rng.uniform(-0.2, 0.2)),
                float(rng.uniform(0, 6.28)),
            ],
            crystal_ior=float(rng.uniform(1.45, 1.65)),
            liquid_ior=float(rng.uniform(1.32, 1.35)),
            glass_ior=float(rng.uniform(1.45, 1.55)),
            roughness=float(rng.uniform(0.03, 0.12)),
            light_energy=float(rng.uniform(180, 260)),
            noise_sigma=float(rng.uniform(0, 2)),
            blur_sigma=float(rng.uniform(0, 0.45)),
            secondaries=[
                dict(
                    position=[
                        float(rng.uniform(-1.1, 1.1)),
                        float(rng.choice([-1, 1]) * rng.uniform(0.7, 1.2)),
                        0.5,
                    ],
                    scale=float(rng.uniform(0.25, 0.45)),
                )
                for _ in range(case["secondary_count"])
            ],
            inclusions=[
                dict(
                    position=[
                        float(rng.uniform(-0.12, 0.12)),
                        float(rng.uniform(-0.12, 0.12)),
                        float(rng.uniform(0.45, 0.65)),
                    ],
                    radius=float(rng.uniform(0.025, 0.045)),
                )
                for _ in range(case["inclusion_count"])
            ],
            bubbles=[
                dict(
                    position=[
                        float(rng.uniform(-1.3, 1.3)),
                        float(rng.uniform(-1.3, 1.3)),
                        float(rng.uniform(0.8, 1.4)),
                    ],
                    radius=float(rng.uniform(0.05, 0.12)),
                )
                for _ in range(case["bubble_count"])
            ],
            frames=[],
        )
        for index in range(protocol["frames_per_scene"]):
            crystal = Crystal(normals, offsets + index * 0.015)
            scene["frames"].append(
                dict(
                    index=index,
                    synthetic_time=index,
                    vertices=crystal.vertices.tolist(),
                    faces=[f.tolist() for f in crystal.faces if len(f) >= 3],
                    geometry_volume=crystal.volume,
                    secondary_visible=index > 0,
                    time_semantics="illustrative_geometry_step_not_physical_time",
                )
            )
        scenes.append(scene)
    if set(families.values()) != {"train", "validation", "test"}:
        raise ValueError("Declare nonempty train/validation/test scene families")
    return scenes


def generate_optical(protocol, output, blender, *, timeout_s=1800):
    scenes = prepare_scenes(protocol)
    blender = Path(blender).resolve()
    version = subprocess.check_output(
        [str(blender), "--version"], text=True
    ).splitlines()[0]
    if not version.startswith("Blender 4.5."):
        raise ValueError("This renderer is verified against Blender 4.5 LTS")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "scripts/blender_optical_scene.py",
        ROOT / "scripts/generate_vision_images.py",
        ROOT / "src/simulation/crystal.py",
        ROOT / protocol["geometry_config"],
    ]
    snapshot = output / "source_snapshot"
    source_hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        source_hashes[str(relative)] = file_sha256(path)
    manifest = dict(
        schema="crystalab_optical_dataset_v1",
        status="running",
        protocol=protocol,
        scenes=scenes,
        source_sha256=source_hashes,
        blender_version=version,
        blender_executable_sha256=file_sha256(blender),
        postprocess_environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            opencv=cv2.__version__,
        ),
        scope="Illustrative optical training scenes; no real KDP kinetics/quality/control validation",
        mask_contract=dict(
            geometric="Unoccluded direct projection with refractive media hidden",
            optical="Emission proxy through original refractive media; target shader changed to opaque emission",
            review_status="synthetic_proxy_requires_visual_QA; not human real labels",
        ),
    )
    save_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        save_json(output / "render_input.json", dict(protocol=protocol, scenes=scenes))
        command = [
            str(blender),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python",
            str(ROOT / "scripts/blender_optical_scene.py"),
            "--",
            "--input",
            str(output / "render_input.json"),
            "--output",
            str(output),
        ]
        with (output / "render.log").open("x") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout_s,
            )
        rows = []
        for scene in scenes:
            for frame in scene["frames"]:
                folder = output / scene["id"] / f"frame_{frame['index']:04d}"
                meta = json.loads((folder / "render.json").read_text())
                image = cv2.imread(str(folder / "rgb_raw.png"))
                if image is None:
                    raise ValueError("Missing rendered RGB")
                rng = np.random.default_rng(scene["seed"] + frame["index"] + 100000)
                if scene["blur_sigma"] > 0:
                    image = cv2.GaussianBlur(image, (3, 3), scene["blur_sigma"])
                image = np.clip(
                    image.astype(float)
                    + rng.normal(0, scene["noise_sigma"], image.shape),
                    0,
                    255,
                ).astype(np.uint8)
                if not cv2.imwrite(str(folder / "image.png"), image):
                    raise OSError("Cannot save processed RGB")
                objects = []
                for item in meta["targets"]:
                    proxy = cv2.imread(str(folder / item["optical_proxy"]), 0)
                    geometric = cv2.imread(str(folder / item["geometric_mask"]), 0)
                    if proxy is None or geometric is None:
                        raise ValueError("Missing mask pass")
                    mask, discarded = select_proxy_mask(
                        proxy, geometric, protocol["optical_mask_threshold"]
                    )
                    mask_path = f"{item['id']}_mask.png"
                    if not cv2.imwrite(str(folder / mask_path), mask):
                        raise OSError("Cannot save mask")
                    area = int(np.count_nonzero(mask))
                    objects.append(
                        dict(
                            item,
                            mask_path=str((folder / mask_path).relative_to(output)),
                            visible_pixels=area,
                            label_eligible=area >= protocol["minimum_label_pixels"],
                            mask_semantics="optical_emission_main_component",
                            discarded_proxy_pixels=discarded,
                        )
                    )
                rows.append(
                    dict(
                        sample_id=f"{scene['id']}:{frame['index']}",
                        scene_id=scene["id"],
                        family_id=scene["family_id"],
                        split=scene["split"],
                        frame_index=frame["index"],
                        synthetic_time=frame["synthetic_time"],
                        evidence_kind="synthetic",
                        image_path=str((folder / "image.png").relative_to(output)),
                        objects=objects,
                        labels=dict(
                            optical_quality=None, physical_defect_response=None
                        ),
                        review_status="unreviewed_synthetic_proxy",
                    )
                )
        with (output / "samples.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        manifest.update(
            status="complete",
            elapsed_seconds=time.perf_counter() - started,
            samples=len(rows),
            samples_by_split=dict(Counter(r["split"] for r in rows)),
            artifact_sha256={
                str(p.relative_to(output)): file_sha256(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and p.name not in {"manifest.json", "render.log"}
            },
        )
        save_json(output / "manifest.json", manifest)
        return audit_optical(output)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "manifest.json", manifest)
        raise


def audit_optical(output):
    output = Path(output)
    m = json.loads((output / "manifest.json").read_text())
    if m["status"] != "complete" or m["schema"] != "crystalab_optical_dataset_v1":
        raise ValueError("Incomplete/unsupported optical dataset")
    for path, h in m["artifact_sha256"].items():
        if (
            Path(path).is_absolute()
            or ".." in Path(path).parts
            or file_sha256(output / path) != h
        ):
            raise ValueError(f"Artifact identity mismatch: {path}")
    for path, h in m["source_sha256"].items():
        if (
            Path(path).is_absolute()
            or ".." in Path(path).parts
            or file_sha256(output / "source_snapshot" / path) != h
        ):
            raise ValueError("Source snapshot identity mismatch")
    rows = [json.loads(v) for v in (output / "samples.jsonl").read_text().splitlines()]
    expected = {(s["id"], f["index"]): (s, f) for s in m["scenes"] for f in s["frames"]}
    if len(rows) != len(expected) or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Missing/duplicate rendered samples")
    hashes, families = {}, {}
    for scene in m["scenes"]:
        if families.setdefault(scene["family_id"], scene["split"]) != scene["split"]:
            raise ValueError("Rendered scene family crosses splits")
    eligible = 0
    per_class = Counter()
    for row in rows:
        scene, frame = expected.pop((row["scene_id"], row["frame_index"]))
        if row["split"] != scene["split"] or row["family_id"] != scene["family_id"]:
            raise ValueError("Scene split changed")
        image_path = Path(row["image_path"])
        if image_path.is_absolute() or ".." in image_path.parts:
            raise ValueError("Unsafe image path")
        image = cv2.imread(str(output / image_path))
        if image is None or image.shape != (
            m["protocol"]["resolution"],
            m["protocol"]["resolution"],
            3,
        ):
            raise ValueError("Invalid rendered image")
        digest = file_sha256(output / row["image_path"])
        if hashes.setdefault(digest, row["split"]) != row["split"]:
            raise ValueError("Identical rendered images cross splits")
        wanted = {
            "seed": 0,
            **{f"inclusion_{i}": 2 for i in range(scene["inclusion_count"])},
        }
        if frame["secondary_visible"]:
            wanted.update(
                {f"secondary_{i}": 1 for i in range(scene["secondary_count"])}
            )
        if len(row["objects"]) != len(wanted) or {
            o["id"] for o in row["objects"]
        } != set(wanted):
            raise ValueError(
                "Rendered target identities differ from scene construction"
            )
        for obj in row["objects"]:
            if (
                type(obj["class_id"]) is not int
                or obj["class_id"] != wanted[obj["id"]]
                or obj["class_name"] != CLASSES[obj["class_id"]]
            ):
                raise ValueError("Rendered class differs from scene construction")
            if obj["mask_semantics"] != "optical_emission_main_component":
                raise ValueError(
                    "Unsupported mask policy; use the archived producer for older datasets"
                )
            mask_path = Path(obj["mask_path"])
            if (
                mask_path.is_absolute()
                or ".." in mask_path.parts
                or mask_path.parent != image_path.parent
            ):
                raise ValueError("Unsafe or inconsistent mask path")
            mask = cv2.imread(str(output / obj["mask_path"]), 0)
            if (
                mask is None
                or mask.shape != image.shape[:2]
                or not set(np.unique(mask)) <= {0, 255}
            ):
                raise ValueError("Invalid binary proxy mask")
            area = int(np.count_nonzero(mask))
            if area != obj["visible_pixels"] or obj["label_eligible"] != (
                area >= m["protocol"]["minimum_label_pixels"]
            ):
                raise ValueError("Proxy visibility count/gate mismatch")
            folder = output / image_path.parent
            if (
                obj["optical_proxy"] != f"{obj['id']}_optical.png"
                or obj["geometric_mask"] != f"{obj['id']}_geometric.png"
            ):
                raise ValueError("Mask pass reference differs from target identity")
            proxy = cv2.imread(str(folder / obj["optical_proxy"]), 0)
            geometric = cv2.imread(str(folder / obj["geometric_mask"]), 0)
            if (
                proxy is None
                or geometric is None
                or proxy.shape != mask.shape
                or geometric.shape != mask.shape
            ):
                raise ValueError("Missing/invalid original mask passes")
            reconstructed, discarded = select_proxy_mask(
                proxy, geometric, m["protocol"]["optical_mask_threshold"]
            )
            if (
                not np.array_equal(reconstructed, mask)
                or discarded != obj["discarded_proxy_pixels"]
            ):
                raise ValueError(
                    "Binary mask differs from original optical/geometric pass reconstruction"
                )
            eligible += int(obj["label_eligible"])
            if obj["label_eligible"]:
                per_class[obj["class_name"]] += 1
    return dict(
        status="pass",
        samples=len(rows),
        scenes=len(m["scenes"]),
        samples_by_split=dict(Counter(r["split"] for r in rows)),
        eligible_object_masks=eligible,
        eligible_masks_by_class=dict(per_class),
        physical_quality_labels=0,
        scope=m["scope"],
    )
