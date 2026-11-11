# Blender optical synthetic-image pipeline

The first optical image path renders actual 3D scenes in Blender/Cycles on CPU,
then applies declared OpenCV blur/noise. It reuses the existing convex crystal
geometry. Material, camera and temporal ranges are illustrative design choices;
they are not calibrated KDP optics, growth kinetics or defect-response laws.

## Renderer and source identity

[Blender 4.5 LTS](https://www.blender.org/releases/4-5/) is pinned in
[`configs/blender_runtime.json`](../configs/blender_runtime.json). The development
run uses the official Linux 4.5.16 portable archive, verified against the official
SHA256 listing before extraction. The runtime is under `/tmp/`, outside Git; use
an explicit `--blender` path. No system installation or GPU is required.

Generation records the binary hash/version, full protocol and resolved scene
parameters, actual source snapshots, geometry, raw passes and processed images.
It refuses an existing output directory and retains a failed manifest/log when
the renderer or postprocessing fails. The runtime archive must be reinstalled
after temporary storage is cleared.

## Scene and timing contract

The small smoke protocol is
[`configs/synthetic_vision_smoke.json`](../configs/synthetic_vision_smoke.json):
six base scene groups, three frames each, split 12/3/3 by whole scene family.
It varies crystal shape/orientation, camera pose/scale, refractive materials,
roughness, illumination, bubbles, secondary crystals and inclusion candidates.
Related frames remain together; changing a seed cannot move a related family
across splits.

Secondary crystals become present after the initial frame. Facet offsets advance
under an explicitly illustrative geometry-step rule. Inclusion placement is a
rendering construction. Neither behavior models a validated response to temperature
or agitation. Time remains `illustrative_geometry_step_not_physical_time`, with
no asserted conversion to a real growth run. Quality and physical defect-response
labels remain null.

## Three different mask meanings

| Artifact | Definition and limit |
|---|---|
| RGB image | Cycles render with glass/liquid/crystal materials, followed by recorded blur/noise. RGB uses denoising and tone mapping; this is an image-generation assumption. |
| Geometric mask | Target rendered alone as opaque emission, with other objects and refractive media hidden. It is an unoccluded direct projection, not the observed boundary through solution. |
| Optical emission proxy | Target shader changed to opaque emission, seen through the original refractive media against a dark scene. Target transmission differs from RGB, so this is an annotation proxy rather than exact empirical visibility. |
| Binary training candidate | Thresholded proxy component with greatest overlap with the geometric projection. Reflected ghost components are excluded and their pixel count recorded. Small masks remain in metadata but are marked below the labeling gate. |

The optical proxy, geometric mask and binary candidate are all retained. Do not
silently call direct projection the visible optical mask. The initial classes
are seed, secondary crystal and inclusion candidate. A rendered inclusion mask
does not establish a real inclusion or optical-quality measurement. Mask eligibility
is a pixel-visibility convention, not a physical defect threshold.

## Generate and audit

With an explicit verified Blender runtime:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/generate_vision_images.py \
  --config configs/synthetic_vision_smoke.json \
  --blender /tmp/crystalab-blender-runtime/blender-4.5.16-linux-x64/blender \
  --output data/synthetic_vision_smoke_v2
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/generate_vision_images.py \
  --output data/synthetic_vision_smoke_v2 --audit
```

The audit checks artifact identity, image shape, scene membership/splits, unique
sample IDs, duplicate-image leakage and mask binary/count/visibility consistency.
Visual RGB/mask review is a separate gate; these checks cannot establish realistic
optics or real-domain transfer. The six-scene smoke batch is not enough to claim
a trained, generalizable vision model.

## Visual failure history and next gate

The first 18-sample render (`synthetic_vision_smoke_v1`) completed its software
checks but failed visual review: noisy RGB and reflected ghost mask components.
It is retained as rejected development evidence. The revised protocol uses more
samples, RGB denoising/tone mapping and an observation view inside the vessel walls;
postprocessing separates the main optical component from reflections.

Only inspect development scenes for generator tuning. Real held-out images remain
excluded. Before model fitting, inspect revised RGB and overlays, preserve any
failed cases and choose a declared synthetic training cohort. Report synthetic
holdout performance separately from independently reviewed real data. The real
[annotation workflow](ANNOTATIONS.md) remains a distinct gate.
