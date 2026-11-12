# Bounded CPU/Vulkan flow comparison

The local AMD Radeon 860M can execute fp64 compute through Vulkan. A completed
comparison on the existing fixed-domain flow/solute solver passes its frozen
accuracy checks, but the hybrid GPU implementation is **slower** at every tested
size. This closes a first CPU/GPU experiment, not the full pitch HPC deliverable.
Heat transport, thermal/solutal feedback, growth and a physically justified
nucleation-risk map remain open. This does not enable GPU YOLO training in the
existing CPU PyTorch environment.

## Case and implementation

The [protocol](../configs/vulkan_flow_experiment.json) was frozen before timing:
SHA256 `a0b16ccb0bfa80449f297b4c4c5bff3a2fcf7b1ab356a32b4d3f8b3447e805e8`.
It uses the independently referenced three-component periodic ABC decay solution
already used by the [flow verification](FLOW_MODEL.md), plus a nonuniform passive
solute field. The cubic domain is 0.01 m wide; grids are 16³, 24³ and 32³.
Four steps of 0.03125 s advance 0.125 s. Constant density, viscosity and solute
diffusivity come from the existing [SI reference](../configs/physical_reference.json).
The chosen concentration field is a numerical test, not a measured or calibrated
thermodynamic state. There is no crystal interface or actuator in this case.

The optional [Vulkan backend](../src/simulation/vulkan_spmv.py) sends CSR sparse
products in momentum/pressure CG solves to an actual fp64 GPU. Assembly, Jacobi
preconditioning, CG vectors/reductions, nonlinear terms and solute remain CPU.
The [scoped integration](../src/simulation/linear.py) leaves the default CPU path
available and propagates GPU failure. It rejects CPU/llvmpipe fallback. Both
paths retain `rtol=2e-13`, `maxiter=4000` and the independent CPU residual check
at `3e-12 * ||rhs||`. All tested systems exceed the existing small direct-solve
threshold. No tolerance was loosened for performance.

Static matrices are cached by content, including shape; mutation changes the key.
Each product copies the vector into coherent mapped memory, submits compute,
waits for its fence and copies the result to NumPy. Host/shader barriers declare
visibility. Cache allocation has a 256 MiB buffer budget. These transfers and
hash/assembly work are included in the measured stepping time.

Hardware: AMD Ryzen AI 7 350 CPU, integrated AMD Radeon 860M Graphics
(`RADV KRACKAN1`, vendor 4098, device 4372), Mesa driver 26.2.5. Compute uses a
128-thread workgroup and shaderFloat64. The optional Python binding is pinned to
`vulkan==1.3.275.1`; the cached wheel was verified against official PyPI SHA256
`e1e0ddf57d3a7d19f79ebf1e192b20dbd378172b027cad4f495d961b51409586`.
The system shaderc shared library compiles our GLSL shader. This used existing
drivers and a project-local optional dependency, with no global driver install.

## Actual results

Each backend/grid has one complete warmup and three measured fresh-process runs,
24 runs total. Measured order alternates CPU/GPU and GPU/CPU. CPU math-library
thread environment variables are fixed to one. No matrix cache survives a run.
Medians in seconds:

| Grid | CPU construction + stepping | GPU construction + stepping | CPU stepping | GPU stepping | GPU stepping / CPU |
|---|---:|---:|---:|---:|---:|
| 16³ | 0.07489 | 0.34931 | 0.04027 | 0.19825 | 4.92 |
| 24³ | 0.24725 | 0.85585 | 0.12506 | 0.62449 | 4.99 |
| 32³ | 0.57034 | 1.95629 | 0.31765 | 1.58977 | 5.00 |

Construction includes GPU context, shader compilation and pipeline creation.
Both timing boundaries exclude process startup/imports, post-run diagnostics,
archive writes and teardown. The step-only result still includes first-use matrix
uploads. Raw individual times are preserved in `results/vulkan_abc_solute_v2`.
This is a small local timing experiment, not an isolated hardware throughput,
energy or general GPU-performance benchmark. The transfer/submission design is
a plausible contributor to the slowdown; no profiling study isolates its cause.

Every GPU run records 16 solves, two distinct uploaded matrices and respectively
475, 689 or 900 dispatches. Allocated Vulkan buffer memory is 851,840; 2,875,264;
and 6,815,616 bytes. This includes allocation alignment, not pipeline/driver
memory. Median process RSS in KiB is CPU/GPU 94,956/183,144; 127,472/183,208;
and 187,760/238,860. RSS is the whole process high-water mark, including diagnostics
and archives. Shared integrated-GPU RAM may overlap RSS; these are not additive
dedicated-VRAM measurements.

The largest CPU/GPU differences across all warmup/measured pairs are:

- Relative velocity: `2.112e-16`.
- Relative dynamic pressure: `2.012e-14`.
- Concentration: `8.527e-14 kg/m³`.
- Dimensionless concentration-excess indicator: `4.264e-16`.

Velocity error against analytic ABC decay decreases from 0.001303 to 0.000691.
Pressure error is about 0.0341–0.0343 at the fixed timestep. All runs satisfy the
declared reference, divergence, scalar bounds and mass checks. This is a bounded
accuracy check; it is not a new spatial/temporal convergence study. Sparse-product
probes include sizes around the 128-thread boundary and a known SPD solution,
matrix mutation and invalid-vector rejection.

The saved map is `max(C - initial_volume_mean_C, 0) / initial_volume_mean_C`.
It marks positive concentration heterogeneity only. It is not supersaturation,
a nucleation probability, an inclusion predictor or a calibrated physical risk.

## Reproduction and independent audit

```bash
# Requires the optional binding, system libshaderc and access to an actual GPU.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/benchmark_vulkan_flow.py \
  --protocol configs/vulkan_flow_experiment.json \
  --output results/vulkan_abc_solute_new

.venv/bin/python scripts/audit_vulkan_flow.py \
  --study results/vulkan_abc_solute_v2 \
  --output results/vulkan_abc_solute_new_audit.json
```

Choose unused output paths. The independent auditor imports neither the solver,
GPU backend nor benchmark producer. It reconstructs analytic fields, periodic
aliases, divergence, initial cell averages, scalar mass/bounds, map values,
12 CPU/GPU comparisons and timing aggregates from all 24 saved field files.
It verifies producer source hashes. It does not independently remeasure wall time
or device activity. The [checkpoint](VULKAN_FLOW_CHECKPOINT.json) records exact
identities, results and verification evidence.

The first study attempt retained a JSON serialization failure before any measured
run; version 2 fixes conversion of NumPy booleans without changing the protocol
or numerical criteria. Earlier pipeline-creation probes exposed an embedded CFFI
stage/string lifetime error; retaining the stage owner fixed it before the study.
No failed output was silently overwritten or counted as an accepted comparison.

## Next HPC work

Keep CPU as the default for these workloads. Any faster GPU implementation needs
another frozen, comparable experiment, including transfers and accuracy, before
adoption. Avoid repeated host round trips by considering solver vectors and
reductions together; the current result does not establish that approach's speed.

The pitch still needs sourced heat capacity/conductivity and thermal boundary
conditions, a declared heat/mass coupling and verified thermal transport. A
physical risk map needs its own supported definition and calibration. Neither
more image training nor this numerical comparison supplies those inputs. Qiskit
remains a separate chemically justified impurity-cluster study.

Method references: [Vulkan synchronization specification](https://docs.vulkan.org/spec/latest/chapters/synchronization.html),
[shaderc C API](https://github.com/google/shaderc/blob/main/libshaderc/include/shaderc/shaderc.h),
[Python binding](https://github.com/realitix/vulkan),
[versioned PyPI release](https://pypi.org/project/vulkan/1.3.275.1/).
