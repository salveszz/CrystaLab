# Focused 3D liquid grid

The historical result directories cited below were cleared on 2026-11-07.
The final benchmark summary/provenance is retained in
[NUMERICAL_CHECKPOINT.json](NUMERICAL_CHECKPOINT.json); fresh runs regenerate raw artifacts.

The crystal remains a continuous convex face-plane solid. Only the liquid mesh
changes: smaller Cartesian boxes resolve the crystal neighborhood and larger
boxes cover the smoother far field. This is a fixed rectilinear mesh, not dynamic
adaptive refinement or a different crystal geometry.

`grid_focus_width=0` selects a uniform grid. A positive value specifies the full
physical width of a central band around the configured seed center on each axis.
Approximately 60% of that axis's cells cover the band uniformly. Geometrically
stretched tails fill the remaining distance to each wall; all edges remain
strictly ordered and the tank dimensions remain unchanged. Bands are clipped at
walls, and an axis covered by the band stays uniform.

`configs/focused.json` uses a 3.6-unit band and 40 cells on each axis. Its central
spacing is 0.15 units, versus 0.25 for the old uniform 40x32x32 tank. It uses 64,000
cells rather than the much larger globally uniform mesh required for comparable
near-crystal resolution. This is an implementation choice to test, not an
experimental length scale or a claim that all settings are resolved.

`scripts/run_simulation.py` selects this configuration by default. The original
`configs/default.json` remains the uniform reference used by the original
convergence protocol; select it explicitly with `--config` when reproducing that
experiment. `SimulationConfig()` also retains lightweight uniform defaults;
constructing it without a study configuration is not a convergence certificate.

## Accounting and transport

- Cell capacities use actual box volumes. Solid intersections, liquid fractions,
  cell centroids and aperture centroids follow actual edge coordinates.
- Diffusive conductance and the displacement potential use actual distances
  between neighboring Cartesian centers. Reconstructed gradients use actual
  liquid centroids; conservative edge fluxes retain their paired signs.
- Slope limits use actual wet-box bounds. Moving-boundary first moments and
  midpoint predictions therefore work on unequal cells as well as uniform ones.
- `cell_volumes[z,y,x]` stores all box volumes. The scalar `cell_volume` is the
  minimum box volume, used for a conservative continuity-error normalization.
  Likewise `dx/dy/dz` are minimum axis spacings, not universal cell widths.
- The legacy `boundary_layer_cells` setting multiplies the nominal mean width
  `max(width/nx, depth/ny, height/nz)`. The study converts a fixed physical depth
  of 0.75 to that setting. `focused.json` uses 3.0, preserving the same depth as
  the old default's 1.5 on its coarser uniform grid.

The final field archive contains `x_edges`, `y_edges`, `z_edges`, `cell_volumes`,
liquid centroids and open-face centroids. Slice rendering uses these edges with
`pcolormesh`; uniform-pixel images would misplace stretched cells.

## Controlled refinement

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_convergence.py \
  --study-config configs/convergence_focused.json --jobs 5 \
  --output results/convergence_focused
```

The focused protocol declares 24x24x24, 32x32x32 and 40x40x40 grids and the same
120-unit duration, 0.75 sampling depth, timestep levels and 2% working tolerance.
Physical coefficients, tank dimensions and the seed are unchanged. The old
uniform protocol and its failures remain separate reference experiments; they
must not be relabeled as focused results.

Tests verify ordered edges, tank/solid/liquid volumes, independent centroid
references, affine sampling, bounded/conservative diffusion, cosine-mode
refinement, prescribed layer translation and coupled mass conservation. The
focused-grid checkpoint passed 96 tests. A regression specifically verifies that the
concentration slope limiter uses real focused-cell bounds, rather than boxes
fabricated from indices and minimum spacing. The completed source-matched study
in `results/convergence_focused_v2_561c551/` passes all declared gates: finest
spatial growth/face differences are 0.086%/0.098%, and finest temporal differences
are 0.157%/0.122%. Every case also passes conservation, positivity and final time.
This validates numerical behavior for this setup, not physical KDP parameters.

The mesh does not move its fine band as the crystal grows. Larger seeds,
different locations, faster growth or changed transport coefficients can require
a wider/finer band and a new resolution check. This benchmark alone does not
establish resolution independence across a future randomized training domain.
