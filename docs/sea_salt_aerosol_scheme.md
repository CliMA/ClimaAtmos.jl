# Prognostic Sea Salt Aerosol Scheme

This document describes the design and implementation of the prognostic sea
salt aerosol parameterization added to ClimaAtmos on branch
`zg/aerosol-playground`. It covers every file changed, every new function
written, and the outstanding TODOs for future development.

---

## Motivation

The existing ClimaAtmos aerosol infrastructure reads prescribed sea salt
concentrations from a MERRA-2 NetCDF file via `TimeVaryingInput` and
interpolates them to the model grid at each radiation timestep. These
prescribed fields cannot respond to the model's own winds or SST and cannot
be advected or mixed by the model's transport operators.

This scheme replaces prescribed sea salt with **prognostic tracers**
(`ρSSLT01`–`ρSSLT05`) that are emitted from the ocean surface, transported
and mixed by the model's existing tracer infrastructure, and removed by a
deposition tendency. The five bins cover the dry radius ranges used by the
existing MERRA-2 files.

---

## Size Bin Definitions

Defined in `src/parameterized_tendencies/aerosols/sea_salt.jl`:

| Bin    | Variable  | Dry radius range (μm) |
|--------|-----------|----------------------|
| SSLT01 | `ρSSLT01` | 0.03 – 0.1           |
| SSLT02 | `ρSSLT02` | 0.1  – 0.5           |
| SSLT03 | `ρSSLT03` | 0.5  – 1.5           |
| SSLT04 | `ρSSLT04` | 1.5  – 5.0           |
| SSLT05 | `ρSSLT05` | 5.0  – 10.0          |

These bounds are stored in the compile-time constant `SEA_SALT_BIN_BOUNDS`.

---

## New and Modified Files

### `src/parameterized_tendencies/aerosols/sea_salt.jl` (new)

The core physics file. Contains all sea salt functions.

#### `monin_obukhov_wind_at_height(z_target, ustar, L, uf_params, κ, z₀)`

Reconstructs the mean wind speed at any height `z_target` (m) from
Monin-Obukhov similarity theory:

    u(z) = (u★ / κ) * [log(z / z₀) - ψ_m(z/L)]

where `ψ_m` is the stability correction function from
`SurfaceFluxes.UniversalFunctions`. Called inside `sea_salt_emission_tendency!`
to extrapolate from the model's first layer (~15 m) down to the 10 m reference
height required by the Gong (2003) formula. Pure scalar function — no
allocations, GPU-compatible.

#### `gong2003_dF_dr(r, u_10, ϴ, SST; SST_adj=true)`

Evaluates the Gong (2003) sea salt number emission spectrum at a single dry
radius `r` (μm):

    dF/dr = 1.373 · u₁₀^3.41 · r^(−A) · (1 + 0.057·r^3.45) · 10^(1.607·exp(−B²))

with:
- `A = 4.7 · (1 + ϴr)^(−0.017 r^(−1.44))`
- `B = (0.433 − log₁₀r) / 0.433`

When `SST_adj=true`, applies the sea surface temperature correction from
Jaeglé et al. (2011):

    SST_factor = 0.3 + 0.1·SST − 0.0076·SST² + 0.00021·SST³

Reference: Gong, S. L. (2003), Global Biogeochem. Cycles, 17(4), 1097.

#### `integrate_bin_gong2003(r_lo, r_hi, u_10, theta, SST, ::Val{N})`

Integrates `gong2003_dF_dr` over a radius bin `[r_lo, r_hi]` using an
N-point trapezoidal rule. `Val{N}` makes `N` a compile-time constant,
allowing `ntuple` to produce a statically-sized tuple — required for
GPU compatibility (no heap allocation).

#### `sea_salt_emission_flux(u_10, T_sfc, bin_index)`

Top-level wrapper: looks up bin bounds from `SEA_SALT_BIN_BOUNDS`, sets
`theta = 30` (Gong 2003 default), and calls `integrate_bin_gong2003` with
32 quadrature points.

#### `sea_salt_emission_tendency!(Yₜ, Y, p, t)`

Called from `surface_flux_tendency!` at every timestep. For each active
prognostic aerosol bin:

1. Extrapolates `u_10` from `sfc_conditions.ustar` and
   `sfc_conditions.obukhov_length` using `monin_obukhov_wind_at_height`.
2. Reads `T_sfc` from `sfc_conditions`.
3. Reads `land_sea_mask` from `p.surface_fractions`.
4. Computes the surface number flux and multiplies by `(1 − land_sea_mask)`
   to zero the flux over land.
5. Applies the flux as a bottom boundary condition using
   `boundary_tendency_scalar`, which localises it to the lowest model layer
   via a `DivergenceF2C` operator.

#### `sea_salt_deposition_tendency!(Yₜ, Y, p, t)`

Called from `additional_tendency!` in `remaining_tendency.jl` at every
timestep. Applies a uniform exponential decay across all vertical levels:

    d(ρSSLTxx)/dt = −λ · ρSSLTxx,   λ = log(2) / half_life

Current `half_life = 0.55 days` is a placeholder.

---

### `src/cache/surface_fractions.jl` (new)

#### `surface_fractions_cache(Y, land_sea_mask_file)`

Allocates and populates static 2D surface fraction fields on the bottom-face
space (`axes(Fields.level(Y.f, Fields.half))`). Currently provides:

- **`land_sea_mask`**: land fraction at each surface point (0 = pure ocean,
  1 = pure land). Loaded from `land_sea_mask_file` (a NetCDF file with a
  `land_fraction` variable) if a path is provided; otherwise initialized to
  zero (all ocean). Loading uses `TimeVaryingInput` with
  `InterpolationsRegridder` and evaluates once at `t = 0`.

The NamedTuple returned is stored as `p.surface_fractions` in `AtmosCache`.
Designed to be extended with additional fields (e.g. `desert_fraction` for
dust emission) without touching any other infrastructure.

---

### `src/solver/types.jl`

Added `prognostic_aerosols::PA` as a new type-parameterized field of
`AtmosModel`, with type parameter `PA` (an `NTuple{N, Symbol}`). Carrying `N`
in the type allows `ntuple(f, Val(N))` in the tendency functions to be
resolved at compile time — required for GPU-safe static allocation.

---

### `src/setups/common/prognostic_variables.jl`

#### `prognostic_aerosol_variables(ρ, names::NTuple{N, Symbol})`

Constructs the initial prognostic state entries for all active aerosol bins.
Given names `(:SSLT01, :SSLT02, ...)`, generates a `NamedTuple`
`(ρSSLT01 = 0, ρSSLT02 = 0, ...)` initialized to zero. Called from
`grid_scale_center_variables`, which assembles the full `Y.c` initial
condition.

A zero-argument fallback `prognostic_aerosol_variables(ρ, ::Tuple{})` 
returns an empty NamedTuple when no aerosols are configured, ensuring no
overhead in default runs.

---

### `src/cache/cache.jl`

- Added `land_sea_mask_file = ""` as a new optional argument to `build_cache`.
- Added call to `surface_fractions_cache(Y, land_sea_mask_file)`.
- Added `surface_fractions` to the `AtmosCache` struct (new type parameter
  `SFRAC`) and to the `args` tuple passed to `AtmosCache(args...)`.

---

### `src/prognostic_equations/surface_flux.jl`

Added a call to `sea_salt_emission_tendency!(Yₜ, Y, p, t)` at the end of
`surface_flux_tendency!`. This is the natural location because sea salt
emission is a surface boundary flux, parallel to the existing momentum,
energy, and moisture fluxes.

---

### `src/prognostic_equations/remaining_tendency.jl`

Added a call to `sea_salt_deposition_tendency!(Yₜ, Y, p, t)` in
`additional_tendency!`. Deposition lives here (not in `surface_flux.jl`)
because it acts on the full 3D column, and because future explicit
dry/wet deposition will depend on 3D fields (precipitation, cloud water)
available in `p.precomputed`.

---

### `src/solver/model_getters.jl`

Extended `get_tracers` to parse the new `prognostic_aerosols` config key
and return it as `prognostic_aerosol_names` alongside the existing
`aerosol_names` and `time_varying_trace_gas_names`.

---

### `src/solver/type_getters.jl`

Reads `land_sea_mask_file` from `config.parsed_args` and passes it to
`build_cache`.

---

### `config/default_configs/default_config.yml`

Two new config keys:

```yaml
prognostic_aerosols:
  help: "Which aerosol bins to treat as prognostic tracers
         (e.g., [\"SSLT01\", \"SSLT02\"])."
  value: []

land_sea_mask_file:
  help: "Path to a NetCDF file with a 'land_fraction' variable
         (lon×lat, values 0–1). If empty, the entire surface is
         treated as ocean (sea_fraction = 1 everywhere)."
  value: ""
```

---

### `src/ClimaAtmos.jl`

Added two `include` calls:
- `include(joinpath("parameterized_tendencies", "aerosols", "sea_salt.jl"))`
- `include(joinpath("cache", "surface_fractions.jl"))`

---

## How the tracer infrastructure works

The `ρSSLT01`–`ρSSLT05` fields require no special registration to be
advected, hyperdiffused, or vertically mixed. The `gs_tracer` system in
`src/utils/variable_manipulations.jl` automatically includes any field in
`Y.c` whose name starts with `ρ` (excluding `ρ`, `ρe_tot`, `ρtke`) via
`is_ρ_weighted_name` and `gs_tracer_names`. This means the sea salt bins
receive for free:

- Horizontal and vertical advection (`advection.jl`)
- Horizontal hyperdiffusion (`hyperdiffusion.jl`, with DSS on sphere runs)
- Vertical turbulent diffusion (`vertical_diffusion_boundary_layer.jl`)
- Viscous sponge damping near the model top

---

## Outstanding TODOs

### In `sea_salt_emission_flux`
- **Number → mass flux**: convert from particles m⁻² s⁻¹ to kg m⁻² s⁻¹
  using assumed sea salt density (~2165 kg m⁻³) and bin mean radius.
- **SST-dependent theta**: `theta = 30` is the Gong (2003) default; the
  original paper allows temperature dependence that has not been implemented.

### In `sea_salt_emission_tendency!`
- **Land masking performance**: the current `@.` broadcast evaluates
  `sea_salt_emission_flux` at every surface point even where
  `land_sea_mask = 1`. Skipping ocean-only points requires a different
  loop structure that is not straightforward with ClimaCore's broadcast model.

### In `sea_salt_deposition_tendency!`
- **Explicit dry deposition**: replace exponential decay with a physically
  based dry deposition velocity that depends on near-surface wind speed,
  particle size, and surface layer stability.
- **Wet deposition**: add scavenging by precipitation and cloud liquid water
  using fields from `p.precomputed`.
- **Gravitational (Stokes) settling**: coarse bins SSLT04–SSLT05
  (r > 1.5 μm) have settling velocities of mm s⁻¹ to cm s⁻¹ that dominate
  over turbulent diffusion. These need an explicit downward tendency term.
  Fine bins (SSLT01–SSLT02) are genuinely passive.
- **Bin-specific half-lives**: the current 0.55-day placeholder should be
  replaced with bin-specific values loaded from ClimaParams.

### In `surface_fractions_cache`
- **Desert fraction**: add a `desert_fraction` field for use by a future
  dust emission parameterization, following the same pattern as
  `land_sea_mask`.
