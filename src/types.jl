import ClimaCore.Quadratures.GaussQuadrature as GQ
import StaticArrays as SA
import Thermodynamics as TD
import Dates

import ClimaParams as CP
import ClimaUtilities.ClimaArtifacts: @clima_artifact
import LazyArtifacts
import CloudMicrophysics.Parameters as CMP

"""
    AbstractMicrophysicsModel

Water and microphysics representation carried by the model.

The choice fixes which water tracers are prognostic and which microphysical
conversion rates are computed. Selected by the YAML key `microphysics_model`.

Subtypes:

  - `DryModel`: no water at all (`microphysics_model: "dry"`).
  - `EquilibriumMicrophysics0M`: saturation-adjustment equilibrium with a
    0-moment sink (`"0M"`).
  - `NonEquilibriumMicrophysics1M`: 1-moment non-equilibrium cloud and
    precipitation (`"1M"`).
  - `NonEquilibriumMicrophysics2M`: 2-moment warm-rain microphysics (`"2M"`).
  - `NonEquilibriumMicrophysics2MP3`: 2-moment warm rain with P3 ice (`"2MP3"`).
"""
abstract type AbstractMicrophysicsModel end

"""
    DryModel

Dry dynamics: no water tracers, no latent heating, no microphysics.

Selected by `microphysics_model: "dry"`.
"""
struct DryModel <: AbstractMicrophysicsModel end

"""
    EquilibriumMicrophysics0M

Equilibrium (saturation-adjustment) moisture with a 0-moment precipitation sink.

Only `ρq_tot` is prognostic; cloud liquid and ice are diagnosed by saturation
adjustment, and condensate in excess of a threshold is removed instantaneously.
Selected by `microphysics_model: "0M"`. Requires `use_sgs_quadrature: true`.
"""
struct EquilibriumMicrophysics0M <: AbstractMicrophysicsModel end

"""
    NonEquilibriumMicrophysics1M(;
        n_substeps = 3,
        n_substeps_quad = 2,
        process_options...,
    )

Non-equilibrium 1-moment microphysics with prognostic cloud and precipitation mass.

Carries `ρq_tot`, `ρq_lcl`, `ρq_icl`, `ρq_rai`, and `ρq_sno`, with conversion
rates from CloudMicrophysics.jl. Selected by `microphysics_model: "1M"`.

The substep counts are handed to CloudMicrophysics' averaged tendency
evaluation, which subdivides the dynamics step into substeps and returns the
time-averaged rates; this keeps the explicit sources stable at large `dt`.

`process_options` selects the variant used for each individual process; they are
collected into a `CMP.Microphysics1MOptions`, which documents the full list of
processes and their available variants. Passing `nothing` for a process disables
it.

Any process that is not passed keeps its default, so
`NonEquilibriumMicrophysics1M()` turns every process on with these variants:

| Process                         | Default variant              |
|:------------------------------- |:---------------------------- |
| `cloud_liquid_formation`        | `CloudLiquidFormation()`     |
| `cloud_ice_formation`           | `TemperatureDependent()`     |
| `cloud_ice_melt`                | `CloudIceMelt()`             |
| `rain_autoconversion`           | `Kessler1M()`                |
| `snow_autoconversion`           | `NoSupersaturation()`        |
| `rain_condensation_evaporation` | `RainEvaporation()`          |
| `snow_deposition_sublimation`   | `DepositionAndSublimation()` |
| `snow_melt`                     | `SnowMelt()`                 |
| `cloud_liquid_rain_accretion`   | `CloudLiquidRainAccretion()` |
| `cloud_liquid_snow_accretion`   | `CloudLiquidSnowAccretion()` |
| `cloud_ice_rain_accretion`      | `CloudIceRainAccretion()`    |
| `cloud_ice_snow_accretion`      | `CloudIceSnowAccretion()`    |
| `rain_snow_accretion`           | `RainSnowAccretion()`        |

These match the defaults of the corresponding config keys, so a model built here
and one built from an unmodified configuration file agree (see
`get_microphysics_1m_options`). All variants except `cloud_ice_formation` are
also the `CMP.Microphysics1MOptions` defaults.

# Fields

  - `n_substeps`: Number of substeps used when the tendencies are evaluated at
    grid-mean conditions (no SGS quadrature) [-].
  - `n_substeps_quad`: Number of substeps used when the tendencies are integrated
    over the SGS quadrature [-].
  - `processes`: Per-process variant selection, a `CMP.Microphysics1MOptions`.

# Examples

```julia
import ClimaAtmos as CA
import CloudMicrophysics.Parameters as CMP

model = CA.NonEquilibriumMicrophysics1M(;
    n_substeps = 1,
    rain_autoconversion = CMP.PrescribedNd(),
    cloud_ice_formation = CMP.ConstantTimescale(),
    rain_snow_accretion = nothing,  # process off
)
```
"""
struct NonEquilibriumMicrophysics1M{OPT} <: AbstractMicrophysicsModel
    n_substeps::Int  # number of microphysics substeps
    n_substeps_quad::Int  # number of microphysics substeps with sgs quadrature
    processes::OPT  # per-process variant selection (CMP.Microphysics1MOptions)
    function NonEquilibriumMicrophysics1M(;
        n_substeps = 3,
        n_substeps_quad = 2,
        process_options...,
    )
        # `cloud_ice_formation` overrides the CloudMicrophysics default
        # (`ConstantTimescale`) so that these defaults match `default_config.yml`
        processes = CMP.Microphysics1MOptions(;
            cloud_ice_formation = CMP.TemperatureDependent(),
            process_options...,
        )
        return new{typeof(processes)}(n_substeps, n_substeps_quad, processes)
    end
end

"""
    NonEquilibriumMicrophysics2M

Two-moment warm-rain microphysics: prognostic mass and number concentrations.

Cloud liquid and rain carry both mass and number, so the droplet size
distribution responds to aerosol and dynamical forcing. There are no ice
processes. Selected by `microphysics_model: "2M"`.
"""
struct NonEquilibriumMicrophysics2M <: AbstractMicrophysicsModel end

"""
    NonEquilibriumMicrophysics2MP3

Two-moment warm rain combined with the P3 predicted-particle-properties ice scheme.

Extends `NonEquilibriumMicrophysics2M` with prognostic ice mass, ice
number, rime mass, and rime volume. Selected by `microphysics_model: "2MP3"`.
"""
struct NonEquilibriumMicrophysics2MP3 <: AbstractMicrophysicsModel end

"""
    NonEquilibriumMicrophysics

Union of the microphysics models that carry prognostic condensate, i.e. all
models for which cloud water is *not* diagnosed by saturation adjustment.
"""
const NonEquilibriumMicrophysics = Union{
    NonEquilibriumMicrophysics1M,
    NonEquilibriumMicrophysics2M,
    NonEquilibriumMicrophysics2MP3,
}

"""
    MoistMicrophysics

Union of every microphysics model that carries water, i.e. all subtypes of
`AbstractMicrophysicsModel` except `DryModel`.
"""
const MoistMicrophysics = Union{
    EquilibriumMicrophysics0M,
    NonEquilibriumMicrophysics1M,
    NonEquilibriumMicrophysics2M,
    NonEquilibriumMicrophysics2MP3,
}

"""
    TracerNonnegativityMethod

Strategy for keeping the microphysical tracers nonnegative.

The constrained tracers are the condensate tracers carried by the microphysics
model (`ρq_lcl`, `ρq_icl`, `ρq_rai`, `ρq_sno`); `q_tot` is included as well when
the `qtot` type parameter is `true`.

Subtypes:

  - `TracerNonnegativityElementConstraint{qtot}`: redistribute tracer mass
    instantaneously within a spectral element, i.e. horizontally.
  - `TracerNonnegativityVaporConstraint{qtot}`: redistribute tracer mass
    instantaneously between vapor (`q_vap = q_tot - q_cond`) and each tracer.
  - `TracerNonnegativityVaporTendency`: exchange mass between vapor and each
    tracer gradually, through a tendency.
  - `TracerNonnegativityVerticalWaterBorrowing`: redistribute tracer mass
    vertically with ClimaCore's `VerticalMassBorrowingLimiter`. The `qtot` type
    parameter is fixed to `false` for this method.

`qtot` is `true` when `q_tot` is among the constrained tracers, `false` otherwise.

# Constructor

    TracerNonnegativityMethod(method::String; include_qtot = false)

Build the method selected by `method`.

# Arguments

  - `method`: One of
      + `"elementwise_constraint"` → `TracerNonnegativityElementConstraint{include_qtot}()`,
      + `"vapor_constraint"` → `TracerNonnegativityVaporConstraint{include_qtot}()`,
      + `"vapor_tendency"` → `TracerNonnegativityVaporTendency()`,
      + `"vertical_water_borrowing"` → `TracerNonnegativityVerticalWaterBorrowing()`.

# Keyword Arguments

  - `include_qtot = false`: Whether `q_tot` is also constrained. Passing `true`
    with `"vapor_tendency"` or `"vertical_water_borrowing"` is an error, because
    those methods do not support it.

# Notes

In YAML configs the equivalent key is `tracer_nonnegativity_method`, where the
`include_qtot = true` variants are spelled by appending `_qtot` to the method
name (e.g. `vapor_constraint_qtot`).

# Examples

```julia
method = ClimaAtmos.TracerNonnegativityMethod("vapor_constraint"; include_qtot = true)
```
"""
abstract type TracerNonnegativityMethod end

"""
    TracerNonnegativityConstraint{qtot}

Tracer-nonnegativity methods that act as instantaneous constraints, i.e. that
clip and redistribute mass in place rather than through a tendency.

Subtypes: `TracerNonnegativityElementConstraint`,
`TracerNonnegativityVaporConstraint`, and
`TracerNonnegativityVerticalWaterBorrowing`. See
`TracerNonnegativityMethod` for the meaning of `qtot`.
"""
abstract type TracerNonnegativityConstraint{qtot} <: TracerNonnegativityMethod end

"""
    TracerNonnegativityElementConstraint{qtot}

Restore nonnegativity by redistributing tracer mass horizontally within each
spectral element. See `TracerNonnegativityMethod`.
"""
struct TracerNonnegativityElementConstraint{qtot} <: TracerNonnegativityConstraint{qtot} end

"""
    TracerNonnegativityVaporConstraint{qtot}

Restore nonnegativity by moving mass between water vapor and the offending
tracer at the same point. See `TracerNonnegativityMethod`.
"""
struct TracerNonnegativityVaporConstraint{qtot} <: TracerNonnegativityConstraint{qtot} end

"""
    TracerNonnegativityVaporTendency

Restore nonnegativity gradually, through a relaxation tendency that exchanges
mass between water vapor and each tracer. See `TracerNonnegativityMethod`.
"""
struct TracerNonnegativityVaporTendency <: TracerNonnegativityMethod end

"""
    TracerNonnegativityVerticalWaterBorrowing

Restore nonnegativity by borrowing tracer mass from the level below, using
ClimaCore's `VerticalMassBorrowingLimiter` with a threshold of zero. See
`TracerNonnegativityMethod`.
"""
struct TracerNonnegativityVerticalWaterBorrowing <: TracerNonnegativityConstraint{false} end

function TracerNonnegativityMethod(method::String; include_qtot::Bool = false)
    if method == "elementwise_constraint"
        return TracerNonnegativityElementConstraint{include_qtot}()
    elseif method == "vapor_constraint"
        return TracerNonnegativityVaporConstraint{include_qtot}()
    elseif method == "vapor_tendency"
        include_qtot &&
            error("TracerNonnegativityVaporTendency does not support `include_qtot = true`")
        return TracerNonnegativityVaporTendency()
    elseif method == "vertical_water_borrowing"
        include_qtot &&
            error("TracerNonnegativityVerticalWaterBorrowing does not support \
                `include_qtot = true`")
        return TracerNonnegativityVerticalWaterBorrowing()
    else
        error("Invalid tracer nonnegativity method: $method")
    end
end

"""
    AbstractSGSamplingType

Sampling strategy for the subgrid-scale distribution of temperature and total
water when cloud fraction and microphysical tendencies are evaluated.

Subtypes:

  - `SGSMean`: evaluate at the grid mean only.
  - `SGSQuadrature`: integrate over the SGS distribution with Gauss-Hermite
    quadrature (see `SGSQuadrature`).
"""
abstract type AbstractSGSamplingType end

"""
    SGSMean

Evaluate subgrid-scale diagnostics at the grid-mean state, without sampling the
SGS distribution.
"""
struct SGSMean <: AbstractSGSamplingType end



"""
    AbstractCloudModel

Strategy for diagnosing the cloud fraction. Selected by the YAML key `cloud_model`.

Subtypes:

  - `GridScaleCloud`: cloud fraction from grid-mean conditions (`"grid_scale"`).
  - `QuadratureCloud`: cloud fraction from the hybrid quadrature-moment formula
    (`"quadrature"`).
  - `MLCloud`: cloud fraction from a neural network (`"MLCloud"`).
"""
abstract type AbstractCloudModel end

"""
    GridScaleCloud

Diagnose the cloud fraction from grid-mean conditions: a grid box is either
fully cloudy or fully clear. Selected by `cloud_model: "grid_scale"`.
"""
struct GridScaleCloud <: AbstractCloudModel end

"""
    QuadratureCloud

Diagnose the cloud fraction with the hybrid quadrature-moment formula, which
integrates saturation over an assumed subgrid-scale joint distribution of
temperature and total water. Selected by `cloud_model: "quadrature"`.
"""
struct QuadratureCloud <: AbstractCloudModel end

"""
    AbstractSGSVarianceModel

Strategy for the subgrid-scale (co)variances of `(θ_li, q_tot)` that feed the SGS
quadrature (cloud fraction, saturation adjustment, microphysics tendencies).
Selected by the YAML key `sgs_variance_model`.

Subtypes:

  - `SGSVarianceVertical`: the historical closure, `2 C ℓ² (∂_z ψ)²`, from vertical
    gradients and the master mixing length only (`"vertical"`).
  - `SGSVariance3D`: a scale-aware closure adding a horizontal turbulent term (with
    the horizontal mixing length `ℓ_h`) and a resolved-gradient "geometric" term
    `c_g [(c_Δz Δz)² (∂_z ψ)² + (c_Δx Δx_h)² |∇_h ψ|²]` that does not vanish in
    stable stratification and grows with the horizontal grid spacing (`"3d"`).
"""
abstract type AbstractSGSVarianceModel end

"""
    SGSVarianceVertical

Vertical-gradient-only SGS variance closure; see [`AbstractSGSVarianceModel`](@ref).
"""
struct SGSVarianceVertical <: AbstractSGSVarianceModel end

"""
    SGSVariance3D

Three-dimensional, scale-aware SGS variance closure; see
[`AbstractSGSVarianceModel`](@ref).
"""
struct SGSVariance3D <: AbstractSGSVarianceModel end

"""
    AbstractTqCorrelationModel

Strategy for the subgrid-scale correlation `corr(T′, q_tot′)` used to sample the
joint SGS PDF. Selected by the YAML key `tq_correlation_model`.

Subtypes:

  - `ConstantTqCorrelation`: the prescribed parameter `Tq_correlation_coefficient`
    (`"constant"`).
  - `DiagnosedTqCorrelation`: `corr = Cov / sqrt(Var_θ Var_q)` from the gradient-based
    (co)variances, falling back to the prescribed value where both variances are
    below a floor (`"diagnosed"`; requires an `SGSVariance3D`-capable covariance
    cache carrying `ᶜT′q′`).
"""
abstract type AbstractTqCorrelationModel end

"""
    ConstantTqCorrelation

Prescribed constant T–q correlation; see [`AbstractTqCorrelationModel`](@ref).
"""
struct ConstantTqCorrelation <: AbstractTqCorrelationModel end

"""
    DiagnosedTqCorrelation

T–q correlation diagnosed from the SGS covariance; see
[`AbstractTqCorrelationModel`](@ref).
"""
struct DiagnosedTqCorrelation <: AbstractTqCorrelationModel end


"""
    MLCloud{M}

Diagnose the cloud fraction with a machine-learning model.

`M` is the type of the wrapped network, which is made GPU-friendly by
`MLCloud_constructor`. Selected by `cloud_model: "MLCloud"`, whose weights are
read from the `cloud_fraction_nn` artifact.

# Fields

  - `model`: The callable network mapping local thermodynamic inputs to a cloud
    fraction in `[0, 1]` [-].
"""
struct MLCloud{M} <: AbstractCloudModel
    model::M
end

"""
    MLCloud_constructor(model)

Wrap a neural network in an `MLCloud` after converting its arrays to
`StaticArrays.SArray`, so that the model can be evaluated inside GPU kernels.
"""
function MLCloud_constructor(model)
    static_model = Adapt.adapt_structure(SA.SArray, model)
    return MLCloud{typeof(static_model)}(static_model)
end

"""
    AbstractInsolation

Source of the top-of-atmosphere solar flux and cosine of the solar zenith angle
used by the radiative transfer solver. Selected by the YAML key `insolation`.

Subtypes:

  - `IdealizedInsolation`: annual-mean insolation without a diurnal cycle
    (`"idealized"`).
  - `TimeVaryingInsolation`: orbital insolation evaluated at the current date
    (`"timevarying"`).
  - `RCEMIPIIInsolation`: the fixed RCEMIP-II values (`"rcemipii"`).
  - `ExternalTVInsolation`: time-varying values read from a column forcing file
    (`"externaldriventv"`).
  - `Larcform1Insolation`: polar night, i.e. no incoming solar flux (`"larcform1"`).
"""
abstract type AbstractInsolation end

"""
    IdealizedInsolation

Annual-mean insolation without a diurnal cycle, following the approximation of
[OGorman2008](@cite): a uniform TOA flux of 680 W/m² and a latitude-dependent
cosine of the zenith angle `μ = (1 + 0.3 (1 - 3 sin²ϕ)) / 2`. Flat-space
geometries are treated as being on the equator.
"""
struct IdealizedInsolation <: AbstractInsolation end

"""
    RCEMIPIIInsolation

Uniform, time-invariant insolation prescribed by the RCEMIP-II protocol
[Wing2018](@cite): a TOA flux of 551.58 W/m² with a solar zenith angle of
42.05°.
"""
struct RCEMIPIIInsolation <: AbstractInsolation end

"""
    ExternalTVInsolation

Take time-varying `coszen` and downwelling shortwave `rsdt` from a column
forcing file; the TOA flux is reconstructed as `rsdt / coszen`.
"""
struct ExternalTVInsolation <: AbstractInsolation end

"""
    Larcform1Insolation

Polar-night insolation for the LARCFORM-1 setup: zero TOA flux, with the cosine
of the zenith angle set to `eps(FT)` because RRTMGP requires a positive value.
"""
struct Larcform1Insolation <: AbstractInsolation end

"""
    TimeVaryingInsolation(; start_date = nothing, latitude = nothing, longitude = nothing)

Compute insolation from the orbital parameters at the current simulation date.

When `latitude`/`longitude` are `nothing`, lat/lon are taken from the grid for
`LatLongZPoint` coordinates and fall back to `(0, 0)` for flat-space columns
(the default global behavior). When provided, the explicit lat/lon are used
instead — useful for single-column setups whose coordinate system doesn't
carry lat/lon (e.g. ARM VARANAL).

# Fields

  - `start_date`: `DateTime` used to convert a non-`ITime` simulation time `t`
    into a date; unused when `t isa ITime`. `nothing` when not needed.
  - `latitude`: Latitude override [degrees], or `nothing` to use the grid.
  - `longitude`: Longitude override [degrees], or `nothing` to use the grid.

# Examples

```julia
insolation = ClimaAtmos.TimeVaryingInsolation(; latitude = 36.6, longitude = -97.5)
```
"""
struct TimeVaryingInsolation{SD, LAT, LON} <: AbstractInsolation
    start_date::SD
    latitude::LAT
    longitude::LON
end
TimeVaryingInsolation(;
    start_date = nothing,
    latitude = nothing,
    longitude = nothing,
) = TimeVaryingInsolation(start_date, latitude, longitude)

"""
    AbstractCloudInRadiation

Describe how cloud properties should be set in radiation.

This is only relevant for RRTMGP.
"""
abstract type AbstractCloudInRadiation end

"""
    InteractiveCloudInRadiation

Use the cloud properties computed by the model, so that clouds and radiation
interact. Selected by `prescribe_clouds_in_radiation: false`.
"""
struct InteractiveCloudInRadiation <: AbstractCloudInRadiation end

"""
    PrescribedCloudInRadiation

Use monthly-average cloud properties from ERA5 in the radiative transfer, so
that the model's own clouds do not feed back on radiation. Selected by
`prescribe_clouds_in_radiation: true`, and only honored for all-sky radiation.
"""
struct PrescribedCloudInRadiation <: AbstractCloudInRadiation end

### -------------------- ###
### Hyperdiffusion model ###
### -------------------- ###

"""
    Hyperdiffusion{FT}(; ν₄_vorticity_coeff, divergence_damping_factor, prandtl_number)

Fourth-order horizontal hyperdiffusion applied to velocity and scalars.

The coefficients are resolution-aware: the hyperviscosity is
`ν₄_vorticity = ν₄_vorticity_coeff * h³`, where `h` is the mean nodal distance
of the horizontal grid, and the scalar hyperdiffusivity is
`ν₄_scalar = ν₄_vorticity / prandtl_number` (see `ν₄`).

# Fields

  - `ν₄_vorticity_coeff`: Resolution-independent vorticity hyperviscosity
    coefficient [m/s].
  - `divergence_damping_factor`: Multiplier on the divergent part of the
    momentum hyperdiffusion relative to the rotational part [-].
  - `prandtl_number`: Ratio of the vorticity hyperviscosity to the scalar
    hyperdiffusivity [-].

# Examples

```julia
hyperdiff = ClimaAtmos.Hyperdiffusion{Float32}(;
    ν₄_vorticity_coeff = 0.150,
    divergence_damping_factor = 5,
    prandtl_number = 1.0,
)
```

Selected by `hyperdiff: "Hyperdiffusion"`, whose coefficients come from the
`vorticity_hyperdiffusion_coefficient`, `divergence_damping_factor`, and
`hyperdiffusion_prandtl_number` config keys. See also
`cam_se_hyperdiffusion`.
"""
@kwdef struct Hyperdiffusion{FT}
    ν₄_vorticity_coeff::FT
    divergence_damping_factor::FT
    prandtl_number::FT
end

"""
    cam_se_hyperdiffusion(::Type{FT})

Return a `Hyperdiffusion` with the CAM-SE preset coefficients.

These match the hyperviscosity coefficients in equations A18 and A19 of
[Lauritzen et al. (2018)](https://doi.org/10.1029/2017MS001257), rescaled from
the CAM-SE ne30 grid spacing to this model's `h³` scaling by the factor
`(1.1e5 / (sqrt(4π / 6) * 6.371e6 / (3 * 30)))^3 ≈ 1.238`.

Selected by `hyperdiff: "CAM_SE"`.
"""
cam_se_hyperdiffusion(::Type{FT}) where {FT} =
    Hyperdiffusion{FT}(;
        ν₄_vorticity_coeff = 0.150 * 1.238,
        divergence_damping_factor = 5,
        prandtl_number = 0.2,
    )

### ------------------------------------ ###
### Prescribed vertical diffusion models ###
### ------------------------------------ ###

"""
    AbstractVerticalDiffusion

Prescribed (non-EDMF) vertical diffusion closure for the boundary layer.
Selected by the YAML key `vert_diff`; `~` disables vertical diffusion.

Subtypes:

  - `VerticalDiffusion`: surface-driven diffusivity that decays above the
    boundary layer (`"VerticalDiffusion"`).
  - `DecayWithHeightDiffusion`: diffusivity decaying exponentially with height
    (`"DecayWithHeightDiffusion"`).

Both are parameterized by a boolean `DM` selecting whether momentum diffusion
is switched off; query it with `disable_momentum_vertical_diffusion`.
"""
abstract type AbstractVerticalDiffusion end

"""
    VerticalDiffusion{DM, FT}(; C_E)
    VerticalDiffusion{FT}(; disable_momentum_vertical_diffusion, C_E)

Boundary-layer diffusion with a surface-driven eddy diffusivity.

The diffusivity is `K_E = C_E ‖u_a‖ z_a` below 850 hPa, where `‖u_a‖` and `z_a`
are the wind speed and height of the lowest model level, and it decays as
`K_E exp(-((p_pbl - p) / p_strato)²)` above, with `p_pbl = 850 hPa` and
`p_strato = 100 hPa`.

`DM` is a boolean type parameter: when `true`, the closure diffuses scalars
only and leaves momentum untouched.

# Fields

  - `C_E`: Dimensionless coefficient scaling the surface-driven diffusivity [-].
"""
@kwdef struct VerticalDiffusion{DM, FT} <: AbstractVerticalDiffusion
    C_E::FT
end
VerticalDiffusion{FT}(; disable_momentum_vertical_diffusion, C_E) where {FT} =
    VerticalDiffusion{disable_momentum_vertical_diffusion, FT}(; C_E)

"""
    disable_momentum_vertical_diffusion(vertical_diffusion)

Return `true` when the vertical diffusion model diffuses scalars only.

Defined for `VerticalDiffusion`, `DecayWithHeightDiffusion`, and `nothing`
(which returns `false`).
"""
disable_momentum_vertical_diffusion(::VerticalDiffusion{DM}) where {DM} = DM

"""
    DecayWithHeightDiffusion{DM, FT}(; H, D₀)
    DecayWithHeightDiffusion{FT}(; disable_momentum_vertical_diffusion, H, D₀)

Vertical diffusion with a diffusivity that decays exponentially with height
above the surface, `K = D₀ exp(-(z - z_sfc) / H)`.

`DM` is a boolean type parameter: when `true`, the closure diffuses scalars
only and leaves momentum untouched.

# Fields

  - `H`: Decay scale height of the diffusivity [m].
  - `D₀`: Diffusivity at the surface [m²/s].
"""
@kwdef struct DecayWithHeightDiffusion{DM, FT} <: AbstractVerticalDiffusion
    H::FT
    D₀::FT
end
DecayWithHeightDiffusion{FT}(; disable_momentum_vertical_diffusion, H, D₀) where {FT} =
    DecayWithHeightDiffusion{disable_momentum_vertical_diffusion, FT}(; H, D₀)

disable_momentum_vertical_diffusion(::DecayWithHeightDiffusion{DM}) where {DM} = DM
disable_momentum_vertical_diffusion(::Nothing) = false


### --------------------- ###
### Eddy Viscosity Models ###
### --------------------- ###

"""
    EddyViscosityModel

Large-eddy-simulation closure providing a subgrid-scale eddy viscosity and
diffusivity, used instead of (or alongside) the EDMF turbulence-convection
schemes at LES resolutions.

Subtypes:

  - `SmagorinskyLilly`: Smagorinsky-Lilly closure, selected by the YAML key
    `smagorinsky_lilly`.
  - `AnisotropicMinimumDissipation`: AMD closure, selected by `amd_les: true`.
  - `ConstantHorizontalDiffusion`: spatially uniform horizontal scalar
    diffusivity, selected by `constant_horizontal_diffusion: true`.
"""
abstract type EddyViscosityModel end

"""
    SmagorinskyLilly{AXES}

Smagorinsky-Lilly eddy viscosity model.

`AXES` is a symbol indicating along which axes the model is applied. It can be

  - `:UVW` (all axes)
  - `:UV` (horizontal axes)
  - `:W` (vertical axis)
  - `:UV_W` (horizontal and vertical axes treated separately).

# Examples

Construct a model instance by passing the selected axes as a keyword argument:

```julia
smagorinsky_lilly = SmagorinskyLilly(; axes = :UV_W)
```
"""
struct SmagorinskyLilly{AXES} <: EddyViscosityModel end

function SmagorinskyLilly(; axes::Symbol)
    @assert axes in (:UVW, :UV, :W, :UV_W) "axes must be one of :UVW, :UV, :W, or :UV_W, got :$axes"
    return SmagorinskyLilly{axes}()
end

"""
    is_smagorinsky_UVW_coupled(model)

Check if the Smagorinsky model is coupled along all axes.
"""
is_smagorinsky_UVW_coupled(::SmagorinskyLilly{AXES}) where {AXES} = AXES == :UVW
is_smagorinsky_UVW_coupled(::Nothing) = false

"""
    is_smagorinsky_vertical(model)

Check if the Smagorinsky model is applied along the vertical axis.

See also `is_smagorinsky_horizontal`.
"""
is_smagorinsky_vertical(::SmagorinskyLilly{AXES}) where {AXES} =
    AXES == :UVW || AXES == :W || AXES == :UV_W
is_smagorinsky_vertical(::Nothing) = false

"""
    is_smagorinsky_horizontal(model)

Check if the Smagorinsky model is applied along the horizontal axes.

See also `is_smagorinsky_vertical`.
"""
is_smagorinsky_horizontal(::SmagorinskyLilly{AXES}) where {AXES} =
    AXES == :UVW || AXES == :UV || AXES == :UV_W
is_smagorinsky_horizontal(::Nothing) = false

"""
    AnisotropicMinimumDissipation{FT}(; c_amd)

Anisotropic Minimum Dissipation (AMD) subgrid-scale closure of [Akbar2016](@cite).

The eddy viscosity and diffusivity are built from velocity and scalar gradients
scaled by the anisotropic filter widths, and are clipped at zero so the closure
is purely dissipative. Enabled by `amd_les: true`.

# Fields

  - `c_amd`: Poincaré coefficient multiplying the AMD viscosity and
    diffusivity [-].

# Examples

```julia
les = ClimaAtmos.AnisotropicMinimumDissipation{Float32}(; c_amd = 0.3)
```
"""
@kwdef struct AnisotropicMinimumDissipation{FT} <: EddyViscosityModel
    c_amd::FT
end

"""
    ConstantHorizontalDiffusion{FT}(; D)

Horizontal diffusion of total energy and grid-scale tracers with a spatially
uniform diffusivity. Momentum is not diffused. Enabled by
`constant_horizontal_diffusion: true`, with `D` taken from the
`constant_horizontal_diffusion_D` parameter.

# Fields

  - `D`: Horizontal diffusivity applied to energy and tracers [m²/s].
"""
@kwdef struct ConstantHorizontalDiffusion{FT} <: EddyViscosityModel
    D::FT
end

### ------------- ###
### Sponge models ###
### ------------- ###

"""
    SpongeModel

Absorbing layer near the model top, used to prevent spurious reflection of
vertically propagating waves off the rigid lid.

Subtypes:

  - `ViscousSponge`: damp the horizontal Laplacian of the prognostic fields.
  - `RayleighSponge`: damp the fields themselves.

Both are switched on by the YAML keys `viscous_sponge` and `rayleigh_sponge`,
which take their coefficients from the model parameters.
"""
abstract type SpongeModel end
Base.broadcastable(x::SpongeModel) = tuple(x)

"""
    ViscousSponge{FT}(; zd, κ₂)

Viscous sponge model; damp variables in proportion to their horizontal Laplacian.

Above the damping height `zd`, the sponge adds the tendency

```math
∂χ/∂t = β ∇ₕ·(∇ₕ χ),   z > zd
```

where `β = κ₂ ζ` and `χ ∈ {uₕ, u₃, ρe_tot, GS_TRACERS}`. The grid-scale tracers
`GS_TRACERS` depend on the microphysics model and may include `ρq_tot`,
`ρq_lcl`, `ρq_icl`, and so on; energy is diffused through the total specific
enthalpy. With `PrognosticEDMFX` the sponge is additionally applied to the
updraft vertical velocities `u₃ʲ`. The damping function is

```math
ζ(z) = sin²(π (z - zd) / (2 (zmax - zd)))
```

with `zmax` the domain top height, so that damping ramps up smoothly from `zd`.

# Fields

  - `zd`: Lower damping height; the sponge is inactive below it [m]. This is an
    absolute altitude, so it does not follow the domain and must be set
    explicitly for domains whose top is not near 30 km.
  - `κ₂`: Damping coefficient [m²/s].

# Examples

```julia
# Apply damping above 20 km with κ₂ = 10⁶ m²/s
sponge = ClimaAtmos.ViscousSponge{Float32}(; zd = 20_000, κ₂ = 1e6)
```
"""
@kwdef struct ViscousSponge{FT} <: SpongeModel
    # Lower damping height, in meters
    zd::FT
    # Damping coefficient, in m²/s
    κ₂::FT
end

"""
    ViscousSponge(params)

Build a `ViscousSponge` from the model parameters, reading `zd_viscous` and
`kappa_2_sponge`. Used when the config sets `viscous_sponge: true`.
"""
ViscousSponge(params) = ViscousSponge(;
    zd = params.zd_viscous,
    κ₂ = params.kappa_2_sponge,
)

"""
    RayleighSponge{FT}(; zd, α_uₕ = 0, α_w = 1, α_tracer = 0)

Rayleigh sponge model; damp variables in proportion to their own value.

Above the damping height `zd`, the sponge adds the tendency

```math
∂χ/∂t = -β χ,   z > zd
```

where `β = α_χ ζ` and the damping function is

```math
ζ(z) = sin²(π (z - zd) / (2 (zmax - zd)))
```

with `zmax` the domain top height. The damped variables are the horizontal
velocity `uₕ` (rate `α_uₕ`), the vertical velocity `u₃` (rate `α_w`), and, when
they are prognostic, `ρtke` and the `PrognosticEDMFX` subdomain scalars `mseʲ`,
`q_totʲ`, and the subdomain microphysics and passive tracers (rate `α_tracer`).
Subdomain scalars are relaxed toward their grid-mean value rather than toward
zero, so only the subgrid-scale departure is damped.

By default only the vertical velocity is damped (`α_uₕ = 0`, `α_w = 1`,
`α_tracer = 0`).

# Fields

  - `zd`: Lower damping height; the sponge is inactive below it [m]. This is an
    absolute altitude, so it does not follow the domain and must be set
    explicitly for domains whose top is not near 30 km.
  - `α_uₕ`: Damping rate for the horizontal velocity, `0` by default [1/s].
  - `α_w`: Damping rate for the vertical velocity, `1` by default [1/s].
  - `α_tracer`: Damping rate for `ρtke` and the subdomain scalars, `0` by
    default [1/s].

# Examples

```julia
# Apply damping to vertical velocity above 20 km
sponge = ClimaAtmos.RayleighSponge{Float32}(; zd = 20_000)
```
"""
@kwdef struct RayleighSponge{FT} <: SpongeModel
    # Lower damping height, in meters
    zd::FT
    # Damping coefficient for horizontal velocity, by default 0 (no damping)
    α_uₕ::FT = 0
    # Damping coefficient for vertical velocity, by default 1 (full damping)
    α_w::FT = 1
    # Damping coefficient for tracer variables, by default 0 (no damping)
    α_tracer::FT = 0
end

"""
    RayleighSponge(params)

Build a `RayleighSponge` from the model parameters, reading `zd_rayleigh`,
`alpha_rayleigh_uh`, `alpha_rayleigh_w`, and `alpha_rayleigh_tracer`. Used when
the config sets `rayleigh_sponge: true`.
"""
RayleighSponge(params) = RayleighSponge(;
    zd = params.zd_rayleigh,
    α_uₕ = params.alpha_rayleigh_uh,
    α_w = params.alpha_rayleigh_w,
    α_tracer = params.alpha_rayleigh_tracer,
)


### ------------------- ###
### Gravity wave models ###
### ------------------- ###

"""
    AbstractGravityWave

Parameterized drag exerted by unresolved gravity waves.

Subtypes:

  - `NonOrographicGravityWave`: convectively and frontally generated wave
    spectrum, switched on by `non_orographic_gravity_wave: true`.
  - `OrographicGravityWave`: waves generated by flow over unresolved topography,
    switched on by the `orographic_gravity_wave` key.
"""
abstract type AbstractGravityWave end

"""
    BeresSourceParams{FT}(; Q0_threshold, beres_scale_factor, σ_x, ν_min, ν_max, n_ν, kwargs...)

Parameters for the convective gravity-wave source spectrum of [beres2004](@cite).

When supplied as the `beres_source` field of `NonOrographicGravityWave`, the
Beres convective spectrum is launched on top of the Alexander-Dunkerton
background spectrum in every column whose EDMF convective heating exceeds
`Q0_threshold` and whose heating layer is deeper than `h_heat_min`. There is no
latitude gate: the source is placed wherever the EDMF scheme produces deep
convective heating.

# Fields

  - `Q0_threshold`: Minimum convective heating rate that activates the source [K/s].
  - `beres_scale_factor`: Dimensionless efficiency ℰ, absorbing the `ρ₀/(Lτ)`
    normalization, the `|Q_t(ν)|²` weight, and tuning [-].
  - `σ_x`: Horizontal half-width of a convective cell [m].
  - `ν_min`: Lowest wave frequency in the quadrature (period ≈ 120 min) [1/s].
  - `ν_max`: Highest wave frequency in the quadrature (period ≈ 10 min) [1/s].
  - `n_ν`: Number of frequency quadrature points. Must satisfy
    `(n_ν - 1) % 4 == 0`, i.e. `n_ν ∈ {5, 9, 13, ...}`, for the composite Boole
    rule; the inner constructor errors otherwise [-].
  - `h_heat_min`: Minimum heating depth that activates the source, which filters
    out shallow convection [m].
  - `z_bot_floor`: Lowest allowed heating-layer base, which excludes the
    boundary-layer signal from `Q_conv` [m].
  - `beres_steady_source`: Whether to include the steady (`ν = 0`) stationary
    component, which deposits momentum only if a `c ≈ 0` phase-speed bin exists.
  - `beres_steady_dc_frac`: Weight of the artificial steady component,
    `Q_t(0)² = dc_frac · ν_min` [-].
  - `beres_L_system`: Largest system scale, which sets `k_min = 2π/L` in the
    even-folded heating profile used by the steady source [m].
  - `heating_latent`: Whether to source the in-cloud heating from the latent
    heating `Q_lat = Σ_p L_p R_p` (requires 1-moment microphysics with
    `PrognosticEDMFX`) instead of the dry-static-energy budget `Q₁`.
  - `detailed_diagnostics`: Whether to expose the `nogw_*` source-internal
    extended diagnostics.
  - `n_h_avg`: Number of heating depths to average over, to smooth the resonance
    in the source spectrum; `1` disables averaging [-].
  - `Δh_frac`: Fractional half-range of that averaging, `h ± Δh_frac · h` [-].
"""
Base.@kwdef struct BeresSourceParams{FT}
    # --- Main parameters ---
    Q0_threshold::FT             # K/s, min heating rate to activate Beres
    beres_scale_factor::FT       # dimensionless efficiency ℰ; knobs for ρ₀/(Lτ) normalization, the |Q_t(ν)|² weight, and tuning
    σ_x::FT                      # m, convective cell horizontal half-width
    ν_min::FT                    # 1/s, min wave frequency (period ~120 min)
    ν_max::FT                    # 1/s, max wave frequency (period ~10 min)
    n_ν::Int                     # frequency quadrature points (must be 4k+1: 5, 9, 13...)
    h_heat_min::FT = FT(1000.0)  # m, min heating depth to activate (filters shallow convection)
    z_bot_floor::FT = FT(2000.0) # m, min allowed z_bot (excludes PBL signal in Q_conv)
    beres_steady_source::Bool = true # boolean flag for steady (ν=0) stationary component: deposits only if a c≈0 bin exits
    beres_steady_dc_frac::FT = FT(1.0) # artificial steady DC weight: Q_t(0)² = dc_frac·ν_min
    beres_L_system::FT = FT(1.0e6)     # m, largest system scale; sets k_min=2π/L in even-folded H; for the steady-state source
    heating_latent::Bool = false       # source in-cloud heating from latent Q_lat=Σ L_p R_p (1M+PrognosticEDMFX) vs DSE-Q₁
    detailed_diagnostics::Bool = false # expose nogw_* source-internal extended diagnostics

    # --- h-averaging (resonance smoothing; default off) ---
    n_h_avg::Int = 1      # number of h values to average over (1 = no averaging)
    Δh_frac::FT = FT(0.1) # fractional half-range for averaging: h ± Δh_frac·h

    function BeresSourceParams{FT}(args...) where {FT}
        obj = new{FT}(args...)
        if (obj.n_ν - 1) % 4 != 0
            error(
                "BeresSourceParams: n_ν must satisfy (n_ν - 1) % 4 == 0 " *
                "(i.e. n_ν ∈ {5, 9, 13, ...}) for composite Boole's rule, " *
                "got n_ν = $(obj.n_ν)",
            )
        end
        return obj
    end
end

"""
    NonOrographicGravityWave{FT, BS}(; source_pressure, damp_pressure, ..., beres_source = nothing)

Non-orographic gravity-wave drag with the launch spectrum of [alexander1999](@cite).

A spectrum of waves is launched at a fixed source level and propagates
vertically until it breaks; the resulting momentum-flux divergence is applied
to the horizontal velocity. The launch amplitude varies with latitude so that
the tropics can be treated separately from the extratropics. Switched on by
`non_orographic_gravity_wave: true`, with the parameters taken from
`params.non_orographic_gravity_wave_params`.

`BS` is the type of the optional convective source: `nothing` gives the
background spectrum only, while a `BeresSourceParams` adds the Beres convective
source on top of it wherever the EDMF scheme convects.

# Fields

  - `source_pressure`: Pressure of the launch level, used on spherical grids [Pa].
  - `damp_pressure`: Pressure above which the waves are damped [Pa].
  - `source_height`: Height of the launch level, used on single columns [m].
  - `Bw`: Amplitude of the broad (westward) part of the launch spectrum [m²/s²].
  - `Bn`: Amplitude of the narrow part of the launch spectrum [m²/s²].
  - `dc`: Phase-speed grid spacing [m/s].
  - `cmax`: Largest resolved phase speed; the grid spans `-cmax:dc:cmax` [m/s].
  - `c0`: Reference phase speed about which the spectrum is centered [m/s].
  - `nk`: Number of horizontal wave bands [-].
  - `cw`: Phase-speed half-width of the broad spectrum outside the tropics [m/s].
  - `cw_tropics`: Phase-speed half-width of the broad spectrum in the tropics [m/s].
  - `cn`: Phase-speed half-width of the narrow spectrum [m/s].
  - `Bt_0`: Background total source momentum flux [Pa].
  - `Bt_n`: Additional source momentum flux in the northern hemisphere [Pa].
  - `Bt_s`: Additional source momentum flux in the southern hemisphere [Pa].
  - `Bt_eq`: Source momentum flux at the equator [Pa].
  - `ϕ0_n`: Central latitude of the northern-hemisphere transition [degrees].
  - `ϕ0_s`: Central latitude of the southern-hemisphere transition [degrees].
  - `dϕ_n`: Northern edge of the tropical band [degrees].
  - `dϕ_s`: Southern edge of the tropical band [degrees].
  - `beres_source`: `nothing`, or a `BeresSourceParams` adding the convective
    source of [beres2004](@cite).
"""
Base.@kwdef struct NonOrographicGravityWave{FT, BS} <: AbstractGravityWave
    source_pressure::FT
    damp_pressure::FT
    source_height::FT
    Bw::FT
    Bn::FT
    dc::FT
    cmax::FT
    c0::FT
    nk::FT
    cw::FT
    cw_tropics::FT
    cn::FT
    Bt_0::FT
    Bt_n::FT
    Bt_s::FT
    Bt_eq::FT
    ϕ0_n::FT
    ϕ0_s::FT
    dϕ_n::FT
    dϕ_s::FT
    beres_source::BS = nothing  # nothing → AD background only; BeresSourceParams → adds the Beres convective source on top of AD wherever EDMF convects
end

"""
    OrographicGravityWave

Drag exerted by gravity waves generated by flow over unresolved topography.

Subtypes:

  - `FullOrographicGravityWave`: the propagating plus blocked drag of
    [garner2005](@cite), selected by `orographic_gravity_wave: "raw_topo"` or
    `"gfdl_restart"`.
  - `LinearOrographicGravityWave`: idealized variant with a user-supplied drag
    input, selected by `orographic_gravity_wave: "linear"`.

Every subtype carries a `topo_info` field, a `Val` that selects how the
subgrid orographic drag tensor is obtained (see `get_topo_info`).
"""
abstract type OrographicGravityWave <: AbstractGravityWave end

"""
    LinearOrographicGravityWave{S}(; topo_info = Val(:linear))

Orographic gravity-wave drag driven by an analytical drag input, for idealized
tests. Selected by `orographic_gravity_wave: "linear"`.

# Fields

  - `topo_info`: `Val(:linear)`, selecting the analytical drag input in
    `get_topo_info`.
"""
Base.@kwdef struct LinearOrographicGravityWave{S} <: OrographicGravityWave
    topo_info::S = Val(:linear)
end

"""
    FullOrographicGravityWave{FT, S, T}(; γ, ϵ, β, h_frac, ρscale, L0, a0, a1, Fr_crit, topo_info, topography)

Orographic gravity-wave drag following [garner2005](@cite), combining the drag
of vertically propagating waves with the drag of low-level blocked flow.

The subgrid obstacle distribution is summarized by the orographic tensor and
the effective obstacle heights supplied through `topo_info`. Selected by
`orographic_gravity_wave: "raw_topo"` or `"gfdl_restart"`, with the shape
parameters taken from `params.orographic_gravity_wave_params`.

# Fields

  - `γ`: Exponent relating obstacle width to height, `L ∝ h^γ` [-].
  - `ϵ`: Exponent of the obstacle number density, `n(h) ∝ h^(-ϵ)` [-].
  - `β`: Obstacle shape exponent in `L(z) = L_b (1 - z/h)^β`; `β = 1` is
    triangular, `β < 1` blunt, and `β > 1` pointy [-].
  - `h_frac`: Fraction setting the blocking threshold, `h_crit = h_frac · V/N` [-].
  - `ρscale`: Reference density used to make the drag dimensional [kg/m³].
  - `L0`: Reference obstacle width [m].
  - `a0`: Coefficient of the propagating-wave drag [-].
  - `a1`: Coefficient of the non-propagating (blocked) drag [-].
  - `Fr_crit`: Critical Froude number separating the two regimes [-].
  - `topo_info`: `Val(:raw_topo)` or `Val(:gfdl_restart)`, selecting how the
    orographic drag tensor is built (see `get_topo_info`).
  - `topography`: `Val` of the configured `topography` key, used when the drag
    tensor is computed on the fly.
"""
Base.@kwdef struct FullOrographicGravityWave{FT, S, T} <: OrographicGravityWave
    γ::FT
    ϵ::FT
    β::FT
    h_frac::FT
    ρscale::FT
    L0::FT
    a0::FT
    a1::FT
    Fr_crit::FT
    topo_info::S
    topography::T
end

"""
    AbstractForcing

Prescribed large-scale forcing imposed on a column or limited-area domain.

`LargeScaleSubsidence` is currently the only subtype; the other forcing objects
in this file (`LargeScaleAdvection`, `ExternalDrivenTVForcing`,
`ISDACForcing`, `HeldSuarezForcing`) are dispatched on directly and are not
part of this hierarchy.
"""
abstract type AbstractForcing end

"""
    HeldSuarezForcing

Held-Suarez idealized forcing: Newtonian relaxation of temperature toward a
prescribed radiative-equilibrium profile plus Rayleigh friction on the
low-level winds.

It is passed through the `radiation_mode` slot rather than as a forcing,
because it replaces radiation in the dry dynamical-core benchmark. Selected by
`rad: "held_suarez"`.
"""
struct HeldSuarezForcing end

"""
    LargeScaleSubsidence{T}

Prescribed large-scale subsidence, advecting scalars vertically with a
specified subsidence velocity profile.

Total enthalpy and `ρq_tot` are subsided, as are `ρq_lcl` and `ρq_icl` for
non-equilibrium microphysics; rain and snow are not. The profile is supplied by
the setup (e.g. `Setups.Bomex`), not by a YAML key.

# Fields

  - `prof`: Callable `prof(z)` returning the subsidence velocity, negative for
    descent [m/s].
"""
struct LargeScaleSubsidence{T} <: AbstractForcing
    prof::T
end
# TODO: is this a forcing?
"""
    LargeScaleAdvection{PT, PQ}

Prescribed large-scale horizontal advective tendencies of temperature and total
water, used in single-column setups. Supplied by the setup through
`Setups.large_scale_advection_forcing`.

# Fields

  - `prof_dTdt`: Callable `prof_dTdt(thermo_params, p, t, z)` returning the
    large-scale temperature tendency, typically a cooling [K/s].
  - `prof_dqtdt`: Callable `prof_dqtdt(thermo_params, p, t, z)` returning the
    large-scale total-water tendency, typically a drying [kg/kg/s].
"""
struct LargeScaleAdvection{PT, PQ}
    prof_dTdt::PT # Set large-scale cooling
    prof_dqtdt::PQ # Set large-scale drying
end
"""
    ExternalDrivenTVForcing{CD, F, M}

Generic time-varying forcing read from column forcing data through the
`ColumnDatasets` interface (an on-disk ClimaColumn file or an in-memory
source). Its `forcing`
is a tuple of composed [`AbstractForcingTerm`](@ref)s (horizontal advection,
vertical fluctuation, nudging, subsidence). Only data required by the composed
terms is loaded, and missing data for a composed term is a loud error.

`time_interpolation_method` sets how the file's `TimeVaryingInput`s behave in
time; it defaults to the dataset format's method (plain `LinearInterpolation`,
which errors out of range so a finite campaign cannot fabricate forcing). A
case whose file stores one repeating period passes
`ColumnDatasets.periodic_calendar_method()` instead.

Surface-temperature and insolation requirements are derived from the resolved
`AtmosModel` during cache construction rather than from the forcing terms.

# Fields

  - `dataset`: The `ColumnDatasets.ColumnDataset` handle for the forcing file.
  - `forcing`: Tuple of composed forcing terms, validated at construction.
  - `time_interpolation_method`: Time-interpolation method handed to the file's
    `TimeVaryingInput`s.

# Constructor

    ExternalDrivenTVForcing(dataset::ColumnDatasets.ColumnDataset; forcing, time_interpolation_method)
    ExternalDrivenTVForcing(path::String; kwargs...)

The `path` method opens the file as a `ColumnDataset` and forwards the keyword
arguments. `forcing` defaults to `default_forcing_terms()` and
`time_interpolation_method` to the dataset format's own method. Runscripts
typically call `ExternalDrivenTVForcing(path; forcing = (...,))`.
"""
struct ExternalDrivenTVForcing{
    CD <: ColumnDatasets.AbstractColumnData,
    F <: Tuple,
    M,
}
    dataset::CD
    forcing::F
    time_interpolation_method::M
end
function ExternalDrivenTVForcing(
    dataset::ColumnDatasets.AbstractColumnData;
    forcing = default_forcing_terms(),
    time_interpolation_method = ColumnDatasets.time_interpolation_method(dataset),
)
    forcing = Tuple(forcing)
    validate_forcing_terms(forcing)
    return ExternalDrivenTVForcing{
        typeof(dataset),
        typeof(forcing),
        typeof(time_interpolation_method),
    }(
        dataset,
        forcing,
        time_interpolation_method,
    )
end
function ExternalDrivenTVForcing(path::String; kwargs...)
    return ExternalDrivenTVForcing(ColumnDatasets.ColumnDataset(path); kwargs...)
end

"""
    ISDACForcing

Analytic large-scale forcing for the ISDAC mixed-phase Arctic stratocumulus
case. Selected by `external_forcing: "ISDAC"`, and supplied automatically by
`Setups.ISDAC`.
"""
struct ISDACForcing end

"""
    AbstractEnvBuoyGradClosure

Closure used to convert environmental thermodynamic gradients into a buoyancy
gradient for the EDMF mixing-length and TKE budgets.

`BuoyGradMean` is currently the only subtype: it evaluates the buoyancy
gradient from the mean environmental state, weighting the dry and cloudy
branches by the cloud fraction.
"""
abstract type AbstractEnvBuoyGradClosure end

"""
    BuoyGradMean

Compute the environmental buoyancy gradient from the mean environmental state.
See `AbstractEnvBuoyGradClosure` and `buoyancy_gradients`.
"""
struct BuoyGradMean <: AbstractEnvBuoyGradClosure end

Base.broadcastable(x::BuoyGradMean) = tuple(x)

"""
    EnvBuoyGradVars{FT}

Environmental state bundled for the buoyancy-gradient computation in
`buoyancy_gradients`.

# Fields

  - `T`: Environmental temperature [K].
  - `ρ`: Environmental air density [kg/m³].
  - `q_tot`: Total specific humidity [kg/kg].
  - `q_liq`: Liquid water specific humidity [kg/kg].
  - `q_ice`: Ice water specific humidity [kg/kg].
  - `cf`: Environmental cloud fraction [-].
  - `∂qt∂z`: Vertical gradient of total specific humidity [1/m].
  - `∂θli∂z`: Vertical gradient of liquid-ice potential temperature [K/m].

# Constructor

    EnvBuoyGradVars(T, ρ, q_tot, q_liq, q_ice, cf, ∂qt∂z_∂θli∂z)

Convenience method taking the two gradients as a single `NamedTuple`
`(; ∂qt∂z, ∂θli∂z)`, which is how they are carried through the EDMF cache.
"""
@kwdef struct EnvBuoyGradVars{FT}
    T::FT
    ρ::FT
    q_tot::FT
    q_liq::FT
    q_ice::FT
    cf::FT
    ∂qt∂z::FT
    ∂θli∂z::FT
end

function EnvBuoyGradVars(
    T,
    ρ,
    q_tot,
    q_liq,
    q_ice,
    cf,
    ∂qt∂z_∂θli∂z,
)
    (; ∂qt∂z, ∂θli∂z) = ∂qt∂z_∂θli∂z
    return EnvBuoyGradVars(T, ρ, q_tot, q_liq, q_ice, cf, ∂qt∂z, ∂θli∂z)
end

Base.eltype(::EnvBuoyGradVars{FT}) where {FT} = FT
Base.broadcastable(x::EnvBuoyGradVars) = tuple(x)

"""
    MixingLength{FT}(master, wall, tke, buoy, l_grid)

Bundle of the EDMF mixing-length scales at one point, returned by
`mixing_length`.

The individual physical scales are blended into `master` by `blend_scales`, and
the result is capped by the grid scale `l_grid`. The scales are kept separately
so they can be output as diagnostics.

# Fields

  - `master`: Final blended mixing length actually used by the closure [m].
  - `wall`: Wall-constrained (surface-layer) length scale [m].
  - `tke`: Turbulent-kinetic-energy production length scale [m].
  - `buoy`: Buoyancy (static-stability) length scale [m].
  - `l_grid`: Grid-resolution length scale `max(Δx_h, Δz)`, `Inf` for single
    columns [m].

# Constructor

    MixingLength(master, wall, tke, buoy, l_grid)

Promote the five arguments to a common float type before constructing.
"""
struct MixingLength{FT}
    master::FT
    wall::FT
    tke::FT
    buoy::FT
    l_grid::FT
end

function MixingLength(master, wall, tke, buoy, l_grid)
    return MixingLength(promote(master, wall, tke, buoy, l_grid)...)
end

"""
    AbstractEDMF

Eddy-diffusivity/mass-flux turbulence-convection scheme. Selected by the YAML
key `turbconv`; `~` disables the scheme entirely.

Subtypes:

  - `EDOnlyEDMFX`: eddy diffusivity only, no mass flux (`"edonly_edmfx"`).
  - `PrognosticEDMFX`: prognostic updraft subdomains plus the environment
    (`"prognostic_edmfx"`).
"""
abstract type AbstractEDMF end

"""
    EDOnlyEDMFX

Eddy-diffusivity-only "EDMF": the mass-flux subdomains are dropped, leaving
TKE-based vertical diffusion. TKE is always prognostic. Selected by
`turbconv: "edonly_edmfx"`.
"""
struct EDOnlyEDMFX <: AbstractEDMF end

"""
    PrognosticEDMFX{N, TKE, FT}

Prognostic eddy-diffusivity/mass-flux scheme with `N` updraft subdomains and an
implicitly defined environment.

Each updraft carries prognostic `ρa`, `u₃`, `mse`, `q_tot`, and the
microphysics tracers, and exchanges mass with the environment through
entrainment and detrainment. `TKE` is a boolean type parameter selecting
prognostic (`true`) or diagnostic (`false`) turbulent kinetic energy.

# Fields

  - `a_half`: Area fraction at which the SGS weight function equals 0.5, i.e. the
    threshold below which subdomain values are smoothly blended toward the
    grid mean [-]. Only meant to be used through `specific`.

See the constructor `PrognosticEDMFX(; n_updrafts, prognostic_tke, area_fraction)`.
"""
struct PrognosticEDMFX{N, TKE, FT} <: AbstractEDMF
    a_half::FT # WARNING: this should never be used outside of `specific`
end
PrognosticEDMFX{N, TKE}(a_half::FT) where {N, TKE, FT} =
    PrognosticEDMFX{N, TKE, FT}(a_half)

"""
    PrognosticEDMFX(; n_updrafts = 1, prognostic_tke = false, area_fraction)

Create a `PrognosticEDMFX` scheme with the given number of updrafts, TKE
treatment, and small-area threshold.

# Keyword Arguments

  - `n_updrafts = 1`: Number of updraft subdomains, which becomes the type
    parameter `N` [-].
  - `prognostic_tke = false`: Whether TKE is prognostic (`true`) or diagnostic
    (`false`); becomes the type parameter `TKE`.
  - `area_fraction`: "Small" area-fraction threshold, passed as `a_half` to
    `sgs_weight_function`. Required; the float type of the scheme is inferred
    from it [-].

# Examples

```julia
turbconv = ClimaAtmos.PrognosticEDMFX(;
    n_updrafts = 1, prognostic_tke = true, area_fraction = 1.0f-5,
)
```
"""
function PrognosticEDMFX(;
    n_updrafts = 1,
    prognostic_tke = false,
    area_fraction::FT,
) where {FT}
    return PrognosticEDMFX{n_updrafts, prognostic_tke, FT}(area_fraction)
end

"""
    n_mass_flux_subdomains(turbconv_model)

Return the number of mass-flux (updraft) subdomains: `N` for
`PrognosticEDMFX{N}`, and zero for `EDOnlyEDMFX` or any other model.
"""
n_mass_flux_subdomains(::PrognosticEDMFX{N}) where {N} = N
n_mass_flux_subdomains(::EDOnlyEDMFX) = 0
n_mass_flux_subdomains(::Any) = 0

"""
    n_prognostic_mass_flux_subdomains(turbconv_model)

Return the number of updraft subdomains whose variables are prognostic: `N` for
`PrognosticEDMFX{N}`, and zero for any other model.
"""
n_prognostic_mass_flux_subdomains(::PrognosticEDMFX{N}) where {N} = N
n_prognostic_mass_flux_subdomains(::Any) = 0

"""
    use_prognostic_tke(turbconv_model)

Return `true` when the turbulence-convection model carries prognostic TKE:
always for `EDOnlyEDMFX`, and according to the `TKE` type parameter for
`PrognosticEDMFX`. Any other model returns `false`.
"""
use_prognostic_tke(::EDOnlyEDMFX) = true
use_prognostic_tke(::PrognosticEDMFX{N, TKE}) where {N, TKE} = TKE
use_prognostic_tke(::Any) = false

"""
    AbstractEntrainmentModel

Closure for the rate at which environmental air is entrained into an EDMF
updraft. Selected by the YAML key `edmfx_entr_model`.

Subtypes:

  - `PiGroupsEntrainment`: rate built from the nondimensional Π groups (`"PiGroups"`).
  - `InvZEntrainment`: rate proportional to `1/z` above the surface (`"Generalized"`).

Subtypes dispatch `entrainment_velocity_scale`; the area-bounding relaxation in
`area_bounding_entr_detr` is applied on top and does not dispatch on the model.
"""
abstract type AbstractEntrainmentModel end

"""
    PiGroupsEntrainment

Entrainment velocity scale built from a linear combination of the
nondimensional Π groups of [Cohen2020](@cite), divided by height above the
surface and multiplied by the upper-area limiter. Selected by
`edmfx_entr_model: "PiGroups"`.
"""
struct PiGroupsEntrainment <: AbstractEntrainmentModel end

"""
    InvZEntrainment

Entrainment velocity scale `entr_coeff / (z - z_sfc)`, multiplied by the
upper-area limiter. Selected by `edmfx_entr_model: "Generalized"`.
"""
struct InvZEntrainment <: AbstractEntrainmentModel end

"""
    AbstractDetrainmentModel

Closure for the rate at which updraft air is detrained into the environment.
Selected by the YAML key `edmfx_detr_model`.

Subtypes:

  - `BuoyancyVelocityDetrainment`: rate from the inverse buoyancy time scale and
    the mass-flux divergence (`"Generalized"`).

Subtypes dispatch `detrainment_rate`, whose fallback for the abstract type
returns zero. Only `BuoyancyVelocityDetrainment` currently defines a method, so
the other subtypes give no dynamical detrainment; the area-bounding relaxation
in `area_bounding_entr_detr` is applied regardless of the model.
"""
abstract type AbstractDetrainmentModel end

"""
    BuoyancyVelocityDetrainment

Detrainment rate combining the clipped inverse buoyancy time scale with the
convergence of the updraft mass flux, multiplied by the lower-area limiter and
clipped at zero. Selected by `edmfx_detr_model: "Generalized"`.
"""
struct BuoyancyVelocityDetrainment <: AbstractDetrainmentModel end

"""
    AbstractTendencyModel

Marker selecting which part of a tendency is applied, used to isolate the
grid-scale and subgrid-scale contributions in debugging and testing.

Subtypes: `UseAllTendency` (both parts), `NoGridScaleTendency` (subgrid-scale
only), and `NoSubgridScaleTendency` (grid-scale only).
"""
abstract type AbstractTendencyModel end

"""
    UseAllTendency

Apply both the grid-scale and the subgrid-scale part of a tendency. See
`AbstractTendencyModel`.
"""
struct UseAllTendency <: AbstractTendencyModel end

"""
    NoGridScaleTendency

Skip the grid-scale part of a tendency. See `AbstractTendencyModel`.
"""
struct NoGridScaleTendency <: AbstractTendencyModel end

"""
    NoSubgridScaleTendency

Skip the subgrid-scale part of a tendency. See `AbstractTendencyModel`.
"""
struct NoSubgridScaleTendency <: AbstractTendencyModel end

# Define broadcasting for types
Base.broadcastable(x::AbstractMicrophysicsModel) = tuple(x)
Base.broadcastable(x::AbstractForcing) = tuple(x)
Base.broadcastable(x::EDOnlyEDMFX) = tuple(x)
Base.broadcastable(x::PrognosticEDMFX) = tuple(x)
Base.broadcastable(x::AbstractEntrainmentModel) = tuple(x)
Base.broadcastable(x::AbstractDetrainmentModel) = tuple(x)
Base.broadcastable(x::AbstractSGSamplingType) = tuple(x)
Base.broadcastable(x::AbstractTendencyModel) = tuple(x)

"""
    RadiationDYCOMS{FT}(; divergence = 3.75e-6, alpha_z = 1.0, kappa = 85.0, F0 = 70.0, F1 = 22.0)

Idealized longwave radiation for the DYCOMS stratocumulus cases of
[Stevens2005](@cite) and [Ackerman2009](@cite).

The net upward flux is parameterized from the liquid-water path above and below
each level, plus a free-tropospheric term above the inversion (located at the
`q_tot = 8 g/kg` isoline) that represents cooling by large-scale divergence.
Selected by `rad: "DYCOMS"`.

# Fields

  - `divergence`: Large-scale horizontal divergence [1/s].
  - `alpha_z`: Coefficient of the free-tropospheric term above the inversion [-].
  - `kappa`: Mass absorption coefficient of cloud liquid water [m²/kg].
  - `F0`: Cloud-top longwave cooling amplitude [W/m²].
  - `F1`: Cloud-base longwave warming amplitude [W/m²].
"""
@kwdef struct RadiationDYCOMS{FT}
    # Large-scale divergence
    divergence::FT = 3.75e-6
    alpha_z::FT = 1.0
    kappa::FT = 85.0
    F0::FT = 70.0
    F1::FT = 22.0
end

"""
    RadiationISDAC{FT}(; F₀ = 72, F₁ = 15, κ = 170)

Idealized longwave radiation for the ISDAC mixed-phase Arctic stratocumulus case.

The net upward flux is `F₀ exp(-κ (LWP_top - LWP_z)) + F₁ exp(-κ LWP_z)`, where
`LWP_z` is the liquid water path integrated from the surface to `z`. Selected
by `rad: "ISDAC"`.

# Fields

  - `F₀`: Cloud-top longwave cooling amplitude [W/m²].
  - `F₁`: Cloud-base longwave warming amplitude [W/m²].
  - `κ`: Mass absorption coefficient of cloud liquid water [m²/kg].
"""
@kwdef struct RadiationISDAC{FT}
    F₀::FT = 72  # W/m²
    F₁::FT = 15  # W/m²
    κ::FT = 170  # m²/kg
end

import AtmosphericProfilesLibrary as APL

"""
    RadiationTRMM_LBA(::Type{FT})

Prescribed radiative heating profile for the TRMM-LBA deep-convection case,
taken from AtmosphericProfilesLibrary.

The stored profile is evaluated as `rad_profile(t, z)` to give a temperature
tendency [K/s], which is converted to an energy tendency. Selected by
`rad: "TRMM_LBA"`.

# Fields

  - `rad_profile`: Callable `(t, z)` returning the radiative heating rate [K/s].
"""
struct RadiationTRMM_LBA{R}
    rad_profile::R
    function RadiationTRMM_LBA(::Type{FT}) where {FT}
        rad_profile = APL.TRMM_LBA_radiation(FT)
        return new{typeof(rad_profile)}(rad_profile)
    end
end

"""
    PrescribedFlow{FT}

Prescribed velocity field that replaces the solved dynamics, used by kinematic
test cases. Selected by the YAML key `prescribed_flow`, which requires flat
topography and the explicit solver.

`ShipwayHill2012VelocityProfile` is currently the only subtype. Subtypes are
callable as `flow(z, t)` and must also define `get_ρu₃qₜ_surface`.
"""
abstract type PrescribedFlow{FT} end

"""
    ShipwayHill2012VelocityProfile{FT}

Prescribed vertical velocity of the kinematic driver of [ShipwayHill2012](@cite).
Selected by `prescribed_flow: "ShipwayHill2012"`.

The instance is callable; see the call method below.
"""
struct ShipwayHill2012VelocityProfile{FT} <: PrescribedFlow{FT} end

"""
    (flow::ShipwayHill2012VelocityProfile{FT})(z, t)

Return the prescribed vertical velocity `w = 1.5 sin(π t / 600 s)` [m/s] during
the first 600 s and zero afterwards. The profile is height-independent, so `z`
is ignored.
"""
function (::ShipwayHill2012VelocityProfile{FT})(z, t) where {FT}
    w1 = FT(1.5)
    t1 = FT(600)
    return t < t1 ? w1 * sinpi(FT(t) / t1) : FT(0)
end

"""
    get_ρu₃qₜ_surface(flow::ShipwayHill2012VelocityProfile, thermo_params, t)

Compute the vertical moisture transport `ρ w qₜ` at the surface implied by the
prescribed flow.

The surface state is currently hard-coded to the Shipway and Hill (2012)
values: `p_sfc = 100 700 Pa`, `θ_sfc = 297.9 K`, and a water vapor mixing ratio
of 0.015 kg/kg.

# Arguments

  - `flow`: The prescribed flow model, see [`PrescribedFlow`](@ref).
  - `thermo_params`: Thermodynamic parameters, needed for the surface air density.
  - `t`: Current simulation time [s].

# Returns

The surface moisture flux as a `WVector` [kg/(m² s)].
"""
function get_ρu₃qₜ_surface(flow::ShipwayHill2012VelocityProfile, thermo_params, t)
    # TODO: Get these values from the setup instead of hardcoding:
    FT = eltype(thermo_params)
    rv_sfc = FT(0.015)  # water vapour mixing ratio at surface (kg/kg)
    q_tot_sfc = rv_sfc / (1 + rv_sfc)  # 0.0148 kg/kg
    p_sfc = FT(100_700)
    θ_sfc = FT(297.9)
    T =
        TD.saturation_adjustment(
            thermo_params,
            TD.pθ_li(),
            p_sfc,
            θ_sfc,
            q_tot_sfc;
            maxiter = 4,
        ).T
    ρ_sfc = TD.air_density(thermo_params, T, p_sfc, q_tot_sfc)  # 1.165 kg/m³
    w_sfc = Geometry.WVector(flow(0, t))
    return ρ_sfc * w_sfc * q_tot_sfc
end

"""
    TestDycoreConsistency

Debugging marker: fill the cache with `NaN`s before each tendency evaluation so
that any quantity the dycore reads without first setting it shows up as a
`NaN`. Selected by `test_dycore_consistency: true`.
"""
struct TestDycoreConsistency end

"""
    ReproducibleRestart

Marker requesting that the simulation be reproducible when restarted from a
restart file, at the cost of deterministically reconstructing cache state that
would otherwise depend on the previous step (the cloud fraction is first
recomputed with `GridScaleCloud` before the Picard iteration). Selected by
`reproducible_restart: true`; disable it for production runs.
"""
struct ReproducibleRestart end

"""
    AbstractTimesteppingMode

Whether a process is integrated explicitly or implicitly.

Subtypes: `Explicit` and `Implicit`. Used for the `diff_mode` numerics option
(config key `implicit_diffusion`) and for
`microphysics_tendency_timestepping` (config key `implicit_microphysics`).
"""
abstract type AbstractTimesteppingMode end

"""
    Explicit

Integrate the process explicitly, as part of the remaining tendency.
"""
struct Explicit <: AbstractTimesteppingMode end

"""
    Implicit

Integrate the process implicitly, as part of the Newton solve, which requires a
corresponding Jacobian block.
"""
struct Implicit <: AbstractTimesteppingMode end
Base.broadcastable(x::AbstractTimesteppingMode) = tuple(x)

"""
    QuasiMonotoneLimiter

Marker selecting ClimaCore's `QuasiMonotoneLimiter` for horizontal tracer
transport, which clips element-wise tracer extrema to those of the upwind
neighborhood. Selected by `apply_sem_quasimonotone_limiter: true`.
"""
struct QuasiMonotoneLimiter end # For dispatching to use the ClimaCore QuasiMonotoneLimiter.

"""
    AbstractScaleBlendingMethod

Method used to combine the candidate EDMF mixing-length scales into a single
master length scale in `blend_scales`. Selected by the YAML key
`edmfx_scale_blending`.

Subtypes:

  - `SmoothMinimumBlending`: Lamb smooth minimum (`"SmoothMinimum"`).
  - `HardMinimumBlending`: plain minimum (`"HardMinimum"`).
"""
abstract type AbstractScaleBlendingMethod end

"""
    SmoothMinimumBlending

Blend the mixing-length scales with the Lamb smooth minimum, a differentiable
approximation to `minimum` controlled by the `smin_ub` and `smin_rm`
parameters. Selected by `edmfx_scale_blending: "SmoothMinimum"`.
"""
struct SmoothMinimumBlending <: AbstractScaleBlendingMethod end

"""
    HardMinimumBlending

Blend the mixing-length scales by taking their plain minimum. Selected by
`edmfx_scale_blending: "HardMinimum"`.
"""
struct HardMinimumBlending <: AbstractScaleBlendingMethod end
Base.broadcastable(x::AbstractScaleBlendingMethod) = tuple(x)

"""
    AtmosNumerics{EN_UP, TR_UP, ED_UP, SG_UP, ED_TR_UP, TDC, RR, LIM, DM, HD}

Numerical options of an `AtmosModel`: upwinding schemes, limiter,
diffusion timestepping mode, hyperdiffusion, and debugging switches.

The upwinding fields hold `Val` symbols so that the scheme is a compile-time
dispatch: `Val(:none)`, `Val(:first_order)`, `Val(:third_order)`, or
`Val(:vanleer_limiter)`. Use the keyword constructor below to pass them as
plain symbols or strings.

# Fields

  - `energy_q_tot_upwinding`: Upwinding for the vertical advection of `ρe_tot`
    and `ρq_tot`.
  - `tracer_upwinding`: Upwinding for the vertical advection of the remaining
    grid-scale tracers.
  - `edmfx_mse_q_tot_upwinding`: Upwinding for the EDMF subdomain `mse`, `q_tot`,
    and TKE advection.
  - `edmfx_sgsflux_upwinding`: Upwinding for the EDMF subgrid-scale mass flux.
  - `edmfx_tracer_upwinding`: Upwinding for the EDMF subdomain tracers.
  - `test_dycore_consistency`: `nothing`, or `TestDycoreConsistency` to fill the
    cache with `NaN`s for debugging.
  - `reproducible_restart`: `nothing`, or `ReproducibleRestart` to make restarts
    reproducible.
  - `limiter`: `nothing`, or `QuasiMonotoneLimiter` for horizontal tracer
    transport.
  - `diff_mode`: `Explicit()` or `Implicit()`, the timestepping mode for vertical
    diffusion.
  - `hyperdiff`: `nothing`, or a `Hyperdiffusion` model.
"""
struct AtmosNumerics{EN_UP, TR_UP, ED_UP, SG_UP, ED_TR_UP, TDC, RR, LIM, DM, HD}
    # Enable specific upwinding schemes for specific equations
    energy_q_tot_upwinding::EN_UP
    tracer_upwinding::TR_UP
    edmfx_mse_q_tot_upwinding::ED_UP
    edmfx_sgsflux_upwinding::SG_UP
    edmfx_tracer_upwinding::ED_TR_UP
    # Add NaNs to certain equations to track down problems
    test_dycore_consistency::TDC
    # Whether the simulation is reproducible when restarting from a restart file
    reproducible_restart::RR
    limiter::LIM
    # Timestepping mode for diffusion: Explicit() or Implicit()
    diff_mode::DM
    # Hyperdiffusion model: nothing or Hyperdiffusion()
    hyperdiff::HD
end
Base.broadcastable(x::AtmosNumerics) = tuple(x)

"""
    AtmosNumerics(; energy_q_tot_upwinding = :vanleer_limiter, tracer_upwinding = :vanleer_limiter,
                  edmfx_mse_q_tot_upwinding = :first_order, edmfx_sgsflux_upwinding = :none,
                  edmfx_tracer_upwinding = :first_order, test_dycore_consistency = nothing,
                  reproducible_restart = nothing, limiter = nothing, diff_mode = Explicit(),
                  hyperdiff = Hyperdiffusion{Float32}(...), kwargs...)

Create an `AtmosNumerics`, converting the upwinding options to `Val`
types for compile-time dispatch.

# Keyword Arguments

  - `energy_q_tot_upwinding = :vanleer_limiter`: Upwinding for `ρe_tot` and
    `ρq_tot` vertical advection. Valid values are `:none`, `:first_order`,
    `:third_order`, and `:vanleer_limiter`, given as a `Symbol`, a `String`, or
    an already-wrapped `Val`.
  - `tracer_upwinding = :vanleer_limiter`: Upwinding for the other grid-scale
    tracers, same valid values.
  - `edmfx_mse_q_tot_upwinding = :first_order`: Upwinding for the EDMF subdomain
    `mse`, `q_tot`, and TKE.
  - `edmfx_sgsflux_upwinding = :none`: Upwinding for the EDMF subgrid-scale mass
    flux.
  - `edmfx_tracer_upwinding = :first_order`: Upwinding for the EDMF subdomain
    tracers.
  - `test_dycore_consistency = nothing`: Pass `TestDycoreConsistency()` to fill
    the cache with `NaN`s.
  - `reproducible_restart = nothing`: Pass `ReproducibleRestart()` for
    reproducible restarts.
  - `limiter = nothing`: Pass `QuasiMonotoneLimiter()` to limit horizontal tracer
    transport.
  - `diff_mode = Explicit()`: Timestepping mode for vertical diffusion.
  - `hyperdiff`: Hyperdiffusion model; defaults to a `Float32` `Hyperdiffusion`
    with the CAM-SE vorticity coefficient, `divergence_damping_factor = 5`, and
    `prandtl_number = 1.0`. Pass `nothing` to disable hyperdiffusion.

!!! warning

    Unrecognized keyword arguments are absorbed by `kwargs...` and silently
    ignored, so a misspelled numerics option is not reported here.

# Examples

```julia
numerics = ClimaAtmos.AtmosNumerics(; tracer_upwinding = :third_order, hyperdiff = nothing)
```
"""
function AtmosNumerics(;
    energy_q_tot_upwinding = :vanleer_limiter,
    tracer_upwinding = :vanleer_limiter,
    edmfx_mse_q_tot_upwinding = :first_order,
    edmfx_sgsflux_upwinding = :none,
    edmfx_tracer_upwinding = :first_order,
    test_dycore_consistency = nothing,
    reproducible_restart = nothing,
    limiter = nothing,
    diff_mode = Explicit(),
    hyperdiff = Hyperdiffusion{Float32}(;
        ν₄_vorticity_coeff = 0.150 * 1.238,
        divergence_damping_factor = 5,
        prandtl_number = 1.0,
    ),
    kwargs...,
)
    # Helper to convert symbols/strings to Val types, or keep Val types as-is
    parse_upwinding(x::Union{Symbol, String}) = Val(Symbol(x))
    parse_upwinding(x::Val) = x

    return AtmosNumerics(
        parse_upwinding(energy_q_tot_upwinding),
        parse_upwinding(tracer_upwinding),
        parse_upwinding(edmfx_mse_q_tot_upwinding),
        parse_upwinding(edmfx_sgsflux_upwinding),
        parse_upwinding(edmfx_tracer_upwinding),
        test_dycore_consistency,
        reproducible_restart,
        limiter,
        diff_mode,
        hyperdiff,
    )
end

"""
    ValTF

`Union{Val{true}, Val{false}}`, the type of a boolean flag lifted to the type
domain so that it can be dispatched on at compile time.
"""
const ValTF = Union{Val{true}, Val{false}}

"""
    EDMFXModel{EEM, EDM, ESMF, ESDF, ENP, EVD, EF, SBM}

Switches and closures of the EDMF scheme, kept separate from the
turbulence-convection model itself (`PrognosticEDMFX` or `EDOnlyEDMFX`) so that
the individual terms can be enabled independently.

The boolean switches are stored as `Val{true}`/`Val{false}` (see `ValTF`) so
that the disabled terms are compiled away; the keyword constructor below
accepts plain `Bool`s.

# Fields

  - `entr_model`: Entrainment closure, an `AbstractEntrainmentModel` or `nothing`.
  - `detr_model`: Detrainment closure, an `AbstractDetrainmentModel` or `nothing`.
  - `sgs_mass_flux`: Whether the subgrid-scale mass flux is applied to the
    grid-mean equations (`edmfx_sgs_mass_flux`).
  - `sgs_diffusive_flux`: Whether the subgrid-scale diffusive flux is applied
    (`edmfx_sgs_diffusive_flux`).
  - `nh_pressure`: Whether the non-hydrostatic pressure drag closure is applied;
    the buoyancy term of the pressure closure is always on (`edmfx_nh_pressure`).
  - `vertical_diffusion`: Whether the prognostic updrafts are vertically diffused
    (`edmfx_vertical_diffusion`).
  - `filter`: Whether negative updraft vertical velocities are relaxed away
    (`edmfx_filter`).
  - `scale_blending_method`: `AbstractScaleBlendingMethod` used to blend the
    mixing-length scales (`edmfx_scale_blending`).
"""
struct EDMFXModel{
    EEM, EDM,
    ESMF <: ValTF, ESDF <: ValTF, ESDFH <: ValTF, ENP <: ValTF, EVD <: ValTF,
    EHD <: ValTF, EF <: ValTF,
    SBM <: AbstractScaleBlendingMethod,
}
    entr_model::EEM
    detr_model::EDM
    sgs_mass_flux::ESMF
    sgs_diffusive_flux::ESDF
    sgs_diffusive_flux_horizontal::ESDFH
    nh_pressure::ENP
    vertical_diffusion::EVD
    horizontal_diffusion::EHD
    filter::EF
    scale_blending_method::SBM
end


# Convenience constructor that converts booleans to Val types
# This outer constructor allows passing booleans, which are converted to Val types
"""
    EDMFXModel(; entr_model = nothing, detr_model = nothing, sgs_mass_flux = false,
               sgs_diffusive_flux = false, nh_pressure = false, vertical_diffusion = false,
               filter = false, scale_blending_method, kwargs...)

Create an `EDMFXModel`, lifting the boolean switches to `Val` types.

# Keyword Arguments

  - `entr_model = nothing`: Entrainment closure, e.g. `InvZEntrainment()`.
  - `detr_model = nothing`: Detrainment closure, e.g. `BuoyancyVelocityDetrainment()`.
  - `sgs_mass_flux = false`: Enable the subgrid-scale mass flux.
  - `sgs_diffusive_flux = false`: Enable the subgrid-scale diffusive flux.
  - `nh_pressure = false`: Enable the non-hydrostatic pressure drag.
  - `vertical_diffusion = false`: Enable vertical diffusion of the updrafts.
  - `filter = false`: Enable relaxation of negative updraft velocities.
  - `scale_blending_method`: Required; an `AbstractScaleBlendingMethod`.

Each boolean may also be given as an already-wrapped `Val{true}`/`Val{false}`.
Unrecognized keyword arguments are absorbed by `kwargs...` and ignored.

# Examples

```julia
edmfx_model = ClimaAtmos.EDMFXModel(;
    entr_model = ClimaAtmos.InvZEntrainment(),
    detr_model = ClimaAtmos.BuoyancyVelocityDetrainment(),
    sgs_mass_flux = true,
    sgs_diffusive_flux = true,
    scale_blending_method = ClimaAtmos.SmoothMinimumBlending(),
)
```
"""
function EDMFXModel(;
    entr_model = nothing,
    detr_model = nothing,
    sgs_mass_flux::Union{Bool, ValTF} = false,
    sgs_diffusive_flux::Union{Bool, ValTF} = false,
    sgs_diffusive_flux_horizontal::Union{Bool, ValTF} = false,
    nh_pressure::Union{Bool, ValTF} = false,
    vertical_diffusion::Union{Bool, ValTF} = false,
    horizontal_diffusion::Union{Bool, ValTF} = false,
    filter::Union{Bool, ValTF} = false,
    scale_blending_method,
    kwargs...,
)
    parse_val_tf(x::Bool) = Val(x)
    parse_val_tf(x::ValTF) = x
    # Convert booleans to Val types, keep Val types as-is
    return EDMFXModel(
        entr_model,
        detr_model,
        parse_val_tf(sgs_mass_flux),
        parse_val_tf(sgs_diffusive_flux),
        parse_val_tf(sgs_diffusive_flux_horizontal),
        parse_val_tf(nh_pressure),
        parse_val_tf(vertical_diffusion),
        parse_val_tf(horizontal_diffusion),
        parse_val_tf(filter),
        scale_blending_method,
    )
end

# Grouped structs to reduce AtmosModel type parameters

"""
    SCMSetup{S, EF, LA, AT, SC}(; subsidence = nothing, external_forcing = nothing,
                                ls_adv = nothing, advection_test = false, scm_coriolis = nothing)

Group of single-column-model and large-eddy-simulation forcings inside an
`AtmosModel`.

These components are primarily used for testing, calibration, and research with
single-column setups; most global runs leave them all at `nothing`. They are
usually supplied by a `Setups` case rather than set by hand.

# Fields

  - `subsidence`: `nothing`, or a `LargeScaleSubsidence`.
  - `external_forcing`: `nothing`, or a forcing object
    (`ExternalDrivenTVForcing`, `ISDACForcing`).
  - `ls_adv`: `nothing`, or a `LargeScaleAdvection`.
  - `advection_test`: Whether to run in pure tracer-advection test mode, in which
    the dynamics are frozen.
  - `scm_coriolis`: `nothing`, or a `NamedTuple` `(; prof_ug, prof_vg, coriolis_param)` prescribing the geostrophic wind and Coriolis parameter.
"""
@kwdef struct SCMSetup{S, EF, LA, AT, SC}
    subsidence::S = nothing
    external_forcing::EF = nothing
    ls_adv::LA = nothing
    advection_test::AT = false
    scm_coriolis::SC = nothing
end


"""
    AbstractChemistryModel

Atmospheric chemistry treatment. Selected by the YAML key `chemistry_model`;
`~` disables chemistry.

`GasPhaseChem` is currently the only subtype.
"""
abstract type AbstractChemistryModel end

"""
    GasPhaseChem

Carry a single passive gas-phase tracer `q_gas_A`, used to exercise the tracer
infrastructure. Selected by `chemistry_model: "passive"`.
"""
struct GasPhaseChem <: AbstractChemistryModel end

"""
    AtmosChem{CM}(; chemistry_model = nothing)

Group of chemistry models inside an `AtmosModel`.

# Fields

  - `chemistry_model`: `nothing`, or an `AbstractChemistryModel` such as
    `GasPhaseChem()`.
"""
@kwdef struct AtmosChem{CM}
    chemistry_model::CM = nothing
end

"""
    AtmosWater{MM, CM, MTTS, TNM, SQ, SVM, TCM, TVL, TVI, TVR, TVS}(; microphysics_model = DryModel(), kwargs...)

Group of moisture, cloud, and microphysics choices inside an
`AtmosModel`.

# Fields

  - `microphysics_model`: An `AbstractMicrophysicsModel`; `DryModel()` by default.
  - `cloud_model`: An `AbstractCloudModel`; `QuadratureCloud()` by default.
  - `microphysics_tendency_timestepping`: `Explicit()`, `Implicit()`, or
    `nothing` when there is no microphysics.
  - `tracer_nonnegativity_method`: `nothing`, or a `TracerNonnegativityMethod`.
  - `sgs_quadrature`: `nothing`, or an `SGSQuadrature` used to integrate cloud
    and microphysics quantities over the subgrid-scale distribution.
  - `sgs_variance_model`: an `AbstractSGSVarianceModel`; `SGSVarianceVertical()` by
    default.
  - `tq_correlation_model`: an `AbstractTqCorrelationModel`; `ConstantTqCorrelation()`
    by default.
  - `terminal_velocity_mode`: `DiagnosticTerminalVelocity()` (the default) or a
    `FixedTerminalVelocity`.

# Examples

```julia
water = ClimaAtmos.AtmosWater(;
    microphysics_model = ClimaAtmos.EquilibriumMicrophysics0M(),
    cloud_model = ClimaAtmos.GridScaleCloud(),
)
```
"""
@kwdef struct AtmosWater{MM, CM, MTTS, TNM, SQ, SVM, TCM, TVL, TVI, TVR, TVS}
    microphysics_model::MM = DryModel()
    cloud_model::CM = QuadratureCloud()
    microphysics_tendency_timestepping::MTTS = nothing
    tracer_nonnegativity_method::TNM = nothing
    sgs_quadrature::SQ = nothing
    sgs_variance_model::SVM = SGSVarianceVertical()
    tq_correlation_model::TCM = ConstantTqCorrelation()
    terminal_velocity_liquid::TVL = FixedTerminalVelocity()
    terminal_velocity_ice::TVI = FixedTerminalVelocity()
    terminal_velocity_rain::TVR = DiagnosticTerminalVelocity()
    terminal_velocity_snow::TVS = FixedTerminalVelocity()
end

"""
    AtmosRadiation{RM, IN}(; radiation_mode = nothing, insolation = IdealizedInsolation())

Group of radiation choices inside an `AtmosModel`.

# Fields

  - `radiation_mode`: `nothing`, an RRTMGP mode (`RRTMGPI.GrayRadiation`,
    `ClearSkyRadiation`, `AllSkyRadiation`,
    `AllSkyRadiationWithClearSkyDiagnostics`), an idealized profile
    (`RadiationDYCOMS`, `RadiationISDAC`, `RadiationTRMM_LBA`), or
    `HeldSuarezForcing()`.
  - `insolation`: An `AbstractInsolation`; `IdealizedInsolation()` by default.
"""
@kwdef struct AtmosRadiation{RM, IN}
    radiation_mode::RM = nothing
    insolation::IN = IdealizedInsolation()
end

"""
    AtmosTurbconv{EDMFX, TCM, SL, AMD, CHD}(; edmfx_model = nothing, turbconv_model = nothing, kwargs...)

Group of turbulence, convection, and LES closures inside an
`AtmosModel`.

# Fields

  - `edmfx_model`: `nothing`, or an `EDMFXModel` holding the EDMF term switches.
  - `turbconv_model`: `nothing`, `PrognosticEDMFX(...)`, or `EDOnlyEDMFX()`.
  - `smagorinsky_lilly`: `nothing`, or a `SmagorinskyLilly`.
  - `amd_les`: `nothing`, or an `AnisotropicMinimumDissipation`.
  - `constant_horizontal_diffusion`: `nothing`, or a
    `ConstantHorizontalDiffusion`.
"""
@kwdef struct AtmosTurbconv{EDMFX, TCM, SL, AMD, CHD}
    edmfx_model::EDMFX = nothing
    turbconv_model::TCM = nothing
    smagorinsky_lilly::SL = nothing
    amd_les::AMD = nothing
    constant_horizontal_diffusion::CHD = nothing
end

"""
    AtmosGravityWave{NOGW, OGW}(; non_orographic_gravity_wave = nothing, orographic_gravity_wave = nothing)

Group of gravity-wave drag parameterizations inside an `AtmosModel`.

# Fields

  - `non_orographic_gravity_wave`: `nothing`, or a `NonOrographicGravityWave`.
  - `orographic_gravity_wave`: `nothing`, or an `OrographicGravityWave`
    (`FullOrographicGravityWave` or `LinearOrographicGravityWave`).
"""
@kwdef struct AtmosGravityWave{NOGW, OGW}
    non_orographic_gravity_wave::NOGW = nothing
    orographic_gravity_wave::OGW = nothing
end

"""
    AtmosSponge{VS, RS}(; viscous_sponge = nothing, rayleigh_sponge = nothing)

Group of model-top sponge layers inside an `AtmosModel`.

# Fields

  - `viscous_sponge`: `nothing`, or a `ViscousSponge`.
  - `rayleigh_sponge`: `nothing`, or a `RayleighSponge`.
"""
@kwdef struct AtmosSponge{VS, RS}
    viscous_sponge::VS = nothing
    rayleigh_sponge::RS = nothing
end

"""
    AtmosSurface{FS, ST, BO, AL}(; flux_scheme, temperature, boundary_overrides, surface_albedo)

Group of surface models inside an `AtmosModel`: the flux closure, the surface
temperature, per-cell boundary overrides, and the albedo.

By default the surface uses fixed exchange coefficients
(`Cd = Ch = 0.0044`), a zonally symmetric analytic SST, no boundary overrides,
and a constant albedo of 0.07.

# Fields

  - `flux_scheme`: a
    [`SurfaceConditions.SurfaceParameterization`](@ref ClimaAtmos.SurfaceConditions.SurfaceParameterization)
    describing the surface flux closure
    ([`MoninObukhov`](@ref ClimaAtmos.SurfaceConditions.MoninObukhov),
    [`ExchangeCoefficients`](@ref ClimaAtmos.SurfaceConditions.ExchangeCoefficients);
    `MoninObukhov` may carry a time-varying `fluxes` callable), or `nothing` to
    skip atmos-side surface updates (e.g. when an external driver overwrites
    `sfc_conditions`). YAML configs may also use
    [`DefaultMoninObukhov`](@ref ClimaAtmos.SurfaceConditions.DefaultMoninObukhov)/[`DefaultExchangeCoefficients`](@ref ClimaAtmos.SurfaceConditions.DefaultExchangeCoefficients)
    markers, which the config-driven `AtmosSurface` constructor resolves against
    `params` eagerly.
  - `temperature`: a
    [`SurfaceConditions.SurfaceTemperature`](@ref ClimaAtmos.SurfaceConditions.SurfaceTemperature)
    ([`AnalyticTemperature`](@ref ClimaAtmos.SurfaceConditions.AnalyticTemperature),
    [`ExternalTemperature`](@ref ClimaAtmos.SurfaceConditions.ExternalTemperature),
    [`SlabOceanTemperature`](@ref ClimaAtmos.SurfaceConditions.SlabOceanTemperature),
    [`CoupledTemperature`](@ref ClimaAtmos.SurfaceConditions.CoupledTemperature)).
  - `boundary_overrides`: a
    [`SurfaceConditions.SurfaceBoundaryOverrides`](@ref ClimaAtmos.SurfaceConditions.SurfaceBoundaryOverrides)
    carrying per-cell defaults for surface pressure / humidity / winds / gustiness
    / beta.
  - `surface_albedo`: a [`SurfaceAlbedoModel`](@ref ClimaAtmos.SurfaceAlbedoModel)
    ([`ConstantAlbedo`](@ref ClimaAtmos.ConstantAlbedo),
    [`RegressionFunctionAlbedo`](@ref ClimaAtmos.RegressionFunctionAlbedo),
    [`CouplerAlbedo`](@ref ClimaAtmos.CouplerAlbedo)).

# Examples

```julia
surface = ClimaAtmos.AtmosSurface(;
    temperature = ClimaAtmos.SurfaceConditions.SlabOceanTemperature{Float32}(),
)
```
"""
@kwdef struct AtmosSurface{FS, ST, BO, AL}
    flux_scheme::FS = SurfaceConditions.ExchangeCoefficients{Float32}(
        Cd = 0.0044, Ch = 0.0044,
    )
    temperature::ST =
        SurfaceConditions.AnalyticTemperature(
            Setups.zonally_symmetric_temperature,
        )
    boundary_overrides::BO = SurfaceConditions.SurfaceBoundaryOverrides()
    surface_albedo::AL = ConstantAlbedo{Float32}(; α = 0.07)
end

"""
    COSPModel{N}(; n_subcolumns = Val(256), overlap = :maximum_random, random_seed = UInt64(1))

Configuration of the COSP satellite simulator, which samples the model's cloud
field into statistically generated subcolumns before computing instrument-like
diagnostics.

Enabled by setting a finite `dt_subcol` in the config; `dt_subcol: Inf`
disables COSP entirely and leaves the model's `cosp` field as `nothing`.

# Fields

  - `n_subcolumns`: Number of SCOPS subcolumns per grid column, held as a `Val`
    so that the subcolumn loop is statically sized [-].
  - `overlap`: Cloud overlap assumption; one of `:maximum`, `:random`, or
    `:maximum_random`.
  - `random_seed`: Seed for the SCOPS overlap selectors, fixed so that the
    subcolumns are reproducible across calls.
"""
@kwdef struct COSPModel{N, O}
    n_subcolumns::Val{N} = Val(100)
    overlap::Val{O} = Val(:maximum_random)
    random_seed::UInt64 = UInt64(1)
end

@inline _cosp_nsubcolumns(::Val{N}) where {N} = N
@inline _cosp_overlap(::Val{O}) where {O} = O

# Add broadcastable for the new grouped types
Base.broadcastable(x::SCMSetup) = tuple(x)
Base.broadcastable(x::AtmosWater) = tuple(x)
Base.broadcastable(x::AtmosRadiation) = tuple(x)
Base.broadcastable(x::AtmosTurbconv) = tuple(x)
Base.broadcastable(x::AtmosGravityWave) = tuple(x)
Base.broadcastable(x::AtmosSponge) = tuple(x)
Base.broadcastable(x::AtmosSurface) = tuple(x)
Base.broadcastable(x::COSPModel) = tuple(x)

# `AtmosX(config::AtmosConfig, ...)` constructors live below the `AtmosConfig`
# struct definition (later in this file) so the type is in scope when those
# methods are parsed.

"""
    AtmosModel{W, SCM, R, TC, PF, GW, VD, SP, SU, NU, CM, COSP}

Complete description of the physics of an atmospheric simulation: which
parameterizations are active and how each is configured.

Components are stored in grouped sub-structs to keep the number of type
parameters manageable, but they can be read either way: `atmos.water.cloud_model`
and `atmos.cloud_model` return the same object, because `Base.getproperty` is
overloaded to forward a grouped property to its owning group (see
`GROUPED_PROPERTY_MAP`). The keyword constructor accepts the same flattened
names.

# Fields

  - `water`: An `AtmosWater` group (moisture, cloud, microphysics).
  - `scm_setup`: An `SCMSetup` group (single-column forcings).
  - `radiation`: An `AtmosRadiation` group (radiation mode, insolation).
  - `turbconv`: An `AtmosTurbconv` group (EDMF and LES closures).
  - `prescribed_flow`: `nothing`, or a `PrescribedFlow` replacing the dynamics.
  - `gravity_wave`: An `AtmosGravityWave` group.
  - `vertical_diffusion`: `nothing`, or an `AbstractVerticalDiffusion`.
  - `sponge`: An `AtmosSponge` group.
  - `surface`: An `AtmosSurface` group.
  - `numerics`: An `AtmosNumerics` group.
  - `chemistry`: An `AtmosChem` group.
  - `cosp`: `nothing`, or a `COSPModel` for the satellite simulator.
  - `disable_surface_flux_tendency`: Whether to skip applying the surface flux
    tendency, independently of whether surface conditions are computed.

See the keyword constructor `AtmosModel(; kwargs...)` below.
"""
struct AtmosModel{W, SCM, R, TC, PF, GW, VD, SP, SU, NU, CM, COSP}
    water::W
    scm_setup::SCM
    radiation::R
    turbconv::TC
    prescribed_flow::PF
    gravity_wave::GW
    vertical_diffusion::VD
    sponge::SP
    surface::SU
    numerics::NU
    chemistry::CM
    cosp::COSP

    # Whether to apply surface flux tendency (independent of surface conditions)
    disable_surface_flux_tendency::Bool
end

# Map grouped struct types to their names in AtmosModel struct
"""
    ATMOS_MODEL_GROUPS

Tuple of `(GroupType, field_name)` pairs mapping each grouped sub-struct type to
the `AtmosModel` field that holds it. It is the single source of truth
for `_create_grouped_struct` and for `GROUPED_PROPERTY_MAP`.

The `ShipwayHill2012VelocityProfile => :prescribed_flow` entry contributes no
forwarded properties, because that type has no fields; `prescribed_flow` is
passed to the constructor as a complete object.
"""
const ATMOS_MODEL_GROUPS = (
    (AtmosWater, :water),
    (AtmosRadiation, :radiation),
    (AtmosTurbconv, :turbconv),
    (ShipwayHill2012VelocityProfile, :prescribed_flow),
    (AtmosGravityWave, :gravity_wave),
    (AtmosSponge, :sponge),
    (AtmosSurface, :surface),
    (AtmosNumerics, :numerics),
    (SCMSetup, :scm_setup),
    (AtmosChem, :chemistry),
)

# Auto-generate map from property_name to group_field
"""
    GROUPED_PROPERTY_MAP

`Dict` mapping each field name of a grouped sub-struct to the name of the
`AtmosModel` field holding that group, e.g.
`:microphysics_model => :water`.

Built automatically from `ATMOS_MODEL_GROUPS`, and used both to forward
property access (`atmos.microphysics_model`) and to route flattened keyword
arguments in `_partition_atmos_model_kwargs`.
"""
const GROUPED_PROPERTY_MAP = Dict{Symbol, Symbol}(
    property => group_field for
    (group_type, group_field) in ATMOS_MODEL_GROUPS for
    property in fieldnames(group_type)
)

"""
    Base.getproperty(atmos::AtmosModel, ::Val{property_name})
    Base.getproperty(atmos::AtmosModel, property_name::Symbol)

Return the requested property of an `AtmosModel`, transparently
forwarding grouped properties to their owning group: `atmos.microphysics_model`
returns `atmos.water.microphysics_model`.

The `Val` method is `@generated` so that the lookup in `GROUPED_PROPERTY_MAP`
happens at compile time and the forwarding costs nothing at run time; the
`Symbol` method simply wraps its argument in a `Val`. Names that are fields of
`AtmosModel` itself are returned directly.
"""
# Forward property access: atmos.microphysics_model → atmos.water.microphysics_model
# Use ::Val constant for @generated compile-time access
@generated function Base.getproperty(
    atmos::AtmosModel,
    ::Val{property_name},
) where {property_name}
    if haskey(GROUPED_PROPERTY_MAP, property_name)
        group_field = GROUPED_PROPERTY_MAP[property_name]
        return quote
            group = getfield(atmos, $(QuoteNode(group_field)))
            getfield(group, $(QuoteNode(property_name)))
        end
    else
        return quote
            getfield(atmos, $(QuoteNode(property_name)))
        end
    end
end

@inline Base.getproperty(atmos::AtmosModel, property_name::Symbol) =
    getproperty(atmos, Val{property_name}())

Base.broadcastable(x::AtmosModel) = tuple(x)

"""
    AtmosModel(; kwargs...)

Create an `AtmosModel`, defaulting to a minimal dry atmosphere.

Every keyword argument is either the name of an `AtmosModel` field (a whole
group, `vertical_diffusion`, `cosp`, `prescribed_flow`, or
`disable_surface_flux_tendency`) or the name of a field of one of the grouped
sub-structs. Flattened names are routed to their owning group through
`GROUPED_PROPERTY_MAP`, so

```julia
AtmosModel(; microphysics_model = EquilibriumMicrophysics0M())
```

is equivalent to passing `water = AtmosWater(; microphysics_model = ...)`.
Passing a complete group object wins: any flattened keywords belonging to that
group are then ignored. Unknown keywords raise an error listing every valid
name.

The resulting model can be read either way:

```julia
model = AtmosModel(; microphysics_model = EquilibriumMicrophysics0M())
model.microphysics_model        # forwarded access
model.water.microphysics_model  # grouped access
```

With no keyword arguments the model is a minimal dry atmosphere:

  - Dry atmosphere: `DryModel()`, with `QuadratureCloud()` but no SGS quadrature.
  - Surface: `AnalyticTemperature` with a zonally symmetric SST, fixed exchange
    coefficients, and a constant albedo of 0.07.
  - `IdealizedInsolation()`, and no radiation, turbulence-convection, gravity
    wave, sponge, or forcing model.
  - Numerics: Van Leer limited upwinding for `ρe_tot`, `ρq_tot`, and the
    tracers, `Explicit()` diffusion, and CAM-SE-like `Float32` hyperdiffusion.

# Keyword Arguments

Grouped into the sub-struct that owns each name; see that struct's docstring
for the full list of admissible values.

  - [`AtmosWater`](@ref): `microphysics_model`, `cloud_model`,
    `microphysics_tendency_timestepping`, `tracer_nonnegativity_method`,
    `sgs_quadrature`, `terminal_velocity_mode`.
  - `SCMSetup`: `subsidence`, `external_forcing`, `ls_adv`,
    `advection_test`, `scm_coriolis`. Normally supplied by a `Setups` case.
  - [`AtmosRadiation`](@ref): `radiation_mode`, `insolation`.
  - [`AtmosTurbconv`](@ref): `edmfx_model`, `turbconv_model`,
    `smagorinsky_lilly`, `amd_les`, `constant_horizontal_diffusion`.
  - [`AtmosGravityWave`](@ref): `non_orographic_gravity_wave`,
    `orographic_gravity_wave`.
  - [`AtmosSponge`](@ref): `viscous_sponge`, `rayleigh_sponge`.
  - [`AtmosSurface`](@ref): `flux_scheme`, `temperature`, `boundary_overrides`,
    `surface_albedo`.
  - `AtmosNumerics`: the five `*_upwinding` options,
    `test_dycore_consistency`, `reproducible_restart`, `limiter`, `diff_mode`,
    `hyperdiff`.
  - [`AtmosChem`](@ref): `chemistry_model`.
  - Ungrouped `AtmosModel` fields: `vertical_diffusion`, `prescribed_flow`,
    `cosp`, and `disable_surface_flux_tendency`.

# Examples

```julia
# Minimal dry model
model = AtmosModel()

# Dry model with Held-Suarez forcing and custom hyperdiffusion
model = AtmosModel(;
    radiation_mode = HeldSuarezForcing(),
    hyperdiff = Hyperdiffusion(;
        ν₄_vorticity_coeff = 1e15,
        divergence_damping_factor = 1.0,
        prandtl_number = 1.0,
    ),
)

# Moist model with all-sky radiation
model = AtmosModel(;
    microphysics_model = EquilibriumMicrophysics0M(),
    radiation_mode = RRTMGPI.AllSkyRadiation(),
)
```

# Default Configuration

The default AtmosModel provides:

  - **Dry atmosphere**: DryModel()
  - **Basic surface**: AnalyticTemperature (zonally-symmetric SST) with default exchange coefficients
  - **Cloud model**: QuadratureCloud() with SGS quadrature
  - **Idealized insolation**: IdealizedInsolation()
  - **Conservative numerics**: First-order upwinding with Explicit() timestepping
  - **No advanced physics**: No radiation, turbulence, or forcing by default

# Available Structs

## AtmosWater

  - `microphysics_model`: DryModel(), EquilibriumMicrophysics0M(), NonEquilibriumMicrophysics1M(), NonEquilibriumMicrophysics2M(), NonEquilibriumMicrophysics2MP3()
  - `cloud_model`: GridScaleCloud(), QuadratureCloud()
  - `microphysics_tendency_timestepping`: Explicit(), Implicit()
  - `sgs_quadrature`: nothing or SGSQuadrature (subgrid-scale quadrature for microphysics tendencies)
  - `terminal_velocity_liquid`: FixedTerminalVelocity (default) or DiagnosticTerminalVelocity
  - `terminal_velocity_ice`: FixedTerminalVelocity (default) or DiagnosticTerminalVelocity
  - `terminal_velocity_rain`: FixedTerminalVelocity or DiagnosticTerminalVelocity (default)
  - `terminal_velocity_snow`: FixedTerminalVelocity (default) or DiagnosticTerminalVelocity

## SCMSetup (Single-Column Model & LES specific - accessed via model.subsidence, model.external_forcing, etc.)

Internal testing and calibration components for single-column setups:

  - `subsidence`: nothing or Bomex_subsidence, Rico_subsidence, DYCOMS_subsidence, etc
  - `external_forcing`: nothing or external forcing objects (ExternalDrivenTVForcing, ISDACForcing)
  - `ls_adv`: nothing or LargeScaleAdvection()
  - `advection_test`: Bool
  - `scm_coriolis`: nothing or NamedTuple `(; prof_ug, prof_vg, coriolis_param)`

## AtmosRadiation

  - `radiation_mode`: Radiation and atmospheric forcing modes

      + Global radiation: RRTMGPI.ClearSkyRadiation(), RRTMGPI.AllSkyRadiation()
      + Atmospheric forcing: HeldSuarezForcing() (for idealized dynamics)
      + SCM-specific: RadiationDYCOMS(), RadiationISDAC(), RadiationTRMM_LBA()

  - `insolation`: IdealizedInsolation(), TimeVaryingInsolation(), etc.

## AtmosTurbconv

  - `edmfx_model`: EDMFXModel()
  - `turbconv_model`: nothing, PrognosticEDMFX(), EDOnlyEDMFX()
  - `smagorinsky_lilly`: nothing or SmagorinskyLilly()
  - `amd_les`: nothing or AnisotropicMinimumDissipation()
  - `constant_horizontal_diffusion`: nothing or ConstantHorizontalDiffusion()

## AtmosGravityWave

  - `non_orographic_gravity_wave`: nothing or NonOrographicGravityWave()
  - `orographic_gravity_wave`: nothing or OrographicGravityWave()

## AtmosSponge

  - `viscous_sponge`: nothing or ViscousSponge()
  - `rayleigh_sponge`: nothing or RayleighSponge()

## AtmosSurface

  - `flux_scheme`: SurfaceConditions.MoninObukhov, SurfaceConditions.ExchangeCoefficients, or a default marker (DefaultMoninObukhov/DefaultExchangeCoefficients), or `nothing` to disable.
  - `temperature`: SurfaceConditions.AnalyticTemperature, ExternalTemperature, SlabOceanTemperature, or CoupledTemperature.
  - `boundary_overrides`: SurfaceConditions.SurfaceBoundaryOverrides
  - `surface_albedo`: ConstantAlbedo(), RegressionFunctionAlbedo(), CouplerAlbedo()

## AtmosNumerics    # Create grouped structs - use provided complete objects or create from individual fields

  - `energy_q_tot_upwinding`, `tracer_upwinding`, `edmfx_mse_q_tot_upwinding`, `edmfx_sgsflux_upwinding`, `edmfx_tracer_upwinding`: Val() upwinding schemes
  - `test_dycore_consistency`: nothing or TestDycoreConsistency() for debugging
  - `limiter`: nothing or QuasiMonotoneLimiter()
  - `vertical_water_borrowing_species`: internal value `nothing` (apply to all tracers; config default is `~`), empty tuple (apply to none; config `[]`), or Tuple{Symbol, ...} from config string/list (e.g. `["ρq_tot"]`) to apply only to those tracers. See config `vertical_water_borrowing_species` in default_config.yml for YAML options.
    (Note: The vertical water borrowing limiter is created in the cache based on `AtmosWaterModel.tracer_nonnegativity_method`)
  - `diff_mode`: Explicit(), Implicit() timestepping mode for diffusion
  - `hyperdiff`: nothing or Hyperdiffusion()

## Top-level Options

  - `vertical_diffusion`: nothing, VerticalDiffusion(), DecayWithHeightDiffusion()
  - `disable_surface_flux_tendency`: Bool
"""
function AtmosModel(; kwargs...)
    group_kwargs, atmos_model_kwargs = _partition_atmos_model_kwargs(kwargs)

    # Create grouped structs - use provided complete objects or create from individual fields
    water = _create_grouped_struct(AtmosWater, atmos_model_kwargs, group_kwargs)
    scm_setup =
        _create_grouped_struct(SCMSetup, atmos_model_kwargs, group_kwargs)
    radiation =
        _create_grouped_struct(AtmosRadiation, atmos_model_kwargs, group_kwargs)
    turbconv =
        _create_grouped_struct(AtmosTurbconv, atmos_model_kwargs, group_kwargs)
    gravity_wave = _create_grouped_struct(
        AtmosGravityWave,
        atmos_model_kwargs,
        group_kwargs,
    )
    sponge =
        _create_grouped_struct(AtmosSponge, atmos_model_kwargs, group_kwargs)
    surface =
        _create_grouped_struct(AtmosSurface, atmos_model_kwargs, group_kwargs)
    numerics =
        _create_grouped_struct(AtmosNumerics, atmos_model_kwargs, group_kwargs)
    chemistry =
        _create_grouped_struct(AtmosChem, atmos_model_kwargs, group_kwargs)

    vertical_diffusion = get(atmos_model_kwargs, :vertical_diffusion, nothing)
    cosp = get(atmos_model_kwargs, :cosp, nothing)
    disable_surface_flux_tendency =
        get(atmos_model_kwargs, :disable_surface_flux_tendency, false)

    prescribed_flow = get(atmos_model_kwargs, :prescribed_flow, nothing)

    return AtmosModel{
        typeof(water),
        typeof(scm_setup),
        typeof(radiation),
        typeof(turbconv),
        typeof(prescribed_flow),
        typeof(gravity_wave),
        typeof(vertical_diffusion),
        typeof(sponge),
        typeof(surface),
        typeof(numerics),
        typeof(chemistry),
        typeof(cosp),
    }(
        water,
        scm_setup,
        radiation,
        turbconv,
        prescribed_flow,
        gravity_wave,
        vertical_diffusion,
        sponge,
        surface,
        numerics,
        chemistry,
        cosp,
        disable_surface_flux_tendency,
    )
end

"""
    _create_grouped_struct(StructType, atmos_model_kwargs, group_kwargs)

Build one grouped sub-struct of an `AtmosModel`.

The `AtmosModel` field owning `StructType` is looked up in
`ATMOS_MODEL_GROUPS`. If the caller passed a complete object under that field
name, it is returned unchanged; otherwise `StructType` is constructed from the
flattened keywords collected for that group. Called from `AtmosModel`.
"""
function _create_grouped_struct(StructType, atmos_model_kwargs, group_kwargs)
    field_name = get(Dict(ATMOS_MODEL_GROUPS), StructType, nothing)
    @assert !isnothing(field_name) "StructType $StructType not found in ATMOS_MODEL_GROUPS"
    complete_object = get(atmos_model_kwargs, field_name, nothing)
    return isnothing(complete_object) ?
           StructType(; group_kwargs[field_name]...) : complete_object
end

"""
    _partition_atmos_model_kwargs(kwargs) -> (group_kwargs, atmos_model_kwargs)

Sort the `AtmosModel` keyword arguments into per-group and direct keywords.

Keywords found in `GROUPED_PROPERTY_MAP` go into `group_kwargs[group_field]`,
keywords naming an `AtmosModel` field go into `atmos_model_kwargs`, and
anything else is collected and reported at once by
`_throw_unknown_atmos_model_argument_error`. Called from `AtmosModel`.

# Returns

A tuple `(group_kwargs, atmos_model_kwargs)`, where `group_kwargs` is a `Dict`
from group field name to a `Dict{Symbol, Any}` of that group's keywords, and
`atmos_model_kwargs` is a `Dict{Symbol, Any}` of the direct keywords.
"""
function _partition_atmos_model_kwargs(kwargs)

    # Merge default minimal model arguments with given kwargs

    # group_kwargs contains a Dict for each group in ATMOS_MODEL_GROUPS
    group_kwargs = Dict(map(ATMOS_MODEL_GROUPS) do (_, group_field)
        group_field => Dict{Symbol, Any}()
    end)

    # Sort kwargs into a hierarchy of dicts matching the AtmosModel struct
    atmos_model_kwargs = Dict{Symbol, Any}()
    unknown_args = Symbol[]

    for (key, value) in pairs(kwargs)
        if haskey(GROUPED_PROPERTY_MAP, key)
            group_field = GROUPED_PROPERTY_MAP[key]
            group_kwargs[group_field][key] = value
        elseif key in fieldnames(AtmosModel)
            atmos_model_kwargs[key] = value
        else
            push!(unknown_args, key)
        end
    end

    # Throw error for all unknown arguments at once
    if !isempty(unknown_args)
        _throw_unknown_atmos_model_argument_error(unknown_args)
    end

    return group_kwargs, atmos_model_kwargs
end

"""
    _throw_unknown_atmos_model_argument_error(unknown_args)

Throw an error naming the unknown `AtmosModel` keyword arguments and listing
every valid one (forwarded group properties plus direct `AtmosModel` fields).
Called from `_partition_atmos_model_kwargs`.
"""
function _throw_unknown_atmos_model_argument_error(unknown_args)
    n_unknown = length(unknown_args)
    plural = n_unknown > 1 ? "s" : ""

    # All valid arguments: forwarded properties + direct AtmosModel fields
    available_forwarded = sort(collect(keys(GROUPED_PROPERTY_MAP)))
    available_direct = sort(collect(fieldnames(AtmosModel)))
    available_all = sort(unique([available_forwarded; available_direct]))

    error(
        "Unknown AtmosModel argument$plural: $(join(unknown_args, ", ")). " *
        "Available arguments:\n  " *
        join(available_all, "\n  "),
    )
end
