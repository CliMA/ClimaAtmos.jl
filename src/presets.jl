module Presets

import ..AtmosModel, ..AtmosSimulation
import ..SphereGrid, ..ColumnGrid
import ..Setups
import ..ClimaAtmosParameters
import ..Parameters as CAP
import ..DryModel
import ..EquilibriumMicrophysics0M
import ..NonEquilibriumMicrophysics1M
import ..GridScaleCloud
import ..SurfaceConditions
import ..IdealizedInsolation
import ..Explicit
import ..PrognosticEDMFX
import ..EDMFXModel
import ..InvZEntrainment, ..BuoyancyVelocityDetrainment
import ..SmoothMinimumBlending

# ============================================================================
# Model presets: A bundle of defaults that the case setup and explicit kwargs
# override. Each returns a NamedTuple of AtmosModel leaf kwargs, passed to the
# `defaults` kwarg: `AtmosModel(grid; setup, defaults = Presets.dry())`.
# ============================================================================

"""
    dry(; kwargs...)

Dry-atmosphere model defaults (`microphysics_model = DryModel()`).
"""
dry(; kwargs...) = (; microphysics_model = DryModel(), kwargs...)

"""
    equil_moist_0m(; kwargs...)

Equilibrium-moisture model defaults: 0-moment microphysics, grid-scale
cloud, zonally-symmetric SST, idealized insolation.
"""
equil_moist_0m(; kwargs...) = (;
    microphysics_model = EquilibriumMicrophysics0M(),
    cloud_model = GridScaleCloud(),
    temperature = SurfaceConditions.AnalyticTemperature(
        Setups.zonally_symmetric_temperature,
    ),
    insolation = IdealizedInsolation(),
    kwargs...,
)

"""
    nonequil_moist_1m(; kwargs...)

[`equil_moist_0m`](@ref) with 1-moment non-equilibrium microphysics and
explicit microphysics tendency timestepping.
"""
nonequil_moist_1m(; kwargs...) = (;
    microphysics_model = NonEquilibriumMicrophysics1M(),
    microphysics_tendency_timestepping = Explicit(),
    cloud_model = GridScaleCloud(),
    temperature = SurfaceConditions.AnalyticTemperature(
        Setups.zonally_symmetric_temperature,
    ),
    insolation = IdealizedInsolation(),
    kwargs...,
)

"""
    prognostic_edmf([FT = Float32]; area_fraction, n_updrafts, prognostic_tke, kwargs...)

Equilibrium-moist model kwargs with the `PrognosticEDMFX`
turbulence-convection scheme: `Generalized` entrainment/detrainment, SGS
mass & diffusive fluxes, non-hydrostatic pressure drag, prognostic updraft
vertical diffusion, and the negative-velocity relaxation filter (matches
the canonical `prognostic_edmfx_*` configs).
"""
function prognostic_edmf(
    ::Type{FT} = Float32;
    area_fraction = FT(1e-5),
    n_updrafts = 1,
    prognostic_tke = true,
    kwargs...,
) where {FT}
    return (;
        microphysics_model = EquilibriumMicrophysics0M(),
        cloud_model = GridScaleCloud(),
        turbconv_model = PrognosticEDMFX(;
            n_updrafts, prognostic_tke, area_fraction,
        ),
        edmfx_model = EDMFXModel(;
            entr_model = InvZEntrainment(),
            detr_model = BuoyancyVelocityDetrainment(),
            sgs_mass_flux = true,
            sgs_diffusive_flux = true,
            nh_pressure = true,
            vertical_diffusion = true,
            filter = true,
            scale_blending_method = SmoothMinimumBlending(),
        ),
        kwargs...,
    )
end

"""
    prognostic_edmf_1m([FT = Float32]; kwargs...)

[`prognostic_edmf`](@ref) with 1-moment non-equilibrium microphysics and explicit
microphysics tendency timestepping (matches the canonical `prognostic_edmfx_*`
configs that use `microphysics_model: "1M"`). All keyword arguments are
forwarded to [`prognostic_edmf`](@ref).
"""
function prognostic_edmf_1m(::Type{FT} = Float32; kwargs...) where {FT}
    return prognostic_edmf(
        FT;
        microphysics_model = NonEquilibriumMicrophysics1M(),
        microphysics_tendency_timestepping = Explicit(),
        kwargs...,
    )
end

# ============================================================================
# Simulation presets — return an AtmosSimulation
# ============================================================================

"""
    aquaplanet([FT = Float32]; grid, params, setup, model_kwargs, kwargs...)

Aquaplanet simulation preset: global [`SphereGrid`](@ref),
[`equil_moist_0m`](@ref) physics, default `DecayingProfile` setup.
"""
function aquaplanet(
    ::Type{FT} = Float32;
    params = ClimaAtmosParameters(FT),
    grid = SphereGrid(FT; radius = CAP.planet_radius(params)),
    setup = nothing,
    model_kwargs = (;),
    kwargs...,
) where {FT}
    model = AtmosModel(
        grid;
        params,
        setup,
        defaults = equil_moist_0m(),
        model_kwargs...,
    )
    return AtmosSimulation(model; kwargs...)
end

end # module
