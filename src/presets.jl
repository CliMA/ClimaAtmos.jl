module Presets

import ..AtmosModel, ..AtmosSimulation
import ..SphereGrid, ..ColumnGrid
import ..Setups
import ..ClimaAtmosParameters
import ..DryModel
import ..EquilibriumMicrophysics0M
import ..NonEquilibriumMicrophysics1M
import ..GridScaleCloud
import ..PrescribedSST
import ..ZonallySymmetricSST
import ..IdealizedInsolation
import ..Explicit
import ..DiagnosticEDMFX, ..PrognosticEDMFX
import ..EDMFXModel
import ..InvZEntrainment, ..BuoyancyVelocityDetrainment
import ..SmoothMinimumBlending

# ============================================================================
# Model presets — return an AtmosModel
# ============================================================================

"""
    dry(; kwargs...)

Dry atmosphere preset (`microphysics_model = DryModel()`).
Keyword arguments are forwarded to [`AtmosModel`](@ref).
"""
function dry(; kwargs...)
    defaults = (; microphysics_model = DryModel())
    return AtmosModel(; defaults..., kwargs...)
end

"""
    equil_moist_0m(; kwargs...)

Equilibrium-moisture preset with 0-moment microphysics, grid-scale cloud,
prescribed zonally-symmetric SST, and idealized insolation.
Keyword arguments are forwarded to [`AtmosModel`](@ref).
"""
function equil_moist_0m(; kwargs...)
    defaults = (;
        microphysics_model = EquilibriumMicrophysics0M(),
        cloud_model = GridScaleCloud(),
        surface_model = PrescribedSST(),
        sfc_temperature = ZonallySymmetricSST(),
        insolation = IdealizedInsolation(),
    )
    return AtmosModel(; defaults..., kwargs...)
end

"""
    nonequil_moist_1m(; kwargs...)

Non-equilibrium-moisture preset with 1-moment microphysics, explicit
microphysics tendency timestepping, grid-scale cloud, prescribed
zonally-symmetric SST, and idealized insolation. Mirrors [`equil_moist_0m`](@ref)
but with 1-moment non-equilibrium microphysics in place of 0-moment equilibrium.
Keyword arguments are forwarded to [`AtmosModel`](@ref).
"""
function nonequil_moist_1m(; kwargs...)
    defaults = (;
        microphysics_model = NonEquilibriumMicrophysics1M(),
        microphysics_tendency_timestepping = Explicit(),
        cloud_model = GridScaleCloud(),
        surface_model = PrescribedSST(),
        sfc_temperature = ZonallySymmetricSST(),
        insolation = IdealizedInsolation(),
    )
    return AtmosModel(; defaults..., kwargs...)
end

"""
    diagnostic_edmf([FT = Float32]; area_fraction, n_updrafts, prognostic_tke, kwargs...)

Equilibrium-moist model with the [`DiagnosticEDMFX`](@ref) turbulence-convection
scheme wired in with `Generalized` entrainment/detrainment, SGS mass & diffusive
fluxes, and non-hydrostatic pressure drag (matches the canonical
`diagnostic_edmfx_*` configs).

`area_fraction` defaults to 1e-5.
Override `microphysics_model` to pair EDMF with a dry or non-equilibrium scheme.
All remaining keyword arguments are forwarded to [`AtmosModel`](@ref).
"""
function diagnostic_edmf(
    ::Type{FT} = Float32;
    area_fraction = FT(1e-5),
    n_updrafts = 1,
    prognostic_tke = true,
    kwargs...,
) where {FT}
    defaults = (;
        microphysics_model = EquilibriumMicrophysics0M(),
        cloud_model = GridScaleCloud(),
        turbconv_model = DiagnosticEDMFX(;
            n_updrafts, prognostic_tke, area_fraction,
        ),
        edmfx_model = EDMFXModel(;
            entr_model = InvZEntrainment(),
            detr_model = BuoyancyVelocityDetrainment(),
            sgs_mass_flux = true,
            sgs_diffusive_flux = true,
            nh_pressure = true,
            scale_blending_method = SmoothMinimumBlending(),
        ),
    )
    return AtmosModel(; defaults..., kwargs...)
end

"""
    prognostic_edmf([FT = Float32]; area_fraction, n_updrafts, prognostic_tke, kwargs...)

Equilibrium-moist model with the [`PrognosticEDMFX`](@ref) turbulence-convection
scheme. In addition to the [`diagnostic_edmf`](@ref) EDMF settings, this also
enables prognostic updraft vertical diffusion and the relaxation filter on
negative updraft velocities (matches the canonical `prognostic_edmfx_*` configs).

All remaining keyword arguments are forwarded to [`AtmosModel`](@ref).
"""
function prognostic_edmf(
    ::Type{FT} = Float32;
    area_fraction = FT(1e-5),
    n_updrafts = 1,
    prognostic_tke = true,
    kwargs...,
) where {FT}
    defaults = (;
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
    )
    return AtmosModel(; defaults..., kwargs...)
end

"""
    prognostic_edmf_1m([FT = Float32]; kwargs...)

[`prognostic_edmf`](@ref) with 1-moment non-equilibrium microphysics and explicit
microphysics tendency timestepping (matches the canonical `prognostic_edmfx_*`
configs that use `microphysics_model: "1M"`). All keyword arguments are
forwarded to [`prognostic_edmf`](@ref) and on to [`AtmosModel`](@ref).
"""
function prognostic_edmf_1m(::Type{FT} = Float32; kwargs...) where {FT}
    defaults = (;
        microphysics_model = NonEquilibriumMicrophysics1M(),
        microphysics_tendency_timestepping = Explicit(),
    )
    return prognostic_edmf(FT; defaults..., kwargs...)
end

# ============================================================================
# Simulation presets — return an AtmosSimulation
# ============================================================================

"""
    aquaplanet([FT = Float32]; kwargs...)

Aquaplanet simulation preset: global [`SphereGrid`](@ref) with
[`equil_moist_0m`](@ref) physics (0M microphysics, prescribed zonally-symmetric
SST, idealized insolation). Uses the default [`DecayingProfile`](@ref) initial
condition from [`AtmosSimulation`](@ref).
Keyword arguments are forwarded to [`AtmosSimulation`](@ref).
"""
function aquaplanet(::Type{FT} = Float32; kwargs...) where {FT}
    defaults = (; model = equil_moist_0m())
    return AtmosSimulation{FT}(; defaults..., kwargs...)
end

"""
    baroclinic_wave([FT = Float32]; kwargs...)

Dry baroclinic-wave simulation preset: global [`SphereGrid`](@ref),
[`DryBaroclinicWave`](@ref) setup, and a dry model with
`disable_surface_flux_tendency = true`. For the moist variant, pass
`setup = Setups.MoistBaroclinicWave()` and
`model = Presets.equil_moist_0m(; disable_surface_flux_tendency = true)`.
Keyword arguments are forwarded to [`AtmosSimulation`](@ref).
"""
function baroclinic_wave(::Type{FT} = Float32; kwargs...) where {FT}
    defaults = (;
        model = dry(; disable_surface_flux_tendency = true),
        setup = Setups.DryBaroclinicWave(),
    )
    return AtmosSimulation{FT}(; defaults..., kwargs...)
end

"""
    bomex([FT = Float32]; kwargs...)

BOMEX shallow-cumulus single-column simulation preset: [`ColumnGrid`](@ref)
(60 uniform levels, z_max = 3 km), [`Setups.Bomex`](@ref) setup,
[`equil_moist_0m`](@ref) physics, `dt = 10 s`, `t_end = 6 h`.

No EDMF turbulence-convection scheme is enabled by default; pass
`model = Presets.diagnostic_edmf(FT)` or `Presets.prognostic_edmf(FT)` to add one.
Keyword arguments are forwarded to [`AtmosSimulation`](@ref).
"""
function bomex(::Type{FT} = Float32; kwargs...) where {FT}
    # Bomex setup needs thermo_params at construction time, so resolve params up front.
    # If the caller passed `params`, use theirs; otherwise build defaults.
    params = haskey(kwargs, :params) ? kwargs[:params] : ClimaAtmosParameters(FT)
    other_kwargs = Base.structdiff(values(kwargs), NamedTuple{(:params,)})
    defaults = (;
        params,
        grid = ColumnGrid(FT; z_elem = 60, z_max = 3000, z_stretch = false),
        setup = Setups.Bomex(;
            prognostic_tke = true,
            thermo_params = params.thermodynamics_params,
        ),
        model = equil_moist_0m(),
        dt = 10,
        t_end = 6 * 3600,
    )
    return AtmosSimulation{FT}(; defaults..., other_kwargs...)
end

end # module
