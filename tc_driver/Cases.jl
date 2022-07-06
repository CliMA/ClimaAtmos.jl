module Cases

import NCDatasets as NC
import OrdinaryDiffEq as ODE
import Thermodynamics as TD

import ClimaCore as CC
import ClimaCore.Operators as CCO
import ClimaCore.Geometry as CCG
import DocStringExtensions

import AtmosphericProfilesLibrary as APL

import Dierckx
import Statistics
import Random
import UnPack

import ..TurbulenceConvection as TC
import ..TurbulenceConvection.Parameters as TCP
const APS = TCP.AbstractTurbulenceConvectionParameters

using ..TurbulenceConvection: pyinterp
using ..TurbulenceConvection: Grid
using ..TurbulenceConvection: real_center_indices
using ..TurbulenceConvection: real_face_indices
using ..TurbulenceConvection: get_inversion

#=
    arr_type(x)
We're keeping this around in case we need
to move some initialized data to the GPU.
In this case, we may be able to modify this
function to do this.
=#
arr_type(x) = x

#####
##### Case types
#####

# For dispatching to inherited class
struct BaseCase end

abstract type AbstractCaseType end

""" [Soares2004](@cite) """
struct Soares <: AbstractCaseType end

""" [Nieuwstadt1993](@cite) """
struct Nieuwstadt <: AbstractCaseType end

struct Bomex <: AbstractCaseType end

""" [Tan2018](@cite) """
struct life_cycle_Tan2018 <: AbstractCaseType end

struct Rico <: AbstractCaseType end

""" [Grabowski2006](@cite) """
struct TRMM_LBA <: AbstractCaseType end

""" [Brown2002](@cite) """
struct ARM_SGP <: AbstractCaseType end

""" [Khairoutdinov2009](@cite) """
struct GATE_III <: AbstractCaseType end

""" [Stevens2005](@cite) """
struct DYCOMS_RF01 <: AbstractCaseType end

""" [Ackerman2009](@cite) """
struct DYCOMS_RF02 <: AbstractCaseType end

struct GABLS <: AbstractCaseType end

struct DryBubble <: AbstractCaseType end

struct LES_driven_SCM <: AbstractCaseType end

#####
##### Radiation and forcing types
#####

struct ForcingNone end
struct ForcingStandard end
struct ForcingDYCOMS_RF01 end
struct ForcingLES end

struct RadiationNone end
struct RadiationDYCOMS_RF01 end
struct RadiationLES end
struct RadiationTRMM_LBA end

"""
    ForcingBase

LES-driven forcing

$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct ForcingBase{T, FT}
    "Coriolis parameter"
    coriolis_param::FT = 0
    "Wind relaxation timescale"
    wind_nudge_τᵣ::FT = 0.0
    "Scalar relaxation lower z"
    scalar_nudge_zᵢ::FT = 0.0
    "Scalar relaxation upper z"
    scalar_nudge_zᵣ::FT = 0.0
    "Scalar maximum relaxation timescale"
    scalar_nudge_τᵣ::FT = 0.0
    "Large-scale divergence (same as in RadiationBase)"
    divergence::FT = 0
end

force_type(::ForcingBase{T}) where {T} = T

Base.@kwdef struct RadiationBase{T, FT}
    "Large-scale divergence (same as in ForcingBase)"
    divergence::FT = 0
    alpha_z::FT = 0
    kappa::FT = 0
    F0::FT = 0
    F1::FT = 0
end

rad_type(::RadiationBase{T}) where {T} = T

Base.@kwdef struct LESData
    "Start time index of LES"
    imin::Int = 0
    "End time index of LES"
    imax::Int = 0
    "Path to LES stats file used to drive SCM"
    les_filename::String = nothing
    "Drive SCM with LES data from t = [end - t_interval_from_end_s, end]"
    t_interval_from_end_s::Float64 = 6 * 3600.0
    "Length of time to average over for SCM initialization"
    initial_condition_averaging_window_s::Float64 = 3600.0
end

#####
##### Radiation and forcing functions
#####

include("Radiation.jl")
include("Forcing.jl")

#####
##### Case methods
#####

get_case(namelist::Dict) = get_case(namelist["meta"]["casename"])
get_case(casename::String) = get_case(Val(Symbol(casename)))
get_case(::Val{:Soares}) = Soares()
get_case(::Val{:Nieuwstadt}) = Nieuwstadt()
get_case(::Val{:Bomex}) = Bomex()
get_case(::Val{:life_cycle_Tan2018}) = life_cycle_Tan2018()
get_case(::Val{:Rico}) = Rico()
get_case(::Val{:TRMM_LBA}) = TRMM_LBA()
get_case(::Val{:ARM_SGP}) = ARM_SGP()
get_case(::Val{:GATE_III}) = GATE_III()
get_case(::Val{:DYCOMS_RF01}) = DYCOMS_RF01()
get_case(::Val{:DYCOMS_RF02}) = DYCOMS_RF02()
get_case(::Val{:GABLS}) = GABLS()
get_case(::Val{:DryBubble}) = DryBubble()
get_case(::Val{:LES_driven_SCM}) = LES_driven_SCM()

get_case_name(case_type::AbstractCaseType) = string(case_type)

#####
##### Case configurations
#####

get_forcing_type(::AbstractCaseType) = ForcingStandard # default
get_forcing_type(::Soares) = ForcingNone
get_forcing_type(::Nieuwstadt) = ForcingNone
get_forcing_type(::DYCOMS_RF01) = ForcingDYCOMS_RF01
get_forcing_type(::DYCOMS_RF02) = ForcingDYCOMS_RF01
get_forcing_type(::DryBubble) = ForcingNone
get_forcing_type(::LES_driven_SCM) = ForcingLES
get_forcing_type(::TRMM_LBA) = ForcingNone

get_radiation_type(::AbstractCaseType) = RadiationNone # default
get_radiation_type(::DYCOMS_RF01) = RadiationDYCOMS_RF01
get_radiation_type(::DYCOMS_RF02) = RadiationDYCOMS_RF01
get_radiation_type(::LES_driven_SCM) = RadiationLES
get_radiation_type(::TRMM_LBA) = RadiationTRMM_LBA

large_scale_divergence(::Union{DYCOMS_RF01, DYCOMS_RF02}) = 3.75e-6

RadiationBase(case::AbstractCaseType, FT) = RadiationBase{Cases.get_radiation_type(case), FT}()

forcing_kwargs(::AbstractCaseType, namelist) = (; coriolis_param = namelist["forcing"]["coriolis"])
forcing_kwargs(case::DYCOMS_RF01, namelist) = (; divergence = large_scale_divergence(case))
forcing_kwargs(case::DYCOMS_RF02, namelist) = (; divergence = large_scale_divergence(case))
les_data_kwarg(::AbstractCaseType, namelist) = ()

ForcingBase(case::AbstractCaseType, FT; kwargs...) = ForcingBase{get_forcing_type(case), FT}(; kwargs...)

#####
##### Default case behavior:
#####

initialize_radiation(::AbstractCaseType, radiation, grid, state, param_set) = nothing

update_forcing(::AbstractCaseType, grid, state, t::Real, param_set) = nothing
initialize_forcing(::AbstractCaseType, forcing, grid::Grid, state, param_set) = initialize(forcing, grid, state)

#####
##### Soares
#####

function surface_ref_state(::Soares, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = eltype(param_set)
    Pg::FT = 1000.0 * 100.0
    qtg::FT = 5.0e-3
    Tg::FT = 300.0
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end
function initialize_profiles(::Soares, grid::Grid, param_set, state; kwargs...)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ
    FT = TC.float_type(state)
    prof_q_tot = APL.Soares_q_tot(FT)
    prof_θ_liq_ice = APL.Soares_θ_liq_ice(FT)
    prof_u = APL.Soares_u(FT)
    prof_tke = APL.Soares_tke(FT)

    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.q_tot[k] = prof_q_tot(z)
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)
        aux_gm.tke[k] = prof_tke(z)
    end
end

function surface_params(case::Soares, surf_ref_state, param_set; Ri_bulk_crit)
    thermo_params = TCP.thermodynamics_params(param_set)
    p_f_surf = TD.air_pressure(thermo_params, surf_ref_state)
    ρ_f_surf = TD.air_density(thermo_params, surf_ref_state)
    FT = eltype(p_f_surf)
    zrough::FT = 0.16 #1.0e-4 0.16 is the value specified in the Nieuwstadt paper.
    Tsurface::FT = 300.0
    qsurface::FT = 5.0e-3
    θ_flux::FT = 6.0e-2
    qt_flux::FT = 2.5e-5
    ts = TD.PhaseEquil_pTq(thermo_params, p_f_surf, Tsurface, qsurface)
    lhf = qt_flux * ρ_f_surf * TD.latent_heat_vapor(thermo_params, ts)
    shf = θ_flux * TD.cp_m(thermo_params, ts) * ρ_f_surf
    ustar::FT = 0.28 # just to initilize grid mean covariances
    kwargs = (; zrough, Tsurface, qsurface, shf, lhf, ustar, Ri_bulk_crit)
    return TC.FixedSurfaceFlux(FT, TC.VariableFrictionVelocity; kwargs...)
end

#####
##### Nieuwstadt
#####

function surface_ref_state(::Nieuwstadt, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = eltype(param_set)
    Pg::FT = 1000.0 * 100.0
    Tg::FT = 300.0
    qtg::FT = 0.0
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end
function initialize_profiles(::Nieuwstadt, grid::Grid, param_set, state; kwargs...)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ

    FT = TC.float_type(state)
    prof_θ_liq_ice = APL.Nieuwstadt_θ_liq_ice(FT)
    prof_u = APL.Nieuwstadt_u(FT)
    prof_tke = APL.Nieuwstadt_tke(FT)

    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)
        aux_gm.tke[k] = prof_tke(z)
    end
end

function surface_params(case::Nieuwstadt, surf_ref_state, param_set; Ri_bulk_crit)
    thermo_params = TCP.thermodynamics_params(param_set)
    p_f_surf = TD.air_pressure(thermo_params, surf_ref_state)
    ρ_f_surf = TD.air_density(thermo_params, surf_ref_state)
    FT = eltype(p_f_surf)
    zrough::FT = 0.16 #1.0e-4 0.16 is the value specified in the Nieuwstadt paper.
    Tsurface::FT = 300.0
    qsurface::FT = 0.0
    θ_flux::FT = 6.0e-2
    lhf::FT = 0.0 # It would be 0.0 if we follow Nieuwstadt.
    ts = TD.PhaseEquil_pTq(thermo_params, p_f_surf, Tsurface, qsurface)
    shf = θ_flux * TD.cp_m(thermo_params, ts) * ρ_f_surf
    ustar::FT = 0.28 # just to initilize grid mean covariances
    kwargs = (; zrough, Tsurface, qsurface, shf, lhf, ustar, Ri_bulk_crit)
    return TC.FixedSurfaceFlux(FT, TC.VariableFrictionVelocity; kwargs...)
end

#####
##### Bomex
#####

function surface_ref_state(::Bomex, param_set::APS, namelist)
    FT = eltype(param_set)
    thermo_params = TCP.thermodynamics_params(param_set)
    Pg::FT = 1.015e5 #Pressure at ground
    Tg::FT = 300.4 #Temperature at ground
    qtg::FT = 0.02245#Total water mixing ratio at surface
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end

function initialize_profiles(::Bomex, grid::Grid, param_set, state; kwargs...)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ

    FT = TC.float_type(state)
    prof_q_tot = APL.Bomex_q_tot(FT)
    prof_θ_liq_ice = APL.Bomex_θ_liq_ice(FT)
    prof_u = APL.Bomex_u(FT)
    prof_tke = APL.Bomex_tke(FT)
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)
        aux_gm.q_tot[k] = prof_q_tot(z)
        aux_gm.tke[k] = prof_tke(z)
    end
end

function surface_params(case::Bomex, surf_ref_state, param_set; Ri_bulk_crit)
    thermo_params = TCP.thermodynamics_params(param_set)
    p_f_surf = TD.air_pressure(thermo_params, surf_ref_state)
    ρ_f_surf = TD.air_density(thermo_params, surf_ref_state)
    FT = eltype(p_f_surf)
    zrough::FT = 1.0e-4
    qsurface::FT = 22.45e-3 # kg/kg
    θ_surface::FT = 299.1
    θ_flux::FT = 8.0e-3
    qt_flux::FT = 5.2e-5
    ts = TD.PhaseEquil_pθq(thermo_params, p_f_surf, θ_surface, qsurface)
    Tsurface = TD.air_temperature(thermo_params, ts)
    lhf = qt_flux * ρ_f_surf * TD.latent_heat_vapor(thermo_params, ts)
    shf = θ_flux * TD.cp_m(thermo_params, ts) * ρ_f_surf
    ustar::FT = 0.28 # m/s
    kwargs = (; zrough, Tsurface, qsurface, shf, lhf, ustar, Ri_bulk_crit)
    return TC.FixedSurfaceFlux(FT, TC.FixedFrictionVelocity; kwargs...)
end

function initialize_forcing(::Bomex, forcing, grid::Grid, state, param_set)
    thermo_params = TCP.thermodynamics_params(param_set)
    initialize(forcing, grid, state)
    prog_gm = TC.center_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    ts_gm = aux_gm.ts
    p_c = aux_gm.p

    FT = TC.float_type(state)
    prof_ug = APL.Bomex_geostrophic_u(FT)
    prof_dTdt = APL.Bomex_dTdt(FT)
    prof_dqtdt = APL.Bomex_dqtdt(FT)
    prof_subsidence = APL.Bomex_subsidence(FT)

    z = CC.Fields.coordinate_field(axes(aux_gm.uₕ_g)).z
    @. aux_gm.uₕ_g = CCG.Covariant12Vector(CCG.UVVector(prof_ug(z), FT(0)))

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        # Geostrophic velocity profiles. vg = 0
        Π = TD.exner(thermo_params, ts_gm[k])
        # Set large-scale cooling
        aux_gm.dTdt_hadv[k] = prof_dTdt(Π, z)
        # Set large-scale drying
        aux_gm.dqtdt_hadv[k] = prof_dqtdt(z)
        #Set large scale subsidence
        aux_gm.subsidence[k] = prof_subsidence(z)
    end
    return nothing
end

#####
##### life_cycle_Tan2018
#####

function surface_ref_state(::life_cycle_Tan2018, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = eltype(param_set)
    Pg::FT = 1.015e5  #Pressure at ground
    Tg::FT = 300.4  #Temperature at ground
    qtg::FT = 0.02245   #Total water mixing ratio at surface
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end
function initialize_profiles(::life_cycle_Tan2018, grid::Grid, param_set, state; kwargs...)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ

    FT = TC.float_type(state)
    prof_q_tot = APL.LifeCycleTan2018_q_tot(FT)
    prof_θ_liq_ice = APL.LifeCycleTan2018_θ_liq_ice(FT)
    prof_u = APL.LifeCycleTan2018_u(FT)
    prof_tke = APL.LifeCycleTan2018_tke(FT)

    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)
        aux_gm.q_tot[k] = prof_q_tot(z)
        aux_gm.tke[k] = prof_tke(z)
    end
end

function surface_params(case::life_cycle_Tan2018, surf_ref_state, param_set; Ri_bulk_crit)
    thermo_params = TCP.thermodynamics_params(param_set)
    p_f_surf = TD.air_pressure(thermo_params, surf_ref_state)
    ρ_f_surf = TD.air_density(thermo_params, surf_ref_state)
    FT = eltype(p_f_surf)
    zrough::FT = 1.0e-4 # not actually used, but initialized to reasonable value
    qsurface::FT = 22.45e-3 # kg/kg
    θ_surface::FT = 299.1
    θ_flux::FT = 8.0e-3
    qt_flux::FT = 5.2e-5
    ts = TD.PhaseEquil_pθq(thermo_params, p_f_surf, θ_surface, qsurface)
    Tsurface = TD.air_temperature(thermo_params, ts)
    lhf0 = qt_flux * ρ_f_surf * TD.latent_heat_vapor(thermo_params, ts)
    shf0 = θ_flux * TD.cp_m(thermo_params, ts) * ρ_f_surf

    weight_factor(t) = FT(0.01) + FT(0.99) * (cos(2 * FT(π) * t / 3600) + 1) / 2
    weight::FT = 1.0
    lhf = t -> lhf0 * (weight * weight_factor(t))
    shf = t -> shf0 * (weight * weight_factor(t))

    ustar::FT = 0.28 # m/s
    kwargs = (; zrough, Tsurface, qsurface, shf, lhf, ustar, Ri_bulk_crit)
    return TC.FixedSurfaceFlux(FT, TC.FixedFrictionVelocity; kwargs...)
end

function initialize_forcing(::life_cycle_Tan2018, forcing, grid::Grid, state, param_set)
    thermo_params = TCP.thermodynamics_params(param_set)
    initialize(forcing, grid, state)
    prog_gm = TC.center_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    p_c = aux_gm.p
    ts_gm = aux_gm.ts

    FT = TC.float_type(state)
    prof_ug = APL.LifeCycleTan2018_geostrophic_u(FT)
    prof_dTdt = APL.LifeCycleTan2018_dTdt(FT)
    prof_dqtdt = APL.LifeCycleTan2018_dqtdt(FT)
    prof_subsidence = APL.LifeCycleTan2018_subsidence(FT)

    aux_gm_uₕ_g = TC.grid_mean_uₕ_g(state)
    TC.set_z!(aux_gm_uₕ_g, prof_ug, x -> FT(0))

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        # Geostrophic velocity profiles. vg = 0
        Π = TD.exner(thermo_params, ts_gm[k])
        # Set large-scale cooling
        aux_gm.dTdt_hadv[k] = prof_dTdt(Π, z)
        # Set large-scale drying
        aux_gm.dqtdt_hadv[k] = prof_dqtdt(z)
        #Set large scale subsidence
        aux_gm.subsidence[k] = prof_subsidence(z)
    end
    return nothing
end

#####
##### Rico
#####

function surface_ref_state(::Rico, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    molmass_ratio = TCP.molmass_ratio(param_set)
    FT = eltype(param_set)
    Pg::FT = 1.0154e5  #Pressure at ground
    Tg::FT = 299.8  #Temperature at ground
    pvg = TD.saturation_vapor_pressure(thermo_params, Tg, TD.Liquid())
    qtg = (1 / molmass_ratio) * pvg / (Pg - pvg)   #Total water mixing ratio at surface
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end
function initialize_profiles(::Rico, grid::Grid, param_set, state; kwargs...)
    thermo_params = TCP.thermodynamics_params(param_set)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    aux_tc = TC.center_aux_turbconv(state)
    p = aux_gm.p
    ρ_c = prog_gm.ρ

    FT = TC.float_type(state)
    prof_u = APL.Rico_u(FT)
    prof_v = APL.Rico_v(FT)
    prof_q_tot = APL.Rico_q_tot(FT)
    prof_θ_liq_ice = APL.Rico_θ_liq_ice(FT)

    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)
        aux_gm.q_tot[k] = prof_q_tot(z)
    end

    # Need to get θ_virt
    @inbounds for k in real_center_indices(grid)
        # Thermo state field cache is not yet
        # defined, so we can't use it yet.
        ts = TD.PhaseEquil_pθq(thermo_params, p[k], aux_gm.θ_liq_ice[k], aux_gm.q_tot[k])
        aux_gm.θ_virt[k] = TD.virtual_pottemp(thermo_params, ts)
    end
    zi = FT(0.6) * get_inversion(grid, state, param_set, FT(0.2))

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.tke[k] = if z <= zi
            1 - z / zi
        else
            FT(0)
        end
    end
end

function surface_params(case::Rico, surf_ref_state, param_set; kwargs...)
    thermo_params = TCP.thermodynamics_params(param_set)
    p_f_surf = TD.air_pressure(thermo_params, surf_ref_state)
    FT = eltype(p_f_surf)

    zrough::FT = 0.00015
    cm0::FT = 0.001229
    ch0::FT = 0.001094
    cq0::FT = 0.001133
    # Adjust for non-IC grid spacing
    grid_adjust(zc_surf) = (log(20 / zrough) / log(zc_surf / zrough))^2
    cm = zc_surf -> cm0 * grid_adjust(zc_surf)
    ch = zc_surf -> ch0 * grid_adjust(zc_surf)
    cq = zc_surf -> cq0 * grid_adjust(zc_surf) # TODO: not yet used..
    Tsurface::FT = 299.8

    # For Rico we provide values of transfer coefficients
    ts = TD.PhaseEquil_pTq(thermo_params, p_f_surf, Tsurface, FT(0)) # TODO: is this correct?
    qsurface = TD.q_vap_saturation(thermo_params, ts)
    kwargs = (; zrough, Tsurface, qsurface, cm, ch, kwargs...)
    return TC.FixedSurfaceCoeffs(FT; kwargs...)
end

function initialize_forcing(::Rico, forcing, grid::Grid, state, param_set)
    thermo_params = TCP.thermodynamics_params(param_set)
    initialize(forcing, grid, state)
    prog_gm = TC.center_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    ts_gm = aux_gm.ts
    p_c = aux_gm.p

    FT = TC.float_type(state)
    prof_ug = APL.Rico_geostrophic_ug(FT)
    prof_vg = APL.Rico_geostrophic_vg(FT)
    prof_dTdt = APL.Rico_dTdt(FT)
    prof_dqtdt = APL.Rico_dqtdt(FT)
    prof_subsidence = APL.Rico_subsidence(FT)

    aux_gm_uₕ_g = TC.grid_mean_uₕ_g(state)
    TC.set_z!(aux_gm_uₕ_g, prof_ug, prof_vg)

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        Π = TD.exner(thermo_params, ts_gm[k])
        aux_gm.dTdt_hadv[k] = prof_dTdt(Π, z) # Set large-scale cooling
        aux_gm.dqtdt_hadv[k] = prof_dqtdt(z) # Set large-scale moistening
        aux_gm.subsidence[k] = prof_subsidence(z) #Set large scale subsidence
    end
    return nothing
end

#####
##### TRMM_LBA
#####

function surface_ref_state(::TRMM_LBA, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    molmass_ratio = TCP.molmass_ratio(param_set)
    FT = eltype(param_set)
    Pg::FT = 991.3 * 100  #Pressure at ground
    Tg::FT = 296.85   # surface values for reference state (RS) which outputs p, ρ
    pvg = TD.saturation_vapor_pressure(thermo_params, Tg, TD.Liquid())
    qtg = (1 / molmass_ratio) * pvg / (Pg - pvg) #Total water mixing ratio at surface
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end
function initialize_profiles(::TRMM_LBA, grid::Grid, param_set, state; kwargs...)
    thermo_params = TCP.thermodynamics_params(param_set)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ
    p = aux_gm.p

    FT = TC.float_type(state)
    # Get profiles from AtmosphericProfilesLibrary.jl
    prof_p = APL.TRMM_LBA_p(FT)
    prof_T = APL.TRMM_LBA_T(FT)
    prof_RH = APL.TRMM_LBA_RH(FT)
    prof_u = APL.TRMM_LBA_u(FT)
    prof_v = APL.TRMM_LBA_v(FT)
    prof_tke = APL.TRMM_LBA_tke(FT)

    zc = grid.zc.z
    molmass_ratio = TCP.molmass_ratio(param_set)
    prog_gm = TC.center_prog_grid_mean(state)

    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)

    aux_gm.T .= prof_T.(zc)

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        pv_star = TD.saturation_vapor_pressure(thermo_params, aux_gm.T[k], TD.Liquid())
        # eq. 37 in pressel et al and the def of RH
        RH = prof_RH(z)
        denom = (prof_p(z) - pv_star + (1 / molmass_ratio) * pv_star * RH / 100)
        qv_star = pv_star * (1 / molmass_ratio) / denom
        aux_gm.q_tot[k] = qv_star * RH / 100
        phase_part = TD.PhasePartition(aux_gm.q_tot[k], FT(0), FT(0)) # initial state is not saturated
        aux_gm.θ_liq_ice[k] = TD.liquid_ice_pottemp_given_pressure(thermo_params, aux_gm.T[k], p[k], phase_part)
        aux_gm.tke[k] = prof_tke(z)
    end
end

function surface_params(case::TRMM_LBA, surf_ref_state, param_set; Ri_bulk_crit)
    thermo_params = TCP.thermodynamics_params(param_set)
    p_f_surf = TD.air_pressure(thermo_params, surf_ref_state)
    FT = eltype(p_f_surf)
    # zrough = 1.0e-4 # not actually used, but initialized to reasonable value
    qsurface::FT = 22.45e-3 # kg/kg
    θ_surface::FT = (273.15 + 23)
    ts = TD.PhaseEquil_pθq(thermo_params, p_f_surf, θ_surface, qsurface)
    Tsurface = TD.air_temperature(thermo_params, ts)
    ustar::FT = 0.28 # this is taken from Bomex -- better option is to approximate from LES tke above the surface
    lhf = t -> 554 * max(0, cos(FT(π) / 2 * ((FT(5.25) * 3600 - t) / FT(5.25) / 3600)))^FT(1.3)
    shf = t -> 270 * max(0, cos(FT(π) / 2 * ((FT(5.25) * 3600 - t) / FT(5.25) / 3600)))^FT(1.5)
    kwargs = (; Tsurface, qsurface, shf, lhf, ustar, Ri_bulk_crit, zero_uv_fluxes = true)
    return TC.FixedSurfaceFlux(FT, TC.FixedFrictionVelocity; kwargs...)
end

RadiationBase(case::TRMM_LBA, FT) = RadiationBase{Cases.get_radiation_type(case), FT}()

initialize_radiation(::TRMM_LBA, radiation, grid::Grid, state, param_set) = initialize(radiation, grid, state)

#####
##### ARM_SGP
#####

function surface_ref_state(::ARM_SGP, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = eltype(param_set)
    Pg::FT = 970.0 * 100 #Pressure at ground
    Tg::FT = 299.0   # surface values for reference state (RS) which outputs  p, ρ
    qtg::FT = 15.2 / 1000 #Total water mixing ratio at surface
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end

function initialize_profiles(::ARM_SGP, grid::Grid, param_set, state; kwargs...)
    thermo_params = TCP.thermodynamics_params(param_set)
    # ARM_SGP inputs
    prog_gm = TC.center_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    ρ_c = prog_gm.ρ
    p = aux_gm.p

    FT = TC.float_type(state)
    prof_u = APL.ARM_SGP_u(FT)
    prof_q_tot = APL.ARM_SGP_q_tot(FT)
    prof_θ_liq_ice = APL.ARM_SGP_θ_liq_ice(FT)
    prof_tke = APL.ARM_SGP_tke(FT)

    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        # TODO figure out how to use ts here
        phase_part = TD.PhasePartition(aux_gm.q_tot[k], aux_gm.q_liq[k], FT(0))
        Π = TD.exner_given_pressure(thermo_params, p[k], phase_part)
        aux_gm.q_tot[k] = prof_q_tot(z)
        aux_gm.T[k] = prof_θ_liq_ice(z) * Π
        aux_gm.θ_liq_ice[k] = TD.liquid_ice_pottemp_given_pressure(thermo_params, aux_gm.T[k], p[k], phase_part)
        aux_gm.tke[k] = prof_tke(z)
    end
end

function surface_params(case::ARM_SGP, surf_ref_state, param_set; Ri_bulk_crit)
    thermo_params = TCP.thermodynamics_params(param_set)
    p_f_surf = TD.air_pressure(thermo_params, surf_ref_state)
    FT = eltype(p_f_surf)
    qsurface::FT = 15.2e-3 # kg/kg
    θ_surface::FT = 299.0
    ts = TD.PhaseEquil_pθq(thermo_params, p_f_surf, θ_surface, qsurface)
    Tsurface = TD.air_temperature(thermo_params, ts)
    ustar::FT = 0.28 # this is taken from Bomex -- better option is to approximate from LES tke above the surface

    t_Sur_in = arr_type(FT[0.0, 4.0, 6.5, 7.5, 10.0, 12.5, 14.5]) .* 3600 #LES time is in sec
    SH = arr_type(FT[-30.0, 90.0, 140.0, 140.0, 100.0, -10, -10]) # W/m^2
    LH = arr_type(FT[5.0, 250.0, 450.0, 500.0, 420.0, 180.0, 0.0]) # W/m^2
    shf = Dierckx.Spline1D(t_Sur_in, SH; k = 1)
    lhf = Dierckx.Spline1D(t_Sur_in, LH; k = 1)

    kwargs = (; Tsurface, qsurface, shf, lhf, ustar, Ri_bulk_crit, zero_uv_fluxes = true)
    return TC.FixedSurfaceFlux(FT, TC.FixedFrictionVelocity; kwargs...)
end

function initialize_forcing(::ARM_SGP, forcing, grid::Grid, state, param_set)
    FT = TC.float_type(state)
    aux_gm_uₕ_g = TC.grid_mean_uₕ_g(state)
    TC.set_z!(aux_gm_uₕ_g, FT(10), FT(0))
    return nothing
end

function update_forcing(::ARM_SGP, grid, state, t::Real, param_set)
    thermo_params = TCP.thermodynamics_params(param_set)
    aux_gm = TC.center_aux_grid_mean(state)
    ts_gm = TC.center_aux_grid_mean(state).ts
    prog_gm = TC.center_prog_grid_mean(state)
    p_c = prog_gm.ρ
    FT = TC.float_type(state)
    @inbounds for k in real_center_indices(grid)
        Π = TD.exner(thermo_params, ts_gm[k])
        z = grid.zc[k].z
        aux_gm.dTdt_hadv[k] = APL.ARM_SGP_dTdt(FT)(t, z)
        aux_gm.dqtdt_hadv[k] = APL.ARM_SGP_dqtdt(FT)(Π, t, z)

    end
end

#####
##### GATE_III
#####

function surface_ref_state(::GATE_III, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = eltype(param_set)
    Pg::FT = 1013.0 * 100  #Pressure at ground
    Tg::FT = 299.184   # surface values for reference state (RS) which outputs p, ρ
    qtg::FT = 16.5 / 1000 #Total water mixing ratio at surface
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end

function initialize_profiles(::GATE_III, grid::Grid, param_set, state; kwargs...)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = TC.float_type(state)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)

    prof_u = APL.GATE_III_u(FT)
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)

    p = aux_gm.p
    ρ_c = prog_gm.ρ
    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.q_tot[k] = APL.GATE_III_q_tot(FT)(z)
        aux_gm.T[k] = APL.GATE_III_T(FT)(z)
        ts = TD.PhaseEquil_pTq(thermo_params, p[k], aux_gm.T[k], aux_gm.q_tot[k])
        aux_gm.θ_liq_ice[k] = TD.liquid_ice_pottemp(thermo_params, ts)
        aux_gm.tke[k] = APL.GATE_III_tke(FT)(z)
    end
end

function surface_params(case::GATE_III, surf_ref_state, param_set; kwargs...)
    thermo_params = TCP.thermodynamics_params(param_set)
    p_f_surf = TD.air_pressure(thermo_params, surf_ref_state)
    FT = eltype(p_f_surf)

    qsurface::FT = 16.5 / 1000.0 # kg/kg
    cm = zc_surf -> FT(0.0012)
    ch = zc_surf -> FT(0.0034337)
    cq = zc_surf -> FT(0.0034337)
    Tsurface::FT = 299.184

    # For GATE_III we provide values of transfer coefficients
    ts = TD.PhaseEquil_pθq(thermo_params, p_f_surf, Tsurface, qsurface)
    qsurface = TD.q_vap_saturation(thermo_params, ts)
    kwargs = (; Tsurface, qsurface, cm, ch, kwargs...)
    return TC.FixedSurfaceCoeffs(FT; kwargs...)
end

function initialize_forcing(::GATE_III, forcing, grid::Grid, state, param_set)
    FT = TC.float_type(state)
    aux_gm = TC.center_aux_grid_mean(state)
    for k in TC.real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.dqtdt_hadv[k] = APL.GATE_III_dqtdt(FT)(z)
        aux_gm.dTdt_hadv[k] = APL.GATE_III_dTdt(FT)(z)
    end
end

#####
##### DYCOMS_RF01
#####

function surface_ref_state(::DYCOMS_RF01, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = eltype(param_set)
    Pg::FT = 1017.8 * 100.0
    qtg::FT = 9.0 / 1000.0
    θ_surf::FT = 289.0
    ts = TD.PhaseEquil_pθq(thermo_params, Pg, θ_surf, qtg)
    Tg = TD.air_temperature(thermo_params, ts)
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end

function initialize_profiles(::DYCOMS_RF01, grid::Grid, param_set, state; kwargs...)
    FT = TC.float_type(state)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)

    prof_u = APL.Dycoms_RF01_u0(FT)
    prof_v = APL.Dycoms_RF01_v0(FT)
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)

    ρ_c = prog_gm.ρ
    p = aux_gm.p
    @inbounds for k in real_center_indices(grid)
        # thetal profile as defined in DYCOMS
        z = grid.zc[k].z
        aux_gm.q_tot[k] = APL.Dycoms_RF01_q_tot(FT)(z)
        aux_gm.θ_liq_ice[k] = APL.Dycoms_RF01_θ_liq_ice(FT)(z)

        # velocity profile (geostrophic)
        aux_gm.tke[k] = APL.Dycoms_RF01_tke(FT)(z)
    end
end

function surface_params(case::DYCOMS_RF01, surf_ref_state, param_set; Ri_bulk_crit)
    FT = eltype(surf_ref_state)
    zrough::FT = 1.0e-4
    ustar::FT = 0.28 # just to initialize grid mean covariances
    shf::FT = 15.0 # sensible heat flux
    lhf::FT = 115.0 # latent heat flux
    Tsurface::FT = 292.5    # K      # i.e. the SST from DYCOMS setup
    qsurface::FT = 13.84e-3 # kg/kg  # TODO - taken from Pycles, maybe it would be better to calculate the q_star(sst) for TurbulenceConvection?
    #density_surface  = 1.22     # kg/m^3

    kwargs = (; zrough, Tsurface, qsurface, shf, lhf, ustar, Ri_bulk_crit)
    return TC.FixedSurfaceFlux(FT, TC.VariableFrictionVelocity; kwargs...)
end

function initialize_forcing(::DYCOMS_RF01, forcing, grid::Grid, state, param_set)
    aux_gm = TC.center_aux_grid_mean(state)
    FT = TC.float_type(state)

    # geostrophic velocity profiles
    aux_gm_uₕ_g = TC.grid_mean_uₕ_g(state)
    TC.set_z!(aux_gm_uₕ_g, FT(7), FT(-5.5))

    # large scale subsidence
    divergence = forcing.divergence
    @inbounds for k in real_center_indices(grid)
        aux_gm.subsidence[k] = -grid.zc[k].z * divergence
    end

    # no large-scale drying
    parent(aux_gm.dqtdt_hadv) .= 0 #kg/(kg * s)
end

function RadiationBase(case::DYCOMS_RF01, FT)
    return RadiationBase{Cases.get_radiation_type(case), FT}(;
        divergence = large_scale_divergence(case),
        alpha_z = 1.0,
        kappa = 85.0,
        F0 = 70.0,
        F1 = 22.0,
    )
end

function initialize_radiation(::DYCOMS_RF01, radiation, grid::Grid, state, param_set)
    aux_gm = TC.center_aux_grid_mean(state)

    # no large-scale drying
    parent(aux_gm.dqtdt_rad) .= 0 #kg/(kg * s)

    # Radiation based on eq. 3 in Stevens et. al., (2005)
    # cloud-top cooling + cloud-base warming + cooling in free troposphere
    update_radiation(radiation, grid, state, 0, param_set)
end

#####
##### DYCOMS_RF02
#####

function surface_ref_state(::DYCOMS_RF02, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = eltype(param_set)
    Pg::FT = 1017.8 * 100.0
    qtg::FT = 9.0 / 1000.0
    θ_surf::FT = 288.3
    ts = TD.PhaseEquil_pθq(thermo_params, Pg, θ_surf, qtg)
    Tg = TD.air_temperature(thermo_params, ts)
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end

function initialize_profiles(::DYCOMS_RF02, grid::Grid, param_set, state; kwargs...)
    FT = TC.float_type(state)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)

    prof_u = APL.Dycoms_RF02_u(FT)
    prof_v = APL.Dycoms_RF02_v(FT)
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)

    p = aux_gm.p
    ρ_c = prog_gm.ρ
    @inbounds for k in real_center_indices(grid)
        # θ_liq_ice profile as defined in DYCOM RF02
        z = grid.zc[k].z
        aux_gm.q_tot[k] = APL.Dycoms_RF02_q_tot(FT)(z)
        aux_gm.θ_liq_ice[k] = APL.Dycoms_RF02_θ_liq_ice(FT)(z)

        # velocity profile
        aux_gm.tke[k] = APL.Dycoms_RF02_tke(FT)(z)
    end
end

function surface_params(case::DYCOMS_RF02, surf_ref_state, param_set; Ri_bulk_crit)
    FT = eltype(surf_ref_state)
    zrough::FT = 1.0e-4  #TODO - not needed?
    ustar::FT = 0.25
    shf::FT = 16.0 # sensible heat flux
    lhf::FT = 93.0 # latent heat flux
    Tsurface::FT = 292.5    # K      # i.e. the SST from DYCOMS setup
    qsurface::FT = 13.84e-3 # kg/kg  # TODO - taken from Pycles, maybe it would be better to calculate the q_star(sst) for TurbulenceConvection?

    kwargs = (; zrough, Tsurface, qsurface, shf, lhf, ustar, Ri_bulk_crit)
    return TC.FixedSurfaceFlux(FT, TC.FixedFrictionVelocity; kwargs...)
end

function initialize_forcing(::DYCOMS_RF02, forcing, grid::Grid, state, param_set)
    FT = TC.float_type(state)
    # the same as in DYCOMS_RF01
    aux_gm = TC.center_aux_grid_mean(state)

    # geostrophic velocity profiles
    aux_gm_uₕ_g = TC.grid_mean_uₕ_g(state)
    TC.set_z!(aux_gm_uₕ_g, FT(5), FT(-5.5))

    # large scale subsidence
    divergence = forcing.divergence
    @inbounds for k in real_center_indices(grid)
        aux_gm.subsidence[k] = -grid.zc[k].z * divergence
    end

    # no large-scale drying
    parent(aux_gm.dqtdt_hadv) .= 0 #kg/(kg * s)
end

function RadiationBase(case::DYCOMS_RF02, FT)
    return RadiationBase{Cases.get_radiation_type(case), FT}(;
        divergence = large_scale_divergence(case),
        alpha_z = 1.0,
        kappa = 85.0,
        F0 = 70.0,
        F1 = 22.0,
    )
end

function initialize_radiation(::DYCOMS_RF02, radiation, grid::Grid, state, param_set)
    # the same as in DYCOMS_RF01
    aux_gm = TC.center_aux_grid_mean(state)

    # no large-scale drying
    parent(aux_gm.dqtdt_rad) .= 0 #kg/(kg * s)

    # Radiation based on eq. 3 in Stevens et. al., (2005)
    # cloud-top cooling + cloud-base warming + cooling in free troposphere
    update_radiation(radiation, grid, state, 0, param_set)
end

#####
##### GABLS
#####

function surface_ref_state(::GABLS, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = eltype(param_set)
    Pg::FT = 1.0e5  #Pressure at ground,
    Tg::FT = 265.0  #Temperature at ground,
    qtg::FT = 0.0
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end
function initialize_profiles(::GABLS, grid::Grid, param_set, state; kwargs...)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ
    FT = TC.float_type(state)

    prof_u = APL.GABLS_u(FT)
    prof_v = APL.GABLS_v(FT)
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        #Set wind velocity profile
        aux_gm.θ_liq_ice[k] = APL.GABLS_θ_liq_ice(FT)(z)
        aux_gm.q_tot[k] = APL.GABLS_q_tot(FT)(z)
        aux_gm.tke[k] = APL.GABLS_tke(FT)(z)
        aux_gm.Hvar[k] = aux_gm.tke[k]
    end
end

function surface_params(case::GABLS, surf_ref_state, param_set; kwargs...)
    FT = eltype(surf_ref_state)
    Tsurface = t -> 265 - (FT(0.25) / 3600) * t
    qsurface::FT = 0.0
    zrough::FT = 0.1

    kwargs = (; Tsurface, qsurface, zrough, kwargs...)
    return TC.MoninObukhovSurface(FT; kwargs...)
end

function initialize_forcing(::GABLS, forcing, grid::Grid, state, param_set)
    FT = TC.float_type(state)
    initialize(forcing, grid, state)
    aux_gm = TC.center_aux_grid_mean(state)

    prof_ug = APL.GABLS_geostrophic_ug(FT)
    prof_vg = APL.GABLS_geostrophic_vg(FT)

    # Geostrophic velocity profiles.
    aux_gm_uₕ_g = TC.grid_mean_uₕ_g(state)
    TC.set_z!(aux_gm_uₕ_g, prof_ug, prof_vg)
    return nothing
end

#####
##### DryBubble
#####

function surface_ref_state(::DryBubble, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = eltype(param_set)
    Pg::FT = 1.0e5  #Pressure at ground
    Tg::FT = 296.0
    qtg::FT = 1.0e-5
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end

function initialize_profiles(::DryBubble, grid::Grid, param_set, state; kwargs...)
    FT = TC.float_type(state)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ

    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, FT(0.01), FT(0))

    # initialize Grid Mean Profiles of thetali and qt
    zc_in = grid.zc.z
    prof_θ_liq_ice = APL.DryBubble_θ_liq_ice(FT)
    aux_gm.θ_liq_ice .= prof_θ_liq_ice.(zc_in)
    parent(prog_gm.ρq_tot) .= 0
    parent(aux_gm.q_tot) .= 0
    parent(aux_gm.tke) .= 0
    parent(aux_gm.Hvar) .= 0
    parent(aux_gm.QTvar) .= 0
    parent(aux_gm.HQTcov) .= 0
end

function surface_params(case::DryBubble, surf_ref_state, param_set; Ri_bulk_crit)
    FT = eltype(surf_ref_state)
    Tsurface::FT = 300.0
    qsurface::FT = 0.0
    shf::FT = 0.0001 # only prevent zero division in SF.jl lmo
    lhf::FT = 0.0001 # only prevent zero division in SF.jl lmo
    ustar::FT = 0.1

    kwargs = (; Tsurface, qsurface, shf, lhf, ustar, Ri_bulk_crit)
    return TC.FixedSurfaceFlux(FT, TC.FixedFrictionVelocity; kwargs...)
end

#####
##### LES_driven_SCM
#####

function forcing_kwargs(::LES_driven_SCM, namelist)
    coriolis_param = namelist["forcing"]["coriolis"]
    les_filename = namelist["meta"]["lesfile"]
    cfsite_number, forcing_model, month, experiment = TC.parse_les_path(les_filename)
    LES_library = TC.get_LES_library()
    les_type = LES_library[forcing_model][string(month, pad = 2)]["cfsite_numbers"][string(cfsite_number, pad = 2)]
    if les_type == "shallow"
        wind_nudge_τᵣ = 6.0 * 3600.0
        scalar_nudge_zᵢ = 3000.0
        scalar_nudge_zᵣ = 3500.0
        scalar_nudge_τᵣ = 24.0 * 3600.0
    elseif les_type == "deep"
        wind_nudge_τᵣ = 3600.0
        scalar_nudge_zᵢ = 16000.0
        scalar_nudge_zᵣ = 20000.0
        scalar_nudge_τᵣ = 2.0 * 3600.0
    end

    return (; wind_nudge_τᵣ, scalar_nudge_zᵢ, scalar_nudge_zᵣ, scalar_nudge_τᵣ, coriolis_param)
end

function les_data_kwarg(::LES_driven_SCM, namelist)
    les_filename = namelist["meta"]["lesfile"]
    # load data here
    LESDat = NC.Dataset(les_filename, "r") do data
        t = data.group["profiles"]["t"][:]
        t_interval_from_end_s = namelist["t_interval_from_end_s"]
        t_from_end_s = t .- t[end]
        # find inds within time interval
        time_interval_bool = findall(>(-t_interval_from_end_s), t_from_end_s)
        imin = time_interval_bool[1]
        imax = time_interval_bool[end]

        LESData(imin, imax, les_filename, t_interval_from_end_s, namelist["initial_condition_averaging_window_s"])
    end
    return (; LESDat)
end

function surface_ref_state(::LES_driven_SCM, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = eltype(param_set)
    les_filename = namelist["meta"]["lesfile"]

    Pg, Tg, qtg = NC.Dataset(les_filename, "r") do data
        pg_str = haskey(data.group["reference"], "p0_full") ? "p0_full" : "p0"
        Pg = data.group["reference"][pg_str][1] #Pressure at ground
        Tg = data.group["reference"]["temperature0"][1] #Temperature at ground
        ql_ground = data.group["reference"]["ql0"][1]
        qv_ground = data.group["reference"]["qv0"][1]
        qi_ground = data.group["reference"]["qi0"][1]
        qtg = ql_ground + qv_ground + qi_ground #Total water mixing ratio at surface
        (FT(Pg), FT(Tg), FT(qtg))
    end
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end

function initialize_profiles(::LES_driven_SCM, grid::Grid, param_set, state; LESDat)

    FT = TC.float_type(state)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)

    nt = NC.Dataset(LESDat.les_filename, "r") do data
        t = data.group["profiles"]["t"][:]
        # define time interval
        half_window = LESDat.initial_condition_averaging_window_s / 2
        t_window_start = (LESDat.t_interval_from_end_s + half_window)
        t_window_end = (LESDat.t_interval_from_end_s - half_window)
        t_from_end_s = Array(t) .- t[end]
        # find inds within time interval
        in_window(x) = -t_window_start <= x <= -t_window_end
        time_interval_bool = findall(in_window, t_from_end_s)
        imin = time_interval_bool[1]
        imax = time_interval_bool[end]
        zc_les = Array(TC.get_nc_data(data, "zc"))

        getvar(var) = pyinterp(vec(grid.zc.z), zc_les, TC.mean_nc_data(data, "profiles", var, imin, imax))

        θ_liq_ice_gm = getvar("thetali_mean")
        q_tot_gm = getvar("qt_mean")
        prog_gm_u_gm = getvar("u_mean")
        prog_gm_v_gm = getvar("v_mean")
        (; zc_les, θ_liq_ice_gm, q_tot_gm, prog_gm_u_gm, prog_gm_v_gm)
    end

    prog_gm_u = copy(aux_gm.q_tot)
    prog_gm_v = copy(aux_gm.q_tot)
    @inbounds for k in real_center_indices(grid)
        aux_gm.θ_liq_ice[k] = nt.θ_liq_ice_gm[k.i]
        aux_gm.q_tot[k] = nt.q_tot_gm[k.i]
        prog_gm_u[k] = nt.prog_gm_u_gm[k.i]
        prog_gm_v[k] = nt.prog_gm_v_gm[k.i]
    end

    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    @. prog_gm_uₕ = CCG.Covariant12Vector(CCG.UVVector(prog_gm_u, prog_gm_v))

    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.tke[k] = if z <= 2500.0
            1 - z / 3000
        else
            FT(0)
        end
    end
end

function surface_params(case::LES_driven_SCM, surf_ref_state, param_set; Ri_bulk_crit, LESDat)
    FT = eltype(surf_ref_state)
    nt = NC.Dataset(LESDat.les_filename, "r") do data
        imin = LESDat.imin
        imax = LESDat.imax

        zrough = FT(1.0e-4)
        Tsurface = Statistics.mean(data.group["timeseries"]["surface_temperature"][:][imin:imax], dims = 1)[1]
        # get surface value of q
        mean_qt_prof = Statistics.mean(data.group["profiles"]["qt_mean"][:][:, imin:imax], dims = 2)[:]
        # TODO: this will need to be changed if/when we don't prescribe surface fluxes
        qsurface = FT(0)
        lhf = FT(Statistics.mean(data.group["timeseries"]["lhf_surface_mean"][:][imin:imax], dims = 1)[1])
        shf = FT(Statistics.mean(data.group["timeseries"]["shf_surface_mean"][:][imin:imax], dims = 1)[1])
        (; zrough, Tsurface, qsurface, lhf, shf)
    end
    UnPack.@unpack zrough, Tsurface, qsurface, lhf, shf = nt

    ustar = FT(0) # TODO: why is initialization missing?
    kwargs = (; zrough, Tsurface, qsurface, shf, lhf, ustar, Ri_bulk_crit)
    return TC.FixedSurfaceFlux(FT, TC.VariableFrictionVelocity; kwargs...)
end

initialize_forcing(::LES_driven_SCM, forcing, grid::Grid, state, param_set; LESDat) =
    initialize(forcing, grid, state, LESDat)

initialize_radiation(::LES_driven_SCM, radiation, grid::Grid, state, param_set; LESDat) =
    initialize(radiation, grid, state, LESDat)

end # module Cases
