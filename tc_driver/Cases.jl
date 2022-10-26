# module Cases

import NCDatasets as NC
import OrdinaryDiffEq as ODE
import Thermodynamics as TD

import ClimaAtmos as CA
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

get_case_name(case_type::AbstractCaseType) = string(case_type)

#####
##### Pressure helper functions for making initial profiles hydrostatic.
#####

"""
    Pressure derivative with height assuming:
    - hydrostatic
    - given θ_liq_ice and q_tot initial profiles
"""
function dp_dz!(p, params, z)
    (; param_set, prof_thermo_var, prof_q_tot, thermo_flag) = params
    thermo_params = TCP.thermodynamics_params(param_set)

    FT = eltype(prof_q_tot(z))

    q_tot = prof_q_tot(z)
    q = TD.PhasePartition(q_tot)

    R_m = TD.gas_constant_air(thermo_params, q)
    grav = TCP.grav(param_set)

    if thermo_flag == "θ_liq_ice"
        θ_liq_ice = prof_thermo_var(z)
        cp_m = TD.cp_m(thermo_params, q)
        MSLP = TCP.MSLP(param_set)
        T = θ_liq_ice * (p / MSLP)^(R_m / cp_m)
    elseif thermo_flag == "temperature"
        T = prof_thermo_var(z)
    else
        error("θ_liq_ice or T must be provided to solve for pressure")
    end

    return -grav * p / R_m / T
end

""" Solving initial value problem for pressure """
function p_ivp(::Type{FT}, params, p_0, z_0, z_max) where {FT}

    z_span = (z_0, z_max)
    prob = ODE.ODEProblem(dp_dz!, p_0, z_span, params)

    sol = ODE.solve(prob, ODE.Tsit5(), reltol = 1e-15, abstol = 1e-15)
    return sol
end

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

    FT = TC.float_type(state)

    # Read in the initial profiles
    prof_q_tot = APL.Soares_q_tot(FT)
    prof_θ_liq_ice = APL.Soares_θ_liq_ice(FT)
    prof_u = APL.Soares_u(FT)
    prof_tke = APL.Soares_tke(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1000 * 100) # TODO - duplicated from surface_ref_state
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; param_set, prof_thermo_var, prof_q_tot, thermo_falg)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)
    p_c = TC.center_aux_grid_mean_p(state)

    # Fill in the grid mean state
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))
    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.q_tot[k] = prof_q_tot(z)
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)
        aux_gm.tke[k] = prof_tke(z)
        p_c[k] = prof_p(z)
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
function initialize_profiles(
    ::Nieuwstadt,
    grid::Grid,
    param_set,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)

    FT = TC.float_type(state)

    # Read in the initial profies
    prof_θ_liq_ice = APL.Nieuwstadt_θ_liq_ice(FT)
    prof_u = APL.Nieuwstadt_u(FT)
    prof_tke = APL.Nieuwstadt_tke(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1000 * 100) # TODO - duplicated from surface_ref_state
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; param_set, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)
    p_c = TC.center_aux_grid_mean_p(state)

    # Fill in the grid mean state
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))
    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)
        aux_gm.tke[k] = prof_tke(z)
        p_c[k] = prof_p(z)
    end
end

function surface_params(
    case::Nieuwstadt,
    surf_ref_state,
    param_set;
    Ri_bulk_crit,
)
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

    FT = TC.float_type(state)

    # Read in the initial profiles
    prof_q_tot = APL.Bomex_q_tot(FT)
    prof_θ_liq_ice = APL.Bomex_θ_liq_ice(FT)
    prof_u = APL.Bomex_u(FT)
    prof_tke = APL.Bomex_tke(FT)

    # Solve the initial value problem
    p_0::FT = FT(1.015e5) # TODO - duplicated from surface_ref_state
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; param_set, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)
    p_c = TC.center_aux_grid_mean_p(state)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))
    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)
        aux_gm.q_tot[k] = prof_q_tot(z)
        aux_gm.tke[k] = prof_tke(z)
        p_c[k] = prof_p(z)
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
function initialize_profiles(
    ::life_cycle_Tan2018,
    grid::Grid,
    param_set,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_q_tot = APL.LifeCycleTan2018_q_tot(FT)
    prof_θ_liq_ice = APL.LifeCycleTan2018_θ_liq_ice(FT)
    prof_u = APL.LifeCycleTan2018_u(FT)
    prof_tke = APL.LifeCycleTan2018_tke(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1.015e5)    # TODO - duplicated from surface_ref_state
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; param_set, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)
    p_c = TC.center_aux_grid_mean_p(state)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))
    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)
        aux_gm.q_tot[k] = prof_q_tot(z)
        aux_gm.tke[k] = prof_tke(z)
        p_c[k] = prof_p(z)
    end
end

function surface_params(
    case::life_cycle_Tan2018,
    surf_ref_state,
    param_set;
    Ri_bulk_crit,
)
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
    p_c = TC.center_aux_grid_mean_p(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_u = APL.Rico_u(FT)
    prof_v = APL.Rico_v(FT)
    prof_q_tot = APL.Rico_q_tot(FT)
    prof_θ_liq_ice = APL.Rico_θ_liq_ice(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1.015e5)    # TODO - duplicated from surface_ref_state
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; param_set, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)
    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)
        aux_gm.q_tot[k] = prof_q_tot(z)
        p_c[k] = prof_p(z)
    end

    # Need to get θ_virt
    @inbounds for k in real_center_indices(grid)
        # Thermo state field cache is not yet
        # defined, so we can't use it yet.
        ts = TD.PhaseEquil_pθq(
            thermo_params,
            p_c[k],
            aux_gm.θ_liq_ice[k],
            aux_gm.q_tot[k],
        )
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

#####
##### TRMM_LBA
#####

function surface_ref_state(::TRMM_LBA, param_set::APS, namelist)
    thermo_params = TCP.thermodynamics_params(param_set)
    molmass_ratio = TCP.molmass_ratio(param_set)
    FT = eltype(param_set)
    Pg::FT = 991.3 * 100  #Pressure at ground
    Tg::FT = 296.85 # surface values for reference state (RS) which outputs p, ρ
    pvg = TD.saturation_vapor_pressure(thermo_params, Tg, TD.Liquid())
    qtg = (1 / molmass_ratio) * pvg / (Pg - pvg) #Total water mixing ratio at surface
    return TD.PhaseEquil_pTq(thermo_params, Pg, Tg, qtg)
end
function TRMM_q_tot_profile(::Type{FT}, param_set) where {FT}

    thermo_params = TCP.thermodynamics_params(param_set)
    molmass_ratio = TCP.molmass_ratio(param_set)

    z_in = APL.TRMM_LBA_z(FT)
    p_in = APL.TRMM_LBA_p(FT)
    T_in = APL.TRMM_LBA_T(FT)
    RH_in = APL.TRMM_LBA_RH(FT)

    # eq. 37 in pressel et al and the def of RH
    q_tot_in = similar(z_in)
    for it in range(1, length = length(z_in))
        z = z_in[it]
        pv_star =
            TD.saturation_vapor_pressure(thermo_params, T_in(z), TD.Liquid())
        denom =
            (p_in(z) - pv_star + (1 / molmass_ratio) * pv_star * RH_in(z) / 100)
        qv_star = pv_star * (1 / molmass_ratio) / denom
        q_tot_in[it] = qv_star * RH_in(z) / 100
    end
    return Dierckx.Spline1D(z_in, q_tot_in; k = 1)
end
function initialize_profiles(
    ::TRMM_LBA,
    grid::Grid,
    param_set,
    state;
    kwargs...,
)
    thermo_params = TCP.thermodynamics_params(param_set)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    p_c = TC.center_aux_grid_mean_p(state)

    FT = TC.float_type(state)

    # Get profiles from AtmosphericProfilesLibrary.jl
    prof_T = APL.TRMM_LBA_T(FT)
    prof_u = APL.TRMM_LBA_u(FT)
    prof_v = APL.TRMM_LBA_v(FT)
    prof_tke = APL.TRMM_LBA_tke(FT)
    prof_q_tot = TRMM_q_tot_profile(FT, param_set)

    # Solve the initial value problem for pressure
    p_0::FT = FT(991.3 * 100)    # TODO - duplicated from surface_ref_state
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_T
    thermo_flag = "temperature"
    params = (; param_set, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)
    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        p_c[k] = prof_p(z)
        aux_gm.q_tot[k] = prof_q_tot(z)
        phase_part = TD.PhasePartition(aux_gm.q_tot[k], FT(0), FT(0)) # initial state is not saturated
        aux_gm.θ_liq_ice[k] = TD.liquid_ice_pottemp_given_pressure(
            thermo_params,
            prof_T(z),
            p_c[k],
            phase_part,
        )
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
    lhf =
        t ->
            554 *
            max(
                0,
                cos(FT(π) / 2 * ((FT(5.25) * 3600 - t) / FT(5.25) / 3600)),
            )^FT(1.3)
    shf =
        t ->
            270 *
            max(
                0,
                cos(FT(π) / 2 * ((FT(5.25) * 3600 - t) / FT(5.25) / 3600)),
            )^FT(1.5)
    kwargs = (;
        Tsurface,
        qsurface,
        shf,
        lhf,
        ustar,
        Ri_bulk_crit,
        zero_uv_fluxes = true,
    )
    return TC.FixedSurfaceFlux(FT, TC.FixedFrictionVelocity; kwargs...)
end


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
    prog_gm = TC.center_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_u = APL.ARM_SGP_u(FT)
    prof_q_tot = APL.ARM_SGP_q_tot(FT)
    prof_θ_liq_ice = APL.ARM_SGP_θ_liq_ice(FT)
    prof_tke = APL.ARM_SGP_tke(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(970 * 100)    # TODO - duplicated from surface_ref_state
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; param_set, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))
    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        # TODO figure out how to use ts here
        p_c[k] = prof_p(z)
        phase_part = TD.PhasePartition(aux_gm.q_tot[k], aux_gm.q_liq[k], FT(0))
        Π = TD.exner_given_pressure(thermo_params, p_c[k], phase_part)
        aux_gm.q_tot[k] = prof_q_tot(z)
        aux_gm.T[k] = prof_θ_liq_ice(z) * Π
        aux_gm.θ_liq_ice[k] = TD.liquid_ice_pottemp_given_pressure(
            thermo_params,
            aux_gm.T[k],
            p_c[k],
            phase_part,
        )
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

    kwargs = (;
        Tsurface,
        qsurface,
        shf,
        lhf,
        ustar,
        Ri_bulk_crit,
        zero_uv_fluxes = true,
    )
    return TC.FixedSurfaceFlux(FT, TC.FixedFrictionVelocity; kwargs...)
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

function initialize_profiles(
    ::GATE_III,
    grid::Grid,
    param_set,
    state;
    kwargs...,
)
    thermo_params = TCP.thermodynamics_params(param_set)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    p_c = TC.center_aux_grid_mean_p(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_q_tot = APL.GATE_III_q_tot(FT)
    prof_T = APL.GATE_III_T(FT)
    prof_tke = APL.GATE_III_tke(FT)
    prof_u = APL.GATE_III_u(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1013 * 100)    # TODO - duplicated from surface_ref_state
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_T
    thermo_falg = "temperature"
    params = (; param_set, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)
    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        aux_gm.q_tot[k] = prof_q_tot(z)
        aux_gm.T[k] = prof_T(z)
        p_c[k] = prof_p(z)
        aux_gm.tke[k] = prof_tke(z)
        ts = TD.PhaseEquil_pTq(
            thermo_params,
            p_c[k],
            aux_gm.T[k],
            aux_gm.q_tot[k],
        )
        aux_gm.θ_liq_ice[k] = TD.liquid_ice_pottemp(thermo_params, ts)
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

function initialize_profiles(
    ::DYCOMS_RF01,
    grid::Grid,
    param_set,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    p_c = TC.center_aux_grid_mean_p(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_u = APL.Dycoms_RF01_u0(FT)
    prof_v = APL.Dycoms_RF01_v0(FT)
    prof_q_tot = APL.Dycoms_RF01_q_tot(FT)
    prof_θ_liq_ice = APL.Dycoms_RF01_θ_liq_ice(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1017.8 * 100)  # TODO - duplicated from surface_ref_state
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; param_set, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)
    @inbounds for k in real_center_indices(grid)
        # thetal profile as defined in DYCOMS
        z = grid.zc[k].z
        aux_gm.q_tot[k] = prof_q_tot(z)
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)

        # velocity profile (geostrophic)
        aux_gm.tke[k] = APL.Dycoms_RF01_tke(FT)(z)
        p_c[k] = prof_p(z)
    end
end

function surface_params(
    case::DYCOMS_RF01,
    surf_ref_state,
    param_set;
    Ri_bulk_crit,
)
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

function initialize_profiles(
    ::DYCOMS_RF02,
    grid::Grid,
    param_set,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    p_c = TC.center_aux_grid_mean_p(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_u = APL.Dycoms_RF02_u(FT)
    prof_v = APL.Dycoms_RF02_v(FT)
    prof_q_tot = APL.Dycoms_RF02_q_tot(FT)
    prof_θ_liq_ice = APL.Dycoms_RF02_θ_liq_ice(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1017.8 * 100)  # TODO - duplicated from surface_ref_state
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; param_set, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)
    @inbounds for k in real_center_indices(grid)
        # θ_liq_ice profile as defined in DYCOM RF02
        z = grid.zc[k].z
        aux_gm.q_tot[k] = prof_q_tot(z)
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)

        # velocity profile
        aux_gm.tke[k] = APL.Dycoms_RF02_tke(FT)(z)
        p_c[k] = prof_p(z)
    end
end

function surface_params(
    case::DYCOMS_RF02,
    surf_ref_state,
    param_set;
    Ri_bulk_crit,
)
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
    p_c = TC.center_aux_grid_mean_p(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_u = APL.GABLS_u(FT)
    prof_v = APL.GABLS_v(FT)
    prof_θ_liq_ice = APL.GABLS_θ_liq_ice(FT)
    prof_q_tot = APL.GABLS_q_tot(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1.0e5)         # TODO - duplicated from surface_ref_state
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; param_set, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)
    @inbounds for k in real_center_indices(grid)
        z = grid.zc[k].z
        #Set wind velocity profile
        aux_gm.θ_liq_ice[k] = prof_θ_liq_ice(z)
        aux_gm.q_tot[k] = prof_q_tot(z)
        aux_gm.tke[k] = APL.GABLS_tke(FT)(z)
        aux_gm.Hvar[k] = aux_gm.tke[k]
        p_c[k] = prof_p(z)
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

# end # module Cases
