module Cases

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

import ..TurbulenceConvection as TC
import ..TurbulenceConvection.Parameters as TCP
const APS = TCP.AbstractTurbulenceConvectionParameters

using ..TurbulenceConvection: Grid
using ..TurbulenceConvection: real_center_indices

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
struct LifeCycleTan2018 <: AbstractCaseType end

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

get_case(casename::String) = get_case(Val(Symbol(casename)))
get_case(::Val{:Soares}) = Soares()
get_case(::Val{:Nieuwstadt}) = Nieuwstadt()
get_case(::Val{:Bomex}) = Bomex()
get_case(::Val{:LifeCycleTan2018}) = LifeCycleTan2018()
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
    (; thermo_params, prof_thermo_var, prof_q_tot, thermo_flag) = params

    FT = eltype(prof_thermo_var(z))

    q_tot = if prof_q_tot ≠ nothing
        prof_q_tot(z)
    else
        FT(0)
    end
    q = TD.PhasePartition(q_tot)

    R_m = TD.gas_constant_air(thermo_params, q)
    grav = TD.Parameters.grav(thermo_params)

    if thermo_flag == "θ_liq_ice"
        θ_liq_ice = prof_thermo_var(z)
        cp_m = TD.cp_m(thermo_params, q)
        MSLP = TD.Parameters.MSLP(thermo_params)
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

function initialize_profiles(
    ::Soares,
    grid::Grid,
    thermo_params,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)

    FT = TC.float_type(state)

    # Read in the initial profiles
    prof_q_tot = APL.Soares_q_tot(FT)
    prof_θ_liq_ice = APL.Soares_θ_liq_ice(FT)
    prof_u = APL.Soares_u(FT)
    prof_tke = APL.Soares_tke(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1000 * 100) # TODO - duplicated from surface_params
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; thermo_params, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)
    p_c = TC.center_aux_grid_mean_p(state)

    # Fill in the grid mean state
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))
    z = CC.Fields.coordinate_field(axes(p_c)).z
    @. aux_gm.q_tot = prof_q_tot(z)
    @. aux_gm.θ_liq_ice = prof_θ_liq_ice(z)
    @. aux_gm.tke = prof_tke(z)
    @. p_c = prof_p(z)
end

function surface_params(case::Soares, thermo_params, moisture_model)
    FT = eltype(thermo_params)
    zrough::FT = 0.16 #1.0e-4 0.16 is the value specified in the Nieuwstadt paper.
    psurface::FT = 1000 * 100
    Tsurface::FT = 300.0
    qsurface::FT = 5.0e-3
    θ_flux::FT = 6.0e-2
    qt_flux::FT = 2.5e-5
    ts = if moisture_model isa CA.DryModel
        TD.PhaseDry_pT(thermo_params, psurface, Tsurface)
    elseif moisture_model isa CA.EquilMoistModel
        TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    end
    ρsurface = TD.air_density(thermo_params, ts)
    lhf = qt_flux * ρsurface * TD.latent_heat_vapor(thermo_params, ts)
    shf = θ_flux * ρsurface * TD.cp_m(thermo_params, ts)
    return TC.FixedSurfaceFlux(zrough, ts, shf, lhf)
end

#####
##### Nieuwstadt
#####

function initialize_profiles(
    ::Nieuwstadt,
    grid::Grid,
    thermo_params,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)

    FT = TC.float_type(state)

    # Read in the initial profies
    prof_θ_liq_ice = APL.Nieuwstadt_θ_liq_ice(FT)
    prof_u = APL.Nieuwstadt_u(FT)
    prof_tke = APL.Nieuwstadt_tke(FT)
    prof_q_tot = nothing

    # Solve the initial value problem for pressure
    p_0::FT = FT(1000 * 100) # TODO - duplicated from surface_params
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; thermo_params, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)
    p_c = TC.center_aux_grid_mean_p(state)

    # Fill in the grid mean state
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))
    z = CC.Fields.coordinate_field(axes(p_c)).z
    @. aux_gm.θ_liq_ice = prof_θ_liq_ice(z)
    @. aux_gm.tke = prof_tke(z)
    @. p_c = prof_p(z)
end

function surface_params(case::Nieuwstadt, thermo_params, moisture_model)
    FT = eltype(thermo_params)
    zrough::FT = 0.16 #1.0e-4 0.16 is the value specified in the Nieuwstadt paper.
    psurface::FT = 1000 * 100
    Tsurface::FT = 300.0
    qsurface::FT = 0.0
    θ_flux::FT = 6.0e-2
    lhf::FT = 0.0 # It would be 0.0 if we follow Nieuwstadt.
    ts = if moisture_model isa CA.DryModel
        TD.PhaseDry_pT(thermo_params, psurface, Tsurface)
    elseif moisture_model isa CA.EquilMoistModel
        TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    end
    shf =
        θ_flux * TD.air_density(thermo_params, ts) * TD.cp_m(thermo_params, ts)
    return TC.FixedSurfaceFlux(zrough, ts, shf, lhf)
end

#####
##### Bomex
#####

function initialize_profiles(
    ::Bomex,
    grid::Grid,
    thermo_params,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)

    FT = TC.float_type(state)

    # Read in the initial profiles
    prof_q_tot = APL.Bomex_q_tot(FT)
    prof_θ_liq_ice = APL.Bomex_θ_liq_ice(FT)
    prof_u = APL.Bomex_u(FT)
    prof_tke = APL.Bomex_tke(FT)

    # Solve the initial value problem
    p_0::FT = FT(1.015e5) # TODO - duplicated from surface_params
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; thermo_params, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)
    p_c = TC.center_aux_grid_mean_p(state)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))
    z = CC.Fields.coordinate_field(axes(p_c)).z
    @. aux_gm.θ_liq_ice = prof_θ_liq_ice(z)
    @. aux_gm.q_tot = prof_q_tot(z)
    @. aux_gm.tke = prof_tke(z)
    @. p_c = prof_p(z)
end

function surface_params(case::Bomex, thermo_params, moisture_model)
    FT = eltype(thermo_params)
    zrough::FT = 1.0e-4
    psurface::FT = 1.015e5
    qsurface::FT = 22.45e-3 # kg/kg
    Tsurface::FT = 300.4 # Equivalent to θsurface = 299.1
    θ_flux::FT = 8.0e-3
    qt_flux::FT = 5.2e-5
    ts = if moisture_model isa CA.DryModel
        TD.PhaseDry_pT(thermo_params, psurface, Tsurface)
    elseif moisture_model isa CA.EquilMoistModel
        TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    end
    ρsurface = TD.air_density(thermo_params, ts)
    lhf = qt_flux * ρsurface * TD.latent_heat_vapor(thermo_params, ts)
    shf = θ_flux * ρsurface * TD.cp_m(thermo_params, ts)
    ustar::FT = 0.28 # m/s
    return TC.FixedSurfaceFluxAndFrictionVelocity(zrough, ts, shf, lhf, ustar)
end

#####
##### LifeCycleTan2018
#####

function initialize_profiles(
    ::LifeCycleTan2018,
    grid::Grid,
    thermo_params,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_q_tot = APL.LifeCycleTan2018_q_tot(FT)
    prof_θ_liq_ice = APL.LifeCycleTan2018_θ_liq_ice(FT)
    prof_u = APL.LifeCycleTan2018_u(FT)
    prof_tke = APL.LifeCycleTan2018_tke(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1.015e5)    # TODO - duplicated from surface_params
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; thermo_params, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)
    p_c = TC.center_aux_grid_mean_p(state)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))
    z = CC.Fields.coordinate_field(axes(p_c)).z
    @. aux_gm.θ_liq_ice = prof_θ_liq_ice(z)
    @. aux_gm.q_tot = prof_q_tot(z)
    @. aux_gm.tke = prof_tke(z)
    @. p_c = prof_p(z)
end

function surface_params(case::LifeCycleTan2018, thermo_params, moisture_model)
    FT = eltype(thermo_params)
    zrough::FT = 1.0e-4 # not actually used, but initialized to reasonable value
    psurface::FT = 1.015e5
    qsurface::FT = 22.45e-3 # kg/kg
    Tsurface::FT = 300.4 # equivalent to θsurface = 299.1
    θ_flux::FT = 8.0e-3
    qt_flux::FT = 5.2e-5
    ts = if moisture_model isa CA.DryModel
        TD.PhaseDry_pT(thermo_params, psurface, Tsurface)
    elseif moisture_model isa CA.EquilMoistModel
        TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    end
    ρsurface = TD.air_density(thermo_params, ts)
    lhf0 = qt_flux * ρsurface * TD.latent_heat_vapor(thermo_params, ts)
    shf0 = θ_flux * ρsurface * TD.cp_m(thermo_params, ts)

    weight_factor(t) = FT(0.01) + FT(0.99) * (cos(2 * FT(π) * t / 3600) + 1) / 2
    weight::FT = 1.0
    lhf = t -> lhf0 * (weight * weight_factor(t))
    shf = t -> shf0 * (weight * weight_factor(t))

    ustar::FT = 0.28 # m/s
    return TC.FixedSurfaceFluxAndFrictionVelocity(zrough, ts, shf, lhf, ustar)
end

#####
##### Rico
#####

function initialize_profiles(
    ::Rico,
    grid::Grid,
    thermo_params,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)
    p_c = TC.center_aux_grid_mean_p(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_u = APL.Rico_u(FT)
    prof_v = APL.Rico_v(FT)
    prof_q_tot = APL.Rico_q_tot(FT)
    prof_θ_liq_ice = APL.Rico_θ_liq_ice(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1.0154e5)    # TODO - duplicated from surface_params
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; thermo_params, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)
    z = CC.Fields.coordinate_field(axes(p_c)).z
    @. aux_gm.θ_liq_ice = prof_θ_liq_ice(z)
    @. aux_gm.q_tot = prof_q_tot(z)
    @. p_c = prof_p(z)

    z = CC.Fields.coordinate_field(axes(p_c)).z
    # Need to get θ_virt
    # Thermo state field cache is not yet
    # defined, so we can't use it yet.
    @. aux_gm.θ_virt = TD.virtual_pottemp(
        thermo_params,
        TD.PhaseEquil_pθq(thermo_params, p_c, aux_gm.θ_liq_ice, aux_gm.q_tot),
    )

    zi = FT(2980.0)
    prof_tke = z -> if z <= zi
        1 - z / zi
    else
        FT(0)
    end
    @. aux_gm.tke = prof_tke(z)
end

function surface_params(case::Rico, thermo_params, moisture_model)
    FT = eltype(thermo_params)
    zrough::FT = 0.00015
    cm0::FT = 0.001229
    ch0::FT = 0.001094
    cq0::FT = 0.001133
    # Adjust for non-IC grid spacing
    grid_adjust(zc_surf) = (log(20 / zrough) / log(zc_surf / zrough))^2
    cm = zc_surf -> cm0 * grid_adjust(zc_surf)
    ch = zc_surf -> ch0 * grid_adjust(zc_surf)
    cq = zc_surf -> cq0 * grid_adjust(zc_surf) # TODO: not yet used..
    psurface::FT = 1.0154e5
    Tsurface::FT = 299.8 # 301.1
    # Saturated surface condtions for a given surface temperature and pressure
    p_sat_surface::FT =
        TD.saturation_vapor_pressure(thermo_params, Tsurface, TD.Liquid())
    ϵ_v::FT =
        TD.Parameters.R_d(thermo_params) / TD.Parameters.R_v(thermo_params)
    qsurface::FT = ϵ_v * p_sat_surface / (psurface - p_sat_surface * (1 - ϵ_v))
    ts = if moisture_model isa CA.DryModel
        TD.PhaseDry_pT(thermo_params, psurface, Tsurface)
    elseif moisture_model isa CA.EquilMoistModel
        TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    end
    # For Rico we provide values of transfer coefficients
    return TC.FixedSurfaceCoeffs(; zrough, ts, ch, cm)
end

#####
##### TRMM_LBA
#####

function TRMM_q_tot_profile(::Type{FT}, thermo_params) where {FT}

    molmass_ratio = TD.Parameters.molmass_ratio(thermo_params)

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
    thermo_params,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)
    p_c = TC.center_aux_grid_mean_p(state)

    FT = TC.float_type(state)

    # Get profiles from AtmosphericProfilesLibrary.jl
    prof_T = APL.TRMM_LBA_T(FT)
    prof_u = APL.TRMM_LBA_u(FT)
    prof_v = APL.TRMM_LBA_v(FT)
    prof_tke = APL.TRMM_LBA_tke(FT)
    prof_q_tot = TRMM_q_tot_profile(FT, thermo_params)

    # Solve the initial value problem for pressure
    p_0::FT = FT(991.3 * 100)    # TODO - duplicated from surface_params
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_T
    thermo_flag = "temperature"
    params = (; thermo_params, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)
    z = CC.Fields.coordinate_field(axes(p_c)).z
    @. p_c = prof_p(z)
    @. aux_gm.q_tot = prof_q_tot(z)
    @. aux_gm.θ_liq_ice = TD.liquid_ice_pottemp_given_pressure(
        thermo_params,
        prof_T(z),
        p_c,
        TD.PhasePartition(aux_gm.q_tot, FT(0), FT(0)), # initial state is not saturated,
    )
    @. aux_gm.tke = prof_tke(z)
end

function surface_params(case::TRMM_LBA, thermo_params, moisture_model)
    FT = eltype(thermo_params)
    # zrough = 1.0e-4 # not actually used, but initialized to reasonable value
    zrough::FT = 0 # actually, used, TODO: should we be using the value above?
    psurface::FT = 991.3 * 100
    qsurface::FT = 22.45e-3 # kg/kg
    Tsurface::FT = 273.15 + 23.7
    ts = if moisture_model isa CA.DryModel
        TD.PhaseDry_pT(thermo_params, psurface, Tsurface)
    elseif moisture_model isa CA.EquilMoistModel
        TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    end
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
    return TC.FixedSurfaceFluxAndFrictionVelocity(zrough, ts, shf, lhf, ustar)
end


#####
##### ARM_SGP
#####

function initialize_profiles(
    ::ARM_SGP,
    grid::Grid,
    thermo_params,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_u = APL.ARM_SGP_u(FT)
    prof_q_tot = APL.ARM_SGP_q_tot(FT)
    prof_θ_liq_ice = APL.ARM_SGP_θ_liq_ice(FT)
    prof_tke = APL.ARM_SGP_tke(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(970 * 100)    # TODO - duplicated from surface_params
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; thermo_params, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    p_c = TC.center_aux_grid_mean_p(state)
    TC.set_z!(prog_gm_uₕ, prof_u, x -> FT(0))
    z = CC.Fields.coordinate_field(axes(prog_gm_uₕ)).z
    # TODO figure out how to use ts here
    @. p_c = prof_p(z)
    @. aux_gm.q_tot = prof_q_tot(z)
    @. aux_gm.T =
        prof_θ_liq_ice(z) * TD.exner_given_pressure(
            thermo_params,
            p_c,
            TD.PhasePartition(aux_gm.q_tot, aux_gm.q_liq, FT(0)),
        )
    @. aux_gm.θ_liq_ice = TD.liquid_ice_pottemp_given_pressure(
        thermo_params,
        aux_gm.T,
        p_c,
        TD.PhasePartition(aux_gm.q_tot, aux_gm.q_liq, FT(0)),
    )
    @. aux_gm.tke = prof_tke(z)
end

function surface_params(case::ARM_SGP, thermo_params, moisture_model)
    FT = eltype(thermo_params)
    psurface::FT = 970 * 100
    qsurface::FT = 15.2e-3 # kg/kg
    θ_surface::FT = 299.0
    ts = if moisture_model isa CA.DryModel
        TD.PhaseDry_pθ(thermo_params, psurface, θ_surface)
    elseif moisture_model isa CA.EquilMoistModel
        TD.PhaseEquil_pθq(thermo_params, psurface, θ_surface, qsurface)
    end
    ustar::FT = 0.28 # this is taken from Bomex -- better option is to approximate from LES tke above the surface

    t_Sur_in = arr_type(FT[0.0, 4.0, 6.5, 7.5, 10.0, 12.5, 14.5]) .* 3600 #LES time is in sec
    SH = arr_type(FT[-30.0, 90.0, 140.0, 140.0, 100.0, -10, -10]) # W/m^2
    LH = arr_type(FT[5.0, 250.0, 450.0, 500.0, 420.0, 180.0, 0.0]) # W/m^2
    shf = Dierckx.Spline1D(t_Sur_in, SH; k = 1)
    lhf = Dierckx.Spline1D(t_Sur_in, LH; k = 1)
    zrough::FT = 0

    return TC.FixedSurfaceFluxAndFrictionVelocity(zrough, ts, shf, lhf, ustar)
end

#####
##### GATE_III
#####

function initialize_profiles(
    ::GATE_III,
    grid::Grid,
    thermo_params,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)
    p_c = TC.center_aux_grid_mean_p(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_q_tot = APL.GATE_III_q_tot(FT)
    prof_T = APL.GATE_III_T(FT)
    prof_tke = APL.GATE_III_tke(FT)
    prof_u = APL.GATE_III_u(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1013 * 100)    # TODO - duplicated from surface_params
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_T
    thermo_flag = "temperature"
    params = (; thermo_params, prof_thermo_var, prof_q_tot, thermo_flag)
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

# TODO: The paper only specifies that Tsurface = 299.88. Where did all of these
# values come from?
function surface_params(case::GATE_III, thermo_params, moisture_model)
    FT = eltype(thermo_params)
    psurface::FT = 1013 * 100
    qsurface::FT = 16.5 / 1000.0 # kg/kg
    cm = zc_surf -> FT(0.0012)
    ch = zc_surf -> FT(0.0034337)
    cq = zc_surf -> FT(0.0034337)
    Tsurface::FT = 299.184

    # For GATE_III we provide values of transfer coefficients
    ts = if moisture_model isa CA.DryModel
        TD.PhaseEquil_pT(thermo_params, psurface, Tsurface)
    elseif moisture_model isa CA.EquilMoistModel
        TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    end
    return TC.FixedSurfaceCoeffs(; zrough = FT(0), ts, ch, cm)
end

#####
##### DYCOMS_RF01
#####

function initialize_profiles(
    ::DYCOMS_RF01,
    grid::Grid,
    thermo_params,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)
    p_c = TC.center_aux_grid_mean_p(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_u = APL.Dycoms_RF01_u0(FT)
    prof_v = APL.Dycoms_RF01_v0(FT)
    prof_q_tot = APL.Dycoms_RF01_q_tot(FT)
    prof_θ_liq_ice = APL.Dycoms_RF01_θ_liq_ice(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1017.8 * 100)  # TODO - duplicated from surface_params
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; thermo_params, prof_thermo_var, prof_q_tot, thermo_flag)
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

function surface_params(case::DYCOMS_RF01, thermo_params, moisture_model)
    FT = eltype(thermo_params)
    zrough::FT = 1.0e-4
    ustar::FT = 0.28 # just to initialize grid mean covariances
    shf::FT = 15.0 # sensible heat flux
    lhf::FT = 115.0 # latent heat flux
    psurface::FT = 1017.8 * 100
    Tsurface::FT = 292.5    # K      # i.e. the SST from DYCOMS setup
    qsurface::FT = 13.84e-3 # kg/kg  # TODO - taken from Pycles, maybe it would be better to calculate the q_star(sst) for TurbulenceConvection?
    #density_surface  = 1.22     # kg/m^3
    ts = if moisture_model isa CA.DryModel
        TD.PhaseDry_pT(thermo_params, psurface, Tsurface)
    elseif moisture_model isa CA.EquilMoistModel
        TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    end
    return TC.FixedSurfaceFlux(zrough, ts, shf, lhf)
end

#####
##### DYCOMS_RF02
#####

function initialize_profiles(
    ::DYCOMS_RF02,
    grid::Grid,
    thermo_params,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)
    p_c = TC.center_aux_grid_mean_p(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_u = APL.Dycoms_RF02_u(FT)
    prof_v = APL.Dycoms_RF02_v(FT)
    prof_q_tot = APL.Dycoms_RF02_q_tot(FT)
    prof_θ_liq_ice = APL.Dycoms_RF02_θ_liq_ice(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1017.8 * 100)  # TODO - duplicated from surface_params
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; thermo_params, prof_thermo_var, prof_q_tot, thermo_flag)
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

function surface_params(case::DYCOMS_RF02, thermo_params, moisture_model)
    FT = eltype(thermo_params)
    zrough::FT = 1.0e-4  #TODO - not needed?
    ustar::FT = 0.25
    shf::FT = 16.0 # sensible heat flux
    lhf::FT = 93.0 # latent heat flux
    psurface::FT = 1017.8 * 100
    Tsurface::FT = 292.5    # K      # i.e. the SST from DYCOMS setup
    qsurface::FT = 13.84e-3 # kg/kg  # TODO - taken from Pycles, maybe it would be better to calculate the q_star(sst) for TurbulenceConvection?
    ts = if moisture_model isa CA.DryModel
        TD.PhaseDry_pT(thermo_params, psurface, Tsurface)
    elseif moisture_model isa CA.EquilMoistModel
        TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    end
    return TC.FixedSurfaceFluxAndFrictionVelocity(zrough, ts, shf, lhf, ustar)
end

#####
##### GABLS
#####

function initialize_profiles(
    ::GABLS,
    grid::Grid,
    thermo_params,
    state;
    kwargs...,
)
    aux_gm = TC.center_aux_grid_mean(state)
    p_c = TC.center_aux_grid_mean_p(state)

    FT = TC.float_type(state)

    # Load the initial profiles
    prof_u = APL.GABLS_u(FT)
    prof_v = APL.GABLS_v(FT)
    prof_θ_liq_ice = APL.GABLS_θ_liq_ice(FT)
    prof_q_tot = APL.GABLS_q_tot(FT)

    # Solve the initial value problem for pressure
    p_0::FT = FT(1.0e5)         # TODO - duplicated from surface_params
    z_0::FT = grid.zf[TC.kf_surface(grid)].z
    z_max::FT = grid.zf[TC.kf_top_of_atmos(grid)].z
    prof_thermo_var = prof_θ_liq_ice
    thermo_flag = "θ_liq_ice"
    params = (; thermo_params, prof_thermo_var, prof_q_tot, thermo_flag)
    prof_p = p_ivp(FT, params, p_0, z_0, z_max)

    # Fill in the grid mean values
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    TC.set_z!(prog_gm_uₕ, prof_u, prof_v)
    prof_tke = APL.GABLS_tke(FT)
    z = CC.Fields.coordinate_field(axes(p_c)).z
    #Set wind velocity profile
    @. aux_gm.θ_liq_ice = prof_θ_liq_ice(z)
    @. aux_gm.q_tot = prof_q_tot(z)
    @. aux_gm.tke = prof_tke(z)
    @. aux_gm.Hvar = aux_gm.tke
    @. p_c = prof_p(z)
end

function surface_params(case::GABLS, thermo_params, moisture_model)
    FT = eltype(thermo_params)
    psurface::FT = 1.0e5
    Tsurface = t -> 265 - (FT(0.25) / 3600) * t
    qsurface::FT = 0.0
    zrough::FT = 0.1
    ts = if moisture_model isa CA.DryModel
        t -> TD.PhaseDry_pT(thermo_params, psurface, Tsurface(t))
    elseif moisture_model isa CA.EquilMoistModel
        t -> TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface(t), qsurface)
    end
    return TC.MoninObukhovSurface(; ts, zrough)
end

end # module Cases
