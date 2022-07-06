import UnPack
import LinearAlgebra as LA
import LinearAlgebra: ×

import TurbulenceConvection as TC
import TurbulenceConvection.Parameters as TCP
const APS = TCP.AbstractTurbulenceConvectionParameters
import Thermodynamics as TD
import ClimaCore as CC
import ClimaCore.Geometry as CCG
import OrdinaryDiffEq as ODE

import CLIMAParameters as CP

include(joinpath(@__DIR__, "dycore_variables.jl"))

#####
##### Methods
#####

####
#### Reference state
####

"""
    compute_ref_state!(
        state,
        grid::Grid,
        param_set::PS;
        ts_g,
    ) where {PS}

TODO: add better docs once the API converges

The reference profiles, given
 - `grid` the grid
 - `param_set` the parameter set
 - `ts_g` the surface reference state (a thermodynamic state)
"""
function compute_ref_state!(state, grid::TC.Grid, param_set::PS; ts_g) where {PS}
    aux_gm = TC.center_aux_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    p_c = aux_gm.p
    ρ_c = prog_gm.ρ
    p_f = aux_gm_f.p
    ρ_f = aux_gm_f.ρ
    compute_ref_state!(p_c, ρ_c, p_f, ρ_f, grid, param_set; ts_g)
end

function compute_ref_state!(
    p_c::CC.Fields.Field,
    ρ_c::CC.Fields.Field,
    p_f::CC.Fields.Field,
    ρ_f::CC.Fields.Field,
    grid::TC.Grid,
    param_set::PS;
    ts_g,
) where {PS}
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = TC.float_type(p_c)
    kf_surf = TC.kf_surface(grid)
    qtg = TD.total_specific_humidity(thermo_params, ts_g)
    Φ = TC.geopotential(param_set, grid.zf[kf_surf].z)
    mse_g = TD.moist_static_energy(thermo_params, ts_g, Φ)
    Pg = TD.air_pressure(thermo_params, ts_g)

    # We are integrating the log pressure so need to take the log of the
    # surface pressure
    logp = log(Pg)

    # Form a right hand side for integrating the hydrostatic equation to
    # determine the reference pressure
    function minus_inv_scale_height(logp, u, z)
        p_ = exp(logp)
        grav = FT(TCP.grav(param_set))
        Φ = TC.geopotential(param_set, z)
        h = TC.enthalpy(mse_g, Φ)
        ts = TD.PhaseEquil_phq(thermo_params, p_, h, qtg)
        R_m = TD.gas_constant_air(thermo_params, ts)
        T = TD.air_temperature(thermo_params, ts)
        return -FT(TCP.grav(param_set)) / (T * R_m)
    end

    # Perform the integration
    z_span = (grid.zmin, grid.zmax)
    prob = ODE.ODEProblem(minus_inv_scale_height, logp, z_span)
    sol = ODE.solve(prob, ODE.Tsit5(), reltol = 1e-12, abstol = 1e-12)
    parent(p_f) .= sol.(vec(grid.zf.z))
    parent(p_c) .= sol.(vec(grid.zc.z))

    p_f .= exp.(p_f)
    p_c .= exp.(p_c)

    # Compute reference state thermodynamic profiles
    @inbounds for k in TC.real_center_indices(grid)
        Φ = TC.geopotential(param_set, grid.zc[k].z)
        h = TC.enthalpy(mse_g, Φ)
        ts = TD.PhaseEquil_phq(thermo_params, p_c[k], h, qtg)
        ρ_c[k] = TD.air_density(thermo_params, ts)
    end

    @inbounds for k in TC.real_face_indices(grid)
        Φ = TC.geopotential(param_set, grid.zf[k].z)
        h = TC.enthalpy(mse_g, Φ)
        ts = TD.PhaseEquil_phq(thermo_params, p_f[k], h, qtg)
        ρ_f[k] = TD.air_density(thermo_params, ts)
    end
    return nothing
end


function set_thermo_state_peq!(state, grid, moisture_model, param_set)
    Ic = CCO.InterpolateF2C()
    thermo_params = TCP.thermodynamics_params(param_set)
    ts_gm = TC.center_aux_grid_mean(state).ts
    prog_gm = TC.center_prog_grid_mean(state)
    prog_gm_f = TC.face_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    p_c = aux_gm.p
    ρ_c = prog_gm.ρ
    C123 = CCG.Covariant123Vector
    @. aux_gm.e_kin = LA.norm_sqr(C123(prog_gm_uₕ) + C123(Ic(prog_gm_f.w))) / 2

    @inbounds for k in TC.real_center_indices(grid)
        thermo_args = if moisture_model isa TC.EquilibriumMoisture
            ()
        elseif moisture_model isa TC.NonEquilibriumMoisture
            (prog_gm.q_liq[k], prog_gm.q_ice[k])
        else
            error("Something went wrong. The moisture_model options are equilibrium or nonequilibrium")
        end
        e_pot = TC.geopotential(param_set, grid.zc.z[k])
        e_int = prog_gm.ρe_tot[k] / ρ_c[k] - aux_gm.e_kin[k] - e_pot
        ts_gm[k] = TC.thermo_state_peq(param_set, p_c[k], e_int, prog_gm.ρq_tot[k] / ρ_c[k], thermo_args...)
    end
    return nothing
end

function set_thermo_state_pθq!(state, grid, moisture_model, param_set)
    Ic = CCO.InterpolateF2C()
    ts_gm = TC.center_aux_grid_mean(state).ts
    prog_gm = TC.center_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    p_c = aux_gm.p
    @inbounds for k in TC.real_center_indices(grid)
        thermo_args = if moisture_model isa TC.EquilibriumMoisture
            ()
        elseif moisture_model isa TC.NonEquilibriumMoisture
            (prog_gm.q_liq[k], prog_gm.q_ice[k])
        else
            error("Something went wrong. The moisture_model options are equilibrium or nonequilibrium")
        end
        ts_gm[k] = TC.thermo_state_pθq(param_set, p_c[k], aux_gm.θ_liq_ice[k], aux_gm.q_tot[k], thermo_args...)
    end
    return nothing
end

function set_grid_mean_from_thermo_state!(param_set, state, grid)
    thermo_params = TCP.thermodynamics_params(param_set)
    Ic = CCO.InterpolateF2C()
    ts_gm = TC.center_aux_grid_mean(state).ts
    prog_gm = TC.center_prog_grid_mean(state)
    prog_gm_f = TC.face_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    ρ_c = prog_gm.ρ
    C123 = CCG.Covariant123Vector
    @. prog_gm.ρe_tot =
        ρ_c * TD.total_energy(
            thermo_params,
            ts_gm,
            LA.norm_sqr(C123(prog_gm_uₕ) + C123(Ic(prog_gm_f.w))) / 2,
            TC.geopotential(param_set, grid.zc.z),
        )
    @. prog_gm.ρq_tot = ρ_c * aux_gm.q_tot

    return nothing
end

function assign_thermo_aux!(state, grid, moisture_model, param_set)
    thermo_params = TCP.thermodynamics_params(param_set)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ts_gm = TC.center_aux_grid_mean(state).ts
    ρ_c = prog_gm.ρ
    @inbounds for k in TC.real_center_indices(grid)
        ts = ts_gm[k]
        aux_gm.q_tot[k] = prog_gm.ρq_tot[k] / ρ_c[k]
        aux_gm.q_liq[k] = TD.liquid_specific_humidity(thermo_params, ts)
        aux_gm.q_ice[k] = TD.ice_specific_humidity(thermo_params, ts)
        aux_gm.T[k] = TD.air_temperature(thermo_params, ts)
        aux_gm.RH[k] = TD.relative_humidity(thermo_params, ts)
        aux_gm.θ_liq_ice[k] = TD.liquid_ice_pottemp(thermo_params, ts)
        aux_gm.h_tot[k] = TC.total_enthalpy(param_set, prog_gm.ρe_tot[k] / ρ_c[k], ts)
    end
    return
end

function ∑stoch_tendencies!(tendencies::FV, prog::FV, params::NT, t::Real) where {NT, FV <: CC.Fields.FieldVector}
    UnPack.@unpack edmf, param_set, surf_params, aux = params

    # set all tendencies to zero
    tends_face = tendencies.face
    tends_cent = tendencies.cent
    parent(tends_face) .= 0
    parent(tends_cent) .= 0

    for inds in TC.iterate_columns(prog.cent)

        state = TC.column_state(prog, aux, tendencies, inds...)
        grid = TC.Grid(state)
        surf = get_surface(surf_params, grid, state, t, param_set)

        # compute updraft stochastic tendencies
        TC.compute_up_stoch_tendencies!(edmf, grid, state, param_set, surf)
    end
end

# Compute the sum of tendencies for the scheme
function ∑tendencies!(tendencies::FV, prog::FV, params::NT, t::Real) where {NT, FV <: CC.Fields.FieldVector}
    UnPack.@unpack edmf, precip_model, param_set, case = params
    UnPack.@unpack surf_params, radiation, forcing, aux, TS = params

    thermo_params = TCP.thermodynamics_params(param_set)
    tends_face = tendencies.face
    tends_cent = tendencies.cent
    parent(tends_face) .= 0
    parent(tends_cent) .= 0

    for inds in TC.iterate_columns(prog.cent)
        state = TC.column_state(prog, aux, tendencies, inds...)
        grid = TC.Grid(state)

        set_thermo_state_peq!(state, grid, edmf.moisture_model, param_set)
        assign_thermo_aux!(state, grid, edmf.moisture_model, param_set)

        aux_gm = TC.center_aux_grid_mean(state)

        @. aux_gm.θ_virt = TD.virtual_pottemp(thermo_params, aux_gm.ts)

        Δt = TS.dt
        surf = get_surface(surf_params, grid, state, t, param_set)

        TC.affect_filter!(edmf, grid, state, param_set, surf, t)

        # Update aux / pre-tendencies filters. TODO: combine these into a function that minimizes traversals
        # Some of these methods should probably live in `compute_tendencies`, when written, but we'll
        # treat them as auxiliary variables for now, until we disentangle the tendency computations.
        Cases.update_forcing(case, grid, state, t, param_set)
        Cases.update_radiation(radiation, grid, state, t, param_set)

        TC.update_aux!(edmf, grid, state, surf, param_set, t, Δt)

        # compute tendencies
        # causes division error in dry bubble first time step
        TC.compute_precipitation_sink_tendencies(precip_model, edmf, grid, state, param_set, Δt)
        TC.compute_precipitation_advection_tendencies(precip_model, edmf, grid, state, param_set)

        TC.compute_turbconv_tendencies!(edmf, grid, state, param_set, surf, Δt)
        compute_gm_tendencies!(edmf, grid, state, surf, radiation, forcing, param_set)
    end

    return nothing
end

function compute_les_Γᵣ(z::FT, τᵣ::FT = 24.0 * 3600.0, zᵢ::FT = 3000.0, zᵣ::FT = 3500.0) where {FT <: Real}
    # returns height-dependent relaxation timescale from eqn. 9 in `Shen et al. 2021`
    if z < zᵢ
        return FT(0)
    elseif zᵢ <= z <= zᵣ
        cos_arg = pi * ((z - zᵢ) / (zᵣ - zᵢ))
        return (FT(0.5) / τᵣ) * (1 - cos(cos_arg))
    elseif z > zᵣ
        return (1 / τᵣ)
    end
end

function compute_gm_tendencies!(
    edmf::TC.EDMFModel,
    grid::TC.Grid,
    state::TC.State,
    surf::TC.SurfaceBase,
    radiation::Cases.RadiationBase,
    force::Cases.ForcingBase,
    param_set::APS,
)
    thermo_params = TCP.thermodynamics_params(param_set)
    Ic = CCO.InterpolateF2C()
    R_d = TCP.R_d(param_set)
    T_0 = TCP.T_0(param_set)
    Lv_0 = TCP.LH_v0(param_set)
    tendencies_gm = TC.center_tendencies_grid_mean(state)
    kc_toa = TC.kc_top_of_atmos(grid)
    kf_surf = TC.kf_surface(grid)
    FT = TC.float_type(state)
    prog_gm = TC.center_prog_grid_mean(state)
    prog_gm_f = TC.face_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    ∇MSE_gm = TC.center_aux_grid_mean(state).∇MSE_gm
    ∇q_tot_gm = TC.center_aux_grid_mean(state).∇q_tot_gm
    aux_en = TC.center_aux_environment(state)
    aux_en_f = TC.face_aux_environment(state)
    aux_up = TC.center_aux_updrafts(state)
    aux_bulk = TC.center_aux_bulk(state)
    ρ_f = aux_gm_f.ρ
    p_c = aux_gm.p
    ρ_c = prog_gm.ρ
    aux_tc = TC.center_aux_turbconv(state)
    ts_gm = TC.center_aux_grid_mean(state).ts

    MSE_gm_toa = aux_gm.h_tot[kc_toa] - aux_gm.e_kin[kc_toa]
    q_tot_gm_toa = prog_gm.ρq_tot[kc_toa] / ρ_c[kc_toa]
    RBe = CCO.RightBiasedC2F(; top = CCO.SetValue(MSE_gm_toa))
    RBq = CCO.RightBiasedC2F(; top = CCO.SetValue(q_tot_gm_toa))
    wvec = CC.Geometry.WVector
    ∇c = CCO.DivergenceF2C()
    @. ∇MSE_gm = ∇c(wvec(RBe(aux_gm.h_tot - aux_gm.e_kin)))
    @. ∇q_tot_gm = ∇c(wvec(RBq(prog_gm.ρq_tot / ρ_c)))

    if edmf.moisture_model isa TC.NonEquilibriumMoisture
        ∇q_liq_gm = TC.center_aux_grid_mean(state).∇q_liq_gm
        ∇q_ice_gm = TC.center_aux_grid_mean(state).∇q_ice_gm
        q_liq_gm_toa = prog_gm.q_liq[kc_toa]
        q_ice_gm_toa = prog_gm.q_ice[kc_toa]
        RBq_liq = CCO.RightBiasedC2F(; top = CCO.SetValue(q_liq_gm_toa))
        RBq_ice = CCO.RightBiasedC2F(; top = CCO.SetValue(q_ice_gm_toa))
        @. ∇q_liq_gm = ∇c(wvec(RBq(prog_gm.q_liq)))
        @. ∇q_ice_gm = ∇c(wvec(RBq(prog_gm.q_ice)))
    end

    # Apply forcing and radiation
    prog_gm_uₕ = TC.grid_mean_uₕ(state)
    aux_gm_uₕ_g = TC.grid_mean_uₕ_g(state)
    # prog_gm_v = TC.grid_mean_v(state)
    tendencies_gm_uₕ = TC.tendencies_grid_mean_uₕ(state)
    # tendencies_gm_v = TC.tendencies_grid_mean_v(state)
    prog_gm_u = TC.physical_grid_mean_u(state)
    prog_gm_v = TC.physical_grid_mean_v(state)

    # Coriolis
    coriolis_param = force.coriolis_param
    # TODO: union split over sphere or box
    # lat = CC.Fields.coordinate_field(axes(ρ_c)).lat
    coords = CC.Fields.coordinate_field(axes(ρ_c))
    coriolis_fn(coord) = CCG.WVector(coriolis_param)
    f = @. CCG.Contravariant3Vector(coriolis_fn(coords))

    C123 = CCG.Covariant123Vector
    C12 = CCG.Contravariant12Vector
    lg = CC.Fields.local_geometry_field(axes(ρ_c))
    @. tendencies_gm_uₕ -= f × (C12(C123(prog_gm_uₕ)) - C12(C123(aux_gm_uₕ_g)))


    @inbounds for k in TC.real_center_indices(grid)
        cp_m = TD.cp_m(thermo_params, ts_gm[k])
        cp_v = TCP.cp_v(param_set)
        cv_v = TCP.cv_v(param_set)
        R_v = TCP.R_v(param_set)
        cv_m = TD.cv_m(thermo_params, ts_gm[k])
        h_v = cv_v * (aux_gm.T[k] - T_0) + Lv_0 - R_v * T_0

        # LS Subsidence
        tendencies_gm.ρe_tot[k] -= ρ_c[k] * aux_gm.subsidence[k] * ∇MSE_gm[k]
        tendencies_gm.ρq_tot[k] -= ρ_c[k] * aux_gm.subsidence[k] * ∇q_tot_gm[k]
        if edmf.moisture_model isa TC.NonEquilibriumMoisture
            tendencies_gm.q_liq[k] -= ∇q_liq_gm[k] * aux_gm.subsidence[k]
            tendencies_gm.q_ice[k] -= ∇q_ice_gm[k] * aux_gm.subsidence[k]
        end
        # Radiation
        if Cases.rad_type(radiation) <: Union{Cases.RadiationDYCOMS_RF01, Cases.RadiationLES, Cases.RadiationTRMM_LBA}
            tendencies_gm.ρe_tot[k] += ρ_c[k] * cv_m * aux_gm.dTdt_rad[k]
        end
        # LS advection
        tendencies_gm.ρq_tot[k] += ρ_c[k] * aux_gm.dqtdt_hadv[k]
        if !(Cases.force_type(force) <: Cases.ForcingDYCOMS_RF01)
            tendencies_gm.ρe_tot[k] += ρ_c[k] * (cp_m * aux_gm.dTdt_hadv[k] + h_v * aux_gm.dqtdt_hadv[k])
        end
        if edmf.moisture_model isa TC.NonEquilibriumMoisture
            tendencies_gm.q_liq[k] += aux_gm.dqldt[k]
            tendencies_gm.q_ice[k] += aux_gm.dqidt[k]
        end

        # LES specific forcings
        if Cases.force_type(force) <: Cases.ForcingLES
            T_fluc = aux_gm.dTdt_fluc[k]

            gm_U_nudge_k = (aux_gm.u_nudge[k] - prog_gm_u[k]) / force.wind_nudge_τᵣ
            gm_V_nudge_k = (aux_gm.v_nudge[k] - prog_gm_v[k]) / force.wind_nudge_τᵣ

            Γᵣ = compute_les_Γᵣ(grid.zc[k].z, force.scalar_nudge_τᵣ, force.scalar_nudge_zᵢ, force.scalar_nudge_zᵣ)
            gm_T_nudge_k = Γᵣ * (aux_gm.T_nudge[k] - aux_gm.T[k])
            gm_q_tot_nudge_k = Γᵣ * (aux_gm.qt_nudge[k] - aux_gm.q_tot[k])
            if edmf.moisture_model isa TC.NonEquilibriumMoisture
                gm_q_liq_nudge_k = Γᵣ * (aux_gm.ql_nudge[k] - prog_gm.q_liq[k])
                gm_q_ice_nudge_k = Γᵣ * (aux_gm.qi_nudge[k] - prog_gm.q_ice[k])
            end

            tendencies_gm.ρq_tot[k] += ρ_c[k] * (gm_q_tot_nudge_k + aux_gm.dqtdt_fluc[k])
            tendencies_gm.ρe_tot[k] +=
                ρ_c[k] * (cv_m * (gm_T_nudge_k + T_fluc) + h_v * (gm_q_tot_nudge_k + aux_gm.dqtdt_fluc[k]))
            tendencies_gm_uₕ[k] += CCG.Covariant12Vector(CCG.UVVector(gm_U_nudge_k, gm_V_nudge_k), lg[k])
            if edmf.moisture_model isa TC.NonEquilibriumMoisture
                tendencies_gm.q_liq[k] += aux_gm.dqldt_hadv[k] + gm_q_liq_nudge_k + aux_gm.dqldt_fluc[k]
                tendencies_gm.q_ice[k] += aux_gm.dqidt_hadv[k] + gm_q_ice_nudge_k + aux_gm.dqidt_fluc[k]
            end
        end

        # Apply precipitation tendencies
        tendencies_gm.ρq_tot[k] +=
            ρ_c[k] * (
                aux_bulk.qt_tendency_precip_formation[k] +
                aux_en.qt_tendency_precip_formation[k] +
                aux_tc.qt_tendency_precip_sinks[k]
            )

        tendencies_gm.ρe_tot[k] +=
            ρ_c[k] * (
                aux_bulk.e_tot_tendency_precip_formation[k] +
                aux_en.e_tot_tendency_precip_formation[k] +
                aux_tc.e_tot_tendency_precip_sinks[k]
            )

        if edmf.moisture_model isa TC.NonEquilibriumMoisture
            tendencies_gm.q_liq[k] += aux_bulk.ql_tendency_precip_formation[k] + aux_en.ql_tendency_precip_formation[k]
            tendencies_gm.q_ice[k] += aux_bulk.qi_tendency_precip_formation[k] + aux_en.qi_tendency_precip_formation[k]
        end
    end
    TC.compute_sgs_flux!(edmf, grid, state, surf, param_set)
    sgs_flux_h_tot = aux_gm_f.sgs_flux_h_tot
    sgs_flux_q_tot = aux_gm_f.sgs_flux_q_tot
    sgs_flux_uₕ = aux_gm_f.sgs_flux_uₕ
    tends_ρe_tot = tendencies_gm.ρe_tot
    tends_ρq_tot = tendencies_gm.ρq_tot
    tends_uₕ = TC.tendencies_grid_mean_uₕ(state)

    ∇sgs = CCO.DivergenceF2C()
    @. tends_ρe_tot += -∇sgs(wvec(sgs_flux_h_tot))
    @. tends_ρq_tot += -∇sgs(wvec(sgs_flux_q_tot))
    @. tends_uₕ += -∇sgs(sgs_flux_uₕ) / ρ_c

    if edmf.moisture_model isa TC.NonEquilibriumMoisture
        sgs_flux_q_liq = aux_gm_f.sgs_flux_q_liq
        sgs_flux_q_ice = aux_gm_f.sgs_flux_q_ice

        tends_q_liq = tendencies_gm.q_liq
        tends_q_ice = tendencies_gm.q_ice
        @. tends_q_liq += -∇sgs(wvec(sgs_flux_q_liq)) / ρ_c
        @. tends_q_ice += -∇sgs(wvec(sgs_flux_q_ice)) / ρ_c
    end

    return nothing
end
