import UnPack
import LinearAlgebra as LA
import LinearAlgebra: ×

import ClimaAtmos as CA
import ClimaAtmos.TurbulenceConvection as TC
import ClimaAtmos.TurbulenceConvection.Parameters as TCP
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
function compute_ref_state!(
    state,
    grid::TC.Grid,
    param_set::PS;
    ts_g,
) where {PS}
    aux_gm = TC.center_aux_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    p_c = TC.center_aux_grid_mean_p(state)
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

    zc = CC.Fields.coordinate_field(axes(ρ_c)).z
    zf = CC.Fields.coordinate_field(axes(ρ_f)).z

    parent(p_c) .= sol.(vec(zc))
    parent(p_f) .= sol.(vec(zf))

    p_f .= exp.(p_f)
    p_c .= exp.(p_c)

    # Compute reference state thermodynamic profiles
    @. ρ_c = TD.air_density(
        thermo_params,
        TD.PhaseEquil_phq(
            thermo_params,
            p_c,
            TC.enthalpy(mse_g, TC.geopotential(param_set, zc)),
            qtg,
        ),
    )
    @. ρ_f = TD.air_density(
        thermo_params,
        TD.PhaseEquil_phq(
            thermo_params,
            p_f,
            TC.enthalpy(mse_g, TC.geopotential(param_set, zf)),
            qtg,
        ),
    )
    return nothing
end

function set_thermo_state_pθq!(Y, p, colidx)
    (; edmf_cache, params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; moisture_model) = edmf_cache.edmf
    ᶜts_gm = p.ᶜts[colidx]
    ᶜρ = Y.c.ρ[colidx]
    ᶜp = p.ᶜp[colidx]
    ρq_tot = Y.c.ρq_tot[colidx]
    θ_liq_ice = edmf_cache.aux.cent.θ_liq_ice[colidx]

    @assert moisture_model isa CA.EquilMoistModel "TODO: add non-equilibrium moisture model support"

    @. ᶜts_gm = TD.PhaseEquil_pθq(thermo_params, ᶜp, θ_liq_ice, ρq_tot / ᶜρ)
    nothing
end

function set_grid_mean_from_thermo_state!(param_set, state, grid)
    thermo_params = TCP.thermodynamics_params(param_set)
    Ic = CCO.InterpolateF2C()
    If = CCO.InterpolateC2F(bottom = CCO.Extrapolate(), top = CCO.Extrapolate())
    ts_gm = TC.center_aux_grid_mean_ts(state)
    prog_gm = TC.center_prog_grid_mean(state)
    prog_gm_f = TC.face_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    prog_gm_uₕ = TC.grid_mean_uₕ(state)

    @. prog_gm.ρ = TD.air_density(thermo_params, ts_gm)
    ρ_c = prog_gm.ρ
    ρ_f = aux_gm_f.ρ

    C123 = CCG.Covariant123Vector
    @. prog_gm.ρe_tot =
        ρ_c * TD.total_energy(
            thermo_params,
            ts_gm,
            LA.norm_sqr(C123(prog_gm_uₕ) + C123(Ic(prog_gm_f.w))) / 2,
            TC.geopotential(param_set, grid.zc.z),
        )

    @. prog_gm.ρq_tot = ρ_c * aux_gm.q_tot
    @. ρ_f = If(ρ_c)

    return nothing
end

function assign_thermo_aux!(state, grid, moisture_model, param_set)
    If = CCO.InterpolateC2F(bottom = CCO.Extrapolate(), top = CCO.Extrapolate())
    thermo_params = TCP.thermodynamics_params(param_set)
    aux_gm = TC.center_aux_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ts_gm = TC.center_aux_grid_mean_ts(state)
    p_c = TC.center_aux_grid_mean_p(state)
    ρ_c = prog_gm.ρ
    ρ_f = aux_gm_f.ρ
    @. ρ_f = If(ρ_c)

    @inbounds for k in TC.real_center_indices(grid)
        ts = ts_gm[k]
        aux_gm.q_tot[k] = prog_gm.ρq_tot[k] / ρ_c[k]
        aux_gm.q_liq[k] = TD.liquid_specific_humidity(thermo_params, ts)
        aux_gm.q_ice[k] = TD.ice_specific_humidity(thermo_params, ts)
        aux_gm.T[k] = TD.air_temperature(thermo_params, ts)
        aux_gm.RH[k] = TD.relative_humidity(thermo_params, ts)
        aux_gm.θ_liq_ice[k] = TD.liquid_ice_pottemp(thermo_params, ts)
        aux_gm.h_tot[k] =
            TC.total_enthalpy(param_set, prog_gm.ρe_tot[k] / ρ_c[k], ts)
        p_c[k] = TD.air_pressure(thermo_params, ts)
        aux_gm.θ_virt[k] = TD.virtual_pottemp(thermo_params, ts)
    end
    return
end

function compute_gm_tendencies!(
    edmf::TC.EDMFModel,
    grid::TC.Grid,
    state::TC.State,
    surf::TC.SurfaceBase,
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
    p_c = TC.center_aux_grid_mean_p(state)
    ρ_c = prog_gm.ρ
    aux_tc = TC.center_aux_turbconv(state)
    ts_gm = TC.center_aux_grid_mean_ts(state)
    ᶜK = TC.center_aux_grid_mean_e_kin(state)

    MSE_gm_toa = aux_gm.h_tot[kc_toa] - ᶜK[kc_toa]
    q_tot_gm_toa = prog_gm.ρq_tot[kc_toa] / ρ_c[kc_toa]
    RBe = CCO.RightBiasedC2F(; top = CCO.SetValue(MSE_gm_toa))
    RBq = CCO.RightBiasedC2F(; top = CCO.SetValue(q_tot_gm_toa))
    wvec = CC.Geometry.WVector
    ∇c = CCO.DivergenceF2C()
    @. ∇MSE_gm = ∇c(wvec(RBe(aux_gm.h_tot - ᶜK)))
    @. ∇q_tot_gm = ∇c(wvec(RBq(prog_gm.ρq_tot / ρ_c)))

    if edmf.moisture_model isa CA.NonEquilMoistModel
        ∇q_liq_gm = TC.center_aux_grid_mean(state).∇q_liq_gm
        ∇q_ice_gm = TC.center_aux_grid_mean(state).∇q_ice_gm
        q_liq_gm_toa = prog_gm.q_liq[kc_toa]
        q_ice_gm_toa = prog_gm.q_ice[kc_toa]
        RBq_liq = CCO.RightBiasedC2F(; top = CCO.SetValue(q_liq_gm_toa))
        RBq_ice = CCO.RightBiasedC2F(; top = CCO.SetValue(q_ice_gm_toa))
        @. ∇q_liq_gm = ∇c(wvec(RBq(prog_gm.q_liq)))
        @. ∇q_ice_gm = ∇c(wvec(RBq(prog_gm.q_ice)))
    end

    # Apply forcing
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

    cp_v = TCP.cp_v(param_set)
    cv_v = TCP.cv_v(param_set)
    R_v = TCP.R_v(param_set)

    # LS Subsidence
    @. tendencies_gm.ρe_tot -= ρ_c * aux_gm.subsidence * ∇MSE_gm
    @. tendencies_gm.ρq_tot -= ρ_c * aux_gm.subsidence * ∇q_tot_gm
    if edmf.moisture_model isa CA.NonEquilMoistModel
        @. tendencies_gm.q_liq -= ∇q_liq_gm * aux_gm.subsidence
        @. tendencies_gm.q_ice -= ∇q_ice_gm * aux_gm.subsidence
    end
    # LS advection
    @. tendencies_gm.ρq_tot += ρ_c * aux_gm.dqtdt_hadv
    # TODO: should `hv` be a thermo function?
    #     (hv = cv_v * (aux_gm.T - T_0) + Lv_0 - R_v * T_0)
    if !(Cases.force_type(force) <: Cases.ForcingDYCOMS_RF01)
        @. tendencies_gm.ρe_tot +=
            ρ_c * (
                TD.cp_m(thermo_params, ts_gm) * aux_gm.dTdt_hadv +
                (cv_v * (aux_gm.T - T_0) + Lv_0 - R_v * T_0) *
                aux_gm.dqtdt_hadv
            )
    end
    if edmf.moisture_model isa CA.NonEquilMoistModel
        @. tendencies_gm.q_liq += aux_gm.dqldt
        @. tendencies_gm.q_ice += aux_gm.dqidt
    end

    # Apply precipitation tendencies
    @. tendencies_gm.ρq_tot +=
        ρ_c * (
            aux_bulk.qt_tendency_precip_formation +
            aux_en.qt_tendency_precip_formation +
            aux_tc.qt_tendency_precip_sinks
        )

    @. tendencies_gm.ρe_tot +=
        ρ_c * (
            aux_bulk.e_tot_tendency_precip_formation +
            aux_en.e_tot_tendency_precip_formation +
            aux_tc.e_tot_tendency_precip_sinks
        )
    if edmf.moisture_model isa CA.NonEquilMoistModel
        @. tendencies_gm.q_liq +=
            aux_bulk.ql_tendency_precip_formation +
            aux_en.ql_tendency_precip_formation
        @. tendencies_gm.q_ice +=
            aux_bulk.qi_tendency_precip_formation +
            aux_en.qi_tendency_precip_formation

        # Additionally apply cloud liquid and ice formation tendencies
        @. tendencies_gm.q_liq +=
            aux_bulk.ql_tendency_noneq + aux_en.ql_tendency_noneq
        @. tendencies_gm.q_ice +=
            aux_bulk.qi_tendency_noneq + aux_en.qi_tendency_noneq
    end

    TC.compute_sgs_flux!(edmf, grid, state, surf, param_set)

    ∇sgs = CCO.DivergenceF2C()
    @. tendencies_gm.ρe_tot += -∇sgs(wvec(aux_gm_f.sgs_flux_h_tot))
    @. tendencies_gm.ρq_tot += -∇sgs(wvec(aux_gm_f.sgs_flux_q_tot))
    @. tendencies_gm_uₕ += -∇sgs(aux_gm_f.sgs_flux_uₕ) / ρ_c

    if edmf.moisture_model isa CA.NonEquilMoistModel
        @. tendencies_gm.q_liq += -∇sgs(wvec(aux_gm_f.sgs_flux_q_liq)) / ρ_c
        @. tendencies_gm.q_ice += -∇sgs(wvec(aux_gm_f.sgs_flux_q_ice)) / ρ_c
    end

    return nothing
end
