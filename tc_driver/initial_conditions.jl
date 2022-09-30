import ClimaAtmos.TurbulenceConvection as TC
import ClimaAtmos.TurbulenceConvection.Parameters as TCP
const APS = TCP.AbstractTurbulenceConvectionParameters

import Thermodynamics as TD

function initialize_edmf(
    edmf::TC.EDMFModel,
    grid::TC.Grid,
    state::TC.State,
    surf_params,
    param_set::APS,
    t::Real,
    case,
)
    thermo_params = TCP.thermodynamics_params(param_set)
    initialize_covariance(edmf, grid, state)
    aux_gm = TC.center_aux_grid_mean(state)
    ts_gm = TC.center_aux_grid_mean_ts(state)
    @. aux_gm.θ_virt = TD.virtual_pottemp(thermo_params, ts_gm)
    surf = get_surface(surf_params, grid, state, t, param_set)
    initialize_updrafts(edmf, grid, state, surf)
    TC.set_edmf_surface_bc(edmf, grid, state, surf, param_set)
    return
end

function initialize_covariance(
    edmf::TC.EDMFModel,
    grid::TC.Grid,
    state::TC.State,
)

    kc_surf = TC.kc_surface(grid)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_en = TC.center_prog_environment(state)
    aux_en = TC.center_aux_environment(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ
    aux_bulk = TC.center_aux_bulk(state)
    ae = 1 .- aux_bulk.area # area of environment

    aux_en.tke .= aux_gm.tke
    aux_en.Hvar .= aux_gm.Hvar
    aux_en.QTvar .= aux_gm.QTvar
    aux_en.HQTcov .= aux_gm.HQTcov

    prog_en.ρatke .= aux_en.tke .* ρ_c .* ae
    if edmf.thermo_covariance_model isa TC.PrognosticThermoCovariances
        prog_en.ρaHvar .= aux_gm.Hvar .* ρ_c .* ae
        prog_en.ρaQTvar .= aux_gm.QTvar .* ρ_c .* ae
        prog_en.ρaHQTcov .= aux_gm.HQTcov .* ρ_c .* ae
    end
    return
end

function initialize_updrafts(edmf, grid, state, surf)
    FT = TC.float_type(state)
    N_up = TC.n_updrafts(edmf)
    kc_surf = TC.kc_surface(grid)
    aux_up = TC.center_aux_updrafts(state)
    prog_gm = TC.center_prog_grid_mean(state)
    aux_up = TC.center_aux_updrafts(state)
    aux_up_f = TC.face_aux_updrafts(state)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_up = TC.center_prog_updrafts(state)
    prog_up_f = TC.face_prog_updrafts(state)
    ρ_c = prog_gm.ρ
    a_min = edmf.minimum_area
    @inbounds for i in 1:N_up
        @inbounds for k in TC.real_face_indices(grid)
            aux_up_f[i].w[k] = 0
            prog_up_f[i].ρaw[k] = 0
        end

        @inbounds for k in TC.real_center_indices(grid)
            aux_up[i].buoy[k] = 0
            # Simple treatment for now, revise when multiple updraft closures
            # become more well defined
            aux_up[i].area[k] = a_min
            aux_up[i].q_tot[k] = aux_gm.q_tot[k]
            aux_up[i].θ_liq_ice[k] = aux_gm.θ_liq_ice[k]
            aux_up[i].q_liq[k] = aux_gm.q_liq[k]
            aux_up[i].q_ice[k] = aux_gm.q_ice[k]
            aux_up[i].T[k] = aux_gm.T[k]
            prog_up[i].ρarea[k] = ρ_c[k] * aux_up[i].area[k]
            prog_up[i].ρaq_tot[k] = prog_up[i].ρarea[k] * aux_up[i].q_tot[k]
            prog_up[i].ρaθ_liq_ice[k] =
                prog_up[i].ρarea[k] * aux_up[i].θ_liq_ice[k]
        end
        if edmf.entr_closure isa TC.PrognosticNoisyRelaxationProcess
            @. prog_up[i].ε_nondim = 0
            @. prog_up[i].δ_nondim = 0
        end

        a_surf = TC.area_surface_bc(surf, edmf, i)
        aux_up[i].area[kc_surf] = a_surf
        prog_up[i].ρarea[kc_surf] = ρ_c[kc_surf] * a_surf
    end
    return
end
