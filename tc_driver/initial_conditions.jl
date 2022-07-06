import TurbulenceConvection as TC
import TurbulenceConvection.Parameters as TCP
const APS = TCP.AbstractTurbulenceConvectionParameters

import Thermodynamics as TD

function initialize_edmf(edmf::TC.EDMFModel, grid::TC.Grid, state::TC.State, surf_params, param_set::APS, t::Real, case)
    thermo_params = TCP.thermodynamics_params(param_set)
    initialize_covariance(edmf, grid, state)
    aux_gm = TC.center_aux_grid_mean(state)
    ts_gm = aux_gm.ts
    @. aux_gm.θ_virt = TD.virtual_pottemp(thermo_params, ts_gm)
    surf = get_surface(surf_params, grid, state, t, param_set)
    if case isa Cases.DryBubble
        initialize_updrafts_DryBubble(edmf, grid, state)
    else
        initialize_updrafts(edmf, grid, state, surf)
    end
    TC.set_edmf_surface_bc(edmf, grid, state, surf, param_set)
    return
end

function initialize_covariance(edmf::TC.EDMFModel, grid::TC.Grid, state::TC.State)

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
    @inbounds for i in 1:N_up
        @inbounds for k in TC.real_face_indices(grid)
            aux_up_f[i].w[k] = 0
            prog_up_f[i].ρaw[k] = 0
        end

        @inbounds for k in TC.real_center_indices(grid)
            aux_up[i].buoy[k] = 0
            # Simple treatment for now, revise when multiple updraft closures
            # become more well defined
            aux_up[i].area[k] = 0
            aux_up[i].q_tot[k] = aux_gm.q_tot[k]
            aux_up[i].θ_liq_ice[k] = aux_gm.θ_liq_ice[k]
            aux_up[i].q_liq[k] = aux_gm.q_liq[k]
            aux_up[i].q_ice[k] = aux_gm.q_ice[k]
            aux_up[i].T[k] = aux_gm.T[k]
            prog_up[i].ρarea[k] = 0
            prog_up[i].ρaq_tot[k] = 0
            prog_up[i].ρaθ_liq_ice[k] = 0
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

import AtmosphericProfilesLibrary
const APL = AtmosphericProfilesLibrary
function initialize_updrafts_DryBubble(edmf, grid, state)

    # criterion 2: b>1e-4
    aux_up = TC.center_aux_updrafts(state)
    aux_up_f = TC.face_aux_updrafts(state)
    aux_gm = TC.center_aux_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    prog_up = TC.center_prog_updrafts(state)
    prog_up_f = TC.face_prog_updrafts(state)
    ρ_0_c = prog_gm.ρ
    ρ_0_f = aux_gm_f.ρ
    N_up = TC.n_updrafts(edmf)
    FT = TC.float_type(state)
    z_in = APL.DryBubble_updrafts_z(FT)
    z_min, z_max = first(z_in), last(z_in)
    prof_θ_liq_ice = APL.DryBubble_updrafts_θ_liq_ice(FT)
    prof_area = APL.DryBubble_updrafts_area(FT)
    prof_w = APL.DryBubble_updrafts_w(FT)
    prof_T = APL.DryBubble_updrafts_T(FT)
    face_bcs = (; bottom = CCO.SetValue(FT(0)), top = CCO.SetValue(FT(0)))
    If = CCO.InterpolateC2F(; face_bcs...)
    @inbounds for i in 1:N_up
        @inbounds for k in TC.real_face_indices(grid)
            if z_min <= grid.zf[k].z <= z_max
                aux_up_f[i].w[k] = eps(FT) # non zero value to aviod zeroing out fields in the filter
            end
        end

        @inbounds for k in TC.real_center_indices(grid)
            z = grid.zc[k].z
            if z_min <= z <= z_max
                aux_up[i].area[k] = prof_area(z)
                aux_up[i].θ_liq_ice[k] = prof_θ_liq_ice(z)
                aux_up[i].q_tot[k] = 0.0
                aux_up[i].q_liq[k] = 0.0
                aux_up[i].q_ice[k] = 0.0

                # for now temperature is provided as diagnostics from LES
                aux_up[i].T[k] = prof_T(z)
                prog_up[i].ρarea[k] = ρ_0_c[k] * aux_up[i].area[k]
                prog_up[i].ρaθ_liq_ice[k] = prog_up[i].ρarea[k] * aux_up[i].θ_liq_ice[k]
                prog_up[i].ρaq_tot[k] = prog_up[i].ρarea[k] * aux_up[i].q_tot[k]
            else
                aux_up[i].area[k] = 0.0
                aux_up[i].θ_liq_ice[k] = aux_gm.θ_liq_ice[k]
                aux_up[i].T[k] = aux_gm.T[k]
                prog_up[i].ρarea[k] = 0.0
                prog_up[i].ρaθ_liq_ice[k] = 0.0
                prog_up[i].ρaq_tot[k] = 0.0
            end
        end
        @. prog_up_f[i].ρaw = If(prog_up[i].ρarea) * aux_up_f[i].w
    end
    return nothing
end
