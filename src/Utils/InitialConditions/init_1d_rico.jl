import TurbulenceConvection
const TC = TurbulenceConvection
import Thermodynamics
const TD = Thermodynamics

function initialize_rico(sim::Simulation1d)
    TC = TurbulenceConvection
    state = sim.state
    grid = sim.grid
    FT = eltype(grid)
    t = FT(0)
    gm = sim.gm
    edmf = sim.edmf
    case = sim.case

    param_set = TC.parameter_set(gm)
    aux_bulk = TC.center_aux_bulk(state)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    aux_tc = TC.center_aux_turbconv(state)
    p0 = TC.center_ref_state(state).p0
    p0_c = TC.center_ref_state(state).p0
    ρ0_c = TC.center_ref_state(state).ρ0
    kc_surf = TC.kc_surface(grid)
    prog_en = TC.center_prog_environment(state)
    aux_en = TC.center_aux_environment(state)
    N_up = TC.n_updrafts(edmf)
    aux_up = TC.center_aux_updrafts(state)
    aux_up_f = TC.face_aux_updrafts(state)
    prog_up = TC.center_prog_updrafts(state)
    prog_up_f = TC.face_prog_updrafts(state)

    @inbounds for k in TC.real_center_indices(grid)
        z = grid.zc[k]
        prog_gm.u[k] = -9.9 + 2.0e-3 * z
        prog_gm.v[k] = -3.8
        #Set Thetal profile
        prog_gm.θ_liq_ice[k] = if z <= 740.0
            297.9
        else
            297.9 + (317.0 - 297.9) / (4000.0 - 740.0) * (z - 740.0)
        end

        #Set qt profile
        prog_gm.q_tot[k] = if z <= 740.0
            (16.0 + (13.8 - 16.0) / 740.0 * z) / 1000.0
        elseif z > 740.0 && z <= 3260.0
            (13.8 + (2.4 - 13.8) / (3260.0 - 740.0) * (z - 740.0)) / 1000.0
        else
            (2.4 + (1.8 - 2.4) / (4000.0 - 3260.0) * (z - 3260.0)) / 1000.0
        end

        ts = TC.thermo_state_pθq(param_set, p0[k], prog_gm.θ_liq_ice[k], prog_gm.q_tot[k])
        aux_tc.θ_virt[k] = TD.virtual_pottemp(ts)
    end
    zi = 0.6 * TC.get_inversion(grid, state, param_set, 0.2)

    @inbounds for k in TC.real_center_indices(grid)
        z = grid.zc[k]
        aux_gm.tke[k] = if z <= zi
            1.0 - z / zi
        else
            0.0
        end

        θ_liq_ice = prog_gm.θ_liq_ice[k]
        q_tot = prog_gm.q_tot[k]
        ts = TC.thermo_state_pθq(param_set, p0_c[k], θ_liq_ice, q_tot)
        aux_gm.q_liq[k] = TD.liquid_specific_humidity(ts)
        aux_gm.q_ice[k] = TD.ice_specific_humidity(ts)
        aux_gm.T[k] = TD.air_temperature(ts)
        ρ = TD.air_density(ts)
        aux_gm.buoy[k] = TC.buoyancy_c(param_set, ρ0_c[k], ρ)
        aux_gm.RH[k] = TD.relative_humidity(ts)

        Π = TD.exner(ts)
        # Geostrophic velocity profiles
        aux_gm.ug[k] = -9.9 + 2.0e-3 * z
        aux_gm.vg[k] = -3.8
        # Set large-scale cooling
        aux_gm.dTdt[k] = (-2.5 / (3600.0 * 24.0)) * Π

        # Set large-scale moistening
        aux_gm.dqtdt[k] = if z <= 2980.0
            (-1.0 + 1.3456 / 2980.0 * z) / 86400.0 / 1000.0   #kg/(kg * s)
        else
            0.3456 / 86400.0 / 1000.0
        end

        #Set large scale subsidence
        aux_gm.subsidence[k] = if z <= 2260.0
            -(0.005 / 2260.0) * z
        else
            -0.005
        end
    end

    # initialize_radiation(sim.case, sim.grid, state, sim.gm, sim.param_set)

    ae = 1 .- aux_bulk.area # area of environment

    aux_en.tke .= aux_gm.tke
    prog_en.ρatke .= aux_en.tke .* ρ0_c .* ae

    TC.get_GMV_CoVar(edmf, grid, state, Val(:tke), Val(:w), Val(:w))
    aux_gm.Hvar .= aux_gm.Hvar[kc_surf] .* aux_gm.tke
    aux_gm.QTvar .= aux_gm.QTvar[kc_surf] .* aux_gm.tke
    aux_gm.HQTcov .= aux_gm.HQTcov[kc_surf] .* aux_gm.tke

    prog_en.ρaHvar .= aux_gm.Hvar .* ρ0_c .* ae
    prog_en.ρaQTvar .= aux_gm.QTvar .* ρ0_c .* ae
    prog_en.ρaHQTcov .= aux_gm.HQTcov .* ρ0_c .* ae

    surf_params = case.surf_params
    parent(aux_tc.prandtl_nvec) .= edmf.prandtl_number
    @inbounds for k in TC.real_center_indices(grid)
        ts = TC.thermo_state_pθq(param_set, p0_c[k], prog_gm.θ_liq_ice[k], prog_gm.q_tot[k])
        aux_tc.θ_virt[k] = TD.virtual_pottemp(ts)
    end
    surf = get_surface(surf_params, grid, state, gm, t, param_set)

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
            aux_up[i].q_tot[k] = prog_gm.q_tot[k]
            aux_up[i].θ_liq_ice[k] = prog_gm.θ_liq_ice[k]
            aux_up[i].q_liq[k] = aux_gm.q_liq[k]
            aux_up[i].q_ice[k] = aux_gm.q_ice[k]
            aux_up[i].T[k] = aux_gm.T[k]
            prog_up[i].ρarea[k] = 0
            prog_up[i].ρaq_tot[k] = 0
            prog_up[i].ρaθ_liq_ice[k] = 0
        end

        a_surf = TC.area_surface_bc(surf, edmf, i)
        aux_up[i].area[kc_surf] = a_surf
        prog_up[i].ρarea[kc_surf] = ρ0_c[kc_surf] * a_surf
    end

    TC.set_edmf_surface_bc(edmf, grid, state, surf, gm)

    sim.skip_io && return nothing
    initialize_io(sim.io_nt.ref_state, sim.Stats)
    io(sim.io_nt.ref_state, sim.Stats, state) # since the reference prog is static

    initialize_io(sim.io_nt.aux, sim.Stats)
    initialize_io(sim.io_nt.diagnostics, sim.Stats)

    # TODO: deprecate
    initialize_io(sim.gm, sim.Stats)
    initialize_io(sim.edmf, sim.Stats)

    open_files(sim.Stats)
    write_simulation_time(sim.Stats, t)

    io(sim.io_nt.aux, sim.Stats, state)
    io(sim.io_nt.diagnostics, sim.Stats, sim.diagnostics)

    # TODO: deprecate
    io(surf, sim.case.surf_params, sim.grid, state, sim.Stats, t)
    close_files(sim.Stats)

    return
end
