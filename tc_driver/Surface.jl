import StaticArrays as SA
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF

function get_surface(
    surf_params::TC.FixedSurfaceFlux,
    grid::TC.Grid,
    state::TC.State,
    t::Real,
    param_set::TCP.AbstractTurbulenceConvectionParameters,
)
    FT = TC.float_type(state)
    surf_flux_params = TCP.surface_fluxes_params(param_set)
    kc_surf = TC.kc_surface(grid)
    kf_surf = TC.kf_surface(grid)
    z_sfc = FT(0)
    z_in = grid.zc[kc_surf].z
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    p_f_surf = aux_gm_f.p[kf_surf]
    Tsurface = TC.surface_temperature(surf_params, t)
    qsurface = TC.surface_q_tot(surf_params, t)
    shf = TC.sensible_heat_flux(surf_params, t)
    lhf = TC.latent_heat_flux(surf_params, t)
    Ri_bulk_crit = surf_params.Ri_bulk_crit
    zrough = surf_params.zrough
    thermo_params = TCP.thermodynamics_params(param_set)

    ts_sfc = TD.PhaseEquil_pTq(thermo_params, p_f_surf, Tsurface, qsurface)
    ts_in = aux_gm.ts[kc_surf]
    scheme = SF.FVScheme()

    bflux = SF.compute_buoyancy_flux(surf_flux_params, shf, lhf, ts_in, ts_sfc, scheme)
    zi = TC.get_inversion(grid, state, param_set, Ri_bulk_crit)
    convective_vel = TC.get_wstar(bflux, zi) # yair here zi in TRMM should be adjusted

    u_sfc = SA.SVector{2, FT}(0, 0)
    # TODO: make correct with topography
    uₕ_gm_surf = TC.physical_grid_mean_uₕ(state)[kc_surf]
    u_in = SA.SVector{2, FT}(uₕ_gm_surf.u, uₕ_gm_surf.v)
    vals_sfc = SF.SurfaceValues(z_sfc, u_sfc, ts_sfc)
    vals_int = SF.InteriorValues(z_in, u_in, ts_in)
    kwargs = (;
        state_in = vals_int,
        state_sfc = vals_sfc,
        shf = shf,
        lhf = lhf,
        z0m = zrough,
        z0b = zrough,
        gustiness = convective_vel,
    )
    sc = if TC.fixed_ustar(surf_params)
        SF.FluxesAndFrictionVelocity{FT}(; kwargs..., ustar = surf_params.ustar)
    else
        SF.Fluxes{FT}(; kwargs...)
    end
    result = SF.surface_conditions(surf_flux_params, sc, scheme)
    return TC.SurfaceBase{FT}(;
        shf = shf,
        lhf = lhf,
        ustar = TC.fixed_ustar(surf_params) ? surf_params.ustar : result.ustar,
        bflux = bflux,
        obukhov_length = result.L_MO,
        cm = result.Cd,
        ch = result.Ch,
        ρu_flux = surf_params.zero_uv_fluxes ? FT(0) : result.ρτxz,
        ρv_flux = surf_params.zero_uv_fluxes ? FT(0) : result.ρτyz,
        ρe_tot_flux = shf + lhf,
        ρq_tot_flux = lhf / TD.latent_heat_vapor(thermo_params, ts_in),
        wstar = convective_vel,
        ρq_liq_flux = FT(0),
        ρq_ice_flux = FT(0),
    )
end

function get_surface(
    surf_params::TC.FixedSurfaceCoeffs,
    grid::TC.Grid,
    state::TC.State,
    t::Real,
    param_set::TCP.AbstractTurbulenceConvectionParameters,
)
    FT = TC.float_type(state)
    surf_flux_params = TCP.surface_fluxes_params(param_set)
    kc_surf = TC.kc_surface(grid)
    kf_surf = TC.kf_surface(grid)
    aux_gm_f = TC.face_aux_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    Tsurface = TC.surface_temperature(surf_params, t)
    qsurface = TC.surface_q_tot(surf_params, t)
    p_f_surf = aux_gm_f.p[kf_surf]
    zrough = surf_params.zrough
    zc_surf = grid.zc[kc_surf].z
    cm = surf_params.cm(zc_surf)
    ch = surf_params.ch(zc_surf)
    Ri_bulk_crit = surf_params.Ri_bulk_crit
    thermo_params = TCP.thermodynamics_params(param_set)

    scheme = SF.FVScheme()
    z_sfc = FT(0)
    z_in = grid.zc[kc_surf].z
    ts_sfc = TD.PhaseEquil_pθq(thermo_params, p_f_surf, Tsurface, qsurface)
    ts_in = aux_gm.ts[kc_surf]
    u_sfc = SA.SVector{2, FT}(0, 0)
    # TODO: make correct with topography
    uₕ_gm_surf = TC.physical_grid_mean_uₕ(state)[kc_surf]
    u_in = SA.SVector{2, FT}(uₕ_gm_surf.u, uₕ_gm_surf.v)
    vals_sfc = SF.SurfaceValues(z_sfc, u_sfc, ts_sfc)
    vals_int = SF.InteriorValues(z_in, u_in, ts_in)
    sc = SF.Coefficients{FT}(state_in = vals_int, state_sfc = vals_sfc, Cd = cm, Ch = ch, z0m = zrough, z0b = zrough)
    result = SF.surface_conditions(surf_flux_params, sc, scheme)
    lhf = result.lhf
    shf = result.shf

    zi = TC.get_inversion(grid, state, param_set, Ri_bulk_crit)
    convective_vel = TC.get_wstar(result.buoy_flux, zi)
    return TC.SurfaceBase{FT}(;
        cm = result.Cd,
        ch = result.Ch,
        obukhov_length = result.L_MO,
        lhf = lhf,
        shf = shf,
        ustar = result.ustar,
        ρu_flux = result.ρτxz,
        ρv_flux = result.ρτyz,
        ρe_tot_flux = shf + lhf,
        ρq_tot_flux = lhf / TD.latent_heat_vapor(thermo_params, ts_in),
        bflux = result.buoy_flux,
        wstar = convective_vel,
        ρq_liq_flux = FT(0),
        ρq_ice_flux = FT(0),
    )
end

function get_surface(
    surf_params::TC.MoninObukhovSurface,
    grid::TC.Grid,
    state::TC.State,
    t::Real,
    param_set::TCP.AbstractTurbulenceConvectionParameters,
)
    surf_flux_params = TCP.surface_fluxes_params(param_set)
    kc_surf = TC.kc_surface(grid)
    kf_surf = TC.kf_surface(grid)
    FT = TC.float_type(state)
    z_sfc = FT(0)
    z_in = grid.zc[kc_surf].z
    prog_gm = TC.center_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    p_f_surf = aux_gm_f.p[kf_surf]
    ts_gm = aux_gm.ts
    Tsurface = TC.surface_temperature(surf_params, t)
    qsurface = TC.surface_q_tot(surf_params, t)
    zrough = surf_params.zrough
    Ri_bulk_crit = surf_params.Ri_bulk_crit
    thermo_params = TCP.thermodynamics_params(param_set)

    scheme = SF.FVScheme()
    ts_sfc = TD.PhaseEquil_pTq(thermo_params, p_f_surf, Tsurface, qsurface)
    ts_in = ts_gm[kc_surf]

    u_sfc = SA.SVector{2, FT}(0, 0)
    # TODO: make correct with topography
    uₕ_gm_surf = TC.physical_grid_mean_uₕ(state)[kc_surf]
    u_in = SA.SVector{2, FT}(uₕ_gm_surf.u, uₕ_gm_surf.v)
    vals_sfc = SF.SurfaceValues(z_sfc, u_sfc, ts_sfc)
    vals_int = SF.InteriorValues(z_in, u_in, ts_in)
    sc = SF.ValuesOnly{FT}(state_in = vals_int, state_sfc = vals_sfc, z0m = zrough, z0b = zrough)
    result = SF.surface_conditions(surf_flux_params, sc, scheme)
    lhf = result.lhf
    shf = result.shf
    zi = TC.get_inversion(grid, state, param_set, Ri_bulk_crit)
    convective_vel = TC.get_wstar(result.buoy_flux, zi)
    return TC.SurfaceBase{FT}(;
        cm = result.Cd,
        ch = result.Ch,
        obukhov_length = result.L_MO,
        lhf = lhf,
        shf = shf,
        ustar = result.ustar,
        ρu_flux = result.ρτxz,
        ρv_flux = result.ρτyz,
        ρe_tot_flux = shf + lhf,
        ρq_tot_flux = lhf / TD.latent_heat_vapor(thermo_params, ts_in),
        bflux = result.buoy_flux,
        wstar = convective_vel,
        ρq_liq_flux = FT(0),
        ρq_ice_flux = FT(0),
    )
end
