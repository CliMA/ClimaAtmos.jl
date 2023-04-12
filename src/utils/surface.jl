import StaticArrays as SA
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import ClimaCore.Fields as Fields
using ClimaCore.Utilities: half
import .TurbulenceConvection as TC
import .TurbulenceConvection.Parameters as TCP
import ..SingleColumnModel
import ..SphericalModel

#####
##### TurbulenceConvection surface utility functions
#####

function get_surface(
    surf_params::Union{
        TC.FixedSurfaceFlux,
        TC.FixedSurfaceFluxAndFrictionVelocity,
    },
    grid::TC.Grid,
    state::TC.State,
    t::Real,
    param_set::TCP.AbstractTurbulenceConvectionParameters,
)
    FT = TC.float_type(state)
    surf_flux_params = TCP.surface_fluxes_params(param_set)
    kc_surf = TC.kc_surface(grid)
    z_sfc = FT(0)
    z_in = grid.zc[kc_surf].z
    shf = TC.sensible_heat_flux(surf_params, t)
    lhf = TC.latent_heat_flux(surf_params, t)
    zrough = surf_params.zrough
    thermo_params = TCP.thermodynamics_params(param_set)

    ts_sfc = TC.surface_thermo_state(surf_params, thermo_params, t)
    ts_in = TC.center_aux_grid_mean_ts(state)[kc_surf]
    scheme = SF.FVScheme()

    u_sfc = SA.SVector{2, FT}(0, 0)
    # TODO: make correct with topography
    uₕ_gm_surf = TC.physical_grid_mean_uₕ(state)[kc_surf]
    u_in = SA.SVector{2, FT}(uₕ_gm_surf.u, uₕ_gm_surf.v)
    vals_sfc = SF.SurfaceValues(z_sfc, u_sfc, ts_sfc)
    vals_int = SF.InteriorValues(z_in, u_in, ts_in)

    bflux = SF.compute_buoyancy_flux(
        surf_flux_params,
        shf,
        lhf,
        ts_in,
        ts_sfc,
        scheme,
    )
    # TODO: In wstar computation the mixed layer depth zi is assumed to be 1km.
    # This should be adjusted in deep convective cases like TRMM.
    convective_vel = TC.get_wstar(bflux)

    kwargs = (;
        state_in = vals_int,
        state_sfc = vals_sfc,
        shf = shf,
        lhf = lhf,
        z0m = zrough,
        z0b = zrough,
        gustiness = convective_vel,
    )
    sc = if surf_params isa TC.FixedSurfaceFluxAndFrictionVelocity
        SF.FluxesAndFrictionVelocity{FT}(; kwargs..., ustar = surf_params.ustar)
    else
        SF.Fluxes{FT}(; kwargs...)
    end
    return SF.surface_conditions(surf_flux_params, sc, scheme)
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
    zrough = surf_params.zrough
    zc_surf = grid.zc[kc_surf].z
    cm = surf_params.cm(zc_surf)
    ch = surf_params.ch(zc_surf)
    thermo_params = TCP.thermodynamics_params(param_set)

    scheme = SF.FVScheme()
    z_sfc = FT(0)
    z_in = grid.zc[kc_surf].z
    ts_sfc = TC.surface_thermo_state(surf_params, thermo_params, t)
    ts_in = TC.center_aux_grid_mean_ts(state)[kc_surf]
    u_sfc = SA.SVector{2, FT}(0, 0)
    # TODO: make correct with topography
    uₕ_gm_surf = TC.physical_grid_mean_uₕ(state)[kc_surf]
    u_in = SA.SVector{2, FT}(uₕ_gm_surf.u, uₕ_gm_surf.v)
    vals_sfc = SF.SurfaceValues(z_sfc, u_sfc, ts_sfc)
    vals_int = SF.InteriorValues(z_in, u_in, ts_in)
    sc = SF.Coefficients{FT}(
        state_in = vals_int,
        state_sfc = vals_sfc,
        Cd = cm,
        Ch = ch,
        z0m = zrough,
        z0b = zrough,
    )
    return SF.surface_conditions(surf_flux_params, sc, scheme)
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
    FT = TC.float_type(state)
    z_sfc = FT(0)
    z_in = grid.zc[kc_surf].z
    ts_gm = TC.center_aux_grid_mean_ts(state)
    zrough = surf_params.zrough
    thermo_params = TCP.thermodynamics_params(param_set)

    scheme = SF.FVScheme()
    ts_sfc = TC.surface_thermo_state(surf_params, thermo_params, t)
    ts_in = ts_gm[kc_surf]

    u_sfc = SA.SVector{2, FT}(0, 0)
    # TODO: make correct with topography
    uₕ_gm_surf = TC.physical_grid_mean_uₕ(state)[kc_surf]
    u_in = SA.SVector{2, FT}(uₕ_gm_surf.u, uₕ_gm_surf.v)
    vals_sfc = SF.SurfaceValues(z_sfc, u_sfc, ts_sfc)
    vals_int = SF.InteriorValues(z_in, u_in, ts_in)
    sc = SF.ValuesOnly{FT}(
        state_in = vals_int,
        state_sfc = vals_sfc,
        z0m = zrough,
        z0b = zrough,
    )
    return SF.surface_conditions(surf_flux_params, sc, scheme)
end

get_surface(::SingleColumnModel, args...) = get_surface(args...)
function get_surface(
    ::SphericalModel,
    surf_params,
    grid::TC.Grid,
    state::TC.State,
    args...,
)
    # TODO: remove this kludge
    sfc_conditions = state.p.sfc_conditions[state.colidx]
    sfc_conditions_inst = Fields._first(sfc_conditions)
    return sfc_conditions_inst
end
