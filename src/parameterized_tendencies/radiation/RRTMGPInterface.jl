module RRTMGPInterface

import ..AbstractCloudInRadiation, ..InteractiveCloudInRadiation

import NCDatasets as NC
using RRTMGP
using ClimaCore: DataLayouts, Spaces, Fields
import Adapt
import ClimaComms
using NVTX
# TODO: Move this file to RRTMGP.jl, once the interface has been settled.
# It will be faster to do interface development in the same repo as experiment
# development, but, since this is just a user-friendly wrapper for RRTMGP.jl, we
# should move it there eventually.

abstract type AbstractRRTMGPMode end
@kwdef struct GrayRadiation <: AbstractRRTMGPMode
    add_isothermal_boundary_layer::Bool = true
    deep_atmosphere::Bool = true
end
@kwdef struct ClearSkyRadiation <: AbstractRRTMGPMode
    idealized_h2o::Bool = false
    add_isothermal_boundary_layer::Bool = true
    aerosol_radiation::Bool = false
    deep_atmosphere::Bool = true
end
@kwdef struct AllSkyRadiation{ACR <: Union{Nothing, AbstractCloudInRadiation}} <:
              AbstractRRTMGPMode
    idealized_h2o::Bool = false
    idealized_clouds::Bool = false
    cloud::ACR = InteractiveCloudInRadiation()
    add_isothermal_boundary_layer::Bool = true
    aerosol_radiation::Bool = false
    """
    Reset the RNG seed before calling RRTMGP to a known value (the timestep number).
    When modeling cloud optics, RRTMGP uses a random number generator.
    Resetting the seed every time RRTMGP is called to a deterministic value ensures that
    the simulation is fully reproducible and can be restarted in a reproducible way.
    Disable this option when running production runs.
    """
    reset_rng_seed::Bool = false
    deep_atmosphere::Bool = true
end
@kwdef struct AllSkyRadiationWithClearSkyDiagnostics{
    ACR <: Union{Nothing, AbstractCloudInRadiation},
} <: AbstractRRTMGPMode
    idealized_h2o::Bool = false
    idealized_clouds::Bool = false
    cloud::ACR = InteractiveCloudInRadiation()
    add_isothermal_boundary_layer::Bool = true
    aerosol_radiation::Bool = false
    """
    Reset the RNG seed before calling RRTMGP to a known value (the timestep number).
    When modeling cloud optics, RRTMGP uses a random number generator.
    Resetting the seed every time RRTMGP is called to a deterministic value ensures that
    the simulation is fully reproducible and can be restarted in a reproducible way.
    Disable this option when running production runs.
    """
    reset_rng_seed::Bool = false
    deep_atmosphere::Bool = true
end


import RRTMGP:
    AbstractInterpolation,
    NoInterpolation,
    ArithmeticMean,
    GeometricMean,
    UniformZ,
    UniformP,
    BestFit,
    AbstractBottomExtrapolation,
    SameAsInterpolation,
    UseSurfaceTempAtBottom,
    HydrostaticBottom,
    requires_z,
    interp!,
    extrap!,
    uniform_z_p

export AbstractInterpolation,
    NoInterpolation,
    ArithmeticMean,
    GeometricMean,
    UniformZ,
    UniformP,
    BestFit,
    AbstractBottomExtrapolation,
    SameAsInterpolation,
    UseSurfaceTempAtBottom,
    HydrostaticBottom,
    requires_z,
    interp!,
    extrap!,
    uniform_z_p

"""
    RRTMGPModel(solver, radiation_mode)

A thin ClimaAtmos-side wrapper over an `RRTMGP.RRTMGPSolver`. RRTMGP owns the
atmospheric state, boundary conditions, RTE workspaces, lookup tables, and flux
buffers; this wrapper only adds a `getproperty` that maps ClimaAtmos's historical
field names (e.g. `center_temperature`, `face_flux`) onto RRTMGP's named getters,
so the per-step physics that writes inputs and reads fluxes stays unchanged.
"""
struct RRTMGPModel{S, M}
    solver::S           # RRTMGP.RRTMGPSolver — owns state, workspaces, and fluxes
    radiation_mode::M   # ClimaAtmos AbstractRRTMGPMode (kept for introspection)
end

# Allow cache to be moved on the CPU. Used by ClimaCoupler to save checkpoints.
Adapt.@adapt_structure RRTMGPModel

# `getproperty` is how the physics code reads and writes the solver's buffers. Each
# historical field name maps to a writable, domain-sized (boundary-layer-excluded)
# device array or view returned by an `RRTMGP` getter.
function Base.getproperty(model::RRTMGPModel, name::Symbol)
    (name === :solver || name === :radiation_mode) && return getfield(model, name)
    return _solver_field(getfield(model, :solver), name)
end

Base.propertynames(::RRTMGPModel, private::Bool = false) =
    private ? (:solver, :radiation_mode) : ()

function _solver_field(solver, name::Symbol)
    # --- net fluxes ---
    name === :face_flux && return RRTMGP.net_flux(solver)
    name === :face_clear_flux && return RRTMGP.clear_net_flux(solver)
    # --- longwave fluxes ---
    name === :face_lw_flux && return RRTMGP.lw_flux_net(solver)
    name === :face_lw_flux_up && return RRTMGP.lw_flux_up(solver)
    name === :face_lw_flux_dn && return RRTMGP.lw_flux_dn(solver)
    name === :face_clear_lw_flux && return RRTMGP.clear_lw_flux(solver)
    name === :face_clear_lw_flux_up && return RRTMGP.clear_lw_flux_up(solver)
    name === :face_clear_lw_flux_dn && return RRTMGP.clear_lw_flux_dn(solver)
    # --- shortwave fluxes ---
    name === :face_sw_flux && return RRTMGP.sw_flux_net(solver)
    name === :face_sw_flux_up && return RRTMGP.sw_flux_up(solver)
    name === :face_sw_flux_dn && return RRTMGP.sw_flux_dn(solver)
    name === :face_sw_direct_flux_dn && return RRTMGP.sw_direct_flux_dn(solver)
    name === :face_clear_sw_flux && return RRTMGP.clear_sw_flux(solver)
    name === :face_clear_sw_flux_up && return RRTMGP.clear_sw_flux_up(solver)
    name === :face_clear_sw_flux_dn && return RRTMGP.clear_sw_flux_dn(solver)
    name === :face_clear_sw_direct_flux_dn &&
        return RRTMGP.clear_sw_direct_flux_dn(solver)
    # --- atmospheric state ---
    name === :center_pressure && return RRTMGP.layer_pressure(solver)
    name === :center_temperature && return RRTMGP.layer_temperature(solver)
    name === :center_relative_humidity &&
        return RRTMGP.layer_relative_humidity(solver)
    name === :surface_temperature && return RRTMGP.surface_temperature(solver)
    name === :center_z && return RRTMGP.center_z(solver)
    name === :face_z && return RRTMGP.face_z(solver)
    name === :latitude && return RRTMGP.latitude(solver)
    # --- boundary conditions ---
    name === :surface_emissivity && return RRTMGP.surface_emissivity(solver)
    name === :cos_zenith && return RRTMGP.cos_zenith(solver)
    name === :toa_flux && return RRTMGP.toa_flux(solver)
    name === :direct_sw_surface_albedo &&
        return RRTMGP.direct_sw_surface_albedo(solver)
    name === :diffuse_sw_surface_albedo &&
        return RRTMGP.diffuse_sw_surface_albedo(solver)
    name === :top_of_atmosphere_lw_flux_dn &&
        return RRTMGP.top_of_atmosphere_lw_flux_dn(solver)
    # --- clouds ---
    name === :center_cloud_liquid_effective_radius &&
        return RRTMGP.cloud_liquid_effective_radius(solver)
    name === :center_cloud_ice_effective_radius &&
        return RRTMGP.cloud_ice_effective_radius(solver)
    name === :center_cloud_liquid_water_path &&
        return RRTMGP.cloud_liquid_water_path(solver)
    name === :center_cloud_ice_water_path &&
        return RRTMGP.cloud_ice_water_path(solver)
    name === :center_cloud_fraction && return RRTMGP.cloud_fraction(solver)
    name === :sw_cloud_cover && return RRTMGP.sw_cloud_cover(solver)
    name === :lw_cloud_cover && return RRTMGP.lw_cloud_cover(solver)
    # --- aerosols ---
    name === :aod_sw_extinction && return RRTMGP.aod_sw_extinction(solver)
    name === :aod_sw_scattering && return RRTMGP.aod_sw_scattering(solver)
    # --- gases and aerosols keyed by chemical/aerosol name (Dict-resolved) ---
    return _named_solver_field(solver, name)
end

# ClimaAtmos lays the MERRA aerosols into the RRTMGP state arrays in this order (which
# must match RRTMGP's `AEROSOL_IDX`); RRTMGP resolves the short names to canonical ones.
const _AEROSOL_SHORT_NAMES = (
    "dust1", "ss1", "so4", "bcpi", "bcpo", "ocpi", "ocpo",
    "dust2", "dust3", "dust4", "dust5", "ss2", "ss3", "ss4", "ss5",
)

# Precomputed maps from ClimaAtmos's historical field-name symbols to the RRTMGP gas or
# aerosol name, so `getproperty` resolves gas VMRs and per-aerosol properties without
# allocating (no `String` conversion + slicing) on every access. The cloud effective radii,
# which also end in `_radius`, are handled by `_solver_field` and never reach here.
const _GAS_VMR_FIELDS = Dict{Symbol, String}(
    Symbol(prefix, g) => g for g in RRTMGP.gas_names_sw() for
    prefix in ("center_volume_mixing_ratio_", "volume_mixing_ratio_")
)
const _AEROSOL_MASS_FIELDS = Dict{Symbol, String}(
    Symbol("center_", a, "_column_mass_density") => a for a in _AEROSOL_SHORT_NAMES
)
const _AEROSOL_RADIUS_FIELDS = Dict{Symbol, String}(
    Symbol("center_", a, "_radius") => a for
    a in _AEROSOL_SHORT_NAMES if occursin("dust", a) || occursin("ss", a)
)

function _named_solver_field(solver, name::Symbol)
    haskey(_GAS_VMR_FIELDS, name) &&
        return RRTMGP.volume_mixing_ratio(solver, _GAS_VMR_FIELDS[name])
    haskey(_AEROSOL_MASS_FIELDS, name) &&
        return RRTMGP.aerosol_column_mass_density(solver, _AEROSOL_MASS_FIELDS[name])
    haskey(_AEROSOL_RADIUS_FIELDS, name) &&
        return RRTMGP.aerosol_radius(solver, _AEROSOL_RADIUS_FIELDS[name])
    error("RRTMGPModel has no field `$name`")
end

get_radiation_method(m::GrayRadiation) = RRTMGP.GrayRadiation()
get_radiation_method(m::ClearSkyRadiation) =
    RRTMGP.ClearSkyRadiation(m.aerosol_radiation)
get_radiation_method(m::AllSkyRadiation) =
    RRTMGP.AllSkyRadiation(m.aerosol_radiation, m.reset_rng_seed)
get_radiation_method(m::AllSkyRadiationWithClearSkyDiagnostics) =
    RRTMGP.AllSkyRadiationWithClearSkyDiagnostics(
        m.aerosol_radiation,
        m.reset_rng_seed,
    )

"""
    RRTMGPModel(params; kwargs...)

A user-friendly interface for `RRTMGP.jl`. Stores an `RRTMGP.RTE.Solver`, along
with all of the data required to use it. Provides easy access to `RRTMGP`'s
inputs and outputs (e.g., `model.center_temperature` and `model.face_flux`).

After constructing an `RRTMGPModel`, use it as follows:

  - update all the inputs that have changed since it was constructed; e.g.,
    `model.center_temperature .= field2array(current_center_temperature_field)`
  - call `update_fluxes!(model)`
  - use the values of any fluxes of interest; e.g.,
    `field2array(face_flux_field) .= model.face_flux`

The `RRTMGPModel` assumes that pressure and temperature live on cell centers,
and internally interpolates data to cell faces when needed.

Every keyword argument that corresponds to an array of cell center or cell face
values can be specified as a scalar (corresponding to a constant value
throughout the atmosphere), or as a 1D array (corresponding to the values in
each column), or as a 2D array with a single row (corresponding to the values in
each level), or as the full 2D array specifying the value at every point.
Similarly, every keyword argument that corresponds to an array of values at the
top/bottom of the atmosphere can be specified as a scalar, or as the full 1D
array.

# Positional Arguments

  - Artifacts stored in `RRTMGPReferenceData/<file_name>`.
    Should be callable with filenames:

      + `lookup_tables/clearsky_lw.nc`
      + `lookup_tables/cloudysky_lw.nc`
      + `lookup_tables/clearsky_sw.nc`
      + `lookup_tables/cloudysky_sw.nc`

  - `FT`: floating-point number type (performance with `Float32` is questionable)
  - `DA`: array type (defaults to `CuArray` when a compatible GPU is available)

# Keyword Arguments

  - `ncol`: number of vertical columns in the domain/extension

  - `domain_nlay`: number of cells (layers) in the domain
  - `radiation_mode`: overall mode for running `RRTMGP`; available options are

      + `GrayRadiation`: uniform absorption across all frequencies
      + `ClearSkyRadiation`: full RRTMGP model, but without clouds
      + `AllSkyRadiation`: full RRTMGP model
      + `AllSkyRadiationWithClearSkyDiagnostics`: computes the fluxes for both
        `AllSkyRadiation` and `ClearSkyRadiation`
  - `interpolation`: method for determining implied values (if there are any);
    see documentation for `AbstractInterpolation` for available options
  - `bottom_extrapolation`: method for determining implied values at the bottom
    cell face (only used when the cell face values are implied); see documentation
    for `AbstractInterpolation` for available options
  - `use_global_means_for_well_mixed_gases`: whether to use a scalar value to
    represent the volume mixing ratio of each well-mixed gas (i.e., a gas that is
    not water vapor or ozone), instead of using an array that represents a
    spatially varying volume mixing ratio
  - `center_pressure` and/or `face_pressure`: air pressure in Pa on cell centers
    and on cell faces (either one or both of these must be specified)
  - `center_temperature` and/or `face_temperature`: air temperature in K on cell
    centers and on cell faces (if `center_pressure` is specified, then
    `center_temperature` must also be specified, and, if `face_pressure` is
    specified, then `face_temperature` must also be specified)
  - `surface_temperature`: temperature of the surface in K (required)
  - `surface_emissivity`: longwave emissivity of the surface (required)
  - `top_of_atmosphere_lw_flux_dn`: incoming longwave radiation in W/m^2
    (assumed to be 0 by default)
  - `direct_sw_surface_albedo`: direct shortwave albedo of the surface
    (required)
  - `diffuse_sw_surface_albedo`: diffuse shortwave albedo of the surface
    (required)
  - `cos_zenith`: cosine of the zenith angle of sun in radians (required)
  - `toa_flux`: irradiance of sun in W/m^2 (required); the incoming
    direct shortwave radiation is given by
    `model.toa_flux .* model.cos_zenith`
  - `top_of_atmosphere_diffuse_sw_flux_dn`: incoming diffuse shortwave
    radiation in W/m^2 (assumed to be 0 by default)
  - arguments only available when `radiation_mode isa GrayRadiation`:

      + `lapse_rate`: a scalar value that specifies the lapse rate throughout the
        atmosphere (required); this is a constant that can't be modified after the
        model is constructed
      + `optical_thickness_parameter`: the longwave optical depth at the surface
        (required)
  - arguments only available when `!(radiation_mode isa GrayRadiation)`:

      + `center_volume_mixing_ratio_h2o`: volume mixing ratio of water vapor on
        cell centers (required)

      + `center_volume_mixing_ratio_o3`: volume mixing ratio of ozone on cell
        centers (required)
      + arguments only available when `use_global_means_for_well_mixed_gases`:

          * `volume_mixing_ratio_<gas_name>` for `gas_name` in `co2`, `n2o`, `co`,
            `ch4`, `o2`, `n2`, `ccl4`, `cfc11`, `cfc12`, `cfc22`, `hfc143a`,
            `hfc125`, `hfc23`, `hfc32`, `hfc134a`, `cf4`, `no2`: a scalar value
            that specifies the volume mixing ratio of each well-mixed gas
            throughout the atmosphere (required)
      + arguments only available when `!use_global_means_for_well_mixed_gases`:

          * `center_volume_mixing_ratio_<gas_name>` for `gas_name` in `co2`,
            `n2o`, `co`,`ch4`, `o2`, `n2`, `ccl4`, `cfc11`, `cfc12`, `cfc22`,
            `hfc143a`, `hfc125`, `hfc23`, `hfc32`, `hfc134a`, `cf4`, `no2`: volume
            mixing ratio of each well-mixed gas on cell centers (required)
      + arguments only available when `!(radiation_mode isa ClearSkyRadiation)`:

          * `center_cloud_liquid_effective_radius`: effective radius of cloud
            liquid water in m on cell centers (required)
          * `center_cloud_ice_effective_radius`: effective radius of cloud ice
            water in m on cell centers (required)
          * `center_cloud_liquid_water_path`: mean path length of cloud liquid
            water in m on cell centers (required)
          * `center_cloud_ice_water_path`: mean path length of cloud ice water in
            m on cell centers (required)
          * `center_cloud_fraction`: cloud fraction on cell centers (required)
          * `ice_roughness`: either 1, 2, or 3, with 3 corresponding to the
            roughest ice (required); this is a constant that can't be modified after
            the model is constructed
      + `latitude`: latitude in degrees (assumed to be 45 by default); used for
        computing the concentration of air in molecules/cm^2
  - arguments only available when
    `requires_z(interpolation) || requires_z(bottom_extrapolation)`:

      + `center_z`: z-coordinate in m at cell centers
      + `face_z`: z-coordinate in m at cell faces
      + `planet_radius`: planet radius (used to compute metric scaling factor)
"""
RRTMGPModel(
    params::RRTMGP.Parameters.ARP,
    context;
    ncol::Int,
    domain_nlay::Int,
    radiation_mode::AbstractRRTMGPMode = ClearSkyRadiation(),
    interpolation::AbstractInterpolation = NoInterpolation(),
    bottom_extrapolation::AbstractBottomExtrapolation = SameAsInterpolation(),
    use_global_means_for_well_mixed_gases::Bool = false,
    kwargs...,
) = _RRTMGPModel(
    params,
    context,
    radiation_mode;
    ncol,
    domain_nlay,
    interpolation,
    bottom_extrapolation,
    use_global_means_for_well_mixed_gases,
    kwargs...,
)

# TODO: make this the new interface for the next breaking release.
function _RRTMGPModel(
    params::RRTMGP.Parameters.ARP,
    context,
    radiation_mode::AbstractRRTMGPMode = ClearSkyRadiation();
    ncol::Int,
    domain_nlay::Int,
    interpolation::AbstractInterpolation = NoInterpolation(),
    bottom_extrapolation::AbstractBottomExtrapolation = SameAsInterpolation(),
    use_global_means_for_well_mixed_gases::Bool = false,
    kwargs...,
)
    device = ClimaComms.device(context)
    DA = ClimaComms.array_type(device)
    FT = typeof(params.grav)
    # turn kwargs into a Dict, so that values can be dynamically popped from it
    dict = Dict(kwargs)
    nlay = domain_nlay + Int(radiation_mode.add_isothermal_boundary_layer)
    grid_params = RRTMGP.RRTMGPGridParams(
        FT;
        context,
        nlay,
        ncol,
        isothermal_boundary_layer = radiation_mode.add_isothermal_boundary_layer,
    )
    op = RRTMGP.Optics.TwoStream(grid_params)
    if use_global_means_for_well_mixed_gases && radiation_mode isa GrayRadiation
        @warn "use_global_means_for_well_mixed_gases is ignored when using \
               GrayRadiation"
    end

    if interpolation isa NoInterpolation
        error("interpolation cannot be NoInterpolation if only center \
               pressures/temperatures are specified")
    end

    radiation_method = get_radiation_method(radiation_mode)
    # Build the lookup tables once and hand them to `RRTMGPSolver` below, so it does not
    # read the (large) NetCDF tables a second time. `lu_kwargs` sizes the input arrays.
    lookup_data = RRTMGP.lookup_tables(grid_params, radiation_method)
    (; lookups, lu_kwargs) = lookup_data

    # `RRTMGP.RRTMGPSolver` (built at the end) owns the radiative sources, the band
    # and broadband flux buffers, and the longwave/shortwave RTE workspaces. Here we
    # build only the boundary conditions and atmospheric state, and seed their inputs
    # from `dict` via `set_input!` (which fills the physical domain and leaves the
    # isothermal boundary layer, if any, for RRTMGP to fill internally).

    # longwave boundary conditions
    sfc_emis = DA{FT}(undef, lu_kwargs.nbnd_lw, ncol)
    set_input!(sfc_emis, "surface_emissivity", domain_nlay, dict)
    name = "top_of_atmosphere_lw_flux_dn"
    if Symbol(name) in keys(dict)
        inc_flux = DA{FT}(undef, ncol)
        set_input!(transpose(inc_flux), name, domain_nlay, dict)
    else
        inc_flux = nothing
    end
    bcs_lw = RRTMGP.BCs.LwBCs(sfc_emis, inc_flux)

    # shortwave boundary conditions
    cos_zenith = DA{FT}(undef, ncol)
    set_input!(cos_zenith, "cos_zenith", domain_nlay, dict)
    toa_flux = DA{FT}(undef, ncol)
    set_input!(toa_flux, "toa_flux", domain_nlay, dict)
    sfc_alb_direct = DA{FT}(undef, lu_kwargs.nbnd_sw, ncol)
    set_input!(sfc_alb_direct, "direct_sw_surface_albedo", domain_nlay, dict)
    sfc_alb_diffuse = DA{FT}(undef, lu_kwargs.nbnd_sw, ncol)
    set_input!(sfc_alb_diffuse, "diffuse_sw_surface_albedo", domain_nlay, dict)
    name = "top_of_atmosphere_diffuse_sw_flux_dn"
    if Symbol(name) in keys(dict)
        @warn "incoming diffuse shortwave fluxes are not yet implemented \
               in RRTMGP.jl; the value of $name will be ignored"
        inc_flux_diffuse = DA{FT}(undef, ncol, ngpt_sw)
        set_input!(transpose(inc_flux_diffuse), name, domain_nlay, dict)
    else
        inc_flux_diffuse = nothing
    end
    bcs_sw = RRTMGP.BCs.SwBCs(
        cos_zenith,
        toa_flux,
        sfc_alb_direct,
        inc_flux_diffuse,
        sfc_alb_diffuse,
    )

    if !(:latitude in keys(dict))
        lon = lat = nothing
    else
        lon = DA{FT}(undef, ncol) # TODO: lon required but unused
        lat = DA{FT}(undef, ncol)
        set_input!(lat, "latitude", domain_nlay, dict)
    end

    p_lev = DA{FT}(undef, nlay + 1, ncol)
    if radiation_mode.deep_atmosphere && :planet_radius in keys(dict)
        metric_scaling = DA{FT}(undef, nlay + 1, ncol)
    else
        metric_scaling = nothing
    end
    t_lev = DA{FT}(undef, nlay + 1, ncol)
    t_sfc = DA{FT}(undef, ncol)
    set_input!(t_sfc, "surface_temperature", domain_nlay, dict)

    if radiation_mode isa GrayRadiation
        p_lay = DA{FT}(undef, nlay, ncol)
        t_lay = DA{FT}(undef, nlay, ncol)
        set_input!(p_lay, "center_pressure", domain_nlay, dict)
        set_input!(t_lay, "center_temperature", domain_nlay, dict)

        z_lev = DA{FT}(undef, nlay + 1, ncol) # TODO: z_lev required but unused

        # lapse_rate is a constant, so don't use set_input! to get it
        :lapse_rate in keys(dict) || throw(UndefKeywordError(:lapse_rate))
        α = pop!(dict, :lapse_rate)
        α isa Real || error("lapse_rate must be a Real")

        d0 = DA{FT}(undef, ncol)
        set_input!(d0, "optical_thickness_parameter", domain_nlay, dict)
        otp = RRTMGP.AtmosphericStates.GrayOpticalThicknessOGorman2008(FT)
        as = RRTMGP.AtmosphericStates.GrayAtmosphericState(
            lat,
            p_lay,
            p_lev,
            t_lay,
            t_lev,
            z_lev,
            t_sfc,
            otp,
        )
    else
        layerdata = DA{FT}(undef, 4, nlay, ncol)
        p_lay = view(layerdata, 2, :, :)
        t_lay = view(layerdata, 3, :, :)
        rh_lay = view(layerdata, 4, :, :)
        set_input!(p_lay, "center_pressure", domain_nlay, dict)
        set_input!(t_lay, "center_temperature", domain_nlay, dict)
        set_input!(rh_lay, "center_relative_humidity", domain_nlay, dict)
        vmr_str = "volume_mixing_ratio_"
        gas_names = filter(
            gas_name ->
                !(gas_name in ("h2o", "h2o_frgn", "h2o_self", "o3")),
            RRTMGP.gas_names_sw(),
        )
        # TODO: This gives the wrong types for CUDA 3.4 and above.
        # gm = use_global_means_for_well_mixed_gases
        # vmr = RRTMGP.Vmrs.init_vmr(ngas, nlay, ncol, FT, DA; gm)
        if use_global_means_for_well_mixed_gases
            vmr = RRTMGP.Vmrs.VmrGM(
                DA{FT}(undef, nlay, ncol),
                DA{FT}(undef, nlay, ncol),
                DA{FT}(undef, lu_kwargs.ngas_sw),
            )
            vmr.vmr .= 0 # TODO: do we need this?
            set_input!(vmr.vmr_h2o, "center_$(vmr_str)h2o", domain_nlay, dict)
            set_input!(vmr.vmr_o3, "center_$(vmr_str)o3", domain_nlay, dict)
            for gas_name in gas_names
                gas_view = view(vmr.vmr, lookups.idx_gases_sw[gas_name])
                set_input!(gas_view, "$vmr_str$gas_name", domain_nlay, dict)
            end
        else
            vmr = RRTMGP.Vmrs.Vmr(DA{FT}(undef, lu_kwargs.ngas_sw, nlay, ncol))
            for gas_name in ["h2o", "o3", gas_names...]
                gas_view = view(vmr.vmr, lookups.idx_gases_sw[gas_name], :, :)
                set_input!(gas_view, "center_$vmr_str$gas_name", domain_nlay, dict)
            end
        end

        if radiation_mode isa ClearSkyRadiation
            cloud_state = nothing
        else
            cld_r_eff_liq = DA{FT}(undef, nlay, ncol)
            set_input!(
                cld_r_eff_liq,
                "center_cloud_liquid_effective_radius",
                domain_nlay,
                dict,
            )
            cld_r_eff_ice = DA{FT}(undef, nlay, ncol)
            set_input!(
                cld_r_eff_ice,
                "center_cloud_ice_effective_radius",
                domain_nlay,
                dict,
            )
            cld_path_liq = DA{FT}(undef, nlay, ncol)
            set_input!(
                cld_path_liq,
                "center_cloud_liquid_water_path",
                domain_nlay,
                dict,
            )
            cld_path_ice = DA{FT}(undef, nlay, ncol)
            set_input!(
                cld_path_ice,
                "center_cloud_ice_water_path",
                domain_nlay,
                dict,
            )
            cld_frac = DA{FT}(undef, nlay, ncol)
            set_input!(cld_frac, "center_cloud_fraction", domain_nlay, dict)
            cld_mask_lw = DA{Bool}(undef, nlay, ncol)
            cld_mask_sw = DA{Bool}(undef, nlay, ncol)
            # cloud covers are outputs (computed by the solve); only allocate them
            cld_cover_sw = DA{FT}(undef, ncol)
            cld_cover_lw = DA{FT}(undef, ncol)
            cld_overlap = RRTMGP.AtmosphericStates.MaxRandomOverlap()

            # ice_roughness is a constant, so don't use set_input! to get it
            if !(:ice_roughness in keys(dict))
                throw(UndefKeywordError(:ice_roughness))
            end
            ice_rgh = pop!(dict, :ice_roughness)
            if !(ice_rgh in (1, 2, 3))
                error("ice_roughness must be either 1, 2, or 3")
            end

            cloud_state = RRTMGP.AtmosphericStates.CloudState(
                cld_r_eff_liq,
                cld_r_eff_ice,
                cld_path_liq,
                cld_path_ice,
                cld_frac,
                cld_cover_sw,
                cld_cover_lw,
                cld_mask_lw,
                cld_mask_sw,
                cld_overlap,
                ice_rgh,
            )
        end

        if radiation_mode.aerosol_radiation
            aod_sw_ext = DA{FT}(undef, ncol)
            aod_sw_sca = DA{FT}(undef, ncol)
            aero_mask = DA{Bool}(undef, nlay, ncol)
            set_input!(aod_sw_ext, "aod_sw_extinction", domain_nlay, dict)
            set_input!(aod_sw_sca, "aod_sw_scattering", domain_nlay, dict)

            n_aerosol_sizes = maximum(values(lookups.idx_aerosize_sw)) # TODO: verify correctness
            n_aerosols = length(lookups.idx_aerosol_sw) # TODO: verify correctness
            # See the lookup table in RRTMGP for the order of aerosols
            aero_size = DA{FT}(undef, n_aerosol_sizes, nlay, ncol)
            aero_mass = DA{FT}(undef, n_aerosols, nlay, ncol)

            aerosol_names = _AEROSOL_SHORT_NAMES
            for (i, name) in enumerate(aerosol_names)
                if occursin("dust", name) || occursin("ss", name)
                    set_input!(
                        view(aero_size, i, :, :),
                        "center_$(name)_radius",
                        domain_nlay,
                        dict,
                    )
                end
            end
            for (i, name) in enumerate(aerosol_names)
                set_input!(
                    view(aero_mass, i, :, :),
                    "center_$(name)_column_mass_density",
                    domain_nlay,
                    dict,
                )
            end
            aerosol_state = RRTMGP.AtmosphericStates.AerosolState(
                aod_sw_ext,
                aod_sw_sca,
                aero_mask,
                aero_size,
                aero_mass,
            )
        else
            aerosol_state = nothing
        end
        as = RRTMGP.AtmosphericStates.AtmosphericState(
            lon,
            lat,
            # layerdata contains `col_dry`, `p_lay`, and `t_lay`
            layerdata,
            p_lev,
            t_lev,
            t_sfc,
            vmr,
            cloud_state,
            aerosol_state,
        )
    end

    if requires_z(interpolation) || requires_z(bottom_extrapolation)
        z_lay = DA{FT}(undef, nlay, ncol)
        set_input!(z_lay, "center_z", domain_nlay, dict)
        z_lev = DA{FT}(undef, nlay + 1, ncol)
        set_input!(z_lev, "face_z", domain_nlay, dict)
        center_z = z_lay
        face_z = z_lev
        if radiation_mode.deep_atmosphere && :planet_radius in keys(dict)
            planet_radius = pop!(dict, :planet_radius)
            # Area ratio appears in denominator of RRTMGP scaling functions,
            # we therefore pass the multiplicative inverse from ClimaAtmos to
            # use mult ops instead of div in RRTMGP GPU kernels.
            metric_scaling .=
                inv.(((z_lev .+ planet_radius) ./ planet_radius) .^ (FT(2)))
        end
    else
        center_z = nothing
        face_z = nothing
    end

    if length(dict) > 0
        @warn string(
            "unused keyword argument",
            length(dict) == 1 ? " " : "s ",
            join(keys(dict), ", ", length(dict) == 2 ? " and " : ", and "),
        )
    end

    # RRTMGP builds and owns the sources, flux buffers, and RTE workspaces. `op` is
    # `TwoStream`, so both bands use scattering optics, matching the previous setup.
    solver = RRTMGP.RRTMGPSolver(
        grid_params,
        radiation_method,
        params,
        bcs_lw,
        bcs_sw,
        as;
        op_lw = op,
        op_sw = op,
        center_z,
        face_z,
        interpolation,
        bottom_extrapolation,
        deep_atmosphere_scaling = metric_scaling,
        lookups = lookup_data,
    )
    return RRTMGPModel(solver, radiation_mode)
end

# This sets `array .= value`, but it allows `array` to be to be a `CuArray`
# while `value` is an `Array` (in which case broadcasting throws an error).
set_array!(array, value::Real, symbol) = fill!(array, value)
function set_array!(array, value::AbstractArray{<:Real}, symbol)
    if ndims(array) == 2
        if size(value) == size(array)
            copyto!(array, value)
        elseif size(value) == (size(array, 1),)
            for col in eachcol(array)
                copyto!(col, value)
            end
        elseif size(value) == (1, size(array, 2))
            for (icol, col) in enumerate(eachcol(array))
                fill!(col, value[1, icol])
            end
        else
            error("expected $symbol to be an array of size $(size(array)), \
                   ($(size(array, 1)),), or (1, $(size(array, 2))); received \
                   an array of size $(size(value))")
        end
    else
        if size(value) == size(array)
            copyto!(array, value)
        else
            error("expected $symbol to be an array of size $(size(array)); \
                   received an array of size $(size(value))")
        end
    end
end

# Seed an RRTMGP input buffer from `dict[name]`. For `center_`/`face_` fields only
# the physical domain is written; the isothermal boundary layer (if any) is filled
# by RRTMGP inside `update_fluxes!`. Mirrors the old `set_and_save!`, minus recording
# a view (the getters now recover every buffer from the solver).
function set_input!(array, name, domain_nlay, dict)
    domain_symbol = Symbol(name)
    domain_symbol in keys(dict) || throw(UndefKeywordError(domain_symbol))
    domain_value = pop!(dict, domain_symbol)
    if startswith(name, "center_") || startswith(name, "face_")
        domain_range =
            startswith(name, "center_") ? (1:domain_nlay) : (1:(domain_nlay + 1))
        set_array!(view(array, domain_range, :), domain_value, domain_symbol)
    else
        set_array!(array, domain_value, domain_symbol)
    end
    return nothing
end

"""
    update_fluxes!(model, seedval)

Run RRTMGP's full radiation update on the wrapped solver — prepare the atmospheric
state (interpolate levels, add the isothermal boundary layer, clip, compute
concentrations), solve the longwave and shortwave problems, and combine them into
the net flux — and return the domain net flux at cell faces (`model.face_flux`).
`seedval` reseeds RRTMGP's cloud-sampling RNG when the mode requests it. After the
call these fluxes are available through `getproperty`:

  - `face_flux`
  - `face_lw_flux`, `face_lw_flux_dn`, `face_lw_flux_up`
  - `face_sw_flux`, `face_sw_flux_dn`, `face_sw_flux_up`, `face_sw_direct_flux_dn`

If `radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics`, the set of available
fluxes also includes

  - `face_clear_flux`
  - `face_clear_lw_flux`, `face_clear_lw_flux_dn`, `face_clear_lw_flux_up`
  - `face_clear_sw_flux`, `face_clear_sw_flux_dn`, `face_clear_sw_flux_up`,
    `face_clear_sw_direct_flux_dn`
"""
NVTX.@annotate update_fluxes!(model::RRTMGPModel, seedval) =
    RRTMGP.update_fluxes!(model.solver, seedval)

include("update_inputs.jl")
end # end module
