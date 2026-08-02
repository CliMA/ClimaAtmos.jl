"""
    RRTMGPInterface

Wrapper around `RRTMGP.jl` that builds and feeds an `RRTMGP.RRTMGPSolver` from ClimaAtmos
fields.

The module owns the ClimaAtmos-facing radiation modes (`AbstractRRTMGPMode` and its
subtypes), the solver constructor `rrtmgp_solver`, and the per-callback input
updates in `update_inputs.jl`. The radiative transfer itself, and the derivations behind
it, belong to `RRTMGP.jl`; see also the radiation docs page, docs/src/radiation.md.
"""
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

"""
    AbstractRRTMGPMode

Overall mode for running RRTMGP, selected by the `rad` configuration argument.

Subtypes:

  - `GrayRadiation` (`rad: gray`): idealized one-band model with uniform absorption.
  - `ClearSkyRadiation` (`rad: clearsky`): full RRTMGP band model with gases but no clouds.
  - `AllSkyRadiation` (`rad: allsky`): full RRTMGP band model, including cloud optics.
  - `AllSkyRadiationWithClearSkyDiagnostics` (`rad: allskywithclear`): as `AllSkyRadiation`,
    and additionally computes the clear-sky fluxes as a diagnostic.

Every subtype carries the flags `add_isothermal_boundary_layer` and `deep_atmosphere`
described under `GrayRadiation`. Instances are passed to `rrtmgp_solver`, which translates
them into the corresponding `RRTMGP` radiation method with `get_radiation_method`.
"""
abstract type AbstractRRTMGPMode end

"""
    GrayRadiation(; add_isothermal_boundary_layer = true, deep_atmosphere = true)

Gray radiation: an idealized one-band model whose optical depth is a prescribed function of
pressure and latitude, requiring no gas, cloud, or aerosol inputs.

The optical thickness profile is RRTMGP's `GrayOpticalThicknessOGorman2008`; the
`lapse_rate` and `optical_thickness_parameter` keyword arguments of `rrtmgp_solver` are
required but currently unused.

# Fields

  - `add_isothermal_boundary_layer`: Whether RRTMGP appends an isothermal layer above the
    model top, so that the domain reaches negligible pressure. RRTMGP allocates and fills it
    internally.
  - `deep_atmosphere`: Whether to scale fluxes by the spherical area ratio `(r/a)²`, which
    matters only for deep-atmosphere configurations. Applied only when `planet_radius` is
    also passed to `rrtmgp_solver`.
"""
@kwdef struct GrayRadiation <: AbstractRRTMGPMode
    add_isothermal_boundary_layer::Bool = true
    deep_atmosphere::Bool = true
end

"""
    ClearSkyRadiation(; idealized_h2o = false, add_isothermal_boundary_layer = true,
                      aerosol_radiation = false, deep_atmosphere = true)

Full RRTMGP correlated k-distribution radiation with gases but without cloud optics.

# Fields

  - `idealized_h2o`: Whether to replace the model's water vapor by an idealized profile of
    prescribed relative humidity, ramped up over the first 30 days (see
    `update_relative_humidity!`).
  - `add_isothermal_boundary_layer`: Whether RRTMGP appends an isothermal layer above the
    model top.
  - `aerosol_radiation`: Whether prescribed aerosols contribute to the shortwave optics.
  - `deep_atmosphere`: Whether to scale fluxes by the spherical area ratio `(r/a)²`.
"""
@kwdef struct ClearSkyRadiation <: AbstractRRTMGPMode
    idealized_h2o::Bool = false
    add_isothermal_boundary_layer::Bool = true
    aerosol_radiation::Bool = false
    deep_atmosphere::Bool = true
end
"""
    AllSkyRadiation{ACR}(; idealized_h2o = false, idealized_clouds = false,
                         cloud = InteractiveCloudInRadiation(),
                         add_isothermal_boundary_layer = true, aerosol_radiation = false,
                         reset_rng_seed = false, deep_atmosphere = true)

Full RRTMGP correlated k-distribution radiation, including cloud optics.

`ACR` is the cloud source, a subtype of `AbstractCloudInRadiation` (or `Nothing`).

# Fields

  - `idealized_h2o`: Whether to replace the model's water vapor by an idealized profile of
    prescribed relative humidity.
  - `idealized_clouds`: Whether to prescribe fixed liquid and ice cloud layers once at solver
    construction, instead of updating cloud properties every radiation step.
  - `cloud`: How cloud properties are obtained: `InteractiveCloudInRadiation` from the model
    state, or `PrescribedCloudInRadiation` from the ERA5 monthly climatology.
  - `add_isothermal_boundary_layer`: Whether RRTMGP appends an isothermal layer above the
    model top.
  - `aerosol_radiation`: Whether prescribed aerosols contribute to the shortwave optics.
  - `reset_rng_seed`: Whether to reset the RNG seed to the timestep number before each RRTMGP
    call. The cloud-optics sampling is stochastic, so a deterministic seed makes runs and
    restarts bitwise reproducible. Leave disabled for production runs.
  - `deep_atmosphere`: Whether to scale fluxes by the spherical area ratio `(r/a)²`.
"""
@kwdef struct AllSkyRadiation{ACR <: Union{Nothing, AbstractCloudInRadiation}} <:
              AbstractRRTMGPMode
    idealized_h2o::Bool = false
    idealized_clouds::Bool = false
    cloud::ACR = InteractiveCloudInRadiation()
    add_isothermal_boundary_layer::Bool = true
    aerosol_radiation::Bool = false
    reset_rng_seed::Bool = false
    deep_atmosphere::Bool = true
end
"""
    AllSkyRadiationWithClearSkyDiagnostics{ACR}(; kwargs...)

As `AllSkyRadiation`, but the solver also computes the clear-sky fluxes, which RRTMGP
exposes through its `clear_*` flux getters for cloud-radiative-effect diagnostics.

The fields and their defaults are identical to `AllSkyRadiation`.
"""
@kwdef struct AllSkyRadiationWithClearSkyDiagnostics{
    ACR <: Union{Nothing, AbstractCloudInRadiation},
} <: AbstractRRTMGPMode
    idealized_h2o::Bool = false
    idealized_clouds::Bool = false
    cloud::ACR = InteractiveCloudInRadiation()
    add_isothermal_boundary_layer::Bool = true
    aerosol_radiation::Bool = false
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

# ClimaAtmos lays the MERRA aerosols into the RRTMGP state arrays in this order (which
# must match RRTMGP's `AEROSOL_IDX`); RRTMGP resolves the short names to canonical ones.
const _AEROSOL_SHORT_NAMES = (
    "dust1", "ss1", "so4", "bcpi", "bcpo", "ocpi", "ocpo",
    "dust2", "dust3", "dust4", "dust5", "ss2", "ss3", "ss4", "ss5",
)

"""
    get_radiation_method(m::AbstractRRTMGPMode)

Translate a ClimaAtmos radiation mode into the corresponding `RRTMGP` radiation method,
forwarding the `aerosol_radiation` and `reset_rng_seed` flags where the method accepts
them.
"""
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
    rrtmgp_solver(params, context, radiation_mode = ClearSkyRadiation(); ncol, domain_nlay,
                  interpolation = NoInterpolation(),
                  bottom_extrapolation = SameAsInterpolation(),
                  use_global_means_for_well_mixed_gases = false, kwargs...)

Construct an `RRTMGP.RRTMGPSolver` and seed all of its inputs, returning the solver.

This is a convenience constructor: it allocates the boundary conditions and the atmospheric
state on the device implied by `context`, fills them from the keyword arguments, and hands
them to `RRTMGP.RRTMGPSolver` along with the lookup tables. Afterwards the solver is used
directly through the `RRTMGP` getters:

```julia
RRTMGP.layer_temperature(solver) .= Fields.field2array(ᶜT)
RRTMGP.update_fluxes!(solver)
Fields.field2array(ᶠradiation_flux) .= RRTMGP.net_flux(solver)
```

Pressure and temperature are given on cell centers; RRTMGP interpolates them to cell faces
with `interpolation` and `bottom_extrapolation`. Because the default `NoInterpolation()`
cannot supply the face values, an interpolation scheme must always be passed.

Each keyword argument that fills an array of cell-center or cell-face values may be given
as a scalar (constant everywhere), a 1D array (one value per level), a 2D array with a
single row (one value per column), or the full 2D array. Arguments for the top or bottom of
the atmosphere may be given as a scalar or as the full 1D array. Values are written only
into the physical domain; the isothermal boundary layer, if any, is filled by RRTMGP.

Unrecognized keyword arguments are reported in a warning rather than silently ignored, and
a missing required one throws an `UndefKeywordError`.

# Arguments

  - `params`: RRTMGP parameter set (`RRTMGP.Parameters.ARP`); its float type sets `FT`.
  - `context`: `ClimaComms` context; its device selects the array type (a GPU array type when
    a compatible GPU is available).
  - `radiation_mode`: The `AbstractRRTMGPMode` to run.

# Keyword Arguments

  - `ncol`: Number of columns.
  - `domain_nlay`: Number of layers in the physical domain, excluding any isothermal boundary
    layer.
  - `interpolation`: Scheme for the implied cell-face values; see `AbstractInterpolation`.
  - `bottom_extrapolation`: Scheme for the implied bottom cell-face value; see
    `AbstractBottomExtrapolation`.
  - `use_global_means_for_well_mixed_gases`: Whether each well-mixed gas (any gas other than
    water vapor and ozone) is represented by a single global value rather than a profile.
    Ignored, with a warning, for `GrayRadiation`.

The remaining inputs are passed through `kwargs...`:

  - `center_pressure`: Air pressure on cell centers [Pa].
  - `center_temperature`: Air temperature on cell centers [K].
  - `surface_temperature`: Surface temperature [K].
  - `surface_emissivity`: Longwave emissivity of the surface, per longwave band [-].
  - `direct_sw_surface_albedo`, `diffuse_sw_surface_albedo`: Direct and diffuse shortwave
    surface albedos, per shortwave band [-].
  - `cos_zenith`: Cosine of the solar zenith angle [-].
  - `toa_flux`: Solar irradiance [W/m²]; the incoming direct shortwave flux is
    `toa_flux .* cos_zenith`.
  - `top_of_atmosphere_lw_flux_dn`: Incoming longwave flux [W/m²]; optional, zero if omitted.
  - `top_of_atmosphere_diffuse_sw_flux_dn`: Incoming diffuse shortwave flux [W/m²]. Accepted
    but ignored, with a warning: incoming diffuse shortwave fluxes are not implemented in
    `RRTMGP.jl`.
  - `latitude`: Latitude [degrees]; optional. Used by RRTMGP to compute the column dry-air
    amount. If omitted, no latitude is stored.
  - `center_z`, `face_z`: Cell-center and cell-face heights [m]; required when
    `requires_z(interpolation) || requires_z(bottom_extrapolation)`.
  - `planet_radius`: Planet radius [m]; optional, and used only alongside `center_z`/`face_z`
    and `radiation_mode.deep_atmosphere` to build the metric scaling factor.

Only for `GrayRadiation`:

  - `lapse_rate`: A scalar lapse rate.
  - `optical_thickness_parameter`: Longwave optical depth at the surface [-].

Both are required and validated, but neither reaches the radiative transfer: the gray
optical depth comes from RRTMGP's `GrayOpticalThicknessOGorman2008` defaults.

Only for the non-gray modes:

  - `center_relative_humidity`: Relative humidity on cell centers [-].
  - `center_volume_mixing_ratio_h2o`, `center_volume_mixing_ratio_o3`: Water vapor and ozone
    volume mixing ratios on cell centers [mol/mol].
  - Well-mixed gases, for each `gas_name` returned by `RRTMGP.gas_names_sw()` other than water
    vapor and ozone (`co2`, `n2o`, `co`, `ch4`, `o2`, `n2`, `ccl4`, `cfc11`, `cfc12`,
    `cfc22`, `hfc143a`, `hfc125`, `hfc23`, `hfc32`, `hfc134a`, `cf4`, `no2`) [mol/mol]:
      + `volume_mixing_ratio_<gas_name>`, a scalar, when
        `use_global_means_for_well_mixed_gases` is true.
      + `center_volume_mixing_ratio_<gas_name>`, a profile, otherwise.
  - When `radiation_mode.aerosol_radiation` is true: `aod_sw_extinction` and
    `aod_sw_scattering` [-], the radii `center_<name>_radius` [μm] of the dust and sea-salt
    bins, and `center_<name>_column_mass_density` [kg/m²] for every species in
    `_AEROSOL_SHORT_NAMES`.

Only for the all-sky modes:

  - `center_cloud_liquid_effective_radius`, `center_cloud_ice_effective_radius`: Effective
    radii of cloud liquid and cloud ice on cell centers [μm].
  - `center_cloud_liquid_water_path`, `center_cloud_ice_water_path`: In-cloud liquid and ice
    water paths on cell centers [g/m²].
  - `center_cloud_fraction`: Cloud fraction on cell centers [-].
  - `ice_roughness`: Ice roughness class, 1, 2, or 3, with 3 the roughest. A constant that
    cannot be changed after construction.

# Returns

An `RRTMGP.RRTMGPSolver`, which owns the radiative sources, the flux buffers, and the
longwave and shortwave RTE workspaces.

See also `update_atmospheric_state!`, which refreshes these inputs every `dt_rad`.
"""
function rrtmgp_solver(
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
    grid_params = RRTMGP.RRTMGPGridParams(
        FT;
        context,
        domain_nlay,
        ncol,
        isothermal_boundary_layer = radiation_mode.add_isothermal_boundary_layer,
    )
    # Total layer count including the isothermal boundary layer (if any). RRTMGP now
    # adds it internally from `domain_nlay`, so read it back for the state allocations.
    nlay = grid_params.nlay
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
    # read the (large) NetCDF tables a second time. The `LookupBundle`'s band/gas counts
    # and name→index maps size the input arrays.
    lookups = RRTMGP.lookup_tables(grid_params, radiation_method)

    # `RRTMGP.RRTMGPSolver` (built at the end) owns the radiative sources, the band
    # and broadband flux buffers, and the longwave/shortwave RTE workspaces. Here we
    # build only the boundary conditions and atmospheric state, and seed their inputs
    # from `dict` via `set_input!` (which fills the physical domain and leaves the
    # isothermal boundary layer, if any, for RRTMGP to fill internally).

    # longwave boundary conditions
    sfc_emis = DA{FT}(undef, lookups.nbnd_lw, ncol)
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
    sfc_alb_direct = DA{FT}(undef, lookups.nbnd_sw, ncol)
    set_input!(sfc_alb_direct, "direct_sw_surface_albedo", domain_nlay, dict)
    sfc_alb_diffuse = DA{FT}(undef, lookups.nbnd_sw, ncol)
    set_input!(sfc_alb_diffuse, "diffuse_sw_surface_albedo", domain_nlay, dict)
    name = "top_of_atmosphere_diffuse_sw_flux_dn"
    if Symbol(name) in keys(dict)
        @warn "incoming diffuse shortwave fluxes are not yet implemented \
               in RRTMGP.jl; the value of $name will be ignored"
        pop!(dict, Symbol(name))
    end
    inc_flux_diffuse = nothing
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
        if use_global_means_for_well_mixed_gases
            vmr = RRTMGP.VolumeMixingRatios.VmrGM(
                DA{FT}(undef, nlay, ncol),
                DA{FT}(undef, nlay, ncol),
                DA{FT}(undef, lookups.ngas_sw),
            )
            vmr.vmr .= 0 # TODO: do we need this?
            set_input!(vmr.vmr_h2o, "center_$(vmr_str)h2o", domain_nlay, dict)
            set_input!(vmr.vmr_o3, "center_$(vmr_str)o3", domain_nlay, dict)
            for gas_name in gas_names
                gas_view = view(vmr.vmr, lookups.idx_gases_sw[gas_name])
                set_input!(gas_view, "$vmr_str$gas_name", domain_nlay, dict)
            end
        else
            vmr = RRTMGP.VolumeMixingRatios.Vmr(
                DA{FT}(undef, lookups.ngas_sw, nlay, ncol),
            )
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
        deep_atmosphere_inverse_scaling = metric_scaling,
        lookups,
    )
    return solver
end

"""
    set_array!(array, value, symbol)

Fill the RRTMGP input `array` from `value`, which may be a scalar or an array.

Unlike broadcasting, this works when `array` lives on a GPU while `value` is a host `Array`.
A 2D target, of shape `(nlevels, ncolumns)`, accepts a value of that full shape, a vector of
one value per level (repeated across columns), or a `1 × ncolumns` row (one value per
column); a target of any other rank requires a value of matching size. Mismatched sizes
throw an error naming `symbol`, the keyword the value came from.

Called from `set_input!`.
"""
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

"""
    set_input!(array, name, domain_nlay, dict)

Seed the RRTMGP input buffer `array` from `dict[Symbol(name)]`, removing the entry from
`dict`; return `nothing`.

Throws an `UndefKeywordError` when the keyword is absent, which is how required inputs are
enforced. For `center_`/`face_` names only the physical domain is written (the first
`domain_nlay` or `domain_nlay + 1` rows); the isothermal boundary layer, if any, is filled
by RRTMGP inside `update_fluxes!`. Popping entries lets `rrtmgp_solver` warn about whatever
keywords are left over.

Called from `rrtmgp_solver`.
"""
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

include("update_inputs.jl")
end # end module
