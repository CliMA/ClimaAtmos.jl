#####
##### Radiation
#####

import ClimaComms
import ClimaCore: DataLayouts, Geometry, Spaces, Fields, Operators
import Insolation
import Thermodynamics as TD
import NCDatasets as NC
import .Parameters as CAP
import RRTMGP
import .RRTMGPInterface as RRTMGPI

import Dates: Year, Date
import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput,
    PeriodicCalendar,
    LinearInterpolation

import Interpolations as Intp
using Statistics: mean

"""
    radiation_model_cache(Y, atmos::AtmosModel, args...)
    radiation_model_cache(Y, radiation_mode, args...; kwargs...)

Allocate the cache (`p.radiation`) required by the configured radiation mode.

The first method forwards to `atmos.radiation_mode`, so the cache contents depend on which
mode is selected by the `rad` configuration argument:

  - `Nothing` and `HeldSuarezForcing`: empty cache; neither performs radiative transfer.
  - `RRTMGPI.AbstractRRTMGPMode` (`gray`, `clearsky`, `allsky`, `allskywithclear`): the
    `RRTMGP` solver, the face radiation flux, and any insolation and cloud caches.
  - `RadiationDYCOMS`, `RadiationTRMM_LBA`, `RadiationISDAC`: the working fields of the
    corresponding idealized single-column radiation profile.

# Returns

A `NamedTuple` that is merged into the simulation cache as `p.radiation`.

See also `radiation_tendency!` and the radiation docs page, docs/src/radiation.md.
"""
radiation_model_cache(Y, atmos::AtmosModel, args...) =
    radiation_model_cache(Y, atmos.radiation_mode, args...)

#####
##### No Radiation
#####

radiation_model_cache(
    Y,
    radiation_mode::Union{Nothing, HeldSuarezForcing};
    args...,
) = (;)

"""
    radiation_tendency!(Yₜ, Y, p, t, radiation_mode)

Add the radiative heating rate to `Yₜ` in place; return `nothing`.

Every mode that computes a flux applies the flux divergence explicitly as
`Yₜ.c.ρe_tot -= ᶜdivᵥ(ᶠradiation_flux)`, with `ᶠradiation_flux` the net *upward* radiative
flux on cell faces [W/m²], so that flux convergence heats the layer. Dispatch:

  - `Nothing` and `HeldSuarezForcing`: no-op. Held-Suarez applies its temperature relaxation
    from `remaining_tendency!` instead (see `held_suarez_forcing_tendency_ρe_tot`).
  - `RRTMGPI.AbstractRRTMGPMode`: uses the flux computed by the radiation callback at the
    `dt_rad` cadence, and also heats the `PrognosticEDMFX` updrafts.
  - `RadiationDYCOMS`, `RadiationISDAC`: build `ᶠradiation_flux` from a liquid-water-path
    parameterization of longwave cooling before taking the divergence.
  - `RadiationTRMM_LBA`: prescribes a heating rate directly, with no flux.

See also `radiation_model_cache` and the radiation docs page, docs/src/radiation.md.
"""
radiation_tendency!(Yₜ, Y, p, t, ::Union{Nothing, HeldSuarezForcing}) = nothing

#####
##### RRTMGP Radiation
#####

"""
    idealized_ozone(z)

Return the idealized ozone volume mixing ratio at altitude `z` [mol/mol], following the
RCEMIP protocol of [Wing2018](@cite).

The profile is analytic in pressure,

```math
O_3(p) = g_1 \\, p^{g_2} \\, e^{-p / g_3},
```

with `p = P₀ exp(-z / H)` in hPa (`P₀ = 1000 hPa`, scale height `H = 7 km`) and empirical
constants `g₁ = 3.6478`, `g₂ = 0.83209`, `g₃ = 11.3515` that yield ppmv, converted here to
a volume mixing ratio.

Used as the ozone input to RRTMGP whenever ozone is not read from a time-varying dataset.
"""
function idealized_ozone(z::FT) where {FT}
    H_EARTH = FT(7000.0)
    P0 = FT(1e5)
    HPA_TO_PA = FT(100.0)
    PPMV_TO_VMR = FT(1e-6)
    p = P0 * exp(-z / H_EARTH) / HPA_TO_PA
    g1 = FT(3.6478)
    g2 = FT(0.83209)
    g3 = FT(11.3515)
    return g1 * p^g2 * exp(-p / g3) * PPMV_TO_VMR
end

"""
    rrtmgp_solver_kwargs(space, include_z)
    rrtmgp_solver_kwargs(space, params, time_varying_trace_gases, radiation_mode,
                         include_z)

Assemble the mode-dependent keyword arguments passed to `RRTMGPInterface.rrtmgp_solver`.

The two-argument method builds the `GrayRadiation` inputs: `lapse_rate`, the
latitude-dependent longwave optical thickness
`optical_thickness_parameter = 7.2 + (1.8 - 7.2) sin²(lat)` [-], and the column latitudes.
Both of the former are required by `RRTMGPInterface.rrtmgp_solver` but are currently
discarded by it, because RRTMGP builds the gray optical depth from its own
`GrayOpticalThicknessOGorman2008` defaults (whose equatorial and polar optical thicknesses,
7.2 and 1.8, are the endpoints reproduced above).
The five-argument method builds the inputs for the RRTMGP band models: the well-mixed
trace-gas volume mixing ratios (fixed values from `params`, or `NaN` placeholders for
gases listed in `time_varying_trace_gases`), the ozone profile from `idealized_ozone`,
cloud inputs when the mode is not `ClearSkyRadiation`, and aerosol radii and column mass
densities when `radiation_mode.aerosol_radiation` is true.

Inputs that are refreshed by the radiation callback are seeded with `NaN` here so that a
missing update shows up immediately rather than as a plausible-looking flux. With
`idealized_clouds`, fixed liquid and ice cloud layers are prescribed instead and are never
updated. When `include_z` is true, the cell-center and cell-face heights are added, along
with `planet_radius` on spherical grids (used for the deep-atmosphere metric scaling).

Latitude falls back to zero (equator) on flat-space grids, whose coordinates carry no
latitude.
"""
function rrtmgp_solver_kwargs(
    space,
    include_z::Bool,
)
    ᶜspace = Spaces.center_space(space)
    ᶠspace = Spaces.face_space(space)
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶠz = Fields.coordinate_field(ᶠspace).z
    if ᶜspace.grid.global_geometry isa Geometry.AbstractSphericalGlobalGeometry
        planet_radius = ᶜspace.grid.global_geometry.radius
    end
    bottom_coords = Fields.coordinate_field(Spaces.level(ᶜspace, 1))
    latitude = if eltype(bottom_coords) <: Geometry.LatLongZPoint
        Fields.field2array(bottom_coords.lat)
    else
        Fields.field2array(zero(bottom_coords.z)) # flat space is on Equator
    end
    kwargs = (;
        lapse_rate = 3.5,
        optical_thickness_parameter = (@. 7.2 + (1.8 - 7.2) * sind(latitude)^2),
        latitude,
    )
    if ᶜspace.grid.global_geometry isa Geometry.AbstractSphericalGlobalGeometry
        zkwargs = (;
            center_z = Fields.field2array(ᶜz),
            face_z = Fields.field2array(ᶠz),
            planet_radius = planet_radius,
        )
    else
        zkwargs = (;
            center_z = Fields.field2array(ᶜz),
            face_z = Fields.field2array(ᶠz),
        )
    end
    return include_z ? (; kwargs..., zkwargs...) : kwargs
end

function rrtmgp_solver_kwargs(
    space,
    params,
    time_varying_trace_gases,
    radiation_mode::RRTMGPI.AbstractRRTMGPMode,
    include_z::Bool,
)
    ᶜspace = Spaces.center_space(space)
    ᶠspace = Spaces.face_space(space)
    FT = Spaces.undertype(space)

    trace_gas_params = CAP.trace_gas_params(params)

    bottom_coords = Fields.coordinate_field(Spaces.level(ᶜspace, 1))
    ᶜΔz = Fields.Δz_field(ᶜspace)
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶠz = Fields.coordinate_field(ᶠspace).z
    if ᶜspace.grid.global_geometry isa Geometry.AbstractSphericalGlobalGeometry
        planet_radius = ᶜspace.grid.global_geometry.radius
    end
    latitude = if eltype(bottom_coords) <: Geometry.LatLongZPoint
        Fields.field2array(bottom_coords.lat)
    else
        Fields.field2array(zero(bottom_coords.z)) # flat space is on Equator
    end
    kwargs = NamedTuple()
    (; aerosol_radiation) = radiation_mode

    center_volume_mixing_ratio_o3 =
        :O3 in Symbol.(time_varying_trace_gases) ? NaN :
        Fields.field2array(idealized_ozone.(ᶜz))

    trace_gas_names = (
        :CO2,
        :N2O,
        :CO,
        :CH4,
        :O2,
        :N2,
        :CCL4,
        :CFC11,
        :CFC12,
        :CFC22,
        :HFC143A,
        :HFC125,
        :HFC23,
        :HFC32,
        :HFC134A,
        :CF4,
        :NO2,
    )
    trace_gas_vmr_names =
        map(
            gas_name -> Symbol(:volume_mixing_ratio_, lowercase(String(gas_name))),
            trace_gas_names,
        )
    trace_gas_vmrs = map(trace_gas_names) do gas_name
        if gas_name in Symbol.(time_varying_trace_gases)
            NaN
        else
            getfield(trace_gas_params, Symbol(gas_name, :_fixed_value))
        end
    end
    kwargs = (;
        use_global_means_for_well_mixed_gases = true,
        center_volume_mixing_ratio_h2o = NaN, # initialize in tendency
        center_relative_humidity = NaN, # initialized in callback
        center_volume_mixing_ratio_o3,
        NamedTuple{trace_gas_vmr_names}(trace_gas_vmrs)...,
        latitude,
    )

    if !(radiation_mode isa RRTMGPI.ClearSkyRadiation)
        kwargs = (; kwargs..., ice_roughness = 2)
        if radiation_mode.idealized_clouds # icy cloud on top and liquid cloud on bottom
            # TODO: can we avoid using DataLayouts with this?
            #     `ᶜis_bottom_cloud = similar(ᶜz, Bool)`
            ᶜis_bottom_cloud = Fields.Field(
                DataLayouts.replace_basetype(Fields.field_values(ᶜz), Bool),
                ᶜspace,
            ) # need to fix several ClimaCore bugs in order to simplify this
            ᶜis_top_cloud = similar(ᶜis_bottom_cloud)
            @. ᶜis_bottom_cloud = ᶜz > 1e3 && ᶜz < 1.5e3
            @. ᶜis_top_cloud = ᶜz > 4e3 && ᶜz < 5e3
            kwargs = (;
                kwargs...,
                center_cloud_liquid_effective_radius = 12,
                center_cloud_ice_effective_radius = 25,
                center_cloud_liquid_water_path = Fields.field2array(
                    @. ifelse(ᶜis_bottom_cloud, FT(0.002) * ᶜΔz, FT(0))
                ),
                center_cloud_ice_water_path = Fields.field2array(
                    @. ifelse(ᶜis_top_cloud, FT(0.001) * ᶜΔz, FT(0))
                ),
                center_cloud_fraction = Fields.field2array(
                    @. ifelse(
                        ᶜis_bottom_cloud | ᶜis_top_cloud,
                        FT(1),
                        0 * ᶜΔz,
                    )
                ),
            )
        else
            kwargs = (;
                kwargs...,
                center_cloud_liquid_water_path = NaN, # initialized in callback
                center_cloud_ice_water_path = NaN, # initialized in callback
                center_cloud_fraction = NaN, # initialized in callback
                center_cloud_liquid_effective_radius = NaN, # initialized in callback
                center_cloud_ice_effective_radius = NaN, # initialized in callback
            )
        end
    end

    if aerosol_radiation
        kwargs = (;
            kwargs...,
            aod_sw_extinction = NaN,
            aod_sw_scattering = NaN,
            # assuming fixed aerosol radius
            center_dust1_radius = 0.55,
            center_dust2_radius = 1.4,
            center_dust3_radius = 2.4,
            center_dust4_radius = 4.5,
            center_dust5_radius = 8,
            center_ss1_radius = 0.55,
            center_ss2_radius = 1.4,
            center_ss3_radius = 2.4,
            center_ss4_radius = 4.5,
            center_ss5_radius = 8,
            center_dust1_column_mass_density = NaN, # initialized in callback
            center_dust2_column_mass_density = NaN, # initialized in callback
            center_dust3_column_mass_density = NaN, # initialized in callback
            center_dust4_column_mass_density = NaN, # initialized in callback
            center_dust5_column_mass_density = NaN, # initialized in callback
            center_ss1_column_mass_density = NaN, # initialized in callback
            center_ss2_column_mass_density = NaN, # initialized in callback
            center_ss3_column_mass_density = NaN, # initialized in callback
            center_ss4_column_mass_density = NaN, # initialized in callback
            center_ss5_column_mass_density = NaN, # initialized in callback
            center_so4_column_mass_density = NaN, # initialized in callback
            center_bcpi_column_mass_density = NaN, # initialized in callback
            center_bcpo_column_mass_density = NaN, # initialized in callback
            center_ocpi_column_mass_density = NaN, # initialized in callback
            center_ocpo_column_mass_density = NaN, # initialized in callback
        )
    end

    if include_z
        if ᶜspace.grid.global_geometry isa
           Geometry.AbstractSphericalGlobalGeometry
            kwargs = (;
                kwargs...,
                center_z = Fields.field2array(ᶜz),
                face_z = Fields.field2array(ᶠz),
                planet_radius = planet_radius,
            )
        else
            kwargs = (;
                kwargs...,
                center_z = Fields.field2array(ᶜz),
                face_z = Fields.field2array(ᶠz),
            )
        end
    end
    return kwargs
end

"""
    radiation_model_cache(Y, radiation_mode::RRTMGPI.AbstractRRTMGPMode, start_date,
                          params, aerosol_names, time_varying_trace_gas_names,
                          insolation_mode; interpolation, bottom_extrapolation)

Build the cache for an RRTMGP radiation mode.

Constructs the `RRTMGP` solver with one column per horizontal node and `Spaces.nlevels`
domain layers, using the mode-specific inputs from `rrtmgp_solver_kwargs`, and allocates
`ᶠradiation_flux`, the net upward radiative flux on cell faces [W/m²] that the radiation
callback fills and `radiation_tendency!` differentiates.

Errors if `aerosol_radiation` is enabled without any of the supported aerosol species in
`aerosol_names`.

# Keyword Arguments

  - `interpolation = RRTMGPI.BestFit()`: Scheme for interpolating the center pressures and
    temperatures to cell faces.
  - `bottom_extrapolation = RRTMGPI.SameAsInterpolation()`: Scheme for the bottom face.

# Returns

A `NamedTuple` with `rrtmgp_solver` and `ᶠradiation_flux`, merged with the insolation cache
(`insolation_cache`) and, for the all-sky modes, the cloud cache (`get_cloud_cache`).
"""
function radiation_model_cache(
    Y,
    radiation_mode::RRTMGPI.AbstractRRTMGPMode,
    start_date,
    params,
    aerosol_names,
    time_varying_trace_gas_names,
    insolation_mode;
    interpolation = RRTMGPI.BestFit(),
    bottom_extrapolation = RRTMGPI.SameAsInterpolation(),
)
    context = ClimaComms.context(axes(Y.c))
    device = context.device
    if !(radiation_mode isa RRTMGPI.GrayRadiation)
        (; aerosol_radiation) = radiation_mode
        if aerosol_radiation && !(any(
            x -> x in aerosol_names,
            [
                "DST01",
                "DST02",
                "DST03",
                "DST04",
                "DST05",
                "SSLT01",
                "SSLT02",
                "SSLT03",
                "SSLT04",
                "SSLT05",
                "SO4",
                "CB1",
                "CB2",
                "OC1",
                "OC2",
            ],
        ))
            error(
                "Need at least one aerosol type when aerosol radiation is turned on",
            )
        end
    end
    FT = Spaces.undertype(axes(Y.c))
    rrtmgp_params = CAP.rrtmgp_params(params)

    include_z =
        RRTMGPI.requires_z(interpolation) ||
        RRTMGPI.requires_z(bottom_extrapolation)

    if radiation_mode isa RRTMGPI.GrayRadiation
        kwargs =
            rrtmgp_solver_kwargs(
                axes(Y.c),
                include_z,
            )
    else
        kwargs =
            rrtmgp_solver_kwargs(
                axes(Y.c),
                params,
                time_varying_trace_gas_names,
                radiation_mode,
                include_z,
            )
    end

    cos_zenith = toa_flux = NaN # initialized in callback

    rrtmgp_solver = RRTMGPI.rrtmgp_solver(
        rrtmgp_params,
        context,
        radiation_mode;
        ncol = length(Spaces.all_nodes(axes(Spaces.level(Y.c, 1)))),
        domain_nlay = Spaces.nlevels(axes(Y.c)),
        interpolation,
        bottom_extrapolation,
        center_pressure = NaN, # initialized in callback
        center_temperature = NaN, # initialized in callback
        surface_temperature = NaN, # initialized in callback
        surface_emissivity = 1,
        direct_sw_surface_albedo = NaN, # initialized in callback
        diffuse_sw_surface_albedo = NaN, # initialized in callback
        cos_zenith,
        toa_flux,
        kwargs...,
    )
    cloud_cache = (;)
    if (radiation_mode isa RRTMGPI.AllSkyRadiation) ||
       (radiation_mode isa RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics)
        cloud_cache = get_cloud_cache(radiation_mode.cloud, Y, start_date)
    end
    return merge(
        (; rrtmgp_solver, ᶠradiation_flux = similar(Y.f, Geometry.WVector{FT})),
        insolation_cache(insolation_mode, Y),
        cloud_cache,
    )
end

"""
    get_cloud_cache(cloud, Y, start_date)

Return the cache needed to supply cloud properties to radiation.

For `InteractiveCloudInRadiation` (and any other setting) the cache is empty, since cloud
properties are diagnosed from the model state. For `PrescribedCloudInRadiation` it holds
the `prescribed_clouds_field` (cloud fraction `cc`, and liquid and ice water contents
`clwc` and `ciwc`) together with the `TimeVaryingInput`s that read them from the ERA5
monthly climatology, interpolated linearly in time on a one-year periodic calendar.
"""
get_cloud_cache(_, _, _) = (;)
function get_cloud_cache(::PrescribedCloudInRadiation, Y, start_date)
    target_space = axes(Y.c)
    prescribed_cloud_names = ("cc", "clwc", "ciwc")
    prescribed_cloud_names_as_symbols = Symbol.(prescribed_cloud_names)
    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())
    timevaryinginputs = [
        TimeVaryingInput(
            AA.era5_cloud_file_path(; context = ClimaComms.context(Y.c)),
            name,
            target_space;
            reference_date = start_date,
            regridder_type = :InterpolationsRegridder,
            regridder_kwargs = (; extrapolation_bc),
            method = LinearInterpolation(PeriodicCalendar(Year(1), Date(2010))),
        ) for name in prescribed_cloud_names
    ]

    prescribed_clouds_field = similar(
        Y.c,
        NamedTuple{
            prescribed_cloud_names_as_symbols,
            NTuple{length(prescribed_cloud_names_as_symbols), eltype(Y.c.ρ)},
        },
    )
    prescribed_cloud_timevaryinginputs =
        (; zip(prescribed_cloud_names_as_symbols, timevaryinginputs)...)
    return (; prescribed_clouds_field, prescribed_cloud_timevaryinginputs)
end

"""
    insolation_cache(insolation_mode, Y)

Return the cache needed by the insolation model.

Only `TimeVaryingInsolation` needs storage: a surface-level field of the `Insolation.jl`
output tuple `(F, S, μ, ζ)`, from which the radiation callback takes the top-of-atmosphere
irradiance `S` [W/m²] and the cosine of the solar zenith angle `μ` [-]. Idealized
insolation modes prescribe both directly and get an empty cache.
"""
insolation_cache(_, _) = (;)
function insolation_cache(::TimeVaryingInsolation, Y)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        insolation_tuple = similar(
            Spaces.level(Y.c, 1),
            @NamedTuple{F::FT, S::FT, μ::FT, ζ::FT}
        )
    )
end

"""
    radiation_tendency!(Yₜ, Y, p, t, ::RRTMGPI.AbstractRRTMGPMode)

Apply the RRTMGP radiative heating to `Yₜ.c.ρe_tot` in place; return `nothing`.

Subtracts the vertical divergence of the net upward flux, `ᶜdivᵥ(ᶠradiation_flux)`, so that
flux convergence heats the layer. The flux itself is not recomputed here: it is refreshed
by the radiation callback every `dt_rad` and held fixed in between, while this tendency is
evaluated explicitly at every timestepper stage.

With `PrognosticEDMFX`, the same grid-mean heating is also applied to each updraft's `mse`
(divided by the updraft density). The grid mean is used as an approximation for updraft
radiation; updrafts are typically absent in the stratosphere, where radiation matters most.
"""
function radiation_tendency!(Yₜ, Y, p, t, ::RRTMGPI.AbstractRRTMGPMode)
    (; ᶠradiation_flux) = p.radiation
    (; turbconv_model) = p.atmos
    @. Yₜ.c.ρe_tot -= ᶜdivᵥ(ᶠradiation_flux)
    # Apply radiation tendency to updrafts in prognostic EDMF. We use the
    # grid-mean radiation as an approximation for updraft radiation.
    # Note: Radiation is not applied to updrafts in diagnostic EDMF because updrafts
    # are typically absent in the stratosphere where radiation is more important.
    if turbconv_model isa PrognosticEDMFX
        (; ᶜρʲs) = p.precomputed
        n = n_mass_flux_subdomains(turbconv_model)
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).mse -= ᶜdivᵥ(ᶠradiation_flux) / ᶜρʲs.:($$j)
        end
    end
    return nothing
end

#####
##### DYCOMS_RF01 and DYCOMS_RF02 radiation
#####

"""
    radiation_model_cache(Y, radiation_mode::RadiationDYCOMS)

Allocate the working fields of the DYCOMS RF01/RF02 idealized radiation profile.

# Returns

A `NamedTuple` holding the extinction field `ᶜκρq`, its definite and indefinite column
integrals `∫_0_∞_κρq` and `ᶠ∫_0_z_κρq`, the inversion-level state `isoline_z_ρ_ρq`, the
face flux `ᶠradiation_flux` [W/m²], and the top-of-atmosphere and surface net energy flux
accumulators used by the diagnostics.
"""
function radiation_model_cache(Y, radiation_mode::RadiationDYCOMS)
    FT = Spaces.undertype(axes(Y.c))
    # The NT type is needed for the `column_reduce!` call below because
    # `column_reduce!` computes the output, `isoline_z_ρ_ρq`, in-place. `Y.c.ρ`
    # and `Y.c.ρq_tot` are input arguments, and `nt1` / `nt2` are output entry
    # values at each point along the vertical column.
    NT = NamedTuple{(:z, :ρ, :ρq_tot), NTuple{3, FT}}
    return (;
        ᶜκρq = similar(Y.c, FT),
        ∫_0_∞_κρq = similar(Spaces.level(Y.c, 1), FT),
        ᶠ∫_0_z_κρq = similar(Y.f, FT),
        isoline_z_ρ_ρq = similar(Spaces.level(Y.c, 1), NT),
        ᶠradiation_flux = similar(Y.f, Geometry.WVector{FT}),
        net_energy_flux_toa = [Geometry.WVector(FT(0))],
        net_energy_flux_sfc = [Geometry.WVector(FT(0))],
    )
end
"""
    radiation_tendency!(Yₜ, Y, p, t, radiation_mode::RadiationDYCOMS)

Apply the DYCOMS RF01/RF02 idealized longwave radiation to `Yₜ.c.ρe_tot` in place; return
`nothing`.

The net upward flux of [Stevens2005](@cite) combines cloud-top cooling, cloud-base warming,
and free-tropospheric warming above the inversion:

```math
F(z) = F_0 e^{-Q(z, ∞)} + F_1 e^{-Q(0, z)} +
       ρ_i \\, c_{p,d} \\, D \\, α_z
       \\left[\\tfrac{1}{4}(z - z_i)^{4/3} + z_i (z - z_i)^{1/3}\\right],
```

where `Q(z₁, z₂) = ∫ κ ρ q_liq dz` is the liquid-water optical path, `D` is the large-scale
divergence [1/s], and the last term applies only above the inversion height `z_i`, taken as
the level whose `q_tot` is closest to 0.008 kg/kg. The tendency is
`-ᶜdivᵥ(ᶠradiation_flux)`, applied explicitly at every stage. Requires a moist
microphysics model.

Two documented departures from the reference remain (see the TODO comments): the extinction
uses the specific content `q_liq` rather than the mixing ratio, and the third term uses the
dry `cp_d` and is clipped to zero below `z_i`, matching the original TurbulenceConvection
implementation.
"""
function radiation_tendency!(Yₜ, Y, p, t, radiation_mode::RadiationDYCOMS)
    @assert !(p.atmos.microphysics_model isa DryModel)

    (; params) = p
    (; ᶜκρq, ∫_0_∞_κρq, ᶠ∫_0_z_κρq, isoline_z_ρ_ρq, ᶠradiation_flux) =
        p.radiation
    (; ᶜq_liq) = p.precomputed
    cp_d = CAP.cp_d(params)
    FT = Spaces.undertype(axes(Y.c))
    NT = NamedTuple{(:z, :ρ, :ρq_tot), NTuple{3, FT}}
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z

    # TODO: According to the paper, we should replace ᶜq_liq
    # with mixing ratio.
    @. ᶜκρq =
        radiation_mode.kappa * Y.c.ρ * ᶜq_liq

    Operators.column_integral_definite!(∫_0_∞_κρq, ᶜκρq)

    Operators.column_integral_indefinite!(ᶠ∫_0_z_κρq, ᶜκρq)

    # Find the values of (z, ρ, q_tot) at the q_tot = 0.008 isoline, i.e., at
    # the level whose value of q_tot is closest to 0.008.
    q_tot_isoline = FT(0.008)
    Operators.column_reduce!(
        (nt1, nt2) ->
            abs(specific.(nt1.ρq_tot, nt1.ρ) - q_tot_isoline) <
            abs(specific.(nt2.ρq_tot, nt2.ρ) - q_tot_isoline) ? nt1 : nt2,
        isoline_z_ρ_ρq,
        Base.broadcasted(NT ∘ tuple, ᶜz, Y.c.ρ, Y.c.ρq_tot),
    )

    zi = isoline_z_ρ_ρq.z
    ρi = isoline_z_ρ_ρq.ρ

    # TODO: According to the paper, we should remove the ifelse condition that
    # clips the third term to 0 below zi, and we should also replace cp_d with
    # cp_m, but this wouldn't match the original code from TurbulenceConvection.
    # Note: ∫_0_z_κρq - ∫_0_∞_κρq = -∫_z_∞_κρq
    @. ᶠradiation_flux = Geometry.WVector(
        radiation_mode.F0 * exp(ᶠ∫_0_z_κρq - ∫_0_∞_κρq) +
        radiation_mode.F1 * exp(-(ᶠ∫_0_z_κρq)) +
        ifelse(
            ᶠz > zi,
            ρi *
            cp_d *
            radiation_mode.divergence *
            radiation_mode.alpha_z *
            (cbrt(ᶠz - zi)^4 / 4 + zi * cbrt(ᶠz - zi)),
            FT(0),
        ),
    )

    @. Yₜ.c.ρe_tot -= ᶜdivᵥ(ᶠradiation_flux)

    return nothing
end

#####
##### TRMM_LBA radiation
#####

"""
    radiation_model_cache(Y, radiation_mode::RadiationTRMM_LBA)

Allocate the working fields of the TRMM_LBA prescribed radiative heating profile.

# Returns

A `NamedTuple` holding the heating-rate field `ᶜdTdt_rad` [K/s] and the top-of-atmosphere
and surface net energy flux accumulators used by the diagnostics.
"""
function radiation_model_cache(Y, radiation_mode::RadiationTRMM_LBA)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜdTdt_rad = similar(Y.c, FT),
        net_energy_flux_toa = [Geometry.WVector(FT(0))],
        net_energy_flux_sfc = [Geometry.WVector(FT(0))],
    )
end

"""
    radiation_tendency!(Yₜ, Y, p, t, radiation_mode::RadiationTRMM_LBA)

Apply the TRMM_LBA prescribed radiative heating to `Yₜ.c.ρe_tot` in place; return
`nothing`.

Unlike the other modes, no flux is computed: the heating rate `ᶜdTdt_rad(t, z)` [K/s] is
read from the `AtmosphericProfilesLibrary` observational profile, which varies with both
height and time of day, and converted to an energy tendency with the moist isochoric heat
capacity, `Yₜ.c.ρe_tot += ρ cv_m dT/dt`. Applied explicitly at every stage.
"""
function radiation_tendency!(Yₜ, Y, p, t, radiation_mode::RadiationTRMM_LBA)
    FT = Spaces.undertype(axes(Y.c))
    (; params) = p
    # TODO: get working (need to add cache / function)
    rad = radiation_mode.rad_profile
    thermo_params = CAP.thermodynamics_params(params)
    ᶜdTdt_rad = p.radiation.ᶜdTdt_rad
    ᶜρ = Y.c.ρ
    (; ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    zc = Fields.coordinate_field(axes(ᶜρ)).z
    @. ᶜdTdt_rad = rad(FT(t), zc)
    @. Yₜ.c.ρe_tot +=
        ᶜρ * TD.cv_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) * ᶜdTdt_rad
    return nothing
end

#####
##### ISDAC radiation
#####

radiation_model_cache(Y, radiation_mode::RadiationISDAC; args...) = (;)  # Don't need a cache for ISDAC

"""
    radiation_tendency!(Yₜ, Y, p, t, radiation_mode::RadiationISDAC)

Apply the ISDAC idealized longwave radiation to `Yₜ.c.ρe_tot` in place; return `nothing`.

The net upward flux is the two-stream, liquid-water-path form

```math
F(z) = F₀ e^{-κ (LWP_{z_t} - LWP_z)} + F₁ e^{-κ LWP_z},
```

with `LWP_z` the liquid water path from the surface to `z` [kg/m²], `LWP_{z_t}` its value at
the domain top, and `κ` the mass extinction coefficient [m²/kg]. The first term is cloud-top
cooling, the second cloud-base warming. The tendency is `-ᶜdivᵥ(ᶠradiation_flux)`, applied
explicitly at every stage. No cache is allocated; the flux is built in `p.scratch`.
"""
function radiation_tendency!(Yₜ, Y, p, t, radiation_mode::RadiationISDAC)
    (; F₀, F₁, κ) = radiation_mode
    (; params, precomputed) = p
    (; ᶜq_liq) = precomputed

    ᶜρq = p.scratch.ᶜtemp_scalar
    @. ᶜρq = Y.c.ρ * ᶜq_liq

    LWP_zₜ = p.scratch.temp_field_level  # column integral of LWP (zₜ = top-of-domain)
    Operators.column_integral_definite!(LWP_zₜ, ᶜρq)

    ᶠLWP_z = p.scratch.ᶠtemp_scalar  # column integral of LWP from 0 to z (z = current level)
    Operators.column_integral_indefinite!(ᶠLWP_z, ᶜρq)

    # TODO: Need to compute flux before `ᶜdivᵥ` until we resolve: https://github.com/CliMA/ClimaCore.jl/issues/1989
    radiation_flux = p.scratch.ᶠtemp_scalar
    @. radiation_flux = F₀ * exp(-κ * (LWP_zₜ - ᶠLWP_z)) + F₁ * exp(-κ * ᶠLWP_z)

    @. Yₜ.c.ρe_tot -= ᶜdivᵥ(Geometry.WVector(
        radiation_flux,
        # F₀ * exp(-κ * (LWP_zₜ - ᶠLWP_z)) + F₁ * exp(-κ * ᶠLWP_z),
    ))  # = -∂F/∂z = ρ cₚ ∂T/∂t (longwave radiation)

    return nothing
end
