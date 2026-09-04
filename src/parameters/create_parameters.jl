import ClimaParams as CP
import RRTMGP.Parameters.RRTMGPParameters
import SurfaceFluxes.Parameters.SurfaceFluxesParameters
import SurfaceFluxes.UniversalFunctions as UF
import Insolation.Parameters.InsolationParameters
import Thermodynamics.Parameters.ThermodynamicsParameters
import CloudMicrophysics as CM
import StaticArrays as SA

"""
    ClimaAtmosParameters(::Type{FT})
    ClimaAtmosParameters(
        toml_dict;
        microphysics_model = nothing,
        has_non_orographic_gw = false,
        has_orographic_gw = false,
        has_beres_source = false,
    )

Construct the parameter set for a ClimaAtmos configuration.

Every value is read from a ClimaParams TOML dictionary. The `FT` method builds
the default dictionary for that float type; the `toml_dict` method takes a
dictionary that may already carry overrides, e.g. one created with
`CP.create_toml_dict(FT; override_file)` from a run or calibration TOML. An
override therefore reaches every parameter set through the same dictionary, and
the sub-parameter sets of Thermodynamics, CloudMicrophysics, SurfaceFluxes,
RRTMGP, and Insolation are constructed from it here.

Sub-parameter sets that the configuration does not need are stored as `nothing`,
which keeps them out of the GPU kernels that capture the parameter set: the
microphysics sets are filtered by `microphysics_model`, and each gravity-wave
set is loaded only when its flag is `true`. Passing `microphysics_model = nothing` keeps all of them.

# Arguments

  - `FT`: Float type of the parameters, `Float32` or `Float64`.
  - `toml_dict`: `ClimaParams` parameter dictionary, including any overrides.

# Keyword Arguments

  - `microphysics_model = nothing`: Microphysics model; selects which
    microphysics parameter sets to keep. `nothing` keeps all of them.
  - `has_non_orographic_gw = false`: Load the non-orographic gravity-wave
    parameters.
  - `has_orographic_gw = false`: Load the orographic gravity-wave parameters.
  - `has_beres_source = false`: Load the Beres convective-source parameters.

# Returns

A `CAP.ClimaAtmosParameters`.

# Examples

```julia
import ClimaAtmos as CA
params = CA.ClimaAtmosParameters(Float64)
```
"""
ClimaAtmosParameters(::Type{FT}; kwargs...) where {FT <: AbstractFloat} =
    ClimaAtmosParameters(CP.create_toml_dict(FT); kwargs...)

function ClimaAtmosParameters(
    toml_dict::TD;
    microphysics_model = nothing,
    has_non_orographic_gw::Bool = false,
    has_orographic_gw::Bool = false,
    has_beres_source::Bool = false,
) where {TD <: CP.ParamDict}
    FT = CP.float_type(toml_dict)

    turbconv_params = TurbulenceConvectionParameters(toml_dict)
    TCP = typeof(turbconv_params)

    thermodynamics_params = ThermodynamicsParameters(toml_dict)
    TP = typeof(thermodynamics_params)

    rrtmgp_params = RRTMGPParameters(toml_dict)
    RP = typeof(rrtmgp_params)

    trace_gas_params = trace_gas_parameters(toml_dict)
    TG = typeof(trace_gas_params)

    insolation_params = InsolationParameters(toml_dict)
    IP = typeof(insolation_params)

    surface_fluxes_params =
        SurfaceFluxesParameters(toml_dict, UF.GryanikParams)
    SFP = typeof(surface_fluxes_params)

    surface_temp_params = SurfaceTemperatureParameters(toml_dict)
    STP = typeof(surface_temp_params)

    microphysics_cloud_params = cloud_parameters(toml_dict)
    MPC = typeof(microphysics_cloud_params)

    microphysics_0m_params = CM.Parameters.Microphysics0MParams(toml_dict)
    microphysics_1m_params = microphys_1m_parameters(
        toml_dict;
        microphysics_1m_options(microphysics_model)...,
    )
    microphysics_2m_params = microphys_2m_parameters(toml_dict)
    microphysics_2mp3_params = get_microphysics_2m_p3_parameters(toml_dict)

    # When a microphysics model is supplied, only keep the parameter sets it
    # actually uses; nullify the rest to save memory.
    if !isnothing(microphysics_model)
        microphysics_model isa EquilibriumMicrophysics0M ||
            (microphysics_0m_params = nothing)
        microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ||
            (microphysics_1m_params = nothing)
        microphysics_model isa
        Union{NonEquilibriumMicrophysics2M, NonEquilibriumMicrophysics2MP3} ||
            (microphysics_2m_params = nothing)
        microphysics_model isa NonEquilibriumMicrophysics2MP3 ||
            (microphysics_2mp3_params = nothing)
    end
    MP0M = typeof(microphysics_0m_params)
    MP1M = typeof(microphysics_1m_params)
    MP2M = typeof(microphysics_2m_params)
    MP2MP3 = typeof(microphysics_2mp3_params)

    vert_diff_params = vert_diff_parameters(toml_dict)
    VDP = typeof(vert_diff_params)

    external_forcing_params = external_forcing_parameters(toml_dict)
    EFP = typeof(external_forcing_params)

    prescribed_aerosol_params = prescribed_aerosol_parameters(toml_dict)
    PAP = typeof(prescribed_aerosol_params)
    # Only load gravity-wave parameters if enabled
    non_orographic_gravity_wave_params =
        has_non_orographic_gw ? NonOrographicGravityWaveParameters(toml_dict) : nothing
    orographic_gravity_wave_params =
        has_orographic_gw ? OrographicGravityWaveParameters(toml_dict) : nothing
    # Beres convective source params load only when the Beres source is enabled
    beres_source_params =
        has_beres_source ? BeresSourceParameters(toml_dict) : nothing
    NOGWP = typeof(non_orographic_gravity_wave_params)
    OGWP = typeof(orographic_gravity_wave_params)
    BSP = typeof(beres_source_params)

    parameters =
        CP.get_parameter_values(toml_dict, atmos_name_map, "ClimaAtmos")
    return CAP.ClimaAtmosParameters{
        FT,
        TP,
        RP,
        TG,
        IP,
        MPC,
        MP0M,
        MP1M,
        MP2M,
        MP2MP3,
        SFP,
        TCP,
        STP,
        VDP,
        EFP,
        PAP,
        NOGWP,
        OGWP,
        BSP,
    }(;
        parameters...,
        thermodynamics_params,
        rrtmgp_params,
        trace_gas_params,
        insolation_params,
        microphysics_cloud_params,
        microphysics_0m_params,
        microphysics_1m_params,
        microphysics_2m_params,
        microphysics_2mp3_params,
        surface_fluxes_params,
        turbconv_params,
        surface_temp_params,
        vert_diff_params,
        external_forcing_params,
        prescribed_aerosol_params,
        non_orographic_gravity_wave_params,
        orographic_gravity_wave_params,
        beres_source_params,
    )
end

"""
    atmos_name_map

Map from ClimaParams parameter names to the field names of
`CAP.ClimaAtmosParameters`.

Only the scalar parameters that ClimaAtmos owns appear here; the sub-parameter
sets are built by their own constructors. A new scalar parameter needs an entry
in this map, a field in the struct, and a definition in ClimaParams.
"""
atmos_name_map = (;
    :f_plane_coriolis_frequency => :f_plane_coriolis_frequency,
    :equator_pole_temperature_gradient_wet => :ΔT_y_wet,
    :angular_velocity_planet_rotation => :Omega,
    :equator_pole_temperature_gradient_dry => :ΔT_y_dry,
    :held_suarez_T_equator_wet => :T_equator_wet,
    :zd_rayleigh => :zd_rayleigh,
    :reference_temperature_exponent => :s_ref,
    :zd_viscous => :zd_viscous,
    :planet_radius => :planet_radius,
    :potential_temp_vertical_gradient => :Δθ_z,
    :C_H => :C_H,
    :c_smag => :c_smag,
    :alpha_rayleigh_w => :alpha_rayleigh_w,
    :alpha_rayleigh_uh => :alpha_rayleigh_uh,
    :alpha_rayleigh_tracer => :alpha_rayleigh_tracer,
    :astronomical_unit => :astro_unit,
    :held_suarez_T_equator_dry => :T_equator_dry,
    :drag_layer_vertical_extent => :σ_b,
    :kappa_2_sponge => :kappa_2_sponge,
    :held_suarez_minimum_temperature => :T_min_hs,
    :idealized_ocean_albedo => :idealized_ocean_albedo,
    :water_refractive_index => :water_refractive_index,
    :D_horizontal_diffusion => :constant_horizontal_diffusion_D,
    :temperature_minimum => :T_min_sgs,
    :specific_humidity_maximum => :q_max_sgs,
    :fixed_cloud_liquid_terminal_velocity => :fixed_cloud_liquid_terminal_velocity,
    :fixed_cloud_ice_terminal_velocity => :fixed_cloud_ice_terminal_velocity,
    :fixed_rain_terminal_velocity => :fixed_rain_terminal_velocity,
    :fixed_snow_terminal_velocity => :fixed_snow_terminal_velocity,
)

"""
    cloud_parameters(FT)
    cloud_parameters(toml_dict)

Assemble the cloud-microphysics parameters shared by all microphysics models.

# Returns

A `NamedTuple` with:

  - `liquid`, `ice`: `CloudMicrophysics` cloud liquid and ice properties.
  - `stokes`, `Ch2022`: Stokes-regime and Chen (2022) terminal-velocity
    parameters.
  - `N_cloud_liquid_droplets`: Prescribed cloud droplet number concentration
    [1/m³].
  - `aml`: Aerosol-ML coefficients, see `aerosol_ml_parameters`.
  - `activation`: `CloudMicrophysics` aerosol-activation parameters.
"""
cloud_parameters(::Type{FT}) where {FT <: AbstractFloat} =
    cloud_parameters(CP.create_toml_dict(FT))

cloud_parameters(toml_dict::CP.ParamDict) = (;
    liquid = CM.Parameters.CloudLiquid(toml_dict),
    ice = CM.Parameters.CloudIce(toml_dict),
    stokes = CM.Parameters.StokesRegimeVelType(toml_dict),
    Ch2022 = CM.Parameters.Chen2022VelType(toml_dict),
    N_cloud_liquid_droplets = CP.get_parameter_values(
        toml_dict,
        "prescribed_cloud_droplet_number_concentration",
        "ClimaAtmos",
    ).prescribed_cloud_droplet_number_concentration,
    aml = aerosol_ml_parameters(toml_dict),
    activation = CM.Parameters.AerosolActivationParameters(toml_dict),
)

"""
    microphysics_1m_options(microphysics_model)

The per-process option selections to build `CMP.Microphysics1MParams` with, as
keyword arguments.
"""
microphysics_1m_options(_) = (;)
function microphysics_1m_options(
    microphysics_model::NonEquilibriumMicrophysics1M,
)
    processes = microphysics_model.processes
    return (;
        (pn => getproperty(processes, pn) for pn in propertynames(processes))...
    )
end

"""
    microphys_1m_parameters(FT; options_kwargs...)
    microphys_1m_parameters(toml_dict; options_kwargs...)

Build the `CloudMicrophysics` 1-moment parameter set.

`options_kwargs` selects the per-process schemes (autoconversion, accretion,
and so on) and is forwarded to `Microphysics1MParams`.
"""
microphys_1m_parameters(
    ::Type{FT};
    options_kwargs...,
) where {FT <: AbstractFloat} =
    microphys_1m_parameters(CP.create_toml_dict(FT); options_kwargs...)

microphys_1m_parameters(
    toml_dict::CP.ParamDict;
    options_kwargs...,
) =
    CM.Parameters.Microphysics1MParams(toml_dict; options_kwargs...)

"""
    microphys_2m_parameters(FT)
    microphys_2m_parameters(toml_dict)

Build the `CloudMicrophysics` 2-moment warm-rain parameter set, without ice.

See `get_microphysics_2m_p3_parameters` for the variant that includes P3 ice.
"""
microphys_2m_parameters(::Type{FT}) where {FT <: AbstractFloat} =
    microphys_2m_parameters(CP.create_toml_dict(FT))

function microphys_2m_parameters(toml_dict::CP.ParamDict)
    CM.Parameters.Microphysics2MParams(toml_dict; with_ice = false)
end

"""
    get_microphysics_2m_p3_parameters(FT)
    get_microphysics_2m_p3_parameters(toml_dict)

Build the parameter set for the 2-moment warm-rain plus P3 ice microphysics
scheme.

# Arguments

  - `FT`: Float type of the parameters, `Float32` or `Float64`.
  - `toml_dict`: `ClimaParams` parameter dictionary, as returned by
    `ClimaParams.create_toml_dict`.
"""
get_microphysics_2m_p3_parameters(::Type{FT}) where {FT <: AbstractFloat} =
    get_microphysics_2m_p3_parameters(CP.create_toml_dict(FT))

function get_microphysics_2m_p3_parameters(toml_dict::CP.ParamDict)
    CM.Parameters.Microphysics2MParams(toml_dict; with_ice = true)
end

"""
    vert_diff_parameters(toml_dict)

Return the parameters of the simple vertical-diffusion schemes as a
`NamedTuple`: `C_E` [-], and the height scale `H` [m] and coefficient `D₀`
[m²/s] of `DecayWithHeightDiffusion`.
"""
function vert_diff_parameters(toml_dict)
    name_map = (; :C_E => :C_E, :H_diffusion => :H, :D_0_diffusion => :D₀)
    return CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
end

"""
    external_forcing_parameters(toml_dict)

Return the GCM-driven external-forcing parameters as a `NamedTuple`: the
momentum and scalar relaxation timescales [s], and the minimum and maximum
heights over which the relaxation is applied [m].
"""
function external_forcing_parameters(toml_dict)
    efp_fields = [
        "gcmdriven_momentum_relaxation_timescale",
        "gcmdriven_scalar_relaxation_timescale",
        "gcmdriven_relaxation_minimum_height",
        "gcmdriven_relaxation_maximum_height",
    ]
    return CP.get_parameter_values(toml_dict, efp_fields, "ClimaAtmos")
end

"""
    aerosol_ml_parameters(toml_dict)

Return the coefficients of the ML-based aerosol-activation correction as a
`NamedTuple`: the reference droplet number `N₀` [1/m³], the calibration
coefficients `α_dust`, `α_seasalt`, `α_SO4`, and `α_q_liq` [-], and the
reference concentrations `c₀_dust`, `c₀_seasalt`, `c₀_SO4` [kg/m³] and `q₀_liq`
[kg/kg].
"""
function aerosol_ml_parameters(toml_dict)
    name_map = (;
        :prescribed_cloud_droplet_number_concentration => :N₀,
        :dust_calibration_coefficient => :α_dust,
        :seasalt_calibration_coefficient => :α_seasalt,
        :ammonium_sulfate_calibration_coefficient => :α_SO4,
        :liquid_water_specific_humidity_calibration_coefficient => :α_q_liq,
        :reference_dust_aerosol_mass_concentration => :c₀_dust,
        :reference_seasalt_aerosol_mass_concentration => :c₀_seasalt,
        :reference_ammonium_sulfate_mass_concentration => :c₀_SO4,
        :reference_liquid_water_specific_humidity => :q₀_liq,
    )
    return CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
end

"""
    prescribed_aerosol_parameters(toml_dict)

Return the optical and hygroscopic properties of the prescribed MERRA-2 aerosol
modes as a `NamedTuple`: the radii of the five sea-salt bins and of the sulfate
mode [m], their hygroscopicity parameters [-], their densities [kg/m³], and the
lognormal widths of the MAM3 coarse and accumulation modes [-].
"""
function prescribed_aerosol_parameters(toml_dict)
    name_map = (;
        :MERRA2_seasalt_aerosol_bin01_radius => :SSLT01_radius,
        :MERRA2_seasalt_aerosol_bin02_radius => :SSLT02_radius,
        :MERRA2_seasalt_aerosol_bin03_radius => :SSLT03_radius,
        :MERRA2_seasalt_aerosol_bin04_radius => :SSLT04_radius,
        :MERRA2_seasalt_aerosol_bin05_radius => :SSLT05_radius,
        :seasalt_aerosol_kappa => :seasalt_kappa,
        :seasalt_aerosol_density => :seasalt_density,
        :mam3_stdev_coarse => :seasalt_std,
        :MERRA2_sulfate_aerosol_radius => :sulfate_radius,
        :sulfate_aerosol_kappa => :sulfate_kappa,
        :sulfate_aerosol_density => :sulfate_density,
        :mam3_stdev_accum => :sulfate_std,
    )
    return CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
end

"""
    trace_gas_parameters(toml_dict)

Return the fixed volume mixing ratios of the radiatively active trace gases as a
`NamedTuple` [mol/mol].

These are the values used wherever a gas is not read from a dataset; see the
Trace Gases page of the documentation.
"""
function trace_gas_parameters(toml_dict)
    name_map = (;
        :CO2_fixed_value => :CO2_fixed_value,
        :N2O_fixed_value => :N2O_fixed_value,
        :CO_fixed_value => :CO_fixed_value,
        :CH4_fixed_value => :CH4_fixed_value,
        :O2_fixed_value => :O2_fixed_value,
        :N2_fixed_value => :N2_fixed_value,
        :CCL4_fixed_value => :CCL4_fixed_value,
        :CFC11_fixed_value => :CFC11_fixed_value,
        :CFC12_fixed_value => :CFC12_fixed_value,
        :CFC22_fixed_value => :CFC22_fixed_value,
        :HFC143A_fixed_value => :HFC143A_fixed_value,
        :HFC125_fixed_value => :HFC125_fixed_value,
        :HFC23_fixed_value => :HFC23_fixed_value,
        :HFC32_fixed_value => :HFC32_fixed_value,
        :HFC134A_fixed_value => :HFC134A_fixed_value,
        :CF4_fixed_value => :CF4_fixed_value,
        :NO2_fixed_value => :NO2_fixed_value,
    )
    return CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
end

"""
    to_svec(x)

Convert arrays to `SVector`s, recursively through `NamedTuple`s, and leave
everything else unchanged.

Parameter vectors arrive from ClimaParams as `Vector`s, which are not `isbits`
and so cannot be captured by GPU kernels.
"""
to_svec(x::AbstractArray) = SA.SVector{length(x)}(x)
to_svec(x) = x
to_svec(x::NamedTuple) = map(x -> to_svec(x), x)

"""
    TurbulenceConvectionParameters(FT, overrides = NamedTuple())
    TurbulenceConvectionParameters(toml_dict, overrides = NamedTuple())

Build the PROPHET (prognostic EDMF) parameter set.

Most values come from ClimaParams through an explicit name map. A few
cloud-fraction release-shape parameters and the updraft sedimentation
coefficient are not yet in ClimaParams' default TOML: they fall back to the
defaults set here, and are read from `toml_dict` only when a run or calibration
TOML defines them. The defaults `margin = abs_margin = sharpness = 1` and
`residual = 0` release the cloud-fraction floor on a one-width saturation
margin guarded by an absolute margin of one floor width, and
`sedimentation_lateral_coeff = 0` disables the lateral sedimentation
correction.

`overrides` is merged last, so it wins over both the TOML values and the
defaults above.
"""
TurbulenceConvectionParameters(
    ::Type{FT},
    overrides = NamedTuple(),
) where {FT <: AbstractFloat} =
    TurbulenceConvectionParameters(CP.create_toml_dict(FT), overrides)

function TurbulenceConvectionParameters(
    toml_dict::CP.ParamDict,
    overrides = NamedTuple(),
)
    name_map = (;
        :min_area_limiter_scale => :min_area_limiter_scale,
        :max_area_limiter_scale => :max_area_limiter_scale,
        :mixing_length_tke_surf_scale => :tke_surf_scale,
        :mixing_length_tke_surf_flux_coeff => :tke_surf_flux_coeff,
        :mixing_length_Ri_crit => :Ri_crit,
        :diagnostic_covariance_coeff => :diagnostic_covariance_coeff,
        :Tq_correlation_coefficient => :Tq_correlation_coefficient,
        :detr_buoy_coeff => :detr_buoy_coeff,
        :EDMF_max_area => :max_area,
        :mixing_length_smin_rm => :smin_rm,
        :entr_coeff => :entr_coeff,
        :entr_inv_length => :entr_inv_length,
        :entr_buoy_coeff => :entr_buoy_coeff,
        :entr_detr_buoy_inv_tau_max => :entr_detr_buoy_inv_tau_max,
        :detr_coeff => :detr_coeff,
        :EDMF_max_surface_area => :max_surface_area,
        :entr_param_vec => :entr_param_vec,
        :turb_entr_param_vec => :turb_entr_param_vec,
        :entr_mult_limiter_coeff => :entr_mult_limiter_coeff,
        :minimum_updraft_top => :min_updraft_top,
        :mixing_length_eddy_viscosity_coefficient => :tke_ed_coeff,
        :mixing_length_smin_ub => :smin_ub,
        :EDMF_min_area => :min_area,
        :detr_vertdiv_coeff => :detr_vertdiv_coeff,
        :detr_massflux_vertdiv_coeff => :detr_massflux_vertdiv_coeff,
        :max_area_limiter_power => :max_area_limiter_power,
        :min_area_limiter_power => :min_area_limiter_power,
        :pressure_normalmode_drag_coeff => :pressure_normalmode_drag_coeff,
        :mixing_length_Prandtl_number_scale => :Prandtl_number_scale,
        :mixing_length_Prandtl_number_0 => :Prandtl_number_0,
        :mixing_length_Prandtl_maximum => :Pr_max,
        :mixing_length_static_stab_coeff => :static_stab_coeff,
        :pressure_normalmode_buoy_coeff1 =>
            :pressure_normalmode_buoy_coeff1,
        :detr_inv_tau => :detr_inv_tau,
        :entr_inv_tau => :entr_inv_tau,
        :entr_detr_limit_inv_tau => :entr_detr_limit_inv_tau,
        :cloud_fraction_param_vec => :cloud_fraction_param_vec,
        :cloud_fraction_steepness_scale => :cloud_fraction_steepness_scale,
        :cloud_fraction_eps_rel => :cloud_fraction_eps_rel,
        :cloud_fraction_sigma_abs => :cloud_fraction_sigma_abs,
        :EDMF_interface_entr_efficiency => :interface_entr_efficiency,
        :EDMF_sfc_mass_flux_ustar_coeff => :sfc_mass_flux_ustar_coeff,
        :EDMF_convective_zi => :convective_zi,
        :EDMF_sfc_mass_flux_cap_fraction => :sfc_mass_flux_cap_fraction,
    )
    parameters = CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
    FT = CP.float_type(toml_dict)
    # Cloud-fraction shape parameters (see `_compute_cloud_fraction`).
    # Not yet in ClimaParams' default toml, so they are fetched only when a
    # run/calibration toml defines them and otherwise fall back to the
    # defaults below (margin = abs_margin = sharpness = 1, residual = 0),
    # which release the floor on a one-width saturation margin guarded by an
    # absolute margin of one floor width.
    # TODO: promote to ClimaParams (and the name_map above) once the
    # release shape has been calibrated.
    release_defaults = (;
        cloud_fraction_floor_release_margin = FT(1),
        cloud_fraction_floor_release_abs_margin = FT(1),
        cloud_fraction_floor_release_sharpness = FT(1),
        cloud_fraction_floor_residual = FT(0),
        # Lateral correction scaling for updraft sedimentation
        # (see `updraft_sedimentation!`). 1.0 = full correction, 0.0 = disabled.
        sedimentation_lateral_coeff = FT(1), # Testing if stable now. To be removed, if yes.
    )
    release_present = filter(collect(keys(release_defaults))) do name
        haskey(toml_dict.data, string(name))
    end
    release_params =
        isempty(release_present) ? (;) :
        CP.get_parameter_values(
            toml_dict,
            String.(release_present),
            "ClimaAtmos",
        )
    parameters =
        merge(parameters, release_defaults, release_params, overrides)
    parameters = to_svec(parameters)
    VFT1 = typeof(parameters.entr_param_vec)
    VFT2 = typeof(parameters.turb_entr_param_vec)
    VTF3 = typeof(parameters.cloud_fraction_param_vec)
    CAP.TurbulenceConvectionParameters{FT, VFT1, VFT2, VTF3}(; parameters...)
end

"""
    SurfaceTemperatureParameters(FT, overrides = NamedTuple())
    SurfaceTemperatureParameters(toml_dict, overrides = NamedTuple())

Build the prescribed analytic sea-surface-temperature parameter set, with
`overrides` merged over the ClimaParams values.
"""
SurfaceTemperatureParameters(
    ::Type{FT},
    overrides = NamedTuple(),
) where {FT <: AbstractFloat} =
    SurfaceTemperatureParameters(CP.create_toml_dict(FT), overrides)

function SurfaceTemperatureParameters(
    toml_dict::CP.ParamDict,
    overrides = NamedTuple(),
)
    name_map = (;
        :SST_mean => :SST_mean,
        :SST_delta => :SST_delta,
        :SST_wavelength => :SST_wavelength,
        :SST_wavelength_latitude => :SST_wavelength_latitude,
    )
    parameters = CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
    parameters = merge(parameters, overrides)
    FT = CP.float_type(toml_dict)
    CAP.SurfaceTemperatureParameters{FT}(; parameters...)
end

"""
    NonOrographicGravityWaveParameters(FT, overrides = NamedTuple())
    NonOrographicGravityWaveParameters(toml_dict, overrides = NamedTuple())

Build the Alexander-Dunkerton non-orographic gravity-wave parameter set, with
`overrides` merged over the ClimaParams values.
"""
NonOrographicGravityWaveParameters(
    ::Type{FT},
    overrides = NamedTuple(),
) where {FT <: AbstractFloat} =
    NonOrographicGravityWaveParameters(CP.create_toml_dict(FT), overrides)

function NonOrographicGravityWaveParameters(
    toml_dict::CP.ParamDict,
    overrides = NamedTuple(),
)
    name_map = (;
        :nogw_source_pressure => :source_pressure,
        :nogw_damp_pressure => :damp_pressure,
        :nogw_source_height => :source_height,
        :nogw_Bw => :Bw,
        :nogw_Bn => :Bn,
        :nogw_dc => :dc,
        :nogw_cmax => :cmax,
        :nogw_c0 => :c0,
        :nogw_nk => :nk,
        :nogw_cw => :cw,
        :nogw_cw_tropics => :cw_tropics,
        :nogw_cn => :cn,
        :nogw_Bt_0 => :Bt_0,
        :nogw_Bt_n => :Bt_n,
        :nogw_Bt_s => :Bt_s,
        :nogw_Bt_eq => :Bt_eq,
        :nogw_phi0_n => :ϕ0_n,
        :nogw_phi0_s => :ϕ0_s,
        :nogw_dphi_n => :dϕ_n,
        :nogw_dphi_s => :dϕ_s,
    )
    parameters = CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
    parameters = merge(parameters, overrides)
    FT = CP.float_type(toml_dict)
    CAP.NonOrographicGravityWaveParameters{FT}(; parameters...)
end


"""
    OrographicGravityWaveParameters(FT, overrides = NamedTuple())
    OrographicGravityWaveParameters(toml_dict, overrides = NamedTuple())

Build the Garner orographic gravity-wave parameter set, with `overrides` merged
over the ClimaParams values.
"""
OrographicGravityWaveParameters(
    ::Type{FT},
    overrides = NamedTuple(),
) where {FT <: AbstractFloat} =
    OrographicGravityWaveParameters(CP.create_toml_dict(FT), overrides)

function OrographicGravityWaveParameters(
    toml_dict::CP.ParamDict,
    overrides = NamedTuple(),
)
    name_map = (;
        :ogw_mountain_height_width_exponent => :γ, # L ∝ h^γ (equation 14, paper suggests γ ≈ 0.4)
        :ogw_number_density_exponent => :ϵ, # number density of orography in a grid cell, n(h) ∝ h^(-ε)
        :ogw_mountain_shape_parameter => :β, # L(z) = L_b(1 - z/h)^β (equation 12), β=1 for triangular mountains and β<1 for blunt mounrains, β>1 for pointy mountains
        :ogw_critical_height_threshold => :h_frac, # h_crit = h_frac * (V / N)
        :ogw_density_scale_factor => :ρscale,
        :ogw_reference_mountain_width => :L0, # L_0 = 80 km
        :ogw_linear_drag_coefficient => :a0, # a_0 = 0.9
        :ogw_nonlinear_drag_coefficient => :a1, # a_1 = 3.0
        :ogw_critical_froude_number => :Fr_crit, # Fr_crit = 0.7
    )
    parameters = CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
    parameters = merge(parameters, overrides)
    FT = CP.float_type(toml_dict)
    CAP.OrographicGravityWaveParameters{FT}(; parameters...)
end


"""
    BeresSourceParameters(FT, overrides = NamedTuple())
    BeresSourceParameters(toml_dict, overrides = NamedTuple())

Build the Beres convective-source parameter set, with `overrides` merged over
the ClimaParams values.
"""
BeresSourceParameters(
    ::Type{FT},
    overrides = NamedTuple(),
) where {FT <: AbstractFloat} =
    BeresSourceParameters(CP.create_toml_dict(FT), overrides)

function BeresSourceParameters(toml_dict::CP.ParamDict, overrides = NamedTuple())
    name_map = (;
        :nogw_beres_Q0_threshold => :Q0_threshold,
        :nogw_beres_scale_factor => :scale_factor,
        :nogw_beres_sigma_x => :σ_x,
        :nogw_beres_nu_min => :ν_min,
        :nogw_beres_nu_max => :ν_max,
        :nogw_beres_n_nu => :n_ν,
        :nogw_beres_h_heat_min => :h_heat_min,
        :nogw_beres_n_h_avg => :n_h_avg,
        :nogw_beres_delta_h_frac => :Δh_frac,
        :nogw_beres_z_bot_floor => :z_bot_floor,
        :nogw_beres_steady_dc_frac => :steady_dc_frac,
        :nogw_beres_L_system => :L_system,
    )
    parameters = CP.get_parameter_values(toml_dict, name_map, "ClimaAtmos")
    parameters = merge(parameters, overrides)
    FT = CP.float_type(toml_dict)
    CAP.BeresSourceParameters{FT}(; parameters...)
end
