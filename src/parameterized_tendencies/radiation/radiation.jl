#####
##### Radiation
#####

import ClimaComms
import ClimaCore: DataLayouts, Geometry, Spaces, Fields, Operators
import Insolation
import Thermodynamics as TD
import .Parameters as CAP
import RRTMGP
import .RRTMGPInterface as RRTMGPI

import Dates: Year
import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput, LinearPeriodFillingInterpolation

import Interpolations
using Statistics: mean

radiation_model_cache(Y, atmos::AtmosModel, args...) =
    radiation_model_cache(Y, atmos.radiation_mode, args...)

#####
##### No Radiation
#####

radiation_model_cache(Y, radiation_mode::Nothing; args...) = (;)
radiation_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

#####
##### RRTMGP Radiation
#####

#########
# Ozone #
#########

function center_vmr_o3(::IdealizedOzone, Y)
    ᶜvolume_mixing_ratio_o3_field =
        idealized_ozone.(Fields.coordinate_field(Y.c).z)
    return Fields.field2array(ᶜvolume_mixing_ratio_o3_field)
end

# Initialized in callback
center_vmr_o3(::PrescribedOzone, _) = NaN

"""
    idealized_ozone(z::FT)

Returns idealized ozone volume mixing ratio (VMR) from Wing et al. 2018.

The ozone profile is calculated as a function of altitude `z` using the following formula:

```math
O_3(z) = g_1 p^{g_2} e^{(-p / g_3)}
```

where:

- `O_3(z)` is the ozone concentration in volume mixing ratio (VMR) at altitude `z`.

- `p` is the pressure at altitude `z` calculated using the hydrostatic equation:
  `p = P_0 exp(-z / H_{Earth})`, where `P_0` is the surface pressure and
  `H_{Earth}` is the scale height of the Earth's atmosphere (assumed to be 7000
  meters).

- `g_1`, `g_2`, and `g_3` are empirical constants.

**References**

- Wing, A. A., et al. (2018). Radiative-convective equilibrium model intercomparison
  project. Geoscientific Model Development, 11(2), 663-690.
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

function radiation_model_cache(
    Y,
    radiation_mode::RRTMGPI.AbstractRRTMGPMode,
    params,
    ozone,
    aerosol_names,
    insolation_mode;
    interpolation = RRTMGPI.BestFit(),
    bottom_extrapolation = RRTMGPI.SameAsInterpolation(),
    data_loader = rrtmgp_data_loader,
)
    context = ClimaComms.context(axes(Y.c))
    device = context.device
    if !(radiation_mode isa RRTMGPI.GrayRadiation)
        (; aerosol_radiation) = radiation_mode
        if aerosol_radiation && !(any(
            x -> x in aerosol_names,
            ["DST01", "SSLT01", "SO4", "CB1", "CB2", "OC1", "OC2"],
        ))
            error(
                "Need at least one aerosol type when aerosol radiation is turned on",
            )
        end
    end
    FT = Spaces.undertype(axes(Y.c))
    DA = ClimaComms.array_type(device){FT}
    rrtmgp_params = CAP.rrtmgp_params(params)

    bottom_coords = Fields.coordinate_field(Spaces.level(Y.c, 1))
    if eltype(bottom_coords) <: Geometry.LatLongZPoint
        latitude = Fields.field2array(bottom_coords.lat)
    else
        latitude = Fields.field2array(zero(bottom_coords.z)) # flat space is on Equator
    end
    local rrtmgp_model
    data_loader(
        RRTMGP.ArtifactPaths.get_input_filename(:gas, :lw),
    ) do input_data
        if radiation_mode isa RRTMGPI.GrayRadiation
            kwargs = (;
                lapse_rate = 3.5,
                optical_thickness_parameter = (@. 7.2 +
                                                  (1.8 - 7.2) *
                                                  sind(latitude)^2),
                latitude,
            )
        else
            center_volume_mixing_ratio_o3 = center_vmr_o3(ozone, Y)

            # the first value for each global mean volume mixing ratio is the
            # present-day value
            input_vmr(name) =
                input_data[name][1] *
                parse(FT, input_data[name].attrib["units"])
            kwargs = (;
                use_global_means_for_well_mixed_gases = true,
                center_volume_mixing_ratio_h2o = NaN, # initialize in tendency
                center_relative_humidity = NaN, # initialized in callback
                center_volume_mixing_ratio_o3,
                volume_mixing_ratio_co2 = input_vmr("carbon_dioxide_GM"),
                volume_mixing_ratio_n2o = input_vmr("nitrous_oxide_GM"),
                volume_mixing_ratio_co = input_vmr("carbon_monoxide_GM"),
                volume_mixing_ratio_ch4 = input_vmr("methane_GM"),
                volume_mixing_ratio_o2 = input_vmr("oxygen_GM"),
                volume_mixing_ratio_n2 = input_vmr("nitrogen_GM"),
                volume_mixing_ratio_ccl4 = input_vmr("carbon_tetrachloride_GM"),
                volume_mixing_ratio_cfc11 = input_vmr("cfc11_GM"),
                volume_mixing_ratio_cfc12 = input_vmr("cfc12_GM"),
                volume_mixing_ratio_cfc22 = input_vmr("hcfc22_GM"),
                volume_mixing_ratio_hfc143a = input_vmr("hfc143a_GM"),
                volume_mixing_ratio_hfc125 = input_vmr("hfc125_GM"),
                volume_mixing_ratio_hfc23 = input_vmr("hfc23_GM"),
                volume_mixing_ratio_hfc32 = input_vmr("hfc32_GM"),
                volume_mixing_ratio_hfc134a = input_vmr("hfc134a_GM"),
                volume_mixing_ratio_cf4 = input_vmr("cf4_GM"),
                volume_mixing_ratio_no2 = 1e-8, # not available in input_data
                latitude,
            )
            if !(radiation_mode isa RRTMGPI.ClearSkyRadiation)
                kwargs = (;
                    kwargs...,
                    center_cloud_liquid_effective_radius = 12,
                    center_cloud_ice_effective_radius = 95,
                    ice_roughness = 2,
                )
                ᶜz = Fields.coordinate_field(Y.c).z
                ᶜΔz = Fields.Δz_field(Y.c)
                if radiation_mode.idealized_clouds # icy cloud on top and wet cloud on bottom
                    # TODO: can we avoid using DataLayouts with this?
                    #     `ᶜis_bottom_cloud = similar(ᶜz, Bool)`
                    ᶜis_bottom_cloud = Fields.Field(
                        DataLayouts.replace_basetype(
                            Fields.field_values(ᶜz),
                            Bool,
                        ),
                        axes(Y.c),
                    ) # need to fix several ClimaCore bugs in order to simplify this
                    ᶜis_top_cloud = similar(ᶜis_bottom_cloud)
                    @. ᶜis_bottom_cloud = ᶜz > 1e3 && ᶜz < 1.5e3
                    @. ᶜis_top_cloud = ᶜz > 4e3 && ᶜz < 5e3
                    kwargs = (;
                        kwargs...,
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
                    )
                end
            end

            if aerosol_radiation
                kwargs = (;
                    kwargs...,
                    # assuming fixed aerosol radius
                    center_dust_radius = 0.2,
                    center_ss_radius = 0.2,
                    center_dust_column_mass_density = NaN, # initialized in callback
                    center_ss_column_mass_density = NaN, # initialized in callback
                    center_so4_column_mass_density = NaN, # initialized in callback
                    center_bcpi_column_mass_density = NaN, # initialized in callback
                    center_bcpo_column_mass_density = NaN, # initialized in callback
                    center_ocpi_column_mass_density = NaN, # initialized in callback
                    center_ocpo_column_mass_density = NaN, # initialized in callback
                )
            end
        end

        if RRTMGPI.requires_z(interpolation) ||
           RRTMGPI.requires_z(bottom_extrapolation)
            kwargs = (;
                kwargs...,
                center_z = Fields.field2array(Fields.coordinate_field(Y.c).z),
                face_z = Fields.field2array(Fields.coordinate_field(Y.f).z),
            )
        end

        cos_zenith = weighted_irradiance = NaN # initialized in callback

        rrtmgp_model = RRTMGPI.RRTMGPModel(
            rrtmgp_params,
            data_loader,
            context;
            ncol = length(Spaces.all_nodes(axes(Spaces.level(Y.c, 1)))),
            domain_nlay = Spaces.nlevels(axes(Y.c)),
            radiation_mode,
            interpolation,
            bottom_extrapolation,
            center_pressure = NaN, # initialized in callback
            center_temperature = NaN, # initialized in callback
            surface_temperature = NaN, # initialized in callback
            surface_emissivity = 1,
            direct_sw_surface_albedo = NaN, # initialized in callback
            diffuse_sw_surface_albedo = NaN, # initialized in callback
            cos_zenith,
            weighted_irradiance,
            kwargs...,
        )
    end
    return merge(
        (; rrtmgp_model, ᶠradiation_flux = similar(Y.f, Geometry.WVector{FT})),
        insolation_cache(insolation_mode, Y),
    )
end

insolation_cache(_, _) = (;)
function insolation_cache(::TimeVaryingInsolation, Y)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        insolation_tuple = similar(Spaces.level(Y.c, 1), Tuple{FT, FT, FT})
    )
end

function radiation_tendency!(Yₜ, Y, p, t, ::RRTMGPI.AbstractRRTMGPMode)
    (; ᶠradiation_flux) = p.radiation
    @. Yₜ.c.ρe_tot -= ᶜdivᵥ(ᶠradiation_flux)
    return nothing
end

#####
##### DYCOMS_RF01 and DYCOMS_RF02 radiation
#####

function radiation_model_cache(Y, radiation_mode::RadiationDYCOMS)
    FT = Spaces.undertype(axes(Y.c))
    NT = NamedTuple{(:z, :ρ, :q_tot), NTuple{3, FT}}
    return (;
        ᶜκρq = similar(Y.c, FT),
        ∫_0_∞_κρq = similar(Spaces.level(Y.c, 1), FT),
        ᶠ∫_0_z_κρq = similar(Y.f, FT),
        isoline_z_ρ_q = similar(Spaces.level(Y.c, 1), NT),
        ᶠradiation_flux = similar(Y.f, Geometry.WVector{FT}),
        net_energy_flux_toa = [Geometry.WVector(FT(0))],
        net_energy_flux_sfc = [Geometry.WVector(FT(0))],
    )
end
function radiation_tendency!(Yₜ, Y, p, t, radiation_mode::RadiationDYCOMS)
    @assert !(p.atmos.moisture_model isa DryModel)

    (; params) = p
    (; ᶜspecific, ᶜts) = p.precomputed
    (; ᶜκρq, ∫_0_∞_κρq, ᶠ∫_0_z_κρq, isoline_z_ρ_q, ᶠradiation_flux) =
        p.radiation
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = CAP.cp_d(params)
    FT = Spaces.undertype(axes(Y.c))
    NT = NamedTuple{(:z, :ρ, :q_tot), NTuple{3, FT}}
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z

    # TODO: According to the paper, we should replace liquid_specific_humidity
    # with TD.mixing_ratios(thermo_params, ᶜts).liq, but this wouldn't
    # match the original code from TurbulenceConvection.
    @. ᶜκρq =
        radiation_mode.kappa *
        Y.c.ρ *
        TD.liquid_specific_humidity(thermo_params, ᶜts)

    Operators.column_integral_definite!(∫_0_∞_κρq, ᶜκρq)

    Operators.column_integral_indefinite!(ᶠ∫_0_z_κρq, ᶜκρq)

    # Find the values of (z, ρ, q_tot) at the q_tot = 0.008 isoline, i.e., at
    # the level whose value of q_tot is closest to 0.008.
    let FT=FT
        Operators.column_reduce!(
            (nt1, nt2) ->
                abs(nt1.q_tot - FT(0.008)) < abs(nt2.q_tot - FT(0.008)) ? nt1 : nt2,
            isoline_z_ρ_q,
            Base.broadcasted(NT ∘ tuple, ᶜz, Y.c.ρ, ᶜspecific.q_tot),
        )
    end

    zi = isoline_z_ρ_q.z
    ρi = isoline_z_ρ_q.ρ

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

function radiation_model_cache(Y, radiation_mode::RadiationTRMM_LBA)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜdTdt_rad = similar(Y.c, FT),
        net_energy_flux_toa = [Geometry.WVector(FT(0))],
        net_energy_flux_sfc = [Geometry.WVector(FT(0))],
    )
end

function radiation_tendency!(Yₜ, Y, p, t, radiation_mode::RadiationTRMM_LBA)
    FT = Spaces.undertype(axes(Y.c))
    (; params) = p
    # TODO: get working (need to add cache / function)
    rad = radiation_mode.rad_profile
    thermo_params = CAP.thermodynamics_params(params)
    ᶜdTdt_rad = p.radiation.ᶜdTdt_rad
    ᶜρ = Y.c.ρ
    ᶜts_gm = p.precomputed.ᶜts
    zc = Fields.coordinate_field(axes(ᶜρ)).z
    @. ᶜdTdt_rad = rad(FT(t), zc)
    @. Yₜ.c.ρe_tot += ᶜρ * TD.cv_m(thermo_params, ᶜts_gm) * ᶜdTdt_rad
    return nothing
end

#####
##### ISDAC radiation
#####

radiation_model_cache(Y, radiation_mode::RadiationISDAC; args...) = (;)  # Don't need a cache for ISDAC
function radiation_tendency!(Yₜ, Y, p, t, radiation_mode::RadiationISDAC)
    (; F₀, F₁, κ) = radiation_mode
    (; params, precomputed) = p
    (; ᶜts) = precomputed
    thermo_params = CAP.thermodynamics_params(params)

    ᶜρq = p.scratch.ᶜtemp_scalar
    @. ᶜρq = Y.c.ρ * TD.liquid_specific_humidity(thermo_params, ᶜts)

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
