import Thermodynamics as TD
import ClimaUtilities
import ClimaCore.Operators
import ClimaUtilities.TimeVaryingInputs: evaluate!
import CloudMicrophysics as CM
import ..Parameters as CAP
import ..PrescribedCloudInRadiation
import ..lazy
import ..specific
import ..NonEquilibriumMicrophysics

update_atmospheric_state!(integrator) =
    update_atmospheric_state!(integrator.p.atmos.radiation_mode, integrator)

function update_atmospheric_state!(radiation_mode::GrayRadiation, integrator)
    # update temperature & pressure
    update_temperature_pressure!(integrator)
    return nothing
end

function update_atmospheric_state!(radiation_mode::R, integrator) where {R}
    # update temperature & pressure
    update_temperature_pressure!(integrator)
    # update relative humidity
    update_relative_humidity!(integrator)
    # update gas concentrations (volume mixing ratios)
    update_volume_mixing_ratios!(integrator)
    # update aerosol concentrations
    update_aerosol_concentrations!(integrator)
    # update cloud properties
    if radiation_mode isa AllSkyRadiation ||
       radiation_mode isa AllSkyRadiationWithClearSkyDiagnostics
        update_cloud_properties!(integrator)
    end
    return nothing
end

import RRTMGP

"""
    update_temperature_pressure!((; u, p, t)::I) where {I}

Update temperature and pressure, given solution `u`, cache `p`
and simulation time `t`. Updates the surface temperature, layer temperature, and layer
pressure inputs of `p.radiation.rrtmgp_solver` (via the `RRTMGP` getters).
"""
function update_temperature_pressure!((; u, p, t)::I) where {I}
    (; ᶜp, ᶜT, sfc_conditions) = p.precomputed
    model = p.radiation.rrtmgp_solver

    # update surface temperature
    RRTMGP.surface_temperature(model) .= Fields.field2array(sfc_conditions.T_sfc)
    # update layer pressure
    RRTMGP.layer_pressure(model) .= Fields.field2array(ᶜp)
    # update layer temperature (RRTMGP clamps it to the lookup-table bounds
    # in its own input preparation, `RRTMGP.clip!`)
    RRTMGP.layer_temperature(model) .= Fields.field2array(ᶜT)
    return nothing
end

"""
    update_relative_humidity!(integrator)

Update relative humidity `ᶜrh`.
"""
function update_relative_humidity!((; u, p, t)::I) where {I}
    (; radiation_mode) = p.atmos
    (; rrtmgp_solver) = p.radiation
    thermo_params = CAP.thermodynamics_params(p.params)
    FT = eltype(thermo_params)
    (; ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    ᶜrh = Fields.array2field(RRTMGP.layer_relative_humidity(rrtmgp_solver), axes(u.c))
    ᶜvmr_h2o = Fields.array2field(
        RRTMGP.volume_mixing_ratio(rrtmgp_solver, "h2o"),
        axes(u.c),
    )
    if radiation_mode.idealized_h2o
        # slowly increase the relative humidity from 0 to 0.6 to account for
        # the fact that we have a very unrealistic initial condition
        max_relative_humidity = FT(0.6)
        t_increasing_humidity = FT(60 * 60 * 24 * 30)
        if FT(t) < t_increasing_humidity
            max_relative_humidity *= FT(t) / t_increasing_humidity
        end
        @. ᶜrh = max_relative_humidity

        # temporarily store ᶜq_tot in ᶜvmr_h2o
        ᶜq_tot = ᶜvmr_h2o
        @. ᶜq_tot =
            max_relative_humidity *
            TD.q_vap_saturation(thermo_params, ᶜT, u.c.ρ, ᶜq_liq, ᶜq_ice)

        # filter ᶜq_tot so that it is monotonically decreasing with z
        for i in 2:Spaces.nlevels(axes(ᶜq_tot))
            level = Fields.field_values(Spaces.level(ᶜq_tot, i))
            prev_level = Fields.field_values(Spaces.level(ᶜq_tot, i - 1))
            @. level = min(level, prev_level)
        end

        # assume that ᶜq_vap = ᶜq_tot when computing ᶜvmr_h2o
        @. ᶜvmr_h2o = TD.vol_vapor_mixing_ratio(thermo_params, ᶜq_tot)
    else
        @. ᶜvmr_h2o =
            TD.vol_vapor_mixing_ratio(
                thermo_params,
                ᶜq_tot_nonneg,
                ᶜq_liq,
                ᶜq_ice,
            )
        @. ᶜrh = min(
            max(
                TD.relative_humidity(
                    thermo_params,
                    ᶜT,
                    ᶜp,
                    ᶜq_tot_nonneg,
                    ᶜq_liq,
                    ᶜq_ice,
                ),
                0,
            ),
            1,
        )
    end

    return nothing
end

"""
    update_volume_mixing_ratios!((; p, t)::I) where {I}

Update volume mixing ratios.
"""
function update_volume_mixing_ratios!((; u, p, t)::I) where {I}
    (; rrtmgp_solver) = p.radiation

    if :o3 in propertynames(p.tracers)
        evaluate!(p.tracers.o3, p.tracers.prescribed_o3_timevaryinginput, t)

        ᶜvmr_o3 = Fields.array2field(
            RRTMGP.volume_mixing_ratio(rrtmgp_solver, "o3"),
            axes(u.c),
        )
        @. ᶜvmr_o3 = p.tracers.o3
    end
    if :co2 in propertynames(p.tracers)
        evaluate!(p.tracers.co2, p.tracers.prescribed_co2_timevaryinginput, t)

        if pkgversion(ClimaUtilities) < v"0.1.21"
            RRTMGP.set_volume_mixing_ratio!(rrtmgp_solver, "co2", p.tracers.co2)
        else
            RRTMGP.set_volume_mixing_ratio!(rrtmgp_solver, "co2", p.tracers.co2[])
        end
    end

    return nothing
end

"""
    update_aerosol_concentrations!((; u, p, t)::I) where {I}

Updates aerosol concentrations for supported aerosol names (dust (5 types),
sea-salt (5 types), sulfates, black-carbon (2 types), organic-carbon (2 types))
"""
function update_aerosol_concentrations!((; u, p, t)::I) where {I}
    (; radiation_mode) = p.atmos
    (; rrtmgp_solver) = p.radiation
    if :prescribed_aerosols_field in propertynames(p.tracers)
        for (key, tv) in pairs(p.tracers.prescribed_aerosol_timevaryinginputs)
            field = getproperty(p.tracers.prescribed_aerosols_field, key)
            evaluate!(field, tv, t)
        end
    end

    # RRTMGP needs effective radius in microns
    if radiation_mode.aerosol_radiation
        ᶜΔz = Fields.Δz_field(u.c)

        more_dust_aerosols = (
            ("dust1", :DST01),
            ("dust2", :DST02),
            ("dust3", :DST03),
            ("dust4", :DST04),
            ("dust5", :DST05),
        )

        more_ssl_aerosols = (
            ("sea_salt1", :SSLT01),
            ("sea_salt2", :SSLT02),
            ("sea_salt3", :SSLT03),
            ("sea_salt4", :SSLT04),
            ("sea_salt5", :SSLT05),
        )

        aerosol_names_pair = [
            more_dust_aerosols...,
            more_ssl_aerosols...,
            ("sulfate", :SO4),
            ("black_carbon_rh", :CB2),
            ("black_carbon", :CB1),
            ("organic_carbon_rh", :OC2),
            ("organic_carbon", :OC1),
        ]

        for (rrtmgp_aerosol_name, prescribed_aerosol_name) in aerosol_names_pair

            ᶜaero_conc = Fields.array2field(
                RRTMGP.aerosol_column_mass_density(rrtmgp_solver, rrtmgp_aerosol_name),
                axes(u.c),
            )
            if prescribed_aerosol_name in
               propertynames(p.tracers.prescribed_aerosols_field)
                aerosol_field = getproperty(
                    p.tracers.prescribed_aerosols_field,
                    prescribed_aerosol_name,
                )
                @. ᶜaero_conc = aerosol_field * u.c.ρ * ᶜΔz
            else
                @. ᶜaero_conc = 0
            end
        end
    end

    return nothing
end

"""
    ᶜcloud_liquid_water_content(microphysics_model, u, ᶜq_liq)
    ᶜcloud_ice_water_content(microphysics_model, u, ᶜq_ice)

Cloud condensate specific contents seen by radiation. With non-equilibrium
microphysics, the precomputed `ᶜq_liq` / `ᶜq_ice` include precipitation
(`q_lcl + q_rai` / `q_icl + q_sno`), so use the prognostic cloud condensate
only; otherwise fall back to the precomputed values.
"""
ᶜcloud_liquid_water_content(::NonEquilibriumMicrophysics, u, ᶜq_liq) =
    @. lazy(max(0, specific(u.c.ρq_lcl, u.c.ρ)))
ᶜcloud_liquid_water_content(microphysics_model, u, ᶜq_liq) = ᶜq_liq
ᶜcloud_ice_water_content(::NonEquilibriumMicrophysics, u, ᶜq_ice) =
    @. lazy(max(0, specific(u.c.ρq_icl, u.c.ρ)))
ᶜcloud_ice_water_content(microphysics_model, u, ᶜq_ice) = ᶜq_ice

"""
    update_cloud_properties((; u, p, t)::I) where {I}

Updates cloud properties:
Updates `cloud_liquid_water_content (ᶜlwp)`, `cloud_ice_water_content (ᶜiwp)`,
`cloud_fraction (ᶜfrac)`, `ᶜliquid_water_mass_concentration`, `ᶜreliq`, `ᶜreice`.
Updates aerosol properties for the following supported symbols:
seasalt_names = [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
dust_names = [:DST01, :DST02, :DST03, :DST04, :DST05]
SO4_names = [:SO4]
When prescribed cloud fields are used, time-varying interpolation is applied using
`ClimaUtilities` functions.
No updates are applied when `radiation_mode.idealized_clouds` is true.
"""
function update_cloud_properties!((; u, p, t)::I) where {I}
    (; radiation_mode) = p.atmos
    (; rrtmgp_solver) = p.radiation
    (; ᶜcloud_fraction, ᶜq_liq, ᶜq_ice) = p.precomputed
    FT = Spaces.undertype(axes(u.c))
    cmc = CAP.microphysics_cloud_params(p.params)

    if :prescribed_clouds_field in propertynames(p.radiation)
        for (key, tv) in pairs(p.radiation.prescribed_cloud_timevaryinginputs)
            field = getproperty(p.radiation.prescribed_clouds_field, key)
            evaluate!(field, tv, t)
        end
    end

    if !radiation_mode.idealized_clouds
        ᶜΔz = Fields.Δz_field(u.c)
        ᶜlwp = Fields.array2field(
            RRTMGP.cloud_liquid_water_path(rrtmgp_solver),
            axes(u.c),
        )
        ᶜiwp = Fields.array2field(
            RRTMGP.cloud_ice_water_path(rrtmgp_solver),
            axes(u.c),
        )
        ᶜfrac =
            Fields.array2field(RRTMGP.cloud_fraction(rrtmgp_solver), axes(u.c))
        ᶜreliq = Fields.array2field(
            RRTMGP.cloud_liquid_effective_radius(rrtmgp_solver),
            axes(u.c),
        )
        ᶜreice = Fields.array2field(
            RRTMGP.cloud_ice_effective_radius(rrtmgp_solver),
            axes(u.c),
        )
        # RRTMGP needs lwp and iwp in g/m^2
        kg_to_g_factor = 1000
        m_to_um_factor = FT(1e6)
        cloud_liquid_water_content =
            radiation_mode.cloud isa PrescribedCloudInRadiation ?
            p.radiation.prescribed_clouds_field.clwc :
            ᶜcloud_liquid_water_content(p.atmos.microphysics_model, u, ᶜq_liq)
        cloud_ice_water_content =
            radiation_mode.cloud isa PrescribedCloudInRadiation ?
            p.radiation.prescribed_clouds_field.ciwc :
            ᶜcloud_ice_water_content(p.atmos.microphysics_model, u, ᶜq_ice)
        cloud_fraction =
            radiation_mode.cloud isa PrescribedCloudInRadiation ?
            p.radiation.prescribed_clouds_field.cc : ᶜcloud_fraction
        @. ᶜlwp =
            kg_to_g_factor * u.c.ρ * cloud_liquid_water_content * ᶜΔz /
            max(cloud_fraction, eps(FT))
        @. ᶜiwp =
            kg_to_g_factor * u.c.ρ * cloud_ice_water_content * ᶜΔz /
            max(cloud_fraction, eps(FT))
        @. ᶜfrac = cloud_fraction
        # RRTMGP needs effective radius in microns
        seasalt_aero_conc = p.scratch.ᶜtemp_scalar
        dust_aero_conc = p.scratch.ᶜtemp_scalar_2
        SO4_aero_conc = p.scratch.ᶜtemp_scalar_3
        @. seasalt_aero_conc = 0
        @. dust_aero_conc = 0
        @. SO4_aero_conc = 0
        # Get aerosol mass concentrations if available
        seasalt_names = [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
        dust_names = [:DST01, :DST02, :DST03, :DST04, :DST05]
        SO4_names = [:SO4]
        if :prescribed_aerosols_field in propertynames(p.tracers)
            aerosol_field = p.tracers.prescribed_aerosols_field
            for aerosol_name in propertynames(aerosol_field)
                if aerosol_name in seasalt_names
                    data = getproperty(aerosol_field, aerosol_name)
                    @. seasalt_aero_conc += data
                elseif aerosol_name in dust_names
                    data = getproperty(aerosol_field, aerosol_name)
                    @. dust_aero_conc += data
                elseif aerosol_name in SO4_names
                    data = getproperty(aerosol_field, aerosol_name)
                    @. SO4_aero_conc += data
                end
            end
        end
        lwp_col = p.scratch.temp_field_level
        ᶜliquid_water_mass_concentration =
            @. lazy(cloud_liquid_water_content * u.c.ρ)
        Operators.column_integral_definite!(
            lwp_col,
            ᶜliquid_water_mass_concentration,
        )

        @. ᶜreliq = ifelse(
            cloud_liquid_water_content > FT(0),
            CM.CloudDiagnostics.effective_radius_Liu_Hallet_97(
                cmc.liquid,
                u.c.ρ,
                max(FT(0), cloud_liquid_water_content) /
                max(eps(FT), cloud_fraction),
                ml_N_cloud_liquid_droplets(
                    (cmc,),
                    dust_aero_conc,
                    seasalt_aero_conc,
                    SO4_aero_conc,
                    lwp_col,
                ),
                FT(0),
                FT(0),
            ) * m_to_um_factor,
            FT(0),
        )

        @. ᶜreice = ifelse(
            cloud_ice_water_content > FT(0),
            CM.CloudDiagnostics.effective_radius_const(cmc.ice) *
            m_to_um_factor,
            FT(0),
        )
    end
    return nothing
end


"""
    ml_N_cloud_liquid_droplets(cmc, c_dust, c_seasalt, c_SO4, q_liq)

  - cmc - a struct with cloud and aerosol parameters
  - c_dust, c_seasalt, c_SO4 - dust, seasalt and ammonium sulfate mass concentrations [kg/kg]
  - q_liq - liquid water specific humidity

Returns the liquid cloud droplet number concentration diagnosed based on the
aerosol loading and cloud liquid water.
"""
function ml_N_cloud_liquid_droplets(cmc, c_dust, c_seasalt, c_SO4, q_liq)
    # We can also add w, T, RH, w' ...
    # Also consider lookind only at around cloud base height
    (; α_dust, α_seasalt, α_SO4, α_q_liq) = cmc.aml
    (; c₀_dust, c₀_seasalt, c₀_SO4, q₀_liq) = cmc.aml
    N₀ = cmc.N_cloud_liquid_droplets

    FT = eltype(N₀)
    return N₀ * (
        FT(1) +
        α_dust * (log(max(c_dust, eps(FT))) - log(c₀_dust)) +
        α_seasalt * (log(max(c_seasalt, eps(FT))) - log(c₀_seasalt)) +
        α_SO4 * (log(max(c_SO4, eps(FT))) - log(c₀_SO4)) +
        α_q_liq * (log(max(q_liq, eps(FT))) - log(q₀_liq))
    )
end
