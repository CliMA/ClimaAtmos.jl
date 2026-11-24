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
import ..NonEquilibriumMicrophysics1M

"""
    update_atmospheric_state!(integrator)
    update_atmospheric_state!(radiation_mode, integrator)

Copy the current model state into the RRTMGP solver's input arrays; return `nothing`.

Called once per radiation callback, i.e. every `dt_rad`, before `RRTMGP.update_fluxes!`.
The one-argument method dispatches on `integrator.p.atmos.radiation_mode`. Which inputs are
refreshed depends on the mode:

  - `GrayRadiation`: temperature and pressure only.
  - All other modes: temperature and pressure, relative humidity and the water vapor volume
    mixing ratio, the prescribed trace gases, and the aerosol column mass densities.
  - `AllSkyRadiation` and `AllSkyRadiationWithClearSkyDiagnostics`: additionally the cloud
    water paths, cloud fraction, and effective radii.

Surface albedo and insolation are *not* set here; the callback sets them separately with
`set_surface_albedo!` and `set_insolation_variables!` after this function returns.

All updates write into arrays owned by `p.radiation.rrtmgp_solver`, accessed through the
`RRTMGP` getters. See the radiation docs page, docs/src/radiation.md.
"""
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
    update_temperature_pressure!(integrator)

Copy the surface temperature, layer temperature, and layer pressure into the RRTMGP solver
inputs; return `nothing`.

Reads `ᶜp`, `ᶜT`, and `sfc_conditions.T_sfc` from `p.precomputed` and writes them through
the `RRTMGP.surface_temperature`, `RRTMGP.layer_pressure`, and `RRTMGP.layer_temperature`
getters. Temperatures outside the lookup-table bounds are clipped by RRTMGP itself, in its
own input preparation, so no clamping is applied here.
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

Update the layer relative humidity and the water vapor volume mixing ratio in the RRTMGP
solver inputs; return `nothing`.

By default both are diagnosed from the model state, with relative humidity clipped to
`[0, 1]`. When `radiation_mode.idealized_h2o` is set, the relative humidity is instead
prescribed as a uniform value that ramps linearly from 0 to 0.6 over the first 30 days (to
absorb the shock of an unrealistic initial condition), and the corresponding `q_tot` is
filtered to be monotonically decreasing with height before being converted to a vapor
volume mixing ratio, assuming `q_vap = q_tot`.

Reads `ᶜT`, `ᶜp`, `ᶜq_tot_nonneg`, `ᶜq_liq`, and `ᶜq_ice` from `p.precomputed`.
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
    update_volume_mixing_ratios!(integrator)

Update the prescribed trace-gas volume mixing ratios in the RRTMGP solver inputs; return
`nothing`.

Only gases configured as time-varying are touched: ozone is evaluated from its
`TimeVaryingInput` into `p.tracers.o3` and copied into the solver's `"o3"` profile, and
carbon dioxide is evaluated and set as a single global mean value. Gases held fixed keep
the values seeded at solver construction. Water vapor is handled by
`update_relative_humidity!`.
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
    update_aerosol_concentrations!(integrator)

Update the prescribed aerosol fields and the RRTMGP aerosol column mass densities; return
`nothing`.

Each prescribed aerosol field is first evaluated from its `TimeVaryingInput` at the current
time. Then, if `radiation_mode.aerosol_radiation` is enabled, the specific mass of each
supported species is converted to a layer column mass density, `ρ q_aero Δz` [kg/m²], and
written into the solver; species absent from the configuration are set to zero.

The supported species are dust (5 size bins), sea salt (5 size bins), sulfate, hydrophilic
and hydrophobic black carbon, and hydrophilic and hydrophobic organic carbon.
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

Return the cloud condensate specific contents seen by radiation [kg/kg].

With `NonEquilibriumMicrophysics` the precomputed `ᶜq_liq` and `ᶜq_ice` also include
precipitation (`q_lcl + q_rai` and `q_icl + q_sno`), which must not contribute to cloud
optics, so these methods return a lazy broadcast of the prognostic cloud condensate alone,
floored at zero. For every other microphysics model the precomputed values already exclude
precipitation and are passed through unchanged.

Called from `update_cloud_properties!`.
"""
ᶜcloud_liquid_water_content(::NonEquilibriumMicrophysics, u, ᶜq_liq) =
    @. lazy(max(0, specific(u.c.ρq_lcl, u.c.ρ)))
ᶜcloud_liquid_water_content(microphysics_model, u, ᶜq_liq) = ᶜq_liq
ᶜcloud_ice_water_content(::NonEquilibriumMicrophysics, u, ᶜq_ice) =
    @. lazy(max(0, specific(u.c.ρq_icl, u.c.ρ)))
ᶜcloud_ice_water_content(microphysics_model, u, ᶜq_ice) = ᶜq_ice

"""
    ᶜsnow_water_content(microphysics_model, u)

Return the prognostic snow specific content seen by radiation [kg/kg].

Only microphysics models that carry prognostic snow (`NonEquilibriumMicrophysics1M`) define a
value; calling this on any other model errors, which is how `include_snow` is enforced
against an incompatible scheme. Called from `update_cloud_properties!`.
"""
ᶜsnow_water_content(::NonEquilibriumMicrophysics1M, u) =
    @. lazy(max(0, specific(u.c.ρq_sno, u.c.ρ)))
ᶜsnow_water_content(microphysics_model, u) = error(
    "include_snow is set but microphysics model $(typeof(microphysics_model)) has no \
     prognostic snow (q_sno); use a 1-moment scheme or set include_snow = false",
)

"""
    update_cloud_properties!(integrator)

Update the cloud inputs of the RRTMGP solver; return `nothing`.

Writes the in-cloud liquid and ice water paths, the cloud fraction, and the liquid and ice
effective radii through the `RRTMGP.cloud_liquid_water_path`, `RRTMGP.cloud_ice_water_path`,
`RRTMGP.cloud_fraction`, `RRTMGP.cloud_liquid_effective_radius`, and
`RRTMGP.cloud_ice_effective_radius` getters. Water paths are grid-mean `ρ q Δz` divided by
the cloud fraction (floored at `eps`) to make them in-cloud values, and are converted to
the units RRTMGP expects: g/m² for the paths and microns for the radii. The paths are
the mean in-cloud values; the shortwave solve scales them by the subgrid inhomogeneity
factor afterwards (`solve_fluxes!`). When `radiation_mode.include_snow` is set,
the in-cloud snow water path (from the prognostic `q_sno` of a non-equilibrium
microphysics scheme, via `ᶜsnow_water_content`) and a placeholder constant snow
effective radius are additionally written through `RRTMGP.cloud_snow_water_path` and
`RRTMGP.cloud_snow_effective_radius`. ** Note: the placeholder snow effective radius is
currently too small, and snow is not currently included in cloud_fraction calculations,
meaning that the way it is currently implemented in radiation is not physically correct
and subject to change in the future.

The liquid effective radius follows the Liu and Hallett (1997) parameterization,
evaluated with a droplet number concentration diagnosed by
`ml_N_cloud_liquid_droplets` from the prescribed sea-salt, dust, and sulfate
mass concentrations and the in-cloud liquid specific humidity; a species with no
prescribed bins enters at its reference concentration (`sum_aerosol_bins!`). The
ice effective radius is a constant. Both radii are zero where the condensate or
the cloud fraction is zero. Cloud condensate and cloud fraction come from the
model state, or from the ERA5 fields in `p.radiation.prescribed_clouds_field`
when `radiation_mode.cloud isa PrescribedCloudInRadiation`; those prescribed
fields are re-evaluated from their `TimeVaryingInput`s first, regardless of the
branch below.

Nothing is written when `radiation_mode.idealized_clouds` is true, since those
cloud layers are prescribed once at solver construction. Uses
`p.scratch.ᶜtemp_scalar`, `ᶜtemp_scalar_2`, and `ᶜtemp_scalar_3`.
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
        zero_FT = zero(FT)
        eps_FT = eps(FT)
        cmc_liquid = cmc.liquid
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
            max(cloud_fraction, eps_FT)
        @. ᶜiwp =
            kg_to_g_factor * u.c.ρ * cloud_ice_water_content * ᶜΔz /
            max(cloud_fraction, eps_FT)
        @. ᶜfrac = cloud_fraction
        # RRTMGP needs effective radius in microns
        aerosol_field =
            :prescribed_aerosols_field in propertynames(p.tracers) ?
            p.tracers.prescribed_aerosols_field : nothing
        (; c₀_dust, c₀_seasalt, c₀_SO4) = cmc.aml
        seasalt_aero_conc = sum_aerosol_bins!(
            p.scratch.ᶜtemp_scalar,
            aerosol_field,
            (:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05),
            c₀_seasalt,
        )
        dust_aero_conc = sum_aerosol_bins!(
            p.scratch.ᶜtemp_scalar_2,
            aerosol_field,
            (:DST01, :DST02, :DST03, :DST04, :DST05),
            c₀_dust,
        )
        SO4_aero_conc = sum_aerosol_bins!(
            p.scratch.ᶜtemp_scalar_3,
            aerosol_field,
            (:SO4,),
            c₀_SO4,
        )
        ᶜq_liq_in = @. lazy(
            max(zero_FT, cloud_liquid_water_content) /
            max(eps_FT, cloud_fraction),
        )

        @. ᶜreliq = ifelse(
            (cloud_liquid_water_content > zero_FT) &
            (cloud_fraction > zero_FT),
            CM.CloudDiagnostics.effective_radius_Liu_Hallet_97(
                cmc_liquid,
                u.c.ρ,
                ᶜq_liq_in,
                ml_N_cloud_liquid_droplets(
                    (cmc,),
                    dust_aero_conc,
                    seasalt_aero_conc,
                    SO4_aero_conc,
                    ᶜq_liq_in,
                ),
                zero_FT,
                zero_FT,
            ) * m_to_um_factor,
            zero_FT,
        )

        reice_um =
            CM.CloudDiagnostics.effective_radius_const(cmc.ice) * m_to_um_factor
        @. ᶜreice = ifelse(
            (cloud_ice_water_content > zero_FT) & (cloud_fraction > zero_FT),
            reice_um,
            zero_FT,
        )

        # Snow (placeholder): write the in-cloud snow water path and a placeholder
        # snow effective radius through the RRTMGP snow getters, gated on include_snow.
        # RRTMGP reuses the ice optics and clamps the radius to the ice-table maximum
        # (~90 µm), so the constant below only matters up to that ceiling. Snow is
        # normalized to an in-cloud value and (via RRTMGP's cloud mask) radiates only in
        # cloudy layers, matching the liquid/ice treatment.
        # Additionally, snow is not currently included in cloud fraction, and therefore
        # using cloud fraction here is erronous and should be changed in the future.
        if radiation_mode.include_snow
            ᶜswp = Fields.array2field(
                RRTMGP.cloud_snow_water_path(rrtmgp_solver),
                axes(u.c),
            )
            ᶜresno = Fields.array2field(
                RRTMGP.cloud_snow_effective_radius(rrtmgp_solver),
                axes(u.c),
            )
            snow_water_content =
                ᶜsnow_water_content(p.atmos.microphysics_model, u)
            # Placeholder constant snow effective radius [μm]; RRTMGP clamps to ~90 µm
            snow_effective_radius_um = FT(90)
            @. ᶜswp =
                kg_to_g_factor * u.c.ρ * snow_water_content * ᶜΔz /
                max(cloud_fraction, eps_FT)
            @. ᶜresno = ifelse(
                (snow_water_content > zero_FT) & (cloud_fraction > zero_FT),
                snow_effective_radius_um,
                zero_FT,
            )
        end
    end
    return nothing
end

"""
    sum_aerosol_bins!(ᶜconc, aerosol_field, bin_names, c₀)

Sum the prescribed aerosol bins named in `bin_names` into `ᶜconc` [kg/kg]; return
`ᶜconc`.

When `aerosol_field` is `nothing` or holds none of the bins, `ᶜconc` is set to the
reference concentration `c₀`, so the species term of `ml_N_cloud_liquid_droplets`
vanishes and an unprescribed species leaves the droplet number at its reference
value. Called from `update_cloud_properties!`.
"""
function sum_aerosol_bins!(ᶜconc, aerosol_field, bin_names, c₀)
    present_names =
        isnothing(aerosol_field) ? () :
        filter(in(bin_names), propertynames(aerosol_field))
    if isempty(present_names)
        fill!(ᶜconc, c₀)
    else
        fill!(ᶜconc, zero(c₀))
        for name in present_names
            ᶜbin = getproperty(aerosol_field, name)
            @. ᶜconc += ᶜbin
        end
    end
    return ᶜconc
end

"""
    ml_N_cloud_liquid_droplets(cmc, c_dust, c_seasalt, c_SO4, q_liq)

Return the cloud droplet number concentration diagnosed from the aerosol loading
and the in-cloud liquid specific humidity [1/m³].

The closure is a bounded log-saturation law about a reference state. The sum

```math
s = Σ_i α_i \\, \\log(\\max(c_i / c_{0,i}, c_{\\min}))
    + α_{q_l} \\log(\\max(q_l / q_{0,l}, c_{\\min}))
```

runs over dust, sea salt, and ammonium sulfate, and

```math
N = N₀ \\exp\\left(Δ_{\\max} \\tanh(s / Δ_{\\max})\\right).
```

The floor ``c_{\\min} = 10^{-3}`` keeps the logarithms finite for vanishing
concentrations, and ``Δ_{\\max} = \\ln(3)`` bounds ``N`` in ``[N₀ / 3, 3 N₀]``;
for small ``|s|``, ``N ≈ N₀ (1 + s)``. The calibration coefficients `α` and
reference values `c₀`, `q₀` come from `cmc.aml`; with the default ``α = 0``,
``N = N₀``.

!!! note "TODO: calibrate the bound and the floor"

    The bound and the floor are provisional values set in the code. Both should
    become ClimaParams parameters when the coefficients are calibrated.

# Arguments

  - `cmc`: Cloud and aerosol parameter set (`CAP.microphysics_cloud_params`),
    providing `cmc.aml` and the reference concentration
    `cmc.N_cloud_liquid_droplets` [1/m³].
  - `c_dust`, `c_seasalt`, `c_SO4`: Dust, sea-salt, and ammonium sulfate mass
    concentrations [kg/kg].
  - `q_liq`: In-cloud liquid water specific humidity [kg/kg], compared against
    `q₀_liq`.
"""
@inline function ml_N_cloud_liquid_droplets(
    cmc,
    c_dust,
    c_seasalt,
    c_SO4,
    q_liq,
)
    (; α_dust, α_seasalt, α_SO4, α_q_liq) = cmc.aml
    (; c₀_dust, c₀_seasalt, c₀_SO4, q₀_liq) = cmc.aml
    N₀ = cmc.N_cloud_liquid_droplets

    FT = eltype(N₀)
    c_min_scale = FT(1e-3)
    s =
        α_dust * log(max(c_dust / c₀_dust, c_min_scale)) +
        α_seasalt * log(max(c_seasalt / c₀_seasalt, c_min_scale)) +
        α_SO4 * log(max(c_SO4 / c₀_SO4, c_min_scale)) +
        α_q_liq * log(max(q_liq / q₀_liq, c_min_scale))

    Δ_max = log(FT(3))
    return N₀ * exp(Δ_max * tanh(s / Δ_max))
end

"""
    solve_fluxes!(integrator)

Run the RRTMGP radiative transfer on the solver's current inputs; return `nothing`.
Called from the radiation callback after `update_atmospheric_state!`, the insolation,
and the surface albedo are set; the net flux is read afterwards with
`RRTMGP.net_flux`.

The McICA key is the timestep index, so the cloud sample is reproducible across
restarts. With `p.radiation.ᶜsw_cloud_optical_depth_factor` present, the two bands are
solved separately: the longwave sees the mean in-cloud water paths that
`update_cloud_properties!` wrote, and the shortwave sees them scaled by the
effective-thickness factor `χ` (`set_sw_cloud_optical_depth_factor!`), which is
derived for reflectance and would understate the emissivity of thin clouds. The
shortwave solve reuses the longwave McICA key, as `RRTMGP.update_fluxes!` does, so
both bands sample the same cloud mask. The scaled paths stay in the solver until the
next callback rewrites them, which is why the factor is only allocated for clouds
that are rewritten every call (`get_cloud_cache`). Without the factor, the solve is
the single `RRTMGP.update_fluxes!` call.
"""
function solve_fluxes!((; u, p, t)::I) where {I}
    (; rrtmgp_solver) = p.radiation
    FT = Spaces.undertype(axes(u.c))
    key = UInt32(floor(FT(t) / p.dt))
    if hasproperty(p.radiation, :ᶜsw_cloud_optical_depth_factor)
        RRTMGP.check_values[] && RRTMGP.validate_inputs(rrtmgp_solver)
        RRTMGP.prepare_atmosphere!(rrtmgp_solver)
        RRTMGP.update_lw_fluxes!(rrtmgp_solver, key)
        ᶜχ = p.radiation.ᶜsw_cloud_optical_depth_factor
        ᶜlwp = Fields.array2field(
            RRTMGP.cloud_liquid_water_path(rrtmgp_solver),
            axes(u.c),
        )
        ᶜiwp = Fields.array2field(
            RRTMGP.cloud_ice_water_path(rrtmgp_solver),
            axes(u.c),
        )
        @. ᶜlwp *= ᶜχ
        @. ᶜiwp *= ᶜχ
        RRTMGP.update_sw_fluxes!(rrtmgp_solver)
        RRTMGP.update_net_fluxes!(rrtmgp_solver)
    else
        RRTMGP.update_fluxes!(rrtmgp_solver, key)
    end
    return nothing
end
