#####
##### Couples the atmospheric model with a slab surface model
#####

using .SurfaceConditions: SurfaceTemperature, SlabOceanTemperature

"""
    surface_precipitation_tendency!(Yₜ, Y, p, t, temperature, microphysics_model)

Add the surface water and energy deposition from precipitation.

Only a `SlabOceanTemperature` surface with a moist microphysics model does
anything; the other methods are no-ops. Decrements `Yₜ.sfc.T` by the
column-integrated precipitation energy tendency divided by the slab heat capacity
per unit area `ρ_ocean cp_ocean depth_ocean`, and decrements `Yₜ.sfc.water` by the
sum of the surface rain and snow fluxes, so precipitation warms or cools the slab
and adds water to it with the sign convention that fluxes are positive upward.

Reads the precomputed `col_integrated_precip_energy_tendency`,
`surface_rain_flux`, and `surface_snow_flux`; `t` is unused. Called from both
`implicit_tendency!` (when microphysics is implicit) and `remaining_tendency!`
(when it is explicit), so the surface deposition always uses the same cached
microphysics sources as the atmospheric water removal, preserving conservation
across IMEX stages.
"""
surface_precipitation_tendency!(Yₜ, Y, p, t, _, _) = nothing

surface_precipitation_tendency!(Yₜ, Y, p, t, ::SlabOceanTemperature, ::DryModel) = nothing

function surface_precipitation_tendency!(
    Yₜ, Y, p, t, slab::SlabOceanTemperature, microphysics_model,
)
    FT = eltype(Y)

    # Surface energy from precipitation
    pet = p.precomputed.col_integrated_precip_energy_tendency
    depth_ocean = slab.depth_ocean
    ρ_ocean = slab.ρ_ocean
    cp_ocean = slab.cp_ocean
    surface_heat_capacity_per_area = ρ_ocean * cp_ocean * depth_ocean
    @. Yₜ.sfc.T -= pet / surface_heat_capacity_per_area

    # Surface water from precipitation (rain + snow)
    P_liq = p.precomputed.surface_rain_flux
    P_snow = p.precomputed.surface_snow_flux
    @. Yₜ.sfc.water -= P_liq + P_snow
end

"""
    surface_temp_tendency!(Yₜ, Y, p, t, temperature::SurfaceTemperature)
    surface_temp_tendency!(Yₜ, Y, p, t, temperature::SlabOceanTemperature)

Add the tendencies of the prognostic surface temperature `Y.sfc.T` and, when
moisture is active, the surface water content `Y.sfc.water`.

All fluxes are positive when directed upward, from the surface to the atmosphere,
and are subtracted from the tendencies, so an upward flux cools and dries the
surface. The first method is a no-op: for any `SurfaceTemperature` other than a
slab ocean, `T` is prescribed or diagnosed. For `SlabOceanTemperature`:

  - `Yₜ.sfc.T` is decremented by the sum of the net upward radiative flux at the
    surface, the upward turbulent energy flux (sensible plus latent heat), and the
    idealized ocean heat-flux divergence `Q` (Q-flux, active only when
    `slab.q_flux`), divided by the slab heat capacity per unit area
    `ρ_ocean cp_ocean depth_ocean`. The Q-flux profile follows Merlis et al. (2013),
    J. Climate 26, https://doi.org/10.1175/JCLI-D-12-00149.1.
  - `Yₜ.sfc.water` is decremented by the upward turbulent water flux (evaporation).

Both turbulent terms are dropped when `p.atmos.disable_surface_flux_tendency` is
`true`, and the radiative term when there is no radiation model. Precipitation
deposition of energy and water is handled separately by
`surface_precipitation_tendency!`. Reads `p.radiation.ᶠradiation_flux` and
`p.precomputed.sfc_conditions`; `t` is unused. Called from
`additional_tendency!`, after the microphysics tendencies.
"""
surface_temp_tendency!(Yₜ, Y, p, t, ::SurfaceTemperature) = nothing

function surface_temp_tendency!(Yₜ, Y, p, t, slab::SlabOceanTemperature)
    FT = eltype(Y)
    params = p.params

    depth_ocean = slab.depth_ocean
    ρ_ocean = slab.ρ_ocean
    cp_ocean = slab.cp_ocean
    q_flux_enabled = slab.q_flux

    # --- ENERGY BALANCE ---
    # Denominator for temperature tendency
    surface_heat_capacity_per_area = ρ_ocean * cp_ocean * depth_ocean

    # 1. Radiative energy surface fluxes
    if !isnothing(p.atmos.radiation_mode)
        # ᶠradiation_flux is positive for net upward flux at the surface
        # (SW_up - SW_down + LW_up - LW_down)
        (; ᶠradiation_flux) = p.radiation
        sfc_rad_e_flux = Spaces.level(ᶠradiation_flux, half).components.data.:1
    else
        sfc_rad_e_flux = 0
    end

    # 2. Turbulent surface energy fluxes (sensible + latent heat) from surface to atmosphere
    if !(p.atmos.disable_surface_flux_tendency)
        turb_e_flux_sfc_to_atm =
            Geometry.WVector.(
                p.precomputed.sfc_conditions.ρ_flux_h_tot,
            ).components.data.:1
    else
        turb_e_flux_sfc_to_atm = 0
    end

    # 3. Idealized Q-fluxes (parameterization of horizontal ocean energy flux divergence),
    # following Merlis et al. (2013), "Hadley Circulation Response to Orbital Precession.
    # Part II: Subtropical Continent.", J. Climate, 26, https://doi.org/10.1175/JCLI-D-12-00149.1
    if q_flux_enabled
        ϕ₀ = slab.ϕ₀
        Q₀ = slab.Q₀
        ϕ = deg2rad.(Fields.level(Fields.coordinate_field(Y.f).lat, half))
        ϕ₀ʳ = FT(deg2rad(ϕ₀))
        Q = @. Q₀ * (1 - 2ϕ^2 / ϕ₀ʳ^2) * exp(-(ϕ^2 / ϕ₀ʳ^2)) / cos(ϕ)
    else
        Q = FT(0)
    end

    # Total energy tendency for surface temperature
    # (precipitation energy/water deposition is handled separately
    # by surface_precipitation_tendency! in both implicit and explicit paths):
    # dT/dt = -(NetRad_upward + TurbFlux_sfc_to_atm + Q_div) / HeatCapacity
    @. Yₜ.sfc.T -=
        (sfc_rad_e_flux + turb_e_flux_sfc_to_atm + Q) /
        surface_heat_capacity_per_area

    # --- WATER BALANCE (if moisture is active) ---
    if !(p.atmos.microphysics_model isa DryModel)
        # Turbulent surface water fluxes (evaporation/condensation)
        if !(p.atmos.disable_surface_flux_tendency)
            sfc_turb_w_flux =
                Geometry.WVector.(
                    p.precomputed.sfc_conditions.ρ_flux_q_tot,
                ).components.data.:1
        else
            sfc_turb_w_flux = 0
        end

        # Water tendency from turbulent fluxes only;
        # precipitation (rain + snow) is handled by surface_precipitation_tendency!.
        @. Yₜ.sfc.water -= sfc_turb_w_flux
    end

end
