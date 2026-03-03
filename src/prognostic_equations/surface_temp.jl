#####
##### Couples the atmospheric model with a slab surface model 
#####

"""
    surface_precipitation_tendency!(Yₜ, Y, p, t, surface_model, microphysics_model)

Applies the surface water and energy deposition from precipitation.

Called from both `implicit_tendency!` (when microphysics is implicit) and
`remaining_tendency!` (when microphysics is explicit), so that the surface
deposition always uses the same cached `ᶜS_ρq_tot` as the atmospheric water
removal, preserving conservation across IMEX stages.
"""
surface_precipitation_tendency!(Yₜ, Y, p, t, _, _) = nothing

function surface_precipitation_tendency!(
    Yₜ, Y, p, t, slab::SlabOceanSST, microphysics_model,
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
    surface_temp_tendency!(Yₜ, Y, p, t, surface_model)

Computes the tendency for the prognostic surface temperature (`Y.sfc.T`) and,
if applicable, prognostic surface water content (`Y.sfc.water`), based on
surface energy and water fluxes. All fluxes are defined as positive when directed
from the surface to the atmosphere (upward).

This function is dispatched based on the type of `surface_model`:
- `::PrescribedSST`: No tendency is computed (surface temperature is fixed).
- `::SlabOceanSST` (slab model): Calculates tendencies due to:
    - Net upward radiative flux at the surface.
    - Net upward turbulent surface energy flux (sensible + latent heat).
    - Net ocean heat flux divergence in the slab (Q-flux), if enabled.
    - For surface water: net upward turbulent surface water flux (evaporation).

Precipitation surface deposition (energy and water) is handled separately by
`surface_precipitation_tendency!`, which is called from both the implicit and
explicit tendency paths.
"""
surface_temp_tendency!(Yₜ, Y, p, t, ::PrescribedSST) = nothing

function surface_temp_tendency!(Yₜ, Y, p, t, slab::SlabOceanSST)
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
