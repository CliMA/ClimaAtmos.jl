#####
##### Couples the atmospheric model with a slab surface model 
#####

"""
    surface_precipitation_tendency!(Yₜ, Y, p, t, surface_model, moisture_model)

Applies the surface water and energy deposition from precipitation.

When microphysics tendencies are treated implicitly, this function is called
from `implicit_tendency!` (rather than from the explicit `surface_temp_tendency!`)
so that both the atmospheric water removal and the surface deposition use the
same cached `ᶜS_ρq_tot`, preserving conservation across IMEX stages.
"""
surface_precipitation_tendency!(Yₜ, Y, p, t, ::PrescribedSST, _) = nothing
surface_precipitation_tendency!(Yₜ, Y, p, t, _, ::DryModel) = nothing
surface_precipitation_tendency!(Yₜ, Y, p, t, ::PrescribedSST, ::DryModel) =
    nothing
surface_precipitation_tendency!(Yₜ, Y, p, t, ::SlabOceanSST, ::DryModel) =
    nothing

function surface_precipitation_tendency!(
    Yₜ, Y, p, t, slab::SlabOceanSST, moisture_model,
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
    - Energy flux due to precipitation.
    - For surface water (if model is not dry):
        - Net upward turbulent surface water flux (evaporation).
        - Precipitation (rain and snow) impact.

The heat capacity of the slab surface is determined by `slab.depth_ocean`,
`slab.ρ_ocean`, and `slab.cp_ocean`.

**Sign Conventions for Flux Variables (all positive = surface to atmosphere/upward):**
- `sfc_rad_e_flux`: Net upward radiative flux (e.g., SW_up - SW_down + LW_up - LW_down).
                   Positive value signifies energy loss for the surface.
- `turb_e_flux_sfc_to_atm` (derived from `p.precomputed.sfc_conditions.ρ_flux_h_tot`):
                   Net upward turbulent energy flux. Positive value signifies energy loss for the surface.
- `Q` (Q-flux): Net ocean heat flux divergence in the slab. Positive value signifies
                local energy loss for the surface slab.
- `pet` (precipitation energy): Energy removed from the surface due to precipitation processes. 
                Positive value signifies energy loss for the surface.
- `P_liq`, `P_snow` (from `p.precomputed.surface_x_flux`): These variables must represent
                the precipitation water flux (positive upward). Thus, for physical
                precipitation *onto* the surface (a downward flux), `P_liq` and `P_snow`
                are *negative*.
- `sfc_turb_w_flux` (derived from `p.precomputed.sfc_conditions.ρ_flux_q_tot`):
                Net upward turbulent water flux (evaporation). Positive value signifies
                water loss for the surface. 

Arguments:
- `Yₜ`: The tendency state vector, where `Yₜ.sfc` components are modified.
- `Y`: The current state vector (used for surface latitude for Q-flux).
- `p`: Cache containing parameters, precomputed fields (radiation fluxes, surface
       conditions, precipitation fluxes), atmospheric model configurations, and
       slab model properties.
- `t`: Current simulation time.
- `surface_model`: The surface model instance.

Modifies surface tendency vector `Yₜ.sfc` in place.
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
                p.precomputed.sfc_conditions.ρ_flux_h_tot
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

    # 4. Energy tendency due to precipitation accumulation
    # When microphysics is implicit, precipitation budget is handled in
    # implicit_tendency! to avoid IMEX stage mismatch (conservation).
    if !(p.atmos.moisture_model isa DryModel) &&
       p.atmos.microphysics_tendency_timestepping != Implicit()

        pet = p.precomputed.col_integrated_precip_energy_tendency

    else
        pet = FT(0)
    end

    # Total energy tendency for surface temperature:
    # dT/dt = -(NetRad_upward + TurbFlux_sfc_to_atm + Q_div - PrecipEnergySource) / HeatCapacity
    @. Yₜ.sfc.T -=
        (sfc_rad_e_flux + turb_e_flux_sfc_to_atm + Q + pet) /
        surface_heat_capacity_per_area

    # --- WATER BALANCE (if moisture is active) ---
    if !(p.atmos.moisture_model isa DryModel)
        # 1. Turbulent surface water fluxes (evaporation/condensation)
        if !(p.atmos.disable_surface_flux_tendency)
            sfc_turb_w_flux =
                Geometry.WVector.(
                    p.precomputed.sfc_conditions.ρ_flux_q_tot
                ).components.data.:1
        else
            sfc_turb_w_flux = 0
        end

        # 2. Precipitation (rain and snow)
        # When microphysics is implicit, precipitation budget is handled in
        # implicit_tendency! to avoid IMEX stage mismatch (conservation).
        if p.atmos.microphysics_tendency_timestepping != Implicit()
            P_liq = p.precomputed.surface_rain_flux
            P_snow = p.precomputed.surface_snow_flux
        else
            P_liq = FT(0)
            P_snow = FT(0)
        end

        # Total water tendency for surface water:
        # d(water)/dt = -(Precip_up + Precip_snow_up + TurbFlux_water_atm_to_sfc) [kg/m²/s]
        @. Yₜ.sfc.water -= P_liq + P_snow + sfc_turb_w_flux
    end

end
