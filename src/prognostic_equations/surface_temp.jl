#####
##### Couples the atmospheric model with a slab surface model 
#####

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

"""
    get_∂F_rad_energy∂T_sfc(T_sfc, α, σ)

Calculate the derivative of the radiative flux with respect to the surface temperature
for the Eisenman sea ice model. This represents the linearization of upward longwave
radiation: d(σT⁴)/dT = 4σT³, accounting for albedo effects.
"""
function get_∂F_rad_energy∂T_sfc(T_sfc, α, σ)
    FT = eltype(T_sfc)
    @. FT(4) * (FT(1) - α) * σ * T_sfc^3
end

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

    # 4. Energy tendency due to precipitation accumulation        
    if !(p.atmos.moisture_model isa DryModel)

        pet = p.conservation_check.col_integrated_precip_energy_tendency

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
                    p.precomputed.sfc_conditions.ρ_flux_q_tot,
                ).components.data.:1
        else
            sfc_turb_w_flux = 0
        end

        # 2. Precipitation (rain and snow, defined negative downward, so positive flux 
        # from surface to atmosphere)
        P_liq = p.precomputed.surface_rain_flux
        P_snow = p.precomputed.surface_snow_flux

        # Total water tendency for surface water:
        # d(water)/dt = -(Precip_up + Precip_snow_up + TurbFlux_water_atm_to_sfc) [kg/m²/s]
        @. Yₜ.sfc.water -= P_liq + P_snow + sfc_turb_w_flux
    end

end

function surface_temp_tendency!(Yₜ, Y, p, t, slab::EisenmanSeaIce)
    # following previous implementation in ClimaCoupler.jl (https://github.com/CliMA/ClimaCoupler.jl/blob/a3b32d169137f7dad2edf33fd2f5e29ebd6d5356/experiments/ClimaEarth/components/ocean/eisenman_seaice.jl#L305)
    FT = eltype(Y)
    params = p.params
    Δt = p.dt # using this to estimate changes using Euler forward step
    # we will divide back out at the end before passing tendencies to the ODE solver

    # ocean params
    (; depth_ocean, ρ_ocean, cp_ocean) = slab

    # Heat capacity of mixed layer
    hρc_ml = ρ_ocean * cp_ocean * depth_ocean
    
    # sea ice params
    (; C0_base, T_base, L_ice, T_freeze, k_ice, T_base) = slab

    # prognostic variables
    (; T, h_ice, T_ml, water) = Y.sfc
    T_sfc = T

    # --- ENERGY BALANCE ---
    # Denominator for temperature tendency
    ml_heat_capacity_per_area = ρ_ocean * cp_ocean * depth_ocean

    # Radiative energy surface fluxes
    # Should we consider difference between ice covered and ice-free here?
    if !isnothing(p.atmos.radiation_mode)
        # ᶠradiation_flux is positive for net upward flux at the surface
        # (SW_up - SW_down + LW_up - LW_down)
        (; ᶠradiation_flux) = p.radiation
        F_rad = Spaces.level(ᶠradiation_flux, half).components.data.:1
    else
        F_rad = 0
    end

    # 2. Turbulent surface energy fluxes (sensible + latent heat) from surface to atmosphere
    if !(p.atmos.disable_surface_flux_tendency)
        F_turb =
            Geometry.WVector.(
                p.precomputed.sfc_conditions.ρ_flux_h_tot,
            ).components.data.:1
    else
        F_turb = 0
    end

    # 3. Energy tendency due to precipitation accumulation        
    if !(p.atmos.moisture_model isa DryModel)
        pet = p.conservation_check.col_integrated_precip_energy_tendency
    else
        pet = FT(0)
    end

    #======================================================================================#
    # Eisenman-Zhang sea ice model thermodynamics
    #======================================================================================#    

    F_atm = @. F_turb + F_rad
    # ice thickness and mixed layer temperature changes due to atmosphereic and ocean fluxes
    ice_covered = parent(h_ice)[1] > 0
    # Note: Ocean Q-fluxes are not implemented
    ocean_qflux = 0
    if ice_covered # ice-covered
        F_base = @. C0_base * (T_ml - T_base)
        #@. Yₜ.sfc.T_ml =  -(F_base) / ml_heat_capacity_per_area 
        #@. Yₜ.sfc.h_ice = (F_atm - F_base) / L_ice 
        ΔT_ml = @. -(F_base - ocean_qflux) * FT(Δt) / (hρc_ml)
        Δh_ice = @. (F_atm - F_base - ocean_qflux) * FT(Δt) / L_ice
        #@. e_base .+= F_base * FT(Δt)
    else # ice-free
        #@. Yₜ.sfc.T_ml = -(F_atm ) / (hρc_ml)
        #@. Yₜ.sfc.h_ice = 0
        ΔT_ml = @. -(F_atm - ocean_qflux) * FT(Δt) / (hρc_ml)
        Δh_ice = 0
    end

    # estimate deltas
    #ΔT_ml = @. Yₜ.sfc.T_ml * Δt
    #Δh_ice = @. Yₜ.sfc.h_ice * Δt

    # T_ml is not allowed to be below freezing
    frazil_ice_formation = (parent(T_ml .+ ΔT_ml)[1] < T_freeze)
    if frazil_ice_formation
        Δh_ice = @. Δh_ice - (T_ml + ΔT_ml - T_freeze) * ml_heat_capacity_per_area / L_ice
        ΔT_ml = @. T_freeze - T_ml
    end

    # adjust ocean temperature if transition to ice-free
    transition_to_icefree = (parent(h_ice)[1] > 0) & (parent(h_ice .+ Δh_ice)[1] <= 0)
    if transition_to_icefree
        ΔT_ml = @. ΔT_ml - (h_ice + Δh_ice) * L_ice / (hρc_ml)
        Δh_ice = @. -h_ice
    end

    if t%3600 == 0
        println("=== EisenmanSeaIce Tendency (t = $t s) ===")
        println("  T_sfc = $(parent(T_sfc)[1]) K")
        println("  T_ml = $(parent(T_ml)[1]) K")
        println("  h_ice = $(parent(h_ice)[1]) m")
        println("  F_rad = $(parent(F_rad)[1]) W/m²")
        println("  F_turb = $(parent(F_turb)[1]) W/m²")
        println("  F_atm (total) = $(parent(F_atm)[1]) W/m²")
        println("  ice_covered = $ice_covered")
        println("  ΔT_ml = $(parent(ΔT_ml)[1]) K")
        println("  Δh_ice = $(parent(Δh_ice)[1]) m")
        println("="^50)
    end

    
    # solve for T_sfcs
    remains_ice_covered = (parent(h_ice .+ Δh_ice)[1] > 0)
    if remains_ice_covered
        # if ice covered, solve implicity (for now one Newton iteration: ΔT_s = - F(T_s) / dF(T_s)/dT_s )
        h = @. h_ice + Δh_ice
        F_conductive = @. k_ice / h * (T_base - T_sfc)
        numerator = @. -F_atm + F_conductive
        denominator = @. k_ice / h + ∂F_atmo∂T_sfc
        δT_sfc = @. numerator / denominator
        surface_melting = (parent(T_sfc .+ δT_sfc)[1] > T_freeze)
        if surface_melting
            δT_sfc = @. T_freeze - T_sfc # NB: T_sfc not storing energy
        end
        # surface is ice-covered, so update T_sfc as ice surface temperature
        T_sfc .+= δT_sfc
        # update surface humidity
        @. q_sfc = TD.q_vap_saturation_generic.(thermo_params, T_sfc, Ya.ρ_sfc, TD.Ice())
    else # ice-free, so update T_sfc as mixed layer temperature
        T_sfc .= T_ml .+ ΔT_ml
        # update surface humidity
        @. q_sfc = TD.q_vap_saturation_generic.(thermo_params, T_sfc, Ya.ρ_sfc, TD.Liquid())
    end

    #Y.T_ml .+= ΔT_ml
    #Y.h_ice .+= Δh_ice
    #Y.T_sfc .= T_sfc
    #Y.q_sfc .= q_sfc

    # compute tendencies
    @. Yₜ.sfc.T_ml = ΔT_ml / FT(Δt)
    @. Yₜ.sfc.h_ice = Δh_ice / FT(Δt)
    @. Yₜ.sfc.T = (T_sfc - Y.sfc.T) / FT(Δt)
    @. Yₜ.sfc.water = 0 # water tendency not implemented yet

end
