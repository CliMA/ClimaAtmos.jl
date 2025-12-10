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

#= Keeping for reference. Developing new EisenmanSeaIce implementation below.
function surface_temp_tendency!(Yₜ, Y, p, t, slab::EisenmanSeaIce)
    FT = eltype(Y.sfc.T)
    params = p.params

    # Eisenman sea ice model parameters
    C0_base = slab.C0_base
    T_base = slab.T_base
    L_ice = slab.L_ice
    T_freeze = slab.T_freeze
    k_ice = slab.k_ice
    
    # Ocean mixed layer parameters
    depth_ocean = slab.depth_ocean
    ρ_ocean = slab.ρ_ocean
    cp_ocean = slab.cp_ocean
    q_flux_enabled = slab.q_flux
    
    # Heat capacity of mixed layer
    hρc_ml = ρ_ocean * cp_ocean * depth_ocean

    # Extract prognostic variables
    h_ice = Y.sfc.h_ice
    T_ml = Y.sfc.T_ml
    T_sfc = Y.sfc.T
    water = Y.sfc.water
    
    # Diagnostic printing
    println("=== EisenmanSeaIce Tendency (t = $t s) ===")
    println("  T_sfc = $(parent(T_sfc)[1]) K")
    println("  T_ml = $(parent(T_ml)[1]) K")
    println("  h_ice = $(parent(h_ice)[1]) m")

    # --- COMPUTE ATMOSPHERIC FLUXES ---
    # 1. Radiative energy surface fluxes (positive upward)
    if !isnothing(p.atmos.radiation_mode)
        # ᶠradiation_flux is positive for net upward flux at the surface 
        # (SW_up - SW_down + LW_up - LW_down)
        (; ᶠradiation_flux) = p.radiation
        F_rad = Spaces.level(ᶠradiation_flux, half).components.data.:1
    else
        F_rad = T_sfc .* FT(0)  # zero field with same structure as surface fields
    end

    # 2. Turbulent surface energy fluxes (sensible + latent heat, positive upward)
    if !(p.atmos.disable_surface_flux_tendency)
        F_turb =
            Geometry.WVector.(
                p.precomputed.sfc_conditions.ρ_flux_h_tot,
            ).components.data.:1
    else
        F_turb = T_sfc .* FT(0)  # zero field with same structure as surface fields
    end

    # Total atmospheric flux (positive upward means energy loss from surface)
    F_atm = @. F_rad + F_turb
    
    println("  F_rad = $(parent(F_rad)[1]) W/m²")
    println("  F_turb = $(parent(F_turb)[1]) W/m²")
    println("  F_atm (total) = $(parent(F_atm)[1]) W/m²")

    # 3. Idealized Q-fluxes (ocean heat flux divergence)
    if q_flux_enabled
        ϕ₀ = slab.ϕ₀
        Q₀ = slab.Q₀
        ϕ = deg2rad.(Fields.level(Fields.coordinate_field(Y.f).lat, half))
        ϕ₀ʳ = FT(deg2rad(ϕ₀))
        ocean_qflux = @. Q₀ * (1 - 2ϕ^2 / ϕ₀ʳ^2) * exp(-(ϕ^2 / ϕ₀ʳ^2)) / cos(ϕ)
    else
        ocean_qflux = T_sfc .* FT(0)  # zero field with same structure as surface fields
    end

    # --- EISENMAN SEA ICE THERMODYNAMICS ---
    # IMPORTANT: The Fortran reference uses explicit forward Euler time stepping,
    # computing increments delta_X = (flux terms) * dt / (constants).
    # We are providing tendencies to an IMEX ODE solver, so we compute
    # dX/dt = (flux terms) / (constants), and the solver handles time integration.
    
    # Sign conventions:
    # - F_atm, F_base: positive = upward = energy loss from surface/ocean
    # - ocean_qflux: positive = heat flux convergence (warming the ocean)
    # - All match the SlabOceanSST convention in this file
    
    # Determine initial ice state
    ice_covered = parent(h_ice)[1] > 0
    
    if ice_covered
        # Ice-covered: ocean below ice is at freezing point
        # Base flux from ocean below to ice base (positive upward = cooling ocean)
        F_base = @. C0_base * (T_ml - T_base)
        
        # Mixed layer temperature tendency (only affected by base flux and ocean Q-flux)
        # Fortran: delta_t_ml = -( F_base + ocean_qflux ) * dt/(depth*RHO_CP)
        # Tendency: dT_ml/dt = -(F_base + ocean_qflux) / (depth*RHO_CP)
        # But wait - in Fortran, ocean_qflux is ADDED, not subtracted in the ice-covered case:
        # "delta_t_ml = - ( ice_basal_flux_const * ( t_ml - t_ice_base ) + ocean_qflux ) * dt/(depth*RHO_CP)"
        # So: dT_ml/dt = -(F_base + ocean_qflux) / hρc_ml
        # Hmm, but that doesn't match the SlabOceanSST case where ocean_qflux is subtracted...
        # Let me re-check SlabOceanSST above... Line 126: "+ Q" in numerator, so -(... + Q)/...
        # And Q is computed the same way for both. So yes, ocean_qflux represents heating (+) or cooling (-)
        # And it should be SUBTRACTED in energy balance: dT/dt = -(upward_fluxes - ocean_heating)/capacity
        # Looking at Fortran again: ocean_qflux is ADDED to corrected_flux before negating
        # corrected_flux =  -SW - LW + sensible + latent (all are INTO ocean, so negative of upward)
        # So corrected_flux is net flux INTO ocean
        # Then: -(corrected_flux + ocean_qflux) means ocean_qflux is ALSO into ocean (heating)
        # In our convention: ocean_qflux > 0 means heating (into ocean)
        # So: dT_ml/dt = -(F_base - ocean_qflux) / hρc_ml
        #                = (-F_base + ocean_qflux) / hρc_ml
        dT_ml_dt = @. -(F_base - ocean_qflux) / hρc_ml
        
        # Ice thickness tendency from atmospheric and base fluxes
        # ClimaCoupler.jl: Δh_ice = (F_atm - F_base - ocean_qflux) * Δt / L_ice
        # where F_atm = F_turb + F_rad (both positive upward = cooling surface)
        # Physical interpretation:
        # - F_atm > F_base: ice surface loses more than base gains → ice thickens
        # - ocean_qflux > 0: heating from below → melts ice
        # So: dh_ice/dt = (F_atm - F_base - ocean_qflux) / L_ice
        # This matches ClimaCoupler exactly!
        dh_ice_dt = @. (F_atm - F_base - ocean_qflux) / L_ice
    else
        # Ice-free: mixed layer absorbs atmospheric flux directly
        # Fortran: delta_t_ml = - ( corrected_flux + ocean_qflux) * dt/(depth*RHO_CP)
        # = -(-F_atm + ocean_qflux) / hρc_ml
        # = (F_atm - ocean_qflux) / hρc_ml
        # But this gives WARMING for positive F_atm (upward flux), which is wrong!
        # Let me reconsider: corrected_flux in Fortran is flux INTO surface
        # corrected_flux = -net_SW - LW + turb (where turb includes sensible+latent upward)
        # Wait, looking at line ~380 in Fortran:
        # "corrected_flux = - net_surf_sw_down - surf_lw_down + alpha_t * CP_AIR + alpha_lw"
        # net_surf_sw_down is DOWNward SW, so -net_surf_sw_down is loss
        # surf_lw_down is DOWNward LW, so -surf_lw_down is loss
        # alpha_t * CP_AIR is upward sensible heat (loss)
        # alpha_lw is upward LW (loss)
        # So corrected_flux = -(SW_down) - (LW_down) + sensible_up + LW_up
        # = upward_net_flux = our F_atm!
        # So corrected_flux = F_atm in our notation!
        # Then: delta_t_ml = -(F_atm + ocean_qflux) / hρc_ml
        # which makes sense: upward flux and ocean divergence both cool the surface
        dT_ml_dt = @. -(F_atm + ocean_qflux) / hρc_ml
        dh_ice_dt = T_sfc .* FT(0)  # zero field with same structure as surface fields
    end

    # --- FRAZIL ICE FORMATION AND COMPLETE MELTING ---
    # The Fortran code handles these as constraints on the explicit Euler step.
    # We can access dt from the cache to properly implement these constraints.
    
    dt = FT(p.cache.dt)
    
    # Compute projected states after one timestep (matching Fortran explicit Euler)
    ΔT_ml = @. dT_ml_dt * dt
    Δh_ice = @. dh_ice_dt * dt
    
    # Frazil ice formation: if T_ml would drop below freezing, form ice instead
    # Fortran: where ( t_ml + delta_t_ml .lt. TFREEZE )
    frazil_formation = @. (T_ml + ΔT_ml) < T_freeze
    Δh_ice = @. ifelse(frazil_formation, 
                       Δh_ice - (T_ml + ΔT_ml - T_freeze) * hρc_ml / L_ice,
                       Δh_ice)
    ΔT_ml = @. ifelse(frazil_formation, 
                      T_freeze - T_ml,
                      ΔT_ml)
    
    # Complete melting: if ice would melt completely, transfer residual energy to ocean
    # Fortran: where ( ( h_ice .gt. 0 ) .and. ( h_ice + delta_h_ice .le. 0 ) )
    complete_melt = @. (h_ice > 0) & ((h_ice + Δh_ice) <= 0)
    ΔT_ml = @. ifelse(complete_melt,
                      ΔT_ml - (h_ice + Δh_ice) * L_ice / hρc_ml,
                      ΔT_ml)
    Δh_ice = @. ifelse(complete_melt,
                       -h_ice,
                       Δh_ice)
    
    # Convert back to rates
    dT_ml_dt = @. ΔT_ml / dt
    dh_ice_dt = @. Δh_ice / dt

    # --- SOLVE FOR SURFACE TEMPERATURE ---
    remains_ice_covered = @. (h_ice + Δh_ice) > FT(1e-6)
    
    # Initialize T_sfc_new
    T_sfc_new = T_sfc .* FT(0)  # create field with correct type
    
    # Ice-covered branch: solve for steady-state ice surface temperature
    # The Fortran uses one Newton iteration to find T_sfc where:
    # flux_ice = corrected_flux (energy balance at ice surface)
    # Residual R = F_atm - F_conductive (should be zero at steady state)
    # Newton step: δT_sfc = -R/(dR/dT_sfc)
    
    # Use projected ice thickness h_ice + Δh_ice for conductive calculation
    h = @. max(h_ice + Δh_ice, FT(1e-6))  # avoid division by zero
    
    # Conductive flux through ice (from base to surface, positive upward)
    F_conductive = @. k_ice / h * (T_base - T_sfc)
    
    # Energy balance residual
    # Fortran: delta_t_ice = ( - corrected_flux + flux_ice ) / ...
    # where corrected_flux = F_atm, flux_ice = F_conductive
    # So: numerator = -F_atm + F_conductive = F_conductive - F_atm
    numerator = @. F_conductive - F_atm
    
    # Derivative of surface energy balance w.r.t. surface temperature
    α_ice = FT(0.7)  # ice albedo
    σ = FT(5.67e-8)  # Stefan-Boltzmann constant
    ∂F_rad∂T_sfc = get_∂F_rad_energy∂T_sfc(T_sfc, α_ice, σ)
    
    # Total atmospheric flux derivative (approximate as just radiation)
    # In full coupling, this would include turbulent flux derivatives
    ∂F_atm∂T_sfc = ∂F_rad∂T_sfc
    
    denominator = @. k_ice / h + ∂F_atm∂T_sfc
    
    δT_sfc = @. numerator / denominator
    
    # Check for surface melting (T_sfc cannot exceed freezing point)
    # Fortran: where ( t_surf + delta_t_ice .gt. TFREEZE ) delta_t_ice = TFREEZE - t_surf
    surface_melting = @. (T_sfc + δT_sfc) > T_freeze
    δT_sfc = @. ifelse(surface_melting, T_freeze - T_sfc, δT_sfc)
    
    # Apply Newton correction for ice-covered points, otherwise use T_ml
    # Fortran ice-free: t_surf = t_ml + delta_t_ml
    T_sfc_new = @. ifelse(remains_ice_covered, 
                          T_sfc + δT_sfc,
                          T_ml + ΔT_ml)
    
    println("  SURFACE TEMPERATURE CALCULATION:")
    println("    remains_ice_covered = $(parent(remains_ice_covered)[1])")
    println("    h (projected) = $(parent(h)[1]) m")
    println("    F_conductive = $(parent(F_conductive)[1]) W/m²")
    println("    numerator (F_cond - F_atm) = $(parent(numerator)[1])")
    println("    ∂F_atm∂T_sfc = $(parent(∂F_atm∂T_sfc)[1])")
    println("    denominator = $(parent(denominator)[1])")
    println("    δT_sfc (Newton) = $(parent(δT_sfc)[1]) K")
    println("    T_sfc_new = $(parent(T_sfc_new)[1]) K")

    # --- COMPUTE TENDENCIES ---
    # For h_ice and T_ml, we computed rates directly from flux balance
    # For T_sfc, the Fortran assumes steady-state (zero-layer model), so it should
    # equilibrate instantly. We approximate this with a relaxation timescale.
    # The Fortran effectively uses τ ~ dt (one timestep), which is ~100-1000s.
    # We use τ=10s as a compromise between stability and responsiveness.
    
    τ_relax = FT(10.0)  # Relaxation timescale [seconds] for ice surface temperature
    
    @. Yₜ.sfc.h_ice = dh_ice_dt
    @. Yₜ.sfc.T_ml = dT_ml_dt
    @. Yₜ.sfc.T = (T_sfc_new - T_sfc) / τ_relax
    
    println("  TENDENCIES:")
    println("    dh_ice/dt = $(parent(Yₜ.sfc.h_ice)[1]) m/s")
    println("    dT_ml/dt = $(parent(Yₜ.sfc.T_ml)[1]) K/s")
    println("    dT_sfc/dt = $(parent(Yₜ.sfc.T)[1]) K/s")
    println("    T_sfc_new (target) = $(parent(T_sfc_new)[1]) K")
    println("    τ_relax = $τ_relax s")
    println("="^50)

    # --- WATER BALANCE (if moisture is active) ---
    if !(p.atmos.moisture_model isa DryModel)
        # 1. Turbulent surface water fluxes (evaporation/condensation)
        if !(p.atmos.disable_surface_flux_tendency)
            sfc_turb_w_flux =
                Geometry.WVector.(
                    p.precomputed.sfc_conditions.ρ_flux_q_tot,
                ).components.data.:1
        else
            sfc_turb_w_flux = T_sfc .* FT(0)  # zero field with same structure as surface fields
        end

        # 2. Precipitation (rain and snow, defined negative downward)
        P_liq = p.precomputed.surface_rain_flux
        P_snow = p.precomputed.surface_snow_flux

        # Total water tendency
        @. Yₜ.sfc.water = -(P_liq + P_snow + sfc_turb_w_flux)
    else
        @. Yₜ.sfc.water = T_sfc .* FT(0)  # zero field with same structure as surface fields
    end
end
=#

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
