#####
##### Eisenman-Zhang thermodynamic 0-layer sea-ice surface model
#####
##### Operator-split update of the prognostic sea-ice/ocean surface state
##### (`Y.sfc.T`, `Y.sfc.h_ice`, `Y.sfc.T_ml`). Ported from the ClimaCoupler
##### implementation (`solve_eisenman_model!`), based on Semtner (1976),
##### Eisenman & Wettlaufer (2009) and Zhang et al. (2021).
#####
##### The update is applied once per timestep by `eisenman_seaice_step!` (a
##### callback), rather than as a smooth tendency, because the freezing /
##### frazil / melt transitions are discontinuous and do not map onto the IMEX
##### tendency integration. `surface_temp_tendency!` is a no-op for this type.
#####

using .SurfaceConditions: EisenmanIceTemperature

"""
    eisenman_seaice_step!(integrator)

Advance the Eisenman-Zhang sea-ice/ocean surface state `Y.sfc` by one timestep
`Δt`, using the surface energy fluxes accumulated in the cache for the current
step. All fluxes are positive when directed from the surface to the atmosphere
(upward).

Per timestep, with `F_atm = F_rad + F_turb` (net upward) and
`hρc = h_ml ρ_ocean cp_ocean`:

 1. Ice-covered (`h_ice > 0`): basal flux `F_base = C0_base (T_ml - T_base)`
    warms/cools the mixed layer and grows/melts ice
    (`Δh_ice = (F_atm - F_base - Q_flux) Δt / L_ice`). Ice-free: `F_atm` cools
    the mixed layer, no ice change.
 2. Frazil: the mixed layer cannot cool below `T_freeze`; any deficit grows ice.
 3. Transition to ice-free: surplus melt energy warms the mixed layer.
 4. Surface temperature: if ice remains, solve the conductive/atmospheric
    balance (one Newton iteration) capped at `T_freeze`; otherwise the surface
    equals the mixed-layer temperature.

The turbulent flux is always included (the model is driven by SH + LH + radiation);
`disable_surface_flux_tendency` only gates the slab-ocean tendency path and has no
effect here. The atmospheric-flux derivative is approximated by the radiative term
`∂F_atm/∂T_sfc ≈ 4 σ T_sfc³` (shortwave albedo term omitted: zero insolation).
"""
function eisenman_seaice_step!(integrator)
    Y = integrator.u
    p = integrator.p
    ice = p.atmos.surface.temperature
    FT = eltype(Y)
    Δt = FT(integrator.dt)

    (; C0_base, T_base, L_ice, T_freeze, k_ice, σ, h_ml, ρ_ocean, cp_ocean, Q_flux) =
        ice
    hρc_ml = h_ml * ρ_ocean * cp_ocean

    # Prognostic surface state, accessed as Fields (so in-place `@.` updates
    # write back into `Y.sfc`). Operating on Fields—rather than the underlying
    # DataLayouts—keeps these compatible with the surface-flux Fields extracted
    # from the cache below, matching `surface_temp_tendency!(::SlabOceanTemperature)`.
    T_sfc = Y.sfc.T
    h_ice = Y.sfc.h_ice
    T_ml = Y.sfc.T_ml
    water = Y.sfc.water

    # Surface energy fluxes from the cache (positive = upward, surface → atmos),
    # read exactly as `surface_temp_tendency!(::SlabOceanTemperature)` does.
    F_rad =
        isnothing(p.atmos.radiation_mode) ? zero(FT) :
        Spaces.level(p.radiation.ᶠradiation_flux, half).components.data.:1
    F_turb =
        Geometry.WVector.(
            p.precomputed.sfc_conditions.ρ_flux_h_tot
        ).components.data.:1
    F_water =
        Geometry.WVector.(
            p.precomputed.sfc_conditions.ρ_flux_q_tot
        ).components.data.:1

    F_atm = @. F_turb + F_rad
    ∂F_atm∂T_sfc = @. FT(4) * σ * T_sfc^3

    # 1. Atmospheric/basal fluxes change the mixed layer and ice thickness.
    ice_covered = @. h_ice > 0
    F_base = @. C0_base * (T_ml - T_base)
    ΔT_ml = @. ifelse(
        ice_covered,
        -(F_base - Q_flux) * Δt / hρc_ml,
        -(F_atm - Q_flux) * Δt / hρc_ml,
    )
    Δh_ice = @. ifelse(ice_covered, (F_atm - F_base - Q_flux) * Δt / L_ice, zero(FT))

    # 2. Frazil ice formation: the mixed layer is not allowed below T_freeze; the
    #    energy deficit grows ice.
    frazil = @. (T_ml + ΔT_ml) < T_freeze
    Δh_ice = @. ifelse(frazil, Δh_ice - (T_ml + ΔT_ml - T_freeze) * hρc_ml / L_ice, Δh_ice)
    ΔT_ml = @. ifelse(frazil, T_freeze - T_ml, ΔT_ml)

    # 3. Transition to ice-free: melt the remaining ice and warm the mixed layer
    #    with the surplus energy.
    transition = @. (h_ice > 0) & ((h_ice + Δh_ice) <= 0)
    ΔT_ml = @. ifelse(transition, ΔT_ml - (h_ice + Δh_ice) * L_ice / hρc_ml, ΔT_ml)
    Δh_ice = @. ifelse(transition, -h_ice, Δh_ice)

    # 4. Surface temperature. If ice remains, balance the conductive flux through
    #    the ice against the atmospheric flux (one Newton iteration), capped at
    #    the freezing point (surface melt). Otherwise the surface equals the
    #    mixed-layer temperature. (Where ice is absent, `h_new <= 0` makes the
    #    ice-branch expression non-finite, but it is discarded by `ifelse`.)
    h_new = @. h_ice + Δh_ice
    remains_ice = @. h_new > 0
    F_cond = @. k_ice / h_new * (T_base - T_sfc)
    ΔT_sfc = @. (-F_atm + F_cond) / (k_ice / h_new + ∂F_atm∂T_sfc)
    T_sfc_ice = @. min(T_sfc + ΔT_sfc, T_freeze)
    T_sfc_new = @. ifelse(remains_ice, T_sfc_ice, T_ml + ΔT_ml)

    # Commit the updates.
    @. T_ml += ΔT_ml
    @. h_ice += Δh_ice
    @. T_sfc = T_sfc_new

    # Surface water budget: turbulent flux removes/adds water (evaporation /
    # sublimation), mirroring the slab-ocean convention.
    @. water -= F_water * Δt

    return nothing
end
