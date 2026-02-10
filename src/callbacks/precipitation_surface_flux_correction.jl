"""
    correct_precipitation_surface_fluxes!(integrator)

Post-step callback that corrects `Y.sfc.water` to account for the IMEX
Newton asymmetry in precipitation fluxes.

The IMEX solver extracts implicit tendencies via `(U - temp)/dtγ` (residual),
which gives atmospheric variables a Newton-corrected tendency but leaves
`sfc.water` with the tendency evaluated at the prediction `U₀`. This creates
an O(dt) per-step mismatch between the sedimentation leaving `ρq_tot` and
the precipitation entering `sfc.water`.

This callback recomputes the bottom-face precipitation flux from the
converged post-step `Y_{n+1}` and corrects `sfc.water` by:

    sfc.water += dt * (flux(Y_{n+1}) - flux(Y_n))

where `flux(Y_n)` is the stale precomputed value used during the step
(stored in `p.precomputed.surface_rain_flux` and `surface_snow_flux`).

After correction, the precomputed fluxes are updated to the new values
so the correction applies cumulatively across steps.
"""
function correct_precipitation_surface_fluxes!(integrator)
    Y = integrator.u
    p = integrator.p
    dt = integrator.dt
    FT = Spaces.undertype(axes(Y.c))

    # Only applies to SlabOceanSST with moisture
    p.atmos.surface_model isa SlabOceanSST || return nothing
    p.atmos.moisture_model isa DryModel && return nothing

    (; surface_rain_flux, surface_snow_flux) = p.precomputed

    # Store old fluxes before recomputation
    old_total_flux = @. surface_rain_flux + surface_snow_flux

    # Recompute fluxes from converged Y_{n+1}
    set_precipitation_surface_fluxes!(
        Y,
        p,
        p.atmos.microphysics_model,
    )

    # Correction: replace stale flux with post-step flux
    new_total_flux = @. surface_rain_flux + surface_snow_flux
    @. Y.sfc.water += FT(dt) * (new_total_flux - old_total_flux)

    return nothing
end
