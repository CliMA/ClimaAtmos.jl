using LinearAlgebra: ×, norm, dot

import .Parameters as CAP
using ClimaCore: Operators, Fields, Limiters, Geometry, Spaces

import ClimaComms
using ClimaCore.Geometry: ⊗

import Thermodynamics as TD

using ClimaCore.Utilities: half

"""
    constrain_state!(Y, p, t)

Apply physical constraints to the state `Y` in place.

Composes all state-only corrective updates that keep prognostic variables in a
physically admissible range:

  - `prescribe_flow!(Y, p, t, p.atmos.prescribed_flow)`: imposes the velocity,
    density, and energy of a 'kinematic driver'-like simulation.
  - `tracer_nonnegativity_constraint!(Y, p, t, p.atmos.water.tracer_nonnegativity_method)`:
    removes negative tracer masses.
  - `enforce_physical_constraints!(Y, p, t, p.atmos)`: grid-mean microphysics and
    EDMF updraft corrections.

Registered with `ClimaTimeSteppers` as the `update_constrain_state` hook and fired
at the cadence set by the `update_constrain_state_every` configuration option
(`"stage"`, `"step"`, or `"dss"`; see `update_constrain_state_signal_handler`).
The `dss!` and `set_precomputed_quantities!` calls are not part of this — the
timestepper runs them through its own `dss!` and `cache!` hooks. Returns `nothing`.
"""
NVTX.@annotate function constrain_state!(Y, p, t)
    prescribe_flow!(Y, p, t, p.atmos.prescribed_flow)
    tracer_nonnegativity_constraint!(Y, p, t, p.atmos.water.tracer_nonnegativity_method)
    enforce_physical_constraints!(Y, p, t, p.atmos)
    return nothing
end

"""
    dss!(Y, p, t)

Perform a weighted Direct Stiffness Summation (DSS) on the center (`Y.c`) and face
(`Y.f`) components of the state, in place.

DSS makes fields continuous across spectral-element boundaries by summing the
contributions to degrees of freedom shared between elements, exchanging data
between MPI ranks where needed. The ghost buffers `p.ghost_buffer.c` and
`p.ghost_buffer.f`, created with `ClimaCore.Spaces.create_dss_buffer`, hold the
communicated data. Nothing is done when `do_dss(axes(Y.c))` is `false`, as for a
single-column or finite-difference-only space.

`t` is unused; it is present because `ClimaTimeSteppers` calls this as its `dss!`
hook. Returns `nothing`.

See also `ClimaCore.Spaces.weighted_dss!`, the underlying `ClimaCore` function.
"""
NVTX.@annotate function dss!(Y, p, t)
    if do_dss(axes(Y.c))
        Spaces.weighted_dss!(Y.c => p.ghost_buffer.c, Y.f => p.ghost_buffer.f)
    end
    return nothing
end

"""
    tracer_nonnegativity_constraint!(Y, p, t, tracer_nonnegativity)

Remove negative water tracer masses from the state `Y` in place.

The fallback method is a no-op; the work is done for a
`TracerNonnegativityConstraint{constrain_qtot}`, which loops over `ρq_lcl`,
`ρq_rai`, `ρq_icl`, `ρq_sno`, and, only when the type parameter `constrain_qtot`
is `true`, `ρq_tot`. Two variants:

  - `TracerNonnegativityElementConstraint`: clips `ρq` to `[0, max(ρq)]` within each
    spectral element, using `p.numerics.tracer_nonnegativity_limiter` to redistribute
    the mass horizontally rather than create it.
  - `TracerNonnegativityVaporConstraint`: sets negative `ρq` to zero pointwise
    wherever `ρq_tot > 0`, i.e., takes the deficit out of the vapor.

When `ρq_tot` itself is clipped, the induced increment is passed to
`enforce_mass_energy_consistency!` so density and total energy stay consistent.
Reads `p.numerics` and `p.scratch`; `t` is unused. Called from `constrain_state!`.
"""
tracer_nonnegativity_constraint!(Y, p, t, _) = nothing
function tracer_nonnegativity_constraint!(Y, p, t,
    tracer_nonnegativity::TracerNonnegativityConstraint{constrain_qtot},
) where {constrain_qtot}
    (; tracer_nonnegativity_limiter) = p.numerics
    (; ᶜtemp_scalar, ᶜtemp_scalar_2) = p.scratch
    ᶜρ = Y.c.ρ
    ᶜρq_tot = Y.c.ρq_tot

    tracer_mass_names = (
        @name(ρq_lcl), @name(ρq_rai), @name(ρq_icl), @name(ρq_sno),
        @name(ρq_tot),
    )

    for name in tracer_mass_names
        MatrixFields.has_field(Y.c, name) || continue
        name == @name(ρq_tot) && !constrain_qtot && continue
        # Compute clipped version of ᶜρq
        ᶜρq = MatrixFields.get_field(Y.c, name)

        if tracer_nonnegativity isa TracerNonnegativityElementConstraint
            if (name == @name(ρq_tot)) && constrain_qtot
                ᶜtemp_scalar_2 .= ᶜρq
            end
            ᶜρq_lim = @. ᶜtemp_scalar = max(0, ᶜρq)
            Limiters.compute_bounds!(tracer_nonnegativity_limiter, ᶜρq_lim, ᶜρ)  # bounds are `extrema(ᶜρq_lim) = (0, max(ᶜρq))`
            Limiters.apply_limiter!(ᶜρq, ᶜρ, tracer_nonnegativity_limiter; warn = false)  # ᶜρq is clipped to bounds, effectively ensuring `0 ≤ ᶜρq`
            if (name == @name(ρq_tot)) && constrain_qtot
                @. ᶜtemp_scalar_2 = ᶜρq - ᶜtemp_scalar_2
                enforce_mass_energy_consistency!(Y, p, ᶜtemp_scalar_2)
            end
        elseif tracer_nonnegativity isa TracerNonnegativityVaporConstraint
            # If `ρq` is negative, set it to 0 (as long as `ρq_tot` is positive), otherwise keep it as is
            @. ᶜρq = ifelse(ᶜρq_tot > 0, max(0, ᶜρq), ᶜρq)
        end

    end

end

"""
    prescribe_flow!(Y, p, t, flow)

Overwrite the velocity, density, and total energy of the state `Y` with the
prescribed 'kinematic driver' flow, in place.

The fallback method is a no-op when `flow` is `nothing`. For a `PrescribedFlow`,
sets `Y.f.u₃` from `flow(z, t)`, clips `Y.c.ρq_tot` to be nonnegative (a negative
value would feed the loop negative `ρq_tot` → smaller `ρ` → more negative `q_tot`),
sets `Y.c.ρ` to the initial dry density plus `ρq_tot`, and resets `Y.c.ρe_tot` to
the total energy of the initial temperature profile at the current kinetic energy.
Both initial profiles come from the Shipway and Hill (2012) setup. Called from
`constrain_state!`. Returns `nothing`.
"""
prescribe_flow!(_, _, _, ::Nothing) = nothing
function prescribe_flow!(Y, p, t, flow::PrescribedFlow)
    (; ᶜΦ) = p.core
    ᶠlg = Fields.local_geometry_field(Y.f)
    z = Fields.coordinate_field(Y.f).z
    @. Y.f.u₃ = C3(Geometry.WVector(flow(z, t)), ᶠlg)

    params = p.params
    thermo_params = CAP.thermodynamics_params(params)
    setup = Setups.ShipwayHill2012(; thermo_params)
    function _shipway_ρ_dry(lg)
        ps = Setups.center_initial_condition(setup, lg, params)
        ρ = Setups.air_density(ps, params)
        return ρ * (1 - ps.q_tot)
    end
    _shipway_T(lg) = Setups.center_initial_condition(setup, lg, params).T
    ᶜρ_init_dry = Setups.initial_condition_field(_shipway_ρ_dry, axes(Y.c))
    ᶜT_init = Setups.initial_condition_field(_shipway_T, axes(Y.c))

    # Clamp ρq_tot to non-negative to prevent the feedback loop:
    # negative ρq_tot → lower ρ → more negative q_tot → blowup
    @. Y.c.ρq_tot = max(Y.c.ρq_tot, 0)
    @. Y.c.ρ = ᶜρ_init_dry + Y.c.ρq_tot
    ᶜq_tot = @. lazy(Y.c.ρq_tot / Y.c.ρ)
    ᶜe_kin = compute_kinetic(Y.c.uₕ, Y.f.u₃)
    # Fix energy to initial temperature
    @. Y.c.ρe_tot = Y.c.ρ * TD.total_energy(thermo_params, ᶜe_kin, ᶜΦ, ᶜT_init, ᶜq_tot)
    return nothing
end
