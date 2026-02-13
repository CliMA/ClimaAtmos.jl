using LinearAlgebra: ×, norm, dot

import .Parameters as CAP
using ClimaCore: Operators, Fields, Limiters, Geometry, Spaces

import ClimaComms
using ClimaCore.Geometry: ⊗

import Thermodynamics as TD

using ClimaCore.Utilities: half

import ClimaCore.Fields: ColumnField

"""
    constrain_state!(Y, p, t)

Apply constraints to the state `Y`.

This function contains constraints that may be applied to the state `Y`,
in order to ensure that the state satisfies certain physical properties.

Currently, these include
- `prescribe_flow!`: used for 'kinematic driver'-like simulations
- `tracer_nonnegativity_constraint!`: used to ensure that tracer fields are non-negative
- `dss!`: used to ensure that fields are continuous at element boundaries
"""
NVTX.@annotate function constrain_state!(Y, p, t)
    prescribe_flow!(Y, p, t, p.atmos.prescribed_flow)
    tracer_nonnegativity_constraint!(Y, p, t, p.atmos.water.tracer_nonnegativity_method)
    dss!(Y, p, t)
    return nothing
end

"""
    dss!(Y, p, t)

Perform a weighted Direct Stiffness Summation (DSS) on components of the state `Y`.

This function applies DSS to `ClimaCore.Field`s (or structures of `Field`s)
typically named `.c` (center-located) and `.f` (face-located) within the state
object `Y`. The DSS operation is essential in `ClimaCore` for ensuring that
fields are C0 continuous across element boundaries. It correctly sums contributions 
to degrees of freedom that are shared between different processes (MPI ranks).

The operation is performed in-place, modifying the fields within `Y` (e.g., `Y.c`, `Y.f`).
Ghost buffers (e.g., `p.ghost_buffer.c`, `p.ghost_buffer.f`), which are themselves
`ClimaCore.Field`s or structures of `Field`s created using
`ClimaCore.Spaces.create_dss_buffer`, are used as temporary storage during
the communication and summation process.

DSS is conditionally performed if `do_dss(axes(Y.c))` returns `true`. This check
typically inspects the `ClimaCore.Spaces.AbstractSpace` of `Y.c` to determine
if its `ClimaComms.context` signifies a distributed parallel environment
requiring inter-process communication.

Arguments:
- `Y`: The state object (e.g., a `NamedTuple` or `ClimaCore.Fields.FieldVector`)
       containing components like `.c` and `.f`. These components are expected to be
       `ClimaCore.Field`s or (possibly nested) `NamedTuple`s of `ClimaCore.Field`s,
       which are modified in-place by this function.
- `p`: A parameter object or cache, expected to contain `p.ghost_buffer`.
       The ghost buffer should mirror the structure of the fields in `Y`
       that undergo DSS and must be compatible with them (typically created via
       `ClimaCore.Spaces.create_dss_buffer(Y)` or by creating buffers for
       individual components like `Y.c` and `Y.f`).
- `t`: The current simulation time. This argument is part of a standard
       function signature in time-stepping loops (common in `ClimaAtmos` and other
       models using `ClimaCore`) but is not be directly used by this function
       if the DSS operation itself is time-independent.

Returns:
- `nothing` (the function modifies fields within `Y` in-place).

See also:
- `ClimaCore.Spaces.weighted_dss!`: The underlying `ClimaCore` function that performs the DSS.
- `ClimaCore.Spaces.create_dss_buffer`: The `ClimaCore` function for creating compatible ghost buffers.
- `ClimaCore.Fields.Field`: The fundamental data type for spatial fields in `ClimaCore`.
- `ClimaCore.Spaces.AbstractSpace`: The type representing the spatial discretization in `ClimaCore`.
- `ClimaComms.context`: Used to determine if the computation is distributed.

# Example (Conceptual within ClimaCore/ClimaAtmos context)
```julia
# Assume Y_state is a FieldVector (or NamedTuple) from an ODE solver state,
# where Y_state.c and Y_state.f are ClimaCore.Field objects or NamedTuples of Fields.
# Assume params.ghost_buffer contains appropriately structured DSS buffers
# created using ClimaCore.Spaces.create_dss_buffer.
# Assume t_current is the current simulation time.

# Example structure for Y and p.ghost_buffer:
# Y_state = (c = center_fields, f = face_fields)
# params = (ghost_buffer = (c = center_dss_buffer, f = face_dss_buffer), ...)

dss!(Y_state, params, t_current)
# The ClimaCore.Field objects within Y_state.c and Y_state.f are now updated
# with DSS applied, ensuring continuity across distributed elements.
```
"""
NVTX.@annotate function dss!(Y, p, t)
    if do_dss(axes(Y.c))
        Spaces.weighted_dss!(Y.c => p.ghost_buffer.c, Y.f => p.ghost_buffer.f)
    end
    return nothing
end

tracer_nonnegativity_constraint!(Y, p, t, _) = nothing
function tracer_nonnegativity_constraint!(Y, p, t,
    tracer_nonnegativity::TracerNonnegativityConstraint{constrain_qtot},
) where {constrain_qtot}
    (; tracer_nonnegativity_limiter) = p.numerics
    (; ᶜtemp_scalar, ᶜtemp_scalar_2) = p.scratch
    ᶜρ = Y.c.ρ
    ᶜρq_tot = Y.c.ρq_tot

    tracer_mass_names = (
        @name(ρq_liq), @name(ρq_rai), @name(ρq_ice), @name(ρq_sno),
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
                ᶜΔρq_tot = similar(ᶜρq)
                ᶜΔρq_tot .= ᶜρq .- ᶜtemp_scalar_2
                thp = CAP.thermodynamics_params(p.params)
                ᶜT = p.precomputed.ᶜT
                ᶜΦ = p.core.ᶜΦ
                @. Y.c.ρ += ᶜΔρq_tot
                @. Y.c.ρe_tot += ᶜΔρq_tot * (TD.internal_energy_vapor(thp, ᶜT) + ᶜΦ)
            end
        elseif tracer_nonnegativity isa TracerNonnegativityVaporConstraint
            # If `ρq` is negative, set it to 0 (as long as `ρq_tot` is positive), otherwise keep it as is
            @. ᶜρq = ifelse(ᶜρq_tot > 0, max(0, ᶜρq), ᶜρq)
        end

    end

end

prescribe_flow!(_, _, _, ::Nothing) = nothing
function prescribe_flow!(Y, p, t, flow::PrescribedFlow)
    (; ᶜΦ) = p.core
    ᶠlg = Fields.local_geometry_field(Y.f)
    z = Fields.coordinate_field(Y.f).z
    @. Y.f.u₃ = C3(Geometry.WVector(flow(z, t)), ᶠlg)

    ### Fix energy to initial temperature
    ᶜlg = Fields.local_geometry_field(Y.c)
    local_state = InitialConditions.ShipwayHill2012()(p.params)
    get_ρ_init_dry(ls) = ls.thermo_state.ρ * (1 - ls.thermo_state.q_tot)
    get_T_init(ls) = TD.air_temperature(ls.thermo_params, ls.thermo_state)
    ᶜρ_init_dry = @. lazy(get_ρ_init_dry(local_state(ᶜlg)))
    ᶜT_init = @. lazy(get_T_init(local_state(ᶜlg)))

    thermo_params = CAP.thermodynamics_params(p.params)

    @. Y.c.ρ = ᶜρ_init_dry + Y.c.ρq_tot
    ᶜq_tot = @. lazy(Y.c.ρq_tot / Y.c.ρ)
    ᶜe_kin = compute_kinetic(Y.c.uₕ, Y.f.u₃)
    @. Y.c.ρe_tot = Y.c.ρ * TD.total_energy(thermo_params, ᶜe_kin, ᶜΦ, ᶜT_init, ᶜq_tot)
    return nothing
end
