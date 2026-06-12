using LinearAlgebra: ×, norm, dot

import .Parameters as CAP
using ClimaCore: Operators, Fields, Limiters, Geometry, Spaces

import ClimaComms
using ClimaCore.Geometry: ⊗

import Thermodynamics as TD

using ClimaCore.Utilities: half

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
    aerosol_nonnegativity_constraint!(Y, p.atmos.prognostic_aerosols)
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
    aerosol_nonnegativity_constraint!(Y, prognostic_aerosols)

Clip prognostic aerosol mass (`Y.c.ρ<name>`) to be nonnegative after each
timestepper stage.

This mirrors the nonnegativity floor that `tracer_nonnegativity_constraint!`
gives the microphysics condensate tracers, but runs whenever prognostic
aerosols are enabled, independent of `tracer_nonnegativity_method` (a
moisture setting). Aerosols share the condensate tracers' vulnerability —
a near-zero background aloft acted on by the sign-indefinite EDMFX SGS flux
divergence — but above the boundary layer TKE ≈ 0, so eddy diffusion cannot
damp overshoot there. Without this floor, negative aerosol mass accumulates
at upper levels and feeds back through the updraft column march
(χʲ entrains the negative grid mean), blowing up in O(10⁴ s)
(see the SSLT notes in edmfx_sgs_flux.jl).

Like `TracerNonnegativityVaporConstraint` for condensate, the clip is local
and not strictly mass-conserving; the clipped negatives are overshoot-sized
(≪ surface emission), so the spurious source is negligible.
"""
aerosol_nonnegativity_constraint!(Y, ::Val{()}) = nothing
@generated function aerosol_nonnegativity_constraint!(
    Y,
    ::Val{names},
) where {names}
    clip_exprs = map(names) do name
        ρχ = Symbol(:ρ, name)
        :(Y.c.$ρχ .= max.(0, Y.c.$ρχ))
    end
    return quote
        $(clip_exprs...)
        return nothing
    end
end

prescribe_flow!(_, _, _, ::Nothing) = nothing
function prescribe_flow!(Y, p, t, flow::PrescribedFlow)
    (; ᶜΦ) = p.core
    ᶠlg = Fields.local_geometry_field(Y.f)
    z = Fields.coordinate_field(Y.f).z
    @. Y.f.u₃ = C3(Geometry.WVector(flow(z, t)), ᶠlg)

    ᶜlg = Fields.local_geometry_field(Y.c)
    thermo_params = CAP.thermodynamics_params(p.params)
    setup = Setups.ShipwayHill2012(; thermo_params)
    function _shipway_ρ_dry(lg)
        ps = Setups.center_initial_condition(setup, lg, p.params)
        ρ = Setups.air_density(ps, p.params)
        return ρ * (1 - ps.q_tot)
    end
    _shipway_T(lg) = Setups.center_initial_condition(setup, lg, p.params).T
    ᶜρ_init_dry = @. lazy(_shipway_ρ_dry(ᶜlg))
    ᶜT_init = @. lazy(_shipway_T(ᶜlg))

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
