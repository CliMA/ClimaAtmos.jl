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

"""
    tracer_nonnegativity_constraint!(Y, p, t, policy)

Apply the per-tracer-class nonnegativity policy to the state `Y`.

`policy` is typically a [`TracerNonnegativityPolicy`](@ref) holding one
[`TracerNonnegativityMethod`](@ref) (or `nothing`) per tracer class. A bare
method is accepted for backwards compatibility and treated as a water-only
policy.

Each class contributes `(tracer_name, method)` pairs and a single loop
applies the method-appropriate kernel per pair, so the enforcement kernels
(elementwise limiter, vapor-gated clip, plain clip) are implemented once and
shared by all classes. Methods enforced elsewhere in the timestep
(`VaporTendency` as a tendency, `VerticalWaterBorrowing` via the `lim!`
hook) fall through every kernel branch and are no-ops here.

The aerosol class requires a floor for stability whenever prognostic
aerosols are on: the EDMFX SGS flux divergence is not sign-preserving for
the grid mean, and above the boundary layer TKE ≈ 0 means no eddy diffusion
damps overshoot, so negative aerosol mass otherwise accumulates aloft and
feeds back through the updraft column march (blowup in O(10⁴ s); see the
SSLT notes in edmfx_sgs_flux.jl). The clip kernel is local and not strictly
mass-conserving; clipped negatives are overshoot-sized (≪ surface emission),
so the spurious source is negligible.
"""
tracer_nonnegativity_constraint!(Y, p, t, _) = nothing
# TODO below is legacy for water only specifications, remove in breaking rel.
tracer_nonnegativity_constraint!(Y, p, t, method::TracerNonnegativityMethod) =
    apply_tracer_nonnegativity!(
        Y,
        p,
        nonnegativity_pairs(method, WATER_NONNEGATIVITY_NAMES),
    )

function tracer_nonnegativity_constraint!(
    Y,
    p,
    t,
    policy::TracerNonnegativityPolicy,
)
    pairs = (
        nonnegativity_pairs(policy.water, WATER_NONNEGATIVITY_NAMES)...,
        nonnegativity_pairs(
            policy.aerosol,
            aerosol_mass_names(p.atmos.prognostic_aerosols),
        )...,
    )
    return apply_tracer_nonnegativity!(Y, p, pairs)
end

nonnegativity_pairs(::Nothing, _) = ()
nonnegativity_pairs(method::TracerNonnegativityMethod, names) =
    map(name -> (name, method), names)

const WATER_NONNEGATIVITY_NAMES =
    (@name(ρq_lcl), @name(ρq_rai), @name(ρq_icl), @name(ρq_sno), @name(ρq_tot))
@generated aerosol_mass_names(::Val{names}) where {names} =
    :($(map(n -> MatrixFields.FieldName(Symbol(:ρ, n)), names)))

constrains_qtot(::TracerNonnegativityConstraint{qtot}) where {qtot} = qtot
constrains_qtot(::TracerNonnegativityMethod) = false

function apply_tracer_nonnegativity!(Y, p, pairs)
    (; tracer_nonnegativity_limiter) = p.numerics
    (; ᶜtemp_scalar, ᶜtemp_scalar_2) = p.scratch
    ᶜρ = Y.c.ρ

    MatrixFields.unrolled_foreach(pairs) do (name, method)
        MatrixFields.has_field(Y.c, name) || return
        name == @name(ρq_tot) && !constrains_qtot(method) && return
        ᶜρq = MatrixFields.get_field(Y.c, name)

        if method isa TracerNonnegativityElementConstraint
            is_qtot = name == @name(ρq_tot)
            if is_qtot
                ᶜtemp_scalar_2 .= ᶜρq
            end
            ᶜρq_lim = @. ᶜtemp_scalar = max(0, ᶜρq)
            Limiters.compute_bounds!(tracer_nonnegativity_limiter, ᶜρq_lim, ᶜρ)  # bounds are `extrema(ᶜρq_lim) = (0, max(ᶜρq))`
            Limiters.apply_limiter!(ᶜρq, ᶜρ, tracer_nonnegativity_limiter; warn = false)  # ᶜρq is clipped to bounds, effectively ensuring `0 ≤ ᶜρq`
            if is_qtot
                @. ᶜtemp_scalar_2 = ᶜρq - ᶜtemp_scalar_2
                enforce_mass_energy_consistency!(Y, p, ᶜtemp_scalar_2)
            end
        elseif method isa TracerNonnegativityVaporConstraint
            # If `ρq` is negative, set it to 0 (as long as `ρq_tot` is positive), otherwise keep it as is
            ᶜρq_tot = Y.c.ρq_tot
            @. ᶜρq = ifelse(ᶜρq_tot > 0, max(0, ᶜρq), ᶜρq)
        elseif method isa TracerNonnegativityClip
            @. ᶜρq = max(0, ᶜρq)
        end
        return nothing
    end

    return nothing
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
