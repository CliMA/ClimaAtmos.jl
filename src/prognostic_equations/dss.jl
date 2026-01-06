using LinearAlgebra: ×, norm, dot

import .Parameters as CAP
using ClimaCore: Operators, Fields, Limiters, Geometry, Spaces

import ClimaComms
using ClimaCore.Geometry: ⊗

import Thermodynamics as TD

using ClimaCore.Utilities: half

import ClimaCore.Fields: ColumnField

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
# params = (ghost_buffer = (c = центр_dss_buffer, f = face_dss_buffer), ...)

dss!(Y_state, params, t_current)
# The ClimaCore.Field objects within Y_state.c and Y_state.f are now updated
# with DSS applied, ensuring continuity across distributed elements.
```
"""

NVTX.@annotate function dss!(Y, p, t)  # TODO: Rename to e.g. `apply_constraints!`
    prescribe_flow!(Y, p, t, p.atmos.prescribed_flow)
    if do_dss(axes(Y.c))
        Spaces.weighted_dss!(Y.c => p.ghost_buffer.c, Y.f => p.ghost_buffer.f)

        scalar_names = CA.scalar_field_names(Y)

        MF.unrolled_foreach(scalar_names) do (name)
            if !MF.is_child_name(name, c.uₕ) && !MF.is_child_name(name, f.u₃)
                ᶜscalar = MF.get_field(Y, name)
                ∑ = sum(Base.Fix2(min, 0), ᶜscalar)
                if ∑ != 0
                    @info "$name is negative by $(-Σ) at t = $t"
                    @info(stacktrace()[1:4])
                end
            end
        end

    end
    return nothing
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
    ᶜts = @. lazy(TD.PhaseEquil_ρTq(thermo_params, Y.c.ρ, ᶜT_init, Y.c.ρq_tot / Y.c.ρ))
    ᶜe_kin = compute_kinetic(Y.c.uₕ, Y.f.u₃)
    @. Y.c.ρe_tot = Y.c.ρ * TD.total_energy(thermo_params, ᶜts, ᶜe_kin, ᶜΦ)
    return nothing
end
