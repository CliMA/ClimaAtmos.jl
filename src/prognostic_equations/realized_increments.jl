#####
##### Realized per-operator tendencies of the split time step
#####

"""
    REALIZED_OPERATORS

Names of the operators of one time step whose realized tendencies
`RealizedIncrements` records, in the order in which they act:

  - `:ark`: the IMEX ARK step (wave core and explicit tendencies), together
    with any change to the state between the end of the previous step and the
    end of the ARK stages.
  - `:vtrans`: post-ARK vertical scalar transport
    (`step_vertical_scalar_transport!`).
  - `:turb`: column turbulence and convection with surface fluxes
    (`step_edmf_and_env_turbulence!` or
    `step_vertical_diffusion_and_surface!`), the prognostic TKE step, and the
    slab ocean turbulent exchange.
  - `:wrepair`: column-conservative water repair
    (`repair_column_water_conservatively!`).
  - `:micro`: coupled microphysics and sedimentation
    (`step_coupled_microphysics_sedimentation!`) and the slab ocean
    precipitation exchange.
  - `:fixer`: state constraints (the EDMF updraft filter, the physical
    constraints between column substeps, `dss!`, and `constrain_state!`), and
    any change not attributed to an earlier operator.

The tendencies of all operators sum to the change of the state over the step
divided by the step `Δt`.
"""
const REALIZED_OPERATORS = (:ark, :vtrans, :turb, :wrepair, :micro, :fixer)

"""
    RealizedIncrements{YT, NT}

Realized tendencies `(Y_after - Y_before) / Δt` of the operators in
`REALIZED_OPERATORS` over the most recent time step.

# Fields

  - `mark`: State after the most recently recorded operator.
  - `tendencies`: `NamedTuple` keyed by `REALIZED_OPERATORS`, each a state-sized
    `FieldVector` holding the realized tendency of that operator [state units
    s⁻¹]. Within a step the entries accumulate increments; at the end of the
    step they are divided by `Δt`. Operators that act several times per step
    (column substeps) accumulate all their increments.
"""
struct RealizedIncrements{YT, NT}
    mark::YT
    tendencies::NT
end

# Box holding the enabled `RealizedIncrements` recorder, or `nothing`, in a
# field named `rc`, which `ClimaCoupler`'s `Checkpointer.CacheIterator` skips,
# keeping the recorder's state-sized buffers out of coupled checkpoints. The
# callers compare `rc` with `nothing` before dispatching on it, so a disabled
# recorder costs no dynamic dispatch or allocation.
mutable struct RealizedIncrementsBox
    rc::Any
end

function RealizedIncrements(Y)
    tendencies = NamedTuple{REALIZED_OPERATORS}(
        ntuple(Val(length(REALIZED_OPERATORS))) do _
            tendency = similar(Y)
            tendency .= zero(eltype(tendency))
            tendency
        end,
    )
    return RealizedIncrements(copy(Y), tendencies)
end

"""
    realized_increments(p)

Return the `RealizedIncrements` recorder bound to `p.scratch.post_step`, or
`nothing` when no recorder is enabled.
"""
@inline realized_increments(p) =
    (hasproperty(p, :scratch) && hasproperty(p.scratch, :post_step)) ?
    p.scratch.post_step.increments.rc : nothing

"""
    enable_realized_increments!(p, Y)

Return the `RealizedIncrements` recorder of `p`, allocating one with its mark
at the state `Y` if none is enabled. Recording starts with the next time step;
until then all tendencies are zero. Called by the realized tendency
diagnostics, so the recorder costs nothing unless such a diagnostic is
requested.
"""
function enable_realized_increments!(p, Y)
    recorder = realized_increments(p)
    isnothing(recorder) || return recorder
    recorder = RealizedIncrements(Y)
    p.scratch.post_step.increments.rc = recorder
    return recorder
end

"""
    record_realized_increment!(Y, p, name::Symbol)

Add the change of `Y` since the last recorded operator to the realized
tendency of operator `name` (one of `REALIZED_OPERATORS`), and move the mark to
`Y`. Does nothing unless a recorder is enabled.
"""
@inline function record_realized_increment!(Y, p, name::Symbol)
    recorder = realized_increments(p)
    recorder === nothing || _record_realized_increment!(recorder, Y, name)
    return nothing
end

function _record_realized_increment!(recorder::RealizedIncrements, Y, name)
    tendency = getproperty(recorder.tendencies, name)
    @. tendency += Y - recorder.mark
    recorder.mark .= Y
    return nothing
end

"""
    begin_realized_increments!(Y, p)

Start recording a time step: zero the tendencies and attribute the change of
`Y` since the end of the previous step to `:ark`. Does nothing unless a
recorder is enabled.
"""
@inline function begin_realized_increments!(Y, p)
    recorder = realized_increments(p)
    recorder === nothing || _begin_realized_increments!(recorder, Y)
    return nothing
end

function _begin_realized_increments!(recorder::RealizedIncrements, Y)
    foreach(recorder.tendencies) do tendency
        tendency .= zero(eltype(tendency))
    end
    _record_realized_increment!(recorder, Y, :ark)
    return nothing
end

"""
    finish_realized_increments!(Y, p, Δt)

Finish recording a time step of length `Δt` [s]: attribute the remaining change
of `Y` to `:fixer` and divide every accumulated increment by `Δt`. Does nothing
unless a recorder is enabled.
"""
@inline function finish_realized_increments!(Y, p, Δt)
    recorder = realized_increments(p)
    recorder === nothing || _finish_realized_increments!(recorder, Y, Δt)
    return nothing
end

function _finish_realized_increments!(recorder::RealizedIncrements, Y, Δt)
    _record_realized_increment!(recorder, Y, :fixer)
    inv_Δt = inv(Δt)
    foreach(recorder.tendencies) do tendency
        tendency .*= inv_Δt
    end
    return nothing
end
