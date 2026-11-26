import ClimaTimeSteppers as CTS
import UnrolledUtilities: unrolled_foreach

"""
    StageOnlyUpdateHandler

`ClimaTimeSteppers.UpdateSignalHandler` that returns `true` on intermediate
stage-ready signals (`EndOfStageSignal`) and `false` at the end of the step
(`EndOfStepSignal`) and on non-stage DSS signals (`WithDSSSignal`).

In `ClimaTimeSteppers`, `EndOfStep <: EndOfStage <: WithDSS <: UpdateSignal`.
Specializing `CTS.needs_update!` on `CTS.EndOfStep` overrides the
`CTS.EndOfStage` method when `EndOfStepSignal` fires at the end of `step_u!`,
avoiding duplicate `constrain_state!` and `set_precomputed_quantities!` calls
before `post_step_callback!` finalizes the post-physics state.
"""
struct StageOnlyUpdateHandler <: CTS.UpdateSignalHandler end

CTS.needs_update!(::StageOnlyUpdateHandler, ::CTS.EndOfStage) = true
CTS.needs_update!(::StageOnlyUpdateHandler, ::CTS.EndOfStep) = false

"""
    WithDSSOnlyUpdateHandler

`ClimaTimeSteppers.UpdateSignalHandler` that returns `true` on stage DSS and
stage-ready signals (`WithDSSSignal`, `EndOfStageSignal`) and `false` at the
end of the step (`EndOfStepSignal`).
"""
struct WithDSSOnlyUpdateHandler <: CTS.UpdateSignalHandler end

CTS.needs_update!(::WithDSSOnlyUpdateHandler, ::CTS.WithDSS) = true
CTS.needs_update!(::WithDSSOnlyUpdateHandler, ::CTS.EndOfStep) = false

"""
    NeverUpdateHandler

`ClimaTimeSteppers.UpdateSignalHandler` that returns `false` for every
`UpdateSignal`.
"""
struct NeverUpdateHandler <: CTS.UpdateSignalHandler end

CTS.needs_update!(::NeverUpdateHandler, ::CTS.UpdateSignal) = false

# Wrapper holding a reference to `integrator.cache` in a field named `rc`.
# `ClimaCoupler`'s `Checkpointer.CacheIterator` skips `:rc` fields during cache
# traversal, keeping the RK stage buffers out of coupled checkpoints.
mutable struct IntegratorCacheBox
    rc::Any
end

# Concrete-typed box holding `(stage_index, T_imp[stage_index])` pairs for
# explicit-diagonal implicit stages (`a_imp[i, i] == 0 && b_imp[i] != 0`).
# Naming the field `:rc` keeps `Checkpointer.CacheIterator` from traversing
# RK stage buffers while avoiding dynamic dispatch inside `implicit_tendency!`.
mutable struct ExplicitImpStageRefBox{YT}
    const rc::Vector{Tuple{Int, YT}}
end

# Concrete-typed box holding `integrator.cache.U` so stage-cache trimming
# identifies ARK stage calls without dynamic dispatch, while `:rc` keeps
# `Checkpointer.CacheIterator` from traversing the stage buffer.
mutable struct StageStateRefBox{YT}
    rc::Union{Nothing, YT}
    StageStateRefBox{YT}(rc) where {YT} = new{YT}(rc)
end

"""
    PostStepPhysicsContext

Per-simulation state stored at `p.scratch.post_step` for post-ARK vertical
transport and column physics.

# Fields

  - `in_stage_loop`: `true` while advancing intermediate ARK stages between
    `initialize_post_step_callback!` and `post_step_callback!`, gating
    stage-cache trimming in `set_precomputed_quantities!`.
  - `stage_imp_idx`: Running index of implicit stages within the current step,
    reset to `0` at initialization and at each `post_step_callback!` call.
  - `cache_box`: Boxed reference to `integrator.cache`, populated during
    callback initialization so stage hooks that receive only `(Y, p, ...)` can
    inspect `T_imp` and `tableau`.
  - `stage_state_ref`: Concrete-typed box holding `integrator.cache.U` so stage
    trimming applies only to ARK stage states and leaves direct cache refreshes
    on `Y` untrimmed.
  - `imp_stage_map`: Tuple mapping 1-based implicit solve counter `k_imp` to the
    tableau stage index `i` with `a_imp[i, i] != 0`.
  - `explicit_imp_stages`: Concrete-typed box of `(i, T_imp[i])` pairs for
    explicit-diagonal stages (`a_imp[i, i] == 0 && b_imp[i] != 0`).
  - `hook!`: Optional callable `hook!(Y, p, t, Δt, integrator_cache)` invoked
    inside `step_post_ark_physics!`.
  - `increments`: Box holding the `RealizedIncrements` recorder of the realized
    tendency diagnostics, or `nothing` (see `enable_realized_increments!`).
"""
mutable struct PostStepPhysicsContext{H, YT}
    in_stage_loop::Bool
    stage_imp_idx::Int
    const cache_box::IntegratorCacheBox
    const stage_state_ref::StageStateRefBox{YT}
    imp_stage_map::NTuple{8, Int}
    const explicit_imp_stages::ExplicitImpStageRefBox{YT}
    const hook!::H
    const increments::RealizedIncrementsBox
end

function PostStepPhysicsContext(;
    hook! = Returns(nothing),
    state_type::Type{YT} = Any,
) where {YT}
    return PostStepPhysicsContext{typeof(hook!), YT}(
        false,
        0,
        IntegratorCacheBox(nothing),
        StageStateRefBox{YT}(nothing),
        ntuple(_ -> 0, Val(8)),
        ExplicitImpStageRefBox{YT}(Tuple{Int, YT}[]),
        hook!,
        RealizedIncrementsBox(nothing),
    )
end

contains_data_layout(::PostStepPhysicsContext) = false

"""
    is_stage_cache_update(Y, p)

Return `true` when `set_precomputed_quantities!` is evaluating an intermediate
ARK stage (`in_stage_loop` is active and `Y` is the integrator's stage buffer
`U`).
"""
@inline function is_stage_cache_update(Y, p)
    hasproperty(p, :scratch) && hasproperty(p.scratch, :post_step) ||
        return false
    post_step = p.scratch.post_step
    post_step.in_stage_loop || return false
    stage_U = post_step.stage_state_ref.rc
    return !isnothing(stage_U) && Y === stage_U
end

"""
    post_step_integrator_cache(p)

Return the `integrator.cache` bound to `p.scratch.post_step`, or `nothing` if
unbound.
"""
@inline post_step_integrator_cache(p) =
    (hasproperty(p, :scratch) && hasproperty(p.scratch, :post_step)) ?
    p.scratch.post_step.cache_box.rc : nothing

"""
    reset_post_step_stage_index!(p)

Reset the implicit-stage counter in `p.scratch.post_step` to `0`.
"""
@inline function reset_post_step_stage_index!(p)
    if hasproperty(p, :scratch) && hasproperty(p.scratch, :post_step)
        p.scratch.post_step.stage_imp_idx = 0
    end
    return nothing
end

"""
    advance_post_step_stage_index!(p)

Increment and return the 1-based implicit-stage counter in
`p.scratch.post_step`.
"""
@inline function advance_post_step_stage_index!(p)
    p.scratch.post_step.stage_imp_idx += 1
    return p.scratch.post_step.stage_imp_idx
end

"""
    has_implicit_stage_tendencies(integrator_cache)

Return `true` when `integrator_cache` has an IMEX tableau with `b_imp` weights
and stage implicit tendencies `T_imp` (both `IMEXARKCache` and
`IMEXSSPRKCache`).
"""
@inline has_implicit_stage_tendencies(integrator_cache) =
    hasproperty(integrator_cache, :tableau) &&
    hasproperty(integrator_cache, :T_imp) &&
    hasproperty(integrator_cache.tableau, :b_imp)

"""
    foreach_b_imp_stage(f, integrator_cache)

Call `f(i, b_imp_i, T_imp_i)` in stage order for each stage `i` whose implicit
Butcher weight `b_imp[i]` is non-zero.

Stages with `b_imp[i] == 0` are skipped before indexing `T_imp`, because
`SparseContainer` omits stages whose column in `a_imp` and entry in `b_imp`
both vanish (such as stage 1 of `ARS343`). Explicit-diagonal stages with
`a_imp[i, i] == 0` and `b_imp[i] != 0` (such as stage 1 of `SSP333`) are
allocated in `T_imp` and visited here.
"""
@inline function foreach_b_imp_stage(f::F, integrator_cache) where {F}
    has_implicit_stage_tendencies(integrator_cache) || return nothing
    b_imp = integrator_cache.tableau.b_imp
    T_imp = integrator_cache.T_imp
    unrolled_foreach(ntuple(identity, CTS.get_val_S(b_imp))) do i
        w_i = b_imp[i]
        if !iszero(w_i)
            f(i, w_i, T_imp[i])
        end
    end
    return nothing
end

function _bind_post_step_stage_refs!(
    post_step::PostStepPhysicsContext{H, YT},
    integrator_cache,
) where {H, YT}
    empty!(post_step.explicit_imp_stages.rc)
    post_step.stage_state_ref.rc =
        (hasproperty(integrator_cache, :U) && integrator_cache.U isa YT) ?
        integrator_cache.U : nothing
    if !has_implicit_stage_tendencies(integrator_cache)
        post_step.imp_stage_map = ntuple(_ -> 0, Val(8))
        return nothing
    end
    a_imp = integrator_cache.tableau.a_imp
    b_imp = integrator_cache.tableau.b_imp
    T_imp = integrator_cache.T_imp
    s = CTS.val_to_int(CTS.get_val_S(a_imp))
    @assert s <= 8 "PostStepPhysicsContext supports at most 8 ARK stages, got $s"
    post_step.imp_stage_map = ntuple(Val(8)) do k
        count = 0
        for i in 1:s
            if !iszero(a_imp[i, i])
                count += 1
                count == k && return i
            end
        end
        return 0
    end
    for i in 1:s
        if iszero(a_imp[i, i]) && !iszero(b_imp[i])
            @assert T_imp[i] isa YT
            push!(post_step.explicit_imp_stages.rc, (i, T_imp[i]))
        end
    end
    return nothing
end

"""
    initialize_post_step_callback!(integrator)

Reset the implicit-stage counter, bind `integrator.cache` into
`integrator.p.scratch.post_step`, and enter stage-loop mode at integrator
initialization or `reinit!`.
"""
function initialize_post_step_callback!(integrator)
    p = integrator.p
    if hasproperty(p, :scratch) && hasproperty(p.scratch, :post_step)
        post_step = p.scratch.post_step
        post_step.stage_imp_idx = 0
        post_step.cache_box.rc = integrator.cache
        _bind_post_step_stage_refs!(post_step, integrator.cache)
        post_step.in_stage_loop = true
    end
    return nothing
end

"""
    step_post_ark_physics!(Y, p, t, Δt, integrator_cache)

Execute post-ARK vertical transport and split column physics over the realized
step `Δt` [s], mutating `Y` and `p` in place prior to post-step `dss!`,
`constrain_state!`, and end-of-step cache finalization.
"""
function step_post_ark_physics!(Y, p, t, Δt, integrator_cache)
    hook! = p.scratch.post_step.hook!
    if hook! isa Returns{Nothing}
        ᶠF̃ =
            !(p.atmos.prescribed_flow isa PrescribedFlow) &&
            !isnothing(integrator_cache) &&
            has_implicit_stage_tendencies(integrator_cache) ?
            step_vertical_scalar_transport!(Y, p, Δt, integrator_cache) :
            nothing
        record_realized_increment!(Y, p, :vtrans)
        set_precomputed_quantities!(Y, p, t; update_precip_velocities = false)
        step_column_physics!(Y, p, t, Δt; ᶠF̃)
    else
        hook!(Y, p, t, Δt, integrator_cache)
    end
    return nothing
end

"""
    post_step_callback!(integrator)

Post-step affect function registered as the unconditional first
`DiscreteCallback` in `AtmosSimulation`.

Exits stage-loop mode, resets the stage-pairing index, advances post-ARK physics
with the realized step `FT(integrator.dt)` (reflecting any `tstops` step
reduction), runs `dss!` and `constrain_state!`, records the realized operator
tendencies when a `RealizedIncrements` recorder is enabled, runs
`set_end_of_step_precomputed_quantities!` (preserving realized turbulent surface
fluxes and bottom precipitation fluxes), and restores `in_stage_loop = true`
ahead of model, checkpoint, saving, and diagnostic callbacks.
"""
NVTX.@annotate function post_step_callback!(integrator)
    p = integrator.p
    hasproperty(p, :scratch) && hasproperty(p.scratch, :post_step) ||
        return nothing
    post_step = p.scratch.post_step
    post_step.in_stage_loop = false
    reset_post_step_stage_index!(p)
    Y = integrator.u
    t = integrator.t
    FT = eltype(p.params)
    Δt = FT(integrator.dt)
    begin_realized_increments!(Y, p)
    step_post_ark_physics!(Y, p, t, Δt, integrator.cache)
    dss!(Y, p, t)
    constrain_state!(Y, p, t)
    finish_realized_increments!(Y, p, Δt)
    set_end_of_step_precomputed_quantities!(Y, p, t)
    post_step.in_stage_loop = true
    return nothing
end

"""
    post_step_discrete_callback()

Build the unconditional `ClimaTimeSteppers.DiscreteCallback` that binds
`integrator.cache` at startup and invokes `post_step_callback!` first after
every integrator step.
"""
function post_step_discrete_callback()
    return CTS.DiscreteCallback(
        (u, t, integrator) -> true,
        post_step_callback!;
        initialize = (cb, u, t, integrator) ->
            initialize_post_step_callback!(integrator),
    )
end
