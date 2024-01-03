import Random
"""
    AutoDebugRun

A type to dispatch for solving in auto-debug mode.

To run this interactively, you can use

julia --project=examples
```julia
using Revise; if !("--config_file" in ARGS)
    push!(ARGS, "--config_file")
    push!(ARGS, "config/model_configs/test_auto_debug.yml")
end; include("examples/hybrid/driver.jl")
```
"""
Base.@kwdef mutable struct AutoDebugRun
    """ Bool indicating that the simulation crashed. """
    crashed::Bool = false
    """ Simulation time that the simulation crashed. """
    t_fail::Float64 = 0
    """ Simulation restart time to replay crashed simulation. """
    t_start_replay::Float64 = 0
    """ Simulation time to start boosting diagnostics. """
    t_boost_diagnostics::Float64 = 0
    """ Cache for dt_save_state_to_disk. """
    dt_save_state_to_disk::Float64
    """ Cached integrator step. """
    step_crash::Int = 0
    """ Cache for output_dir. """
    output_dir::String
    """ Minimum number of steps to replay. """
    n_min_steps_to_replay::Int = 10
    """ The portion of simulation time to replay. For example, for 0.8 we replay the last 80% from `t_last_restart` to `t_fail`. """
    replay_factor::Float64 = 0.8
    """ Auto-debug plotting function. """
    auto_plot::Function = (integrator) -> nothing
    """ Simulate crash at nsteps = 10 for testing purposes. """
    simulate_crash::Bool = false
    """ Bool indicating that we are in replay. """
    in_replay::Bool = false
end

import ClimaCore.Fields as Fields
function find_column(f::Fields.Field, cond)
    _colidx = Fields.ColumnIndex((-1, -1), -1)
    Fields.bycolumn(axes(f)) do colidx
        if cond(f[colidx])
            _colidx = colidx
        end
    end
    found = _colidx.h ≠ -1
    return (_colidx, found)
end
function count_columns(f::Fields.Field, cond)::Int
    c = 0
    Fields.bycolumn(axes(f)) do colidx
        if cond(f[colidx])
            c += 1
        end
    end
    return c
end


function reset_to_last_restart!(integrator, debugger::AutoDebugRun)

    t = debugger.t_start_replay
    # Read restart data:
    (; u, p) = integrator
    day = floor(Int, t / (60 * 60 * 24))
    sec = floor(Int, t % (60 * 60 * 24))
    (; output_dir) = debugger
    @info "Replaying crashed simulation from HDF5 file on day $day second $sec"
    mkpath(joinpath(output_dir))
    restart_file = joinpath(output_dir, "day$day.$sec.hdf5")
    comms_ctx = ClimaComms.context(integrator.u.c)
    reader = InputOutput.HDF5Reader(restart_file, comms_ctx)
    Y = InputOutput.read_field(reader, "Y")
    t_start = InputOutput.HDF5.read_attribute(reader.file, "time")
    @assert t == t_start
    integrator.t = t_start
    integrator.step = Int(integrator.t / integrator.dt)

    # Restart state:
    @. u = Y
    # Re-compute cache
    CA.set_precomputed_quantities!(u, p, t)
    # Base.close(reader) # is this needed?
    return nothing
end

function export_state(debugger::AutoDebugRun, t)
    if debugger.in_replay
        return t ≥ debugger.t_boost_diagnostics
    else
        return false
    end
end

function replay!(integrator, debugger)
    while !isempty(integrator.tstops) && integrator.step != integrator.stepstop
        Random.seed!(1234)
        SciMLBase.step!(integrator)
        if debugger.simulate_crash && integrator.step == 30
            @warn "Simulating crash (during replay)..."
            error("Simulating crash for testing purposes")
        end
    end
    return nothing
end

function solve_run_mode!(integrator, debugger::AutoDebugRun)
    @info "Solving in debug mode, does not support restarted simulations."
    @info "Re-setting random seed for step-wise reproducibility"

    while !isempty(integrator.tstops) && integrator.step != integrator.stepstop
        Random.seed!(1234)
        try
            SciMLBase.step!(integrator)
            if debugger.simulate_crash && integrator.step == 30
                @warn "Simulating crash..."
                error("Simulating crash for testing purposes")
            end
            # maybe_graceful_exit(integrator)
            # TODO Add MPI support (use watchdog file)
            # if comms_ctx.err_code == 1
            #     error("Someone else crashed")
            # end
        catch e1
            # comms_ctx.err_code = 1
            debugger.crashed = true
            debugger.t_fail = integrator.t
            debugger.step_crash = integrator.step
            try
                reset_to_last_restart!(integrator, debugger)
                debugger.in_replay = true
                replay!(integrator, debugger)
            catch e2
                @error "ClimaAtmos simulation crashed. Stacktrace for failed simulation" exception =
                    (e2, catch_backtrace())
            end
            debugger.auto_plot(integrator)
            break # break out of while-loop
        end

        # Update debugger
        (; t, dt) = integrator
        (; dt_save_state_to_disk) = debugger
        # Update t_start_replay
        #  - assumes t_start = 0
        t_start = t * 0
        n_steps = Int(t / dt)
        i = findlast(n -> t - n * dt_save_state_to_disk > 0, 0:n_steps)
        n_replay_steps =
            count(n -> t - n * dt_save_state_to_disk > 0, 0:n_steps)

        # It may be that a simulation exports and then
        # crashes on the very next step, making the replay
        # window very short (maybe even only one step).
        # So, let's ensure that we have at least a N steps
        # since the last replay, if not, let's replay from
        # one back.
        t_last_restart = if isnothing(i)
            t_start
        elseif i == 0
            zero(dt_save_state_to_disk)
        else
            # t_last_step
            # Replay from two dt_save_state_to_disk's ago.
            j =
                (i + debugger.n_min_steps_to_replay) * dt_save_state_to_disk > t ? (i - 1) : i
            j * dt_save_state_to_disk
        end
        # TODO: the logic would be much easier if we
        #       instead specified `nsteps_save_state_to_disk`,
        #       or converted to this very early.
        t_last_step = Int(t_last_restart / dt)

        # TODO: should this instead be 80% between
        #       t_last_restart and t_fail? A good choice
        #       depends on dt_save_state_to_disk, the speed of
        #       the physics, and (maybe) t_fail?
        debugger.t_start_replay = t_last_restart
        (; replay_factor) = debugger
        debugger.t_boost_diagnostics = t_last_restart # can this be logically/heuristically improved?
        # debugger.t_boost_diagnostics = if n_replay_steps > 10
        #     t_last_restart*(1-replay_factor)+t*replay_factor
        # else
        #     t_last_restart
        # end
    end
    finalize!(integrator)
    return integrator.sol
end
