"""
The calibration driver: flat parallelism and staged `T_stops`.

Two things differ from stock ClimaCalibrate, both by overriding rather than forking:

  - `run_iteration` is specialized on `SocratesInterface` so the unit of work is one
    `(member, case)` pair instead of one member. With 11 cases and, say, 4 members, that is 44
    independent tasks over the same worker pool instead of 4 — so worker count is no longer capped
    by ensemble size, and a slow case cannot idle a worker behind it.
  - `calibrate` owns the iteration loop so `DataMisfitController.terminate_at` can be ratcheted
    through `T_stops`, and so `max_iter` can stop the run whether or not the final stop is reached.

Both reuse Layer 1's runner and ClimaCalibrate's own loop primitives (`initialize`,
`load_ekp_struct`, `observation_map_and_update!`, `last_completed_iteration`).
"""

using ClimaCalibrate: ClimaCalibrate
using Distributed: Distributed
using EnsembleKalmanProcesses: EnsembleKalmanProcesses as EKP
using JLD2: JLD2
using Logging: Logging

# --- rebuilding immutable EKP objects ------------------------------------------------------- #

"""
    set_field(x, name, value)

A copy of the immutable struct `x` with field `name` replaced.

Uses the positional constructor Julia generates for a parametric struct, reading `fieldnames` at
runtime — so no field list is duplicated here and upstream additions need no change. If a custom
inner constructor ever shadows the generated one this fails loudly with a `MethodError` rather than
silently doing the wrong thing.

Mutable fields are shared with `x` rather than copied, which is what preserves
`DataMisfitController`'s `iteration` history across a ratchet.
"""
function set_field(x, name::Symbol, value)
    T = typeof(x)
    name in fieldnames(T) ||
        error("$(nameof(T)) has no field `$name`; it has $(fieldnames(T))")
    return T((f === name ? value : getfield(x, f) for f in fieldnames(T))...)
end

"""
    ratchet_terminate_at(ekp, T_stops)

`ekp` with its scheduler's `terminate_at` advanced to the first entry of `T_stops` beyond the
accumulated algorithmic time, or `ekp` unchanged when no advance is due.

`terminate_at` feeds the `DataMisfitController`'s timestep selection, not just its stopping test, so
advancing it in stages gives a staged learning-rate schedule rather than one budget spread over the
whole run.
"""
function ratchet_terminate_at(ekp, T_stops)
    isnothing(T_stops) && return ekp
    isempty(T_stops) && return ekp
    scheduler = ekp.scheduler
    hasproperty(scheduler, :terminate_at) || return ekp
    T = sum(EKP.get_Δt(ekp); init = zero(eltype(EKP.get_Δt(ekp))))
    next = findfirst(>(T), T_stops)
    isnothing(next) && return ekp
    target = T_stops[next]
    target > scheduler.terminate_at || return ekp
    @info "T_stops: advancing terminate_at" from = scheduler.terminate_at to = target T
    return set_field(ekp, :scheduler, set_field(scheduler, :terminate_at, target))
end

"""Accumulated algorithmic time `T = Σ Δt` consumed by `ekp` so far."""
accumulated_T(ekp) = sum(EKP.get_Δt(ekp); init = 0.0)

"""
    use_worker_log(dir)

Send this process's log to `dir/worker_<id>.log`. Does nothing on the driver, whose output belongs on
the terminal.

Call it from the same `@everywhere` block that loads the model code. Without it a worker logs to the
terminal, interleaved with every other worker and with the driver.
"""
function use_worker_log(dir::AbstractString)
    Distributed.myid() == 1 && return nothing
    mkpath(dir)
    path = joinpath(dir, "worker_$(Distributed.myid()).log")
    io = open(path, "w")
    Base.global_logger(Logging.SimpleLogger(io))
    @info "Logging from worker $(Distributed.myid())"
    flush(io)
    return path
end

# --- flat (member, case) iteration ---------------------------------------------------------- #

"""Marker recording that one `(iteration, member, case)` run finished, for resume."""
case_marker_path(interface, iteration, member, case) =
    joinpath(case_output_dir(interface, iteration, member, case), "case_completed")

case_completed(interface, iteration, member, case) =
    isfile(case_marker_path(interface, iteration, member, case))

function mark_case_completed(interface, iteration, member, case)
    path = case_marker_path(interface, iteration, member, case)
    mkpath(dirname(path))
    write(path, "completed")
    return nothing
end

"""
    ClimaCalibrate.Calibration.run_iteration(backend::WorkerBackend, interface::SocratesInterface, ...)

Run one iteration as a flat pool of `(member, case)` tasks.

Only the cases that have not already completed are scheduled, so a resumed iteration re-runs exactly
what is missing rather than a whole member. A member counts as failed if any of its cases failed,
and the iteration aborts if the member failure rate exceeds `backend.failure_rate` — matching stock
ClimaCalibrate's accounting.
"""
function ClimaCalibrate.Calibration.run_iteration(
    backend::ClimaCalibrate.WorkerBackend,
    interface::SocratesInterface,
    iteration,
    ensemble_size,
    output_dir,
)
    tasks = [
        (member, case) for member in 1:ensemble_size for case in interface.cases if
        !case_completed(interface, iteration, member, case)
    ]
    total = ensemble_size * length(interface.cases)
    @info "Iteration $iteration: $(length(tasks))/$total case-runs to do" n_workers =
        length(backend.worker_pool.workers)

    if !isempty(tasks)
        executor = SS.SM.WorkerPoolExecutor(
            backend.worker_pool;
            empty_pool_timeout = backend.empty_pool_timeout,
        )
        run_one = task -> begin
            member, case = task
            SocratesCalibration.run_case_for_member(interface, iteration, member, case)
        end
        results = SS.SM.run_tasks(run_one, tasks, executor)
        for ((member, case), result) in zip(tasks, results)
            isnothing(result) || mark_case_completed(interface, iteration, member, case)
        end
    end

    failed = [
        member for member in 1:ensemble_size if
        any(case -> !case_completed(interface, iteration, member, case), interface.cases)
    ]
    for member in 1:ensemble_size
        member in failed ||
            ClimaCalibrate.write_model_completed(output_dir, iteration, member)
    end
    rate = length(failed) / ensemble_size
    rate > backend.failure_rate && error(
        "Iteration $iteration had a $(round(rate * 100; digits = 2))% member failure rate \
         (members $failed), exceeding the $(backend.failure_rate * 100)% threshold.",
    )
    isempty(failed) || @warn "Failed ensemble members: $failed"
    return nothing
end

# --- the loop ------------------------------------------------------------------------------- #

"""
    clear_iterations!(output_dir) -> n_removed

Remove the calibration state under `output_dir`: every `iteration_XXX` directory and `eki_file.jld2`.

Anything else in `output_dir` — `logs/`, `figures/` — is left alone, so this is safe to call after the
workers have opened their log files.
"""
function clear_iterations!(output_dir::AbstractString)
    isdir(output_dir) || return 0
    removed = 0
    for entry in readdir(output_dir)
        path = joinpath(output_dir, entry)
        if startswith(entry, "iteration_") && isdir(path)
            rm(path; recursive = true)
            removed += 1
        elseif entry == "eki_file.jld2"
            rm(path; force = true)
        end
    end
    return removed
end

"""
    calibrate(backend, ekp, interface; n_iterations, prior, T_stops, max_iter,
              force_termination_at_T, overwrite)

Run the calibration, staging `terminate_at` through `T_stops`.

The loop continues while the accumulated `T` is below the final stop, or while iterations remain and
`force_termination_at_T` is `false`, and stops unconditionally at `max_iter`.

`overwrite` discards a previous calibration's iterations before starting, instead of resuming it. It
acts here rather than when the interface is built, so not calling `calibrate` cannot destroy output,
and it clears only the calibration state — see [`clear_iterations!`](@ref).

Reuses ClimaCalibrate's `initialize`, `load_ekp_struct` and `observation_map_and_update!`, so the
on-disk layout and restart behaviour are the stock ones.
"""
function calibrate(
    backend,
    ekp,
    interface::SocratesInterface;
    prior,
    n_iterations::Int,
    T_stops = nothing,
    max_iter::Int = n_iterations,
    force_termination_at_T::Bool = false,
    overwrite::Bool = false,
)
    output_dir = interface.output_dir
    if overwrite
        removed = clear_iterations!(output_dir)
        removed > 0 &&
            @info "Discarded a previous calibration" output_dir n_iterations = removed
    end
    ensemble_size = EKP.get_N_ens(ekp)
    terminal_T = isnothing(T_stops) ? Inf : Float64(last(T_stops))
    @info "Initializing SOCRATES calibration" ensemble_size n_cases =
        length(interface.cases) n_iterations max_iter T_stops output_dir
    ekp = ClimaCalibrate.initialize(ekp, prior, output_dir)

    iteration = ClimaCalibrate.last_completed_iteration(output_dir) + 1
    while true
        T = accumulated_T(ekp)
        keep_going =
            (T < terminal_T) || (iteration <= n_iterations && !force_termination_at_T)
        keep_going || (@info "Stopping: T = $T reached the final stop $terminal_T"; break)
        iteration > max_iter &&
            (@info "Stopping: reached max_iter = $max_iter (T = $T)"; break)

        @info "Iteration $iteration" T terminate_at =
            hasproperty(ekp.scheduler, :terminate_at) ? ekp.scheduler.terminate_at :
            nothing
        ClimaCalibrate.Calibration.run_iteration(
            backend,
            interface,
            iteration,
            ensemble_size,
            output_dir,
        )

        ekp = ClimaCalibrate.load_ekp_struct(output_dir, iteration)
        ekp = ratchet_terminate_at(ekp, T_stops)
        # Persist the ratchet before the update, so the object the update runs on is the staged one
        # and a restart reloads it.
        JLD2.save_object(ClimaCalibrate.ekp_path(output_dir, iteration), ekp)
        terminate = ClimaCalibrate.observation_map_and_update!(
            ekp,
            output_dir,
            iteration,
            prior,
            interface,
        )
        ekp = ClimaCalibrate.load_ekp_struct(output_dir, iteration + 1)
        iteration += 1
        if !isnothing(terminate)
            @info "Stopping: the EKP scheduler signalled termination"
            break
        end
    end
    return ekp
end

# --- per-iteration housekeeping -------------------------------------------------------------- #

"""
    ClimaCalibrate.analyze_iteration(interface::SocratesInterface, ekp, g_ensemble, prior, output_dir, iteration)

Log progress and delete the completed iteration's model output.

Runs after the ensemble update and after `G_ensemble.jld2` is written, so the diagnostics are no
longer needed. The EKP objects, `G_ensemble.jld2`, each member's `parameters.toml`, and the
completion markers are kept — only the NetCDF output is removed, which is the bulk of the disk.
"""
function ClimaCalibrate.analyze_iteration(
    interface::SocratesInterface,
    ekp,
    g_ensemble,
    prior,
    output_dir,
    iteration,
)
    @info "Iteration $iteration complete" mean_parameters =
        EKP.get_ϕ_mean_final(prior, ekp) error = last(EKP.get_error(ekp)) T =
        accumulated_T(ekp)
    interface.prune_output || return nothing
    removed = 0
    for member in 1:EKP.get_N_ens(ekp), case in interface.cases
        dir = case_output_dir(interface, iteration, member, case)
        isdir(dir) || continue
        for (root, _, files) in walkdir(dir), f in files
            endswith(f, ".nc") && (rm(joinpath(root, f); force = true); removed += 1)
        end
    end
    removed > 0 && @info "Pruned $removed NetCDF files from iteration $iteration"
    return nothing
end
