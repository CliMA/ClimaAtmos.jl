"""
Running SOCRATES cases.

`run_case` runs one; `run_cases` runs a list. Both take an `executor`, which is the **only**
task-scheduling mechanism in this experiment: the calibration layer does not add its own, it builds
a `(member, case)` task list and hands it to [`run_tasks`](@ref) with a
[`WorkerPoolExecutor`](@ref) over its own worker pool. "Run 11 cases on 11 workers" and "run 44
member-case tasks on 30 workers" are therefore the same call with a different task list.
"""

using ClimaAtmos: ClimaAtmos as CA
using Distributed: Distributed

"""
    AbstractExecutor

How a list of independent tasks is executed. Concrete: [`SerialExecutor`](@ref),
[`WorkerPoolExecutor`](@ref).
"""
abstract type AbstractExecutor end

"""
    SerialExecutor()

Run tasks one at a time in the current process. The default, and what a plain REPL session wants.
"""
struct SerialExecutor <: AbstractExecutor end

"""Default seconds to wait on an empty worker pool before giving up."""
const DEFAULT_EMPTY_POOL_TIMEOUT = 7200

"""
    WorkerPoolExecutor(pool; empty_pool_timeout = DEFAULT_EMPTY_POOL_TIMEOUT)

Run tasks across a `Distributed.WorkerPool`, taking the next task as each worker frees up. Any
worker can take any task, so a long task never blocks a short one behind it.

`pool` is supplied by the caller — e.g. `Distributed.WorkerPool(Distributed.workers())`, or
ClimaCalibrate's own pool during a calibration — so this adds no second notion of "the workers".

`empty_pool_timeout` bounds how long a task waits for a worker when the pool is empty. Workers can
join asynchronously, so an empty pool is normal early on; the timer only runs while nothing is in
flight, and expiring it errors rather than hanging forever.
"""
struct WorkerPoolExecutor{P <: Distributed.AbstractWorkerPool} <: AbstractExecutor
    pool::P
    empty_pool_timeout::Int
end

WorkerPoolExecutor(
    pool::Distributed.AbstractWorkerPool;
    empty_pool_timeout::Integer = DEFAULT_EMPTY_POOL_TIMEOUT,
) = WorkerPoolExecutor(pool, Int(empty_pool_timeout))

"""
    run_tasks(f, tasks, executor) -> results

Apply `f` to each element of `tasks`, returning results in `tasks` order. `f` must be callable on
a worker, so it may only close over serializable data.

A task that throws is logged and its result is `nothing`; the remaining tasks still run. Callers
decide what a failure means — the calibration layer turns it into a `NaN` column.
"""
function run_tasks(f, tasks, ::SerialExecutor)
    results = Vector{Any}(nothing, length(tasks))
    for (i, task) in enumerate(tasks)
        try
            results[i] = f(task)
        catch e
            @error "Task failed" task exception = (e, catch_backtrace())
        end
    end
    return results
end

function run_tasks(f, tasks, executor::WorkerPoolExecutor)
    (; pool, empty_pool_timeout) = executor
    results = Vector{Any}(nothing, length(tasks))
    pending = collect(enumerate(tasks))
    # Tasks currently running on checked-out workers. Those workers are absent from the pool but
    # will return, so an empty pool with work in flight is not a stall.
    inflight = Threads.Atomic{Int}(0)
    t_last_available = time()
    @sync while !isempty(pending)
        if isempty(pool.workers)
            # Only treat an empty pool as fatal when nothing is running to replenish it.
            inflight[] > 0 && (t_last_available = time())
            waited = time() - t_last_available
            inflight[] == 0 && waited > empty_pool_timeout && error(
                "No workers available for $(round(Int, waited)) s (timeout \
                 $(empty_pool_timeout) s) with nothing in flight. Ensure workers were started and \
                 can reach this process.",
            )
            sleep(1)
            continue
        end
        t_last_available = time()
        i, task = pop!(pending)
        worker = take!(pool)
        Threads.atomic_add!(inflight, 1)
        @async try
            results[i] = Distributed.remotecall_fetch(f, worker, task)
        catch e
            @error "Task failed on worker $worker" task exception = e
        finally
            Threads.atomic_sub!(inflight, 1)
            put!(pool, worker)
        end
    end
    return results
end

"""
    run_case(case; FT, output_dir, kwargs...) -> output_dir

Build and solve one SOCRATES case, returning the directory its diagnostics were written to.

All keyword arguments of [`socrates_simulation`](@ref) are accepted, plus `FT` (default
`Float64`).

# Examples
```julia
run_case(socrates_case("RF09_Obs"); output_dir = "runs/rf09")
run_case(case; params = "calibrated.toml", output_dir = "runs/calibrated")
run_case(case; grid = socrates_grid(Float64, case; dz_min = 200), output_dir = "runs/coarse")
run_case(case; FT = Float32, t_end = 3600, output_dir = "runs/quick")
```
"""
function run_case(
    case::SocratesCase;
    FT::Type{<:AbstractFloat} = Float64,
    output_dir::AbstractString,
    kwargs...,
)
    simulation = socrates_simulation(FT, case; output_dir, kwargs...)
    # `solve_atmos!` catches a crashing integration and *returns* `:simulation_crashed` rather than
    # throwing, leaving the diagnostics written up to the failure. Without this check a crashed run is
    # indistinguishable from a finished one: it returns an output directory holding a short record,
    # which downstream is scored as though the model simply ran.
    result = CA.solve_atmos!(simulation)
    result.ret_code === :success || error(
        "$(case_name(case)) did not run to completion: ClimaAtmos returned `$(result.ret_code)`. \
         Its output covers only part of $(simulation.t_end) s. The reason is in the simulation's own \
         log, which on a worker is that worker's log file.",
    )
    return simulation.output_dir
end

"""
    run_cases(cases; FT, output_dir, executor, kwargs...) -> Vector

Run several cases, each into `output_dir/<case_name>`, returning each run's output directory (or
`nothing` for a case that failed) in `cases` order.

`executor` defaults to [`SerialExecutor`](@ref); pass a [`WorkerPoolExecutor`](@ref) to spread the
cases over workers.

# Examples
```julia
run_cases(socrates_cases(); output_dir = "runs/all")
run_cases(socrates_cases(); output_dir = "runs/all",
          executor = WorkerPoolExecutor(Distributed.WorkerPool(Distributed.workers())))
```
"""
function run_cases(
    cases::AbstractVector{<:SocratesCase};
    FT::Type{<:AbstractFloat} = Float64,
    output_dir::AbstractString,
    executor::AbstractExecutor = SerialExecutor(),
    kwargs...,
)
    foreach(validate, cases)
    run_one(case) = run_case(case; FT, output_dir = joinpath(output_dir, case_name(case)), kwargs...)
    return run_tasks(run_one, cases, executor)
end
