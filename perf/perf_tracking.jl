#=
Comparison of the step timings of `perf/benchmark.jl` jobs with those of `main`.

On a `climaatmos-ci` PR or merge-queue build, each job compares its timings with
those of the newest `main` commit in its history, posts the comparison as a Buildkite
annotation, and stages its timings in `PERF_STAGING_ROOT/pr-<n>/<job_id>.json`. When
the PR merges, `perf/publish_perf.jl` moves them to `PERF_STORE/<commit>/`, where
later builds find them.

Timings are the `summaries` of `CTS.benchmark_step`, keyed by function name, with
times in ns and memory in bytes. Requires `reproducibility_tests/reproducibility_utils.jl`
(for `discover_pr_number` and `get_commit_sha`).
=#
import JSON
import OrderedCollections
import BenchmarkTools

const PERF_STORE = "/resnick/scratch/esm/slurm-buildkite/climaatmos-perf"
const PERF_STAGING_ROOT = "/resnick/scratch/esm/slurm-buildkite/climaatmos-perf-staging"

# Slowdown above which a change is shown in bold and the annotation is a warning
const SLOWDOWN_FLAG = 0.05

# Fraction of the `step!` time below which a function is left out of the report
const MIN_STEP_FRACTION = 0.05

# Keys that `CTS.benchmark_step` puts in its `summaries`
const BENCHMARKED_FUNCS = (
    "step!",
    "ldiv!",
    "Wfact!",
    "T_imp!",
    "T_exp!",
    "lim!",
    "dss!",
    "constrain_state!",
    "initialize_imp!",
    "cache!",
    "cache_imp!",
)

perf_file(dir, job_id) = joinpath(dir, "$job_id.json")
read_perf_file(file) = JSON.parsefile(file; dicttype = OrderedCollections.OrderedDict)

"""
    perf_baseline(job_id; store = PERF_STORE)

Return the `main` timings of `job_id` to compare with as
`(; commit, timings, in_history)`: those of the newest commit in the history of `HEAD`
with timings for `job_id`, or, if no such commit is found (e.g. in a shallow clone),
the most recently written ones, with `in_history = false`. Return `nothing` if no
commit has timings for `job_id`.
"""
function perf_baseline(job_id; store = PERF_STORE)
    isdir(store) || return nothing
    published =
        filter(commit -> isfile(perf_file(joinpath(store, commit), job_id)), readdir(store))
    isempty(published) && return nothing
    history = try
        split(readchomp(pipeline(`git rev-list --max-count=2000 HEAD`; stderr = devnull)))
    catch
        String[]
    end
    i = findfirst(sha -> first(sha, 7) in published, history)
    in_history = !isnothing(i)
    commit =
        in_history ? first(history[i], 7) :
        argmax(commit -> mtime(perf_file(joinpath(store, commit), job_id)), published)
    timings = read_perf_file(perf_file(joinpath(store, commit), job_id))
    return (; commit, timings, in_history)
end

timing(timings, func, metric) =
    isnothing(timings) || !haskey(timings, func) ? nothing : timings[func][metric]

format_time(t) = isnothing(t) ? "" : BenchmarkTools.prettytime(t)
format_memory(m) = isnothing(m) ? "" : BenchmarkTools.prettymemory(m)
format_change(ratio) = string(ratio > 0 ? "+" : "", round(100 * ratio; digits = 1), "%")

"""
    time_change(timings, main_timings, func)

Return the relative change in the minimum time of `func` from `main_timings` to
`timings`, or `nothing` if `main_timings` has no time for `func`.
"""
function time_change(timings, main_timings, func)
    t_main = timing(main_timings, func, "time_min")
    (isnothing(t_main) || t_main <= 0) && return nothing
    return timing(timings, func, "time_min") / t_main - 1
end

"""
    perf_report(job_id, timings, baseline)

Return `(; markdown, any_slower)`: a one-line summary of the `step!` time that expands
to a Markdown table of the minimum time and memory of each function in `timings` and in
`baseline`, and whether any function is more than `SLOWDOWN_FLAG` slower than in
`baseline`. Functions that take less than `MIN_STEP_FRACTION` of the `step!` time are
left out.
"""
function perf_report(job_id, timings, baseline)
    main_timings = isnothing(baseline) ? nothing : baseline.timings
    step_time = timing(timings, "step!", "time_min")
    slower = String[]
    table = IOBuffer()
    println(
        table,
        "| Function | Time, `main` | Time, this PR | Change ",
        "| Memory, `main` | Memory, this PR |",
    )
    println(table, "|---|---:|---:|---:|---:|---:|")
    for func in BENCHMARKED_FUNCS
        haskey(timings, func) || continue
        timing(timings, func, "time_min") < MIN_STEP_FRACTION * step_time && continue
        ratio = time_change(timings, main_timings, func)
        change = isnothing(ratio) ? "" : format_change(ratio)
        if !isnothing(ratio) && ratio > SLOWDOWN_FLAG
            change = "**$change**"
            push!(slower, func)
        end
        println(
            table,
            "| `$func` | $(format_time(timing(main_timings, func, "time_min"))) | ",
            "$(format_time(timing(timings, func, "time_min"))) | $change | ",
            "$(format_memory(timing(main_timings, func, "memory"))) | ",
            "$(format_memory(timing(timings, func, "memory"))) |",
        )
    end

    # `<summary>` is HTML, so it uses `<code>` rather than backticks
    summary = string(
        "Timestep cost of <code>$job_id</code>: <code>step!</code> takes ",
        format_time(step_time),
    )
    step_change = time_change(timings, main_timings, "step!")
    if isnothing(baseline)
        summary *= ", no <code>main</code> timings to compare with yet"
    elseif !isnothing(step_change)
        summary *= ", $(format_change(step_change)) compared with <code>main</code>"
    end
    if !isempty(slower)
        summary *= string(
            ". More than $(round(Int, 100 * SLOWDOWN_FLAG))% slower: ",
            join(("<code>$func</code>" for func in slower), ", "),
        )
    end

    io = IOBuffer()
    println(io, "<details>\n<summary>$summary</summary>\n")
    if !isnothing(baseline)
        history_note =
            baseline.in_history ? "" : ", which is not in the history of this build"
        print(io, "Compared with `main` at `$(baseline.commit)`$history_note. ")
    end
    println(
        io,
        "Times are the minimum over the benchmark samples. Functions that take less ",
        "than $(round(Int, 100 * MIN_STEP_FRACTION))% of the `step!` time are left out.\n",
    )
    print(io, String(take!(table)))
    println(io, "\n</details>")
    return (; markdown = String(take!(io)), any_slower = !isempty(slower))
end

"""
    stage_perf(job_id, timings, pr_number; staging_root = PERF_STAGING_ROOT)

Write the timings of `job_id` to `staging_root/pr-<pr_number>/<job_id>.json`,
replacing those of an earlier build of the PR.
"""
function stage_perf(job_id, timings, pr_number; staging_root = PERF_STAGING_ROOT)
    file = perf_file(joinpath(staging_root, "pr-$pr_number"), job_id)
    mkpath(dirname(file))
    open(io -> JSON.print(io, timings), file, "w")
    return file
end

"""
    publish_staged_perf(pr_number, commit;
        store = PERF_STORE, staging_root = PERF_STAGING_ROOT)

Move the timings staged for PR `pr_number` to `store/<commit>/`. Idempotent,
returning `:already_published`, `:nothing_staged`, or `:published`.
"""
function publish_staged_perf(
    pr_number,
    commit;
    store = PERF_STORE,
    staging_root = PERF_STAGING_ROOT,
)
    dest = joinpath(store, commit)
    ispath(dest) && return :already_published
    staged = joinpath(staging_root, "pr-$pr_number")
    isdir(staged) || return :nothing_staged
    mkpath(store)
    mv(staged, dest)
    return :published
end

"""
    annotate_build(markdown, style, context)

Post `markdown` as the annotation `context` of the current Buildkite build.
"""
annotate_build(markdown, style, context) =
    run(`buildkite-agent annotate $markdown --style $style --context $context`)

"""
    track_perf(job_id, summaries)

On a `climaatmos-ci` build of a PR or of the merge queue, stage the `summaries` of
`CTS.benchmark_step` for publishing when the PR merges, compare them with `main`, and
post the comparison as a Buildkite annotation. Errors are logged rather than thrown,
so that they do not fail the benchmark.
"""
function track_perf(job_id, summaries)
    get(ENV, "BUILDKITE_PIPELINE_SLUG", "") == "climaatmos-ci" || return nothing
    get(ENV, "UPSTREAM_PACKAGES", "") == "" || return nothing
    # The same form as the published timings
    timings =
        JSON.parse(JSON.json(summaries); dicttype = OrderedCollections.OrderedDict)
    try
        pr_number = discover_pr_number()
        if isnothing(pr_number)
            @warn "Perf: could not determine the PR number; timings not staged."
        else
            file = stage_perf(job_id, timings, pr_number)
            @info "Perf: staged the timings of PR #$pr_number for publishing on merge." file
        end
    catch err
        @error "Perf: could not stage the timings" exception = (err, catch_backtrace())
    end
    try
        report = perf_report(job_id, timings, perf_baseline(job_id))
        println(report.markdown)
        style = report.any_slower ? "warning" : "info"
        annotate_build(report.markdown, style, "perf-$job_id")
    catch err
        @error "Perf: could not compare the timings with main" exception =
            (err, catch_backtrace())
    end
    return nothing
end
