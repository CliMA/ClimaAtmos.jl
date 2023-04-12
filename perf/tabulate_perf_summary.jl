import Plots
import Dates
import JSON
import OrderedCollections
include(joinpath(@__DIR__, "..", "src", "utils", "yaml_helper.jl"))

function sorted_dataset_folder(; dir = pwd())
    matching_paths = String[]
    for file in readdir(dir)
        !ispath(joinpath(dir, file)) && continue
        push!(matching_paths, joinpath(dir, file))
    end
    isempty(matching_paths) && return ""
    # sort by timestamp
    sorted_paths =
        sort(matching_paths; by = f -> Dates.unix2datetime(stat(f).mtime))
    return sorted_paths
end

#=
Performance summaries are structured as follows:

summaries[path|commit][job_id][func][metric]

all metrics can be found in `get_summary`.
=#

ca_dir = joinpath(dirname(@__DIR__))

function get_job_ids(buildkite_yaml; filter_name = "--perf_summary true")
    buildkite_commands = commands_from_yaml(buildkite_yaml; filter_name)
    @assert length(buildkite_commands) > 0 # sanity check
    job_ids = map(buildkite_commands) do bkcs
        strip(first(split(last(split(bkcs, "--job_id ")), " ")), '\"')
    end
    return job_ids
end

function combine_PRs_performance_benchmarks(path)
    job_ids = get_job_ids(
        joinpath(ca_dir, ".buildkite", "pipeline.yml");
        filter_name = "--perf_summary true",
    )
    # Combine summaries into one dict
    summaries = OrderedCollections.OrderedDict()
    for job_id in job_ids
        file = joinpath(path, "perf_benchmark_$job_id.json")
        @debug "collecting file `$file`"
        isfile(file) || continue
        jfile = JSON.parsefile(file; dicttype = OrderedCollections.OrderedDict)
        summaries[job_id] = jfile
    end
    # Save to be moved into main
    open(joinpath(path, "perf_benchmarks.json"), "w") do io
        JSON.print(io, summaries)
    end
    # Clean up individual benchmarks
    for job_id in job_ids
        file = joinpath(ca_dir, "perf_benchmark_$job_id.json")
        rm(file; force = true)
    end
    return summaries
end

function load_old_summaries(paths)
    summaries = OrderedCollections.OrderedDict()
    for path in paths
        file = joinpath(path, "perf_benchmarks.json")
        isfile(file) || continue
        jfile = JSON.parsefile(file; dicttype = OrderedCollections.OrderedDict)
        commit = last(split(path, "climaatmos-main/"))
        summaries[commit] = jfile
    end
    return summaries
end

function performance_history_paths()
    # Note: cluster_data_prefix is also defined elsewhere
    cluster_data_prefix = "/central/scratch/esm/slurm-buildkite/climaatmos-main"
    if ispath(cluster_data_prefix)
        sorted_paths = sorted_dataset_folder(; dir = cluster_data_prefix)
        n = 3
        last_n_paths = sorted_paths[max(end - n + 1, 1):end]
        return last_n_paths
    else
        return String[]
    end
end

perf_hist_paths = performance_history_paths()
@info "Perf history paths: `$perf_hist_paths`"
summaries = load_old_summaries(perf_hist_paths)
summaries["This PR"] = combine_PRs_performance_benchmarks(ca_dir)

# If function names have changed, then this will break, so lets
# gracefully handle this case and throw a warning later.
get_metric(summaries, commit, job_id, func, metric, has_func) =
    has_func[func * commit] ? summaries[commit][job_id][func][metric] : "NA"

function metric_name(metric)
    metric_name_map =
        Dict("t_mean_val" => "Time (ave)", "mem_val" => "Allocations")
    return metric_name_map[metric]
end

#####
##### Tables approach
#####

compute_percent_change(this_PR::Number, main::Number) =
    (this_PR - main) / main * 100

compute_percent_change(this_PR::Number, main::String) =
    (@assert main == "NA"; "NA")

compute_percent_change(this_PR::String, main::Number) =
    (@assert this_PR == "NA"; "NA")

compute_percent_change(this_PR::String, main::String) =
    (@assert this_PR == "NA"; @assert main == "NA"; "NA")

import PrettyTables

function tabulate_summaries(summaries, job_id, metric_tup, funcs, has_func)
    metric = first(metric_tup)
    metric_val = last(metric_tup)
    commits = collect(keys(summaries))

    data_history = map(commits) do commit
        map(funcs) do func
            get_metric(summaries, commit, job_id, func, metric, has_func)
        end
    end

    if length(commits) ≥ 2
        main = map(funcs) do func
            get_metric(
                summaries,
                commits[end - 1],
                job_id,
                func,
                metric_val,
                has_func,
            )
        end
        this_PR = map(funcs) do func
            get_metric(summaries, commits[end], job_id, func, metric_val, has_func)
        end
        percent_change = compute_percent_change.(this_PR, main)
    else
        percent_change = map(funcs) do func
            "Insufficient data for percent change"
        end
    end

    table_data = hcat(funcs, data_history..., percent_change)

    header_names = map(enumerate(commits)) do (i, commit)
        i == length(commit) - 1 ? commit * " (main)" : commit
    end

    header = (
        ["Function", header_names..., "Percent change"],
        ["", ["" for c in commits]..., "(PR-main)/main×100"],
    )

    worsened(data_ij) = !(data_ij isa String) && (data_ij > 0)
    improved(data_ij) = !(data_ij isa String) && (data_ij < 0)

    hl_worsened_pc = PrettyTables.Highlighter(
        (data, i, j) -> worsened(data[i, j]) && j == size(data, 2),
        PrettyTables.crayon"red bold",
    )
    hl_improved_pc = PrettyTables.Highlighter(
        (data, i, j) -> improved(data[i, j]) && j == size(data, 2),
        PrettyTables.crayon"green bold",
    )

    @info "Metric: $(metric_name(last(metric_tup)))"
    PrettyTables.pretty_table(
        table_data;
        header,
        header_crayon = PrettyTables.crayon"yellow bold",
        subheader_crayon = PrettyTables.crayon"blue bold",
        highlighters = (hl_worsened_pc, hl_improved_pc),
        crop = :none,
        alignment = vcat(:l, repeat([:r], size(table_data, 2) - 1)),
    )
end

metric_tups = [
    ("mem", "mem_val"),
    # ("nalloc", "nalloc"),
    ("t_mean", "t_mean_val"),
]

# These functions should match with those in
# the `trials` `Dict` in `perf/benchmark.jl`.
funcs = [
    "Wfact",
    "linsolve",
    "implicit_tendency!",
    "remaining_tendency!",
    "additional_tendency!",
    "hyperdiffusion_tendency!",
    "step!",
]

function compute_has_func(summaries, funcs)
    has_func = OrderedCollections.OrderedDict()
    for job_id in collect(keys(summaries["This PR"]))
        for commit in collect(keys(summaries))
            for func in funcs
                has_func[func * commit] = if haskey(summaries[commit], job_id)
                    haskey(summaries[commit][job_id], func)
                else
                    @warn "Key $job_id not found for commit $commit and func $func."
                    false
                end
            end
        end
    end

    if !all(values(has_func))
        # We can't just throw an error without tracking function
        # names, which seems complicated and brittle. Instead, we
        # warn that function names have changed. This means that
        # we'll have gaps in our performance reports if we don't
        # keep function names up to date, but fixing this should
        # be trivial and smooth since nothing other than fixing
        # the function names is required.
        @info "Missing functions:"
        for key in keys(has_func)
            has_func[key] && continue
            @info "   has_func[$key] = $(has_func[key])"
        end
        @warn string(
            "Perf metrics missing function keys.",
            "It seems that function names have been changed,",
            "and the performance script needs updated.",
        )
    end
    return has_func
end

has_func = compute_has_func(summaries, funcs)

for job_id in collect(keys(summaries["This PR"]))
    @info "##################################### Perf summary for job `$job_id`"
    for metric_tup in metric_tups
        tabulate_summaries(summaries, job_id, metric_tup, funcs, has_func)
    end
end
