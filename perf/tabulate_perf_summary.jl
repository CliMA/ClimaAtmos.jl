import Plots
import Dates
import JSON
import OrderedCollections

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

function get_job_ids(buildkite_yaml; trigger = "benchmark.jl")
    buildkite_commands = readlines(buildkite_yaml)
    filter!(x -> occursin(trigger, x), buildkite_commands)
    # TODO: can we filter this better?
    filter!(x -> occursin("perf_target", x), buildkite_commands)
    @assert length(buildkite_commands) > 0 # sanity check
    job_ids = map(buildkite_commands) do bkcs
        strip(first(split(last(split(bkcs, "--job_id ")), " ")), '\"')
    end
    return job_ids
end

function combine_PRs_performance_benchmarks(path)
    job_ids = get_job_ids(
        joinpath(ca_dir, ".buildkite", "pipeline.yml");
        trigger = "benchmark.jl",
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
        summaries[path] = jfile
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

function get_metric(summaries, commit, job_id, func, metric, has_func)
    # If function names have changed, then this will break, so lets
    # gracefully handle this case and throw a warning later.
    return has_func[func * commit] ? summaries[commit][job_id][func][metric] : 0
end

function metric_name(metric)
    metric_name_map =
        Dict("t_mean_val" => "Time (ave)", "mem_val" => "Allocations")
    return metric_name_map[metric]
end

#####
##### Tables approach
#####

import PrettyTables

function tabulate_summaries(summaries, job_id, metric_tup)
    # These functions should match with those in
    # the `trials` `Dict` in `perf/benchmark.jl`.
    funcs = [
        "Wfact",
        "linsolve",
        "implicit_tendency!",
        "remaining_tendency!",
        "default_remaining_tendency!",
        "hyperdiffusion_tendency!",
        "step!",
    ]
    metric = first(metric_tup)
    metric_val = last(metric_tup)
    commits = collect(keys(summaries))
    has_func = OrderedCollections.OrderedDict()

    for commit in commits
        for func in funcs
            has_func[func * commit] = haskey(summaries[commit][job_id], func)
        end
    end

    if !all(values(has_func))
        # We can't just throw an error without tracking function
        # names, which seems complicated and brittle. Instead, we
        # warn that function names have changed. This means that
        # we'll have gaps in our perfromance reports if we don't
        # keep function names up to date, but fixing this should
        # be trivial and smooth since nothing other than fixing
        # the function names is required.
        @show has_func
        @warn string(
            "Perf metrics missing function keys.",
            "It seems that function names have been changed,",
            "and the performance script needs updated.",
        )
    end

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
        percent_change = (this_PR .- main) ./ main .* 100
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

for job_id in collect(keys(summaries["This PR"]))
    @info "##################################### Perf summary for job `$job_id`"
    for metric_tup in metric_tups
        tabulate_summaries(summaries, job_id, metric_tup)
    end
end
