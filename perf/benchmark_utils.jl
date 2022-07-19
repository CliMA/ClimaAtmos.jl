import StatsBase
import PrettyTables
import BenchmarkTools

#####
##### BenchmarkTools's trial utils
#####

get_summary(trial) = (;
    # Using some BenchmarkTools internals :/
    mem = BenchmarkTools.prettymemory(trial.memory),
    mem_val = trial.memory,
    nalloc = trial.allocs,
    t_min = BenchmarkTools.prettytime(minimum(trial.times)),
    t_max = BenchmarkTools.prettytime(maximum(trial.times)),
    t_mean = BenchmarkTools.prettytime(StatsBase.mean(trial.times)),
    t_mean_val = StatsBase.mean(trial.times),
    t_med = BenchmarkTools.prettytime(StatsBase.median(trial.times)),
    n_samples = length(trial),
)

function tabulate_summary(summary)
    summary_keys = collect(keys(summary))
    mem = map(k -> summary[k].mem, summary_keys)
    nalloc = map(k -> summary[k].nalloc, summary_keys)
    t_mean = map(k -> summary[k].t_mean, summary_keys)
    t_min = map(k -> summary[k].t_min, summary_keys)
    t_max = map(k -> summary[k].t_max, summary_keys)
    t_med = map(k -> summary[k].t_med, summary_keys)
    n_samples = map(k -> summary[k].n_samples, summary_keys)

    table_data = hcat(
        string.(collect(keys(summary))),
        mem,
        nalloc,
        t_min,
        t_max,
        t_mean,
        t_med,
        n_samples,
    )

    header = (
        [
            "Function",
            "Memory",
            "allocs",
            "Time",
            "Time",
            "Time",
            "Time",
            "N-samples",
        ],
        [" ", "estimate", "estimate", "min", "max", "mean", "median", ""],
    )

    PrettyTables.pretty_table(
        table_data;
        header,
        crop = :none,
        alignment = vcat(:l, repeat([:r], length(header[1]) - 1)),
    )
end

function get_trial(f, args, name)
    sample_limit = 10
    f(args...) # compile first
    b = BenchmarkTools.@benchmarkable $f($(args)...)
    println("Benchmarking $name...")
    trial = BenchmarkTools.run(b, samples = sample_limit)
    return trial
end
