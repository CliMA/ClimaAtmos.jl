#=
julia
using Revise; include(joinpath(".dev", "dt_sweep.jl"))
=#

dt_from_config(config) = first(splitext(last(split(config, "dt_"))))
function job(config)
    dt = dt_from_config(config)
    return "
      - label: \"Timestep sweep, dt=$dt\"
        command: >
          julia --color=yes --project=examples examples/hybrid/driver.jl --config_file config/dt_sweep_configs/dt_$config.yml
        artifact_paths: \"dt_$config/*\""
end

dt_file(dt) = joinpath(pkg_dir, "config", "dt_sweep_configs", "dt_$dt.yml")
# Make config files
function write_config_files(dts)
    for dt in dts
        open(dt_file(dt), "w") do io

            println(io, "dt_save_to_disk: \"10days\"")
            println(io, "dt: \"$dt\"")
            println(io, "t_end: \"300days\"")
            println(io, "h_elem: 16")
            println(io, "z_elem: 63")
            println(io, "dz_bottom: 30.0")
            println(io, "dz_top: 3000.0")
            println(io, "z_max: 55000.0")
            println(io, "kappa_4: 2.0e16")
            println(io, "vert_diff: \"true\"")
            println(io, "moist: \"equil\"")
            println(io, "precip_model: \"0M\"")
            println(io, "rayleigh_sponge: true")
            println(io, "forcing: \"held_suarez\"")
            println(io, "job_id: \"dt_sweep_$dt\"")
        end
    end
end
#= Generate a yaml file with given
 - timesteps `dt`,
 - pipeline yaml `pipeline_yml` to copy
 - flag `flag` to copy the pipeline up to (exclusive)
=#
function generate_yml(
    dts;
    flag,
    pipeline_yml = joinpath(@__DIR__, "..", ".buildkite", "pipeline.yml"),
)
    pipeline_lines = readlines(pipeline_yml; keep = true)
    pipeline_config =
        split(first(split(join(pipeline_lines), flag)), "\n")[1:(end - 1)]
    local lines
    mktempdir() do path
        f = joinpath(path, "temp_pipeline.yml")
        open(f, "w") do io
            for l in pipeline_config
                println(io, l)
            end
            println(io, "  - group: \"Timestep sweep\"")
            println(io, "    steps:")
            for dt in dts
                println(io, job(dt))
            end
        end
        lines = readlines(f; keep = true)
    end
    return lines
end

pkg_dir = joinpath(@__DIR__, "..")
mkpath(joinpath(pkg_dir, "config", "dt_sweep_configs"))

# dts = 1:5
# 1  20  40  60  100  140  200  260  320  400
dts = map(i -> Int(max(1, floor(round(i * i * 4; digits = -1)))), 1:3)
@info "Making dt sweep with $dts"
dts = map(dt -> "$(dt)secs", dts)
lines = generate_yml(dts; flag = "- group:")

write_config_files(dts)

open(joinpath(pkg_dir, ".buildkite", "dt_pipeline.yml"), "w") do io
    print(io, join(lines))
end

# Cleanup
for dt in dts
    rm(dt_file(dt))
end
rm(joinpath(pkg_dir, ".buildkite", "dt_pipeline.yml"))
