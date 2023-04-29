#=
This script defines `parsed_args` for performance runs, and allows
options to be overridden in several ways. In short the precedence
for defining `parsed_args` is
    - Highest precedence: args defined in `ARGS`
    - Mid     precedence: args defined in `parsed_args_perf_target` (below)
    - Lowest  precedence: args defined in `cli_defaults(s)`
julia --project=perf/ perf/config_parsed_args.jl --target_job sphere_baroclinic_wave_rhoe_equilmoist

Example of interactive run:
```
julia --project=perf/
push!(ARGS, "--job_id", "flame_perf_target_edmf");
push!(ARGS, "--target_job", "sphere_baroclinic_wave_rhoe_equilmoist_edmf");
push!(ARGS, "--moist", "dry");
push!(ARGS, "--precip_model", "nothing");
push!(ARGS, "--rad", "nothing");
include("perf/config_parsed_args.jl")
```
=#
ca_dir = dirname(@__DIR__)
include(joinpath(ca_dir, "src", "utils", "cli_options.jl"));
include(joinpath(ca_dir, "src", "utils", "yaml_helper.jl"))
(s, _parsed_args) = parse_commandline()
parsed_args_defaults = cli_defaults(s);
dict = parsed_args_per_job_id(; filter_name = "driver.jl");

# Start with performance target, and override anything provided in ARGS
parsed_args_prescribed = parsed_args_from_ARGS(ARGS)

target_job = get(parsed_args_prescribed, "target_job", nothing)
parsed_args_perf_target = isnothing(target_job) ? Dict() : dict[target_job]

parsed_args_perf_target["forcing"] = "held_suarez";
parsed_args_perf_target["vert_diff"] = true;
parsed_args_perf_target["surface_scheme"] = "bulk";
parsed_args_perf_target["moist"] = "equil";
parsed_args_perf_target["enable_threading"] = false;
parsed_args_perf_target["rad"] = "allskywithclear";
parsed_args_perf_target["precip_model"] = "0M";
parsed_args_perf_target["dt"] = "1secs";
parsed_args_perf_target["t_end"] = "10secs";
parsed_args_perf_target["dt_save_to_sol"] = Inf;
parsed_args_perf_target["z_elem"] = 25;
parsed_args_perf_target["h_elem"] = 12;

parsed_args =
    merge(parsed_args_defaults, parsed_args_perf_target, parsed_args_prescribed);
