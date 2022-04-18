import ArgParse
s = ArgParse.ArgParseSettings()
ArgParse.@add_arg_table s begin
    "--FLOAT_TYPE"
    help = "Float type"
    arg_type = String
    default = "Float32"
    "--t_end"
    help = "Simulation end time"
    arg_type = Float64
    "--dt"
    help = "Simulation time step"
    arg_type = Float64
    "--dt_save_to_sol" # TODO: should we default to Inf?
    help = "Time between saving solution, 0 means do not save"
    arg_type = Float64
    default = Float64(60 * 60 * 24)
    "--config"
    help = "Spatial configuration [`SP`: Spherical, `SC`: Single Column, `BO`: Box]"
    arg_type = String
    default = "SP"
    "--HSF"
    help = "Held Suarez Forcing"
    arg_type = Bool
    default = false
    "--topo"
    help = "Use topology"
    arg_type = Bool
    default = false
    "--moist"
    help = "Moisture model [`dry`: dry, `equil`: equilibrium, `non_equil`: non-equilibrium]"
    arg_type = String
    default = "dry"
    "--micro"
    help = "Microphysics scheme [`0M`: 0-moment microphysics, `1M`: 1-moment microphysics]"
    arg_type = String
    default = "dry"
    "--surf_flux"
    help = "Use SurfaceFluxes"
    arg_type = Bool
    default = false
    "--VD"
    help = "Apply vertical diffusion"
    arg_type = Bool
    default = false
    "--rad"
    help = "Radiation model [`nothing`: no radiation, `gray`: gray radiation (RRTMGP), `all_sky`: all-sky radiation (RRTMGP)]"
    arg_type = String
    "--edmf"
    help = "Use Eddy-Diffusivity Mass-Flux (EDMF) model"
    arg_type = Bool
    default = false
    "--DE"
    help = "Solve Dynamical Equations (DE)"
    arg_type = Bool
    default = true
    "--regression_test"
    help = "(Bool) perform regression test"
    arg_type = Bool
    default = false
    "--enable_threading"
    help = "Enable multi-threading. Note: Julia must be launched with (e.g.,) `--threads=8`"
    arg_type = Bool
    default = false
    "--TEST_NAME"
    help = "Job name"
    arg_type = String
    "--output_dir"
    help = "Output directory"
    arg_type = String
    "--job_id"
    help = "Uniquely identifying string for a particular job"
    arg_type = String
end

parsed_args = ArgParse.parse_args(ARGS, s)

function cli_defaults(s::ArgParse.ArgParseSettings)
    defaults = Dict()
    # TODO: Don't use ArgParse internals
    for arg in s.args_table.fields
        defaults[arg.dest_name] = arg.default
    end
    return defaults
end

#= Use the job ID for the output folder =#
function output_directory(
    s::ArgParse.ArgParseSettings,
    parsed_args = ArgParse.parse_args(ARGS, s),
)
    return job_id_from_parsed_args(s, parsed_args)
end

"""
    job_id_from_parsed_args(
        s::ArgParseSettings,
        parsed_args = ArgParse.parse_args(ARGS, s)
    )

Returns a unique name (`String`) given
 - `s::ArgParse.ArgParseSettings` The arg parse settings
 - `parsed_args` The parse arguments

The `ArgParseSettings` are used for truncating
this string based on the default values.
"""
job_id_from_parsed_args(s, parsed_args = ArgParse.parse_args(ARGS, s)) =
    job_id_from_parsed_args(cli_defaults(s), parsed_args)

function job_id_from_parsed_args(defaults::Dict, parsed_args)
    _parsed_args = deepcopy(parsed_args)
    s = ""
    warn = false
    for k in keys(_parsed_args)
        # Skip defaults to alleviate verbose names
        defaults[k] == _parsed_args[k] && continue

        if _parsed_args[k] isa String
            # We don't need keys if the value is a string
            # (alleviate verbose names)
            s *= _parsed_args[k]
        elseif _parsed_args[k] isa Int
            s *= k * "_" * string(_parsed_args[k])
        elseif _parsed_args[k] isa Real
            warn = true
        else
            s *= k * "_" * string(_parsed_args[k])
        end
        s *= "_"
    end
    s = replace(s, "/" => "_")
    s = strip(s, '_')
    warn && @warn "Truncated job ID:$s may not be unique due to use of Real"
    return s
end
