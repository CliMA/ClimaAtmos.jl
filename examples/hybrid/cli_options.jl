import ArgParse
function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--FLOAT_TYPE"
        help = "Float type"
        arg_type = String
        default = "Float32"
        "--t_end"
        help = "Simulation end time. Examples: [`1200days`, `40secs`]"
        arg_type = String
        default = "10days"
        "--dt"
        help = "Simulation time step. Examples: [`10secs`, `1hours`]"
        arg_type = String
        default = "600secs"
        "--dt_save_to_sol"
        help = "Time between saving solution. Examples: [`10days`, `1hours`, `Inf` (do not save)]"
        arg_type = String
        default = "1days"
        "--dt_save_to_disk"
        help = "Time between saving to disk. Examples: [`10secs`, `1hours`, `Inf` (do not save)]"
        arg_type = String
        default = "Inf"
        "--dt_save_restart"
        help = "Time between saving restart files to disk. Examples: [`10secs`, `1hours`, `Inf` (do not save)]"
        arg_type = String
        default = "Inf"
        "--dt_rad"
        help = "Time between calling radiation callback for sphere configurations"
        arg_type = String
        default = "6hours"
        "--config"
        help = "Spatial configuration [`sphere` (default), `column`, `box`]"
        arg_type = String
        default = "sphere"
        "--moist"
        help = "Moisture model [`dry` (default), `equil`, `non_equil`]"
        arg_type = String
        default = "dry"
        "--precip_model"
        help = "Precipitation model [`nothing` (default), `0M`]"
        arg_type = String
        "--forcing"
        help = "Forcing [`nothing` (default), `held_suarez`]"
        arg_type = String
        "--subsidence"
        help = "Subsidence [`nothing` (default), `Bomex`, `LifeCycleTan2018`, `Rico`, `DYCOMS`]"
        arg_type = String
        "--ls_adv"
        help = "Large-scale advection [`nothing` (default), `Bomex`, `LifeCycleTan2018`, `Rico`, `ARM_SGP`, `GATE_III`]"
        arg_type = String
        "--edmf_coriolis"
        help = "EDMF coriolis [`nothing` (default), `Bomex`,`LifeCycleTan2018`,`Rico`,`ARM_SGP`,`DYCOMS_RF01`,`DYCOMS_RF02`,`GABLS`]"
        arg_type = String
        "--vert_diff"
        help = "Vertical diffusion [`false` (default), `VerticalDiffusion`, `true` (defaults to `VerticalDiffusion`)]"
        arg_type = String
        default = "false"
        "--surface_scheme"
        help = "Surface flux scheme [`nothing` (default), `bulk`, `monin_obukhov`]"
        arg_type = String
        "--surface_thermo_state_type"
        help = "Surface thermo state type [`GCMSurfaceThermoState` (default), `PrescribedThermoState`]"
        arg_type = String
        default = "GCMSurfaceThermoState"
        "--C_E"
        help = "Bulk transfer coefficient"
        arg_type = Float64
        default = Float64(0.0044)
        "--coupled"
        help = "Coupled simulation [`false` (default), `true`]"
        arg_type = Bool
        default = false
        "--turbconv"
        help = "Turbulence convection scheme [`nothing` (default), `edmf`]"
        arg_type = String
        "--turbconv_case"
        help = "The case run by Turbulence convection scheme [`Bomex` (default), `Bomex`, `DYCOMS_RF01`, `TRMM_LBA`, `GABLS`]"
        arg_type = String
        "--hyperdiff"
        help = "Hyperdiffusion [`ClimaHyperdiffusion` (default), `TempestHyperdiffusion`, `none` (or `false`)]"
        arg_type = String
        default = "ClimaHyperdiffusion"
        "--enable_qt_hyperdiffusion"
        help = "Enable the hyperdiffusion of specific humidity [`true` (default), `false`] (TODO: reconcile this with `ρe_tot` or remove if instability fixed with limiters)"
        arg_type = Bool
        default = true
        "--idealized_insolation"
        help = "Use idealized insolation in radiation model [`false`, `true` (default)]"
        arg_type = Bool
        default = true
        "--idealized_h2o"
        help = "Use idealized H2O in radiation model [`false` (default), `true`]"
        arg_type = Bool
        default = false
        "--idealized_clouds"
        help = "Use idealized clouds in radiation model [`false` (default), `true`]"
        arg_type = Bool
        default = false
        "--rad"
        help = "Radiation model [`nothing` (default), `gray`, `clearsky`, `allsky`, `allskywithclear`]"
        arg_type = String
        "--energy_name"
        help = "Energy variable name [`rhoe` (default), `rhotheta`]"
        arg_type = String
        default = "rhoe"
        "--perturb_initstate"
        help = "Add a perturbation to the initial condition [`false`, `true` (default)]"
        arg_type = Bool
        default = true
        "--discrete_hydrostatic_balance"
        help = "Set the initial state to discrete hydrostatic balance"
        arg_type = Bool
        default = false
        "--energy_upwinding"
        help = "Energy upwinding mode [`none` (default), `first_order` , `third_order`, `boris_book`, `zalesak`]"
        arg_type = Symbol
        default = :none
        "--tracer_upwinding"
        help = "Tracer upwinding mode [`none` (default), `first_order` , `third_order`, `boris_book`, `zalesak`]"
        arg_type = Symbol
        default = :none  # TODO: change to :zalesak
        "--ode_algo"
        help = "ODE algorithm [`ARS343` (default), `SSP333`, `IMKG343a`, `ODE.Euler`, `ODE.IMEXEuler`, `ODE.Rosenbrock23`, etc.]"
        arg_type = String
        default = "ARS343"
        "--max_newton_iters"
        help = "Maximum number of Newton's method iterations (only for ODE algorithms that use Newton's method)"
        arg_type = Int
        default = 1
        "--use_newton_rtol"
        help = "Whether to check if the current iteration of Newton's method has an error within a relative tolerance, instead of always taking the maximum number of iterations (only for ClimaTimeSteppers.jl)"
        arg_type = Bool
        default = false
        "--newton_rtol"
        help = "Relative tolerance of Newton's method (only for ClimaTimeSteppers.jl; only used when `use_newton_rtol` is `true`)"
        arg_type = Float64
        default = Float64(1e-5)
        "--use_krylov_method"
        help = "Whether to use a Krylov method to solve the linear system in Newton's method (only for ClimaTimeSteppers.jl)"
        arg_type = Bool
        default = false
        "--krylov_rtol"
        help = "Relative tolerance of the Krylov method (only for ClimaTimeSteppers.jl; only used if `use_krylov_method` is `true`)"
        arg_type = Float64
        default = Float64(0.1)
        "--use_dynamic_krylov_rtol"
        help = "Whether to use Eisenstat-Walker forcing instead of a constant relative tolerance in the Krylov method (only for ClimaTimeSteppers.jl)"
        arg_type = Bool
        default = false
        "--eisenstat_walker_forcing_alpha"
        help = "Value of alpha to use for Eisenstat-Walker forcing (only for ClimaTimeSteppers.jl; only used if `use_krylov_method` and `use_dynamic_krylov_rtol` are `true`)"
        arg_type = Float64
        default = Float64(2)
        "--jvp_step_adjustment"
        help = "Amount by which the step size of the forward difference approximation of the Jacobian-vector product in the Krylov method should be scaled (only used if `use_krylov_method` is `true`)"
        arg_type = Float64
        default = Float64(1)
        "--split_ode"
        help = "Use split of ODE problem. Examples: [`true` (implicit, default), `false` (explicit)]"
        arg_type = Bool
        default = true
        "--regression_test"
        help = "(Bool) perform regression test"
        arg_type = Bool
        default = false
        "--enable_threading"
        help = "Enable multi-threading. Note: Julia must be launched with (e.g.,) `--threads=8`"
        arg_type = Bool
        default = true
        "--output_dir"
        help = "Output directory"
        arg_type = String
        "--job_id"
        help = "Uniquely identifying string for a particular job"
        arg_type = String
        "--quicklook_reference_job_id"
        help = "Identifier of job to use as the \"reference\" solution in the quicklook plot; the current job's results get compared to the results of the quicklook job on the main branch (only used if `debugging_tc` is `true`)"
        arg_type = String
        "--trunc_stack_traces"
        help = "Set to `true` to truncate printing of ClimaCore `Field`s"
        arg_type = Bool
        default = true
        "--fps"
        help = "Frames per second for animations"
        arg_type = Int
        default = 5
        "--post_process"
        help = "Post process [`true` (default), `false`]"
        arg_type = Bool
        default = true
        "--h_elem"
        help = "number of elements per edge on a cubed sphere"
        arg_type = Int
        default = 6
        "--x_elem"
        help = "number of horizontal elements in the x-direction"
        arg_type = Int
        default = 6
        "--y_elem"
        help = "number of horizontal elements in the y-direction"
        arg_type = Int
        default = 6
        "--z_elem"
        help = "number of vertical elements"
        arg_type = Int
        default = 10
        "--nh_poly"
        help = "Horizontal polynomial degree. Note: The number of quadrature points in 1D within each horizontal element is then Nq = <--nh_poly> + 1"
        arg_type = Int
        default = 3
        "--bubble"
        help = "Enable bubble correction for more accurate surface areas"
        arg_type = Bool
        default = false
        "--x_max"
        help = "Model domain size, x direction. Default: 300km"
        arg_type = Float64
        default = Float64(300e3)
        "--y_max"
        help = "Model domain size, y direction. Default: 300km"
        arg_type = Float64
        default = Float64(300e3)
        "--z_max"
        help = "Model top height. Default: 30km"
        arg_type = Float64
        default = Float64(30e3)
        "--z_stretch"
        help = "Stretch grid in z-direction. [`true` (default), `false`]"
        arg_type = Bool
        default = true
        "--dz_bottom"
        help = "Model bottom grid depth. Default: 500m"
        arg_type = Float64
        default = Float64(500)
        "--dz_top"
        help = "Model top grid depth. Default: 5000m"
        arg_type = Float64
        default = Float64(5000)
        "--kappa_4"
        help = "Hyperdiffusion parameter"
        arg_type = Float64
        default = Float64(2e17)
        "--rayleigh_sponge"
        help = "Rayleigh sponge [`true`, `false` (default)]"
        arg_type = Bool
        default = false
        "--viscous_sponge"
        help = "Viscous sponge [`true`, `false` (default)]"
        arg_type = Bool
        default = false
        "--zd_rayleigh"
        help = "Rayleigh sponge height"
        arg_type = Float64
        default = Float64(15e3)
        "--alpha_rayleigh_uh"
        help = "Rayleigh sponge coefficient for horizontal velocity"
        arg_type = Float64
        default = Float64(1e-4)
        "--alpha_rayleigh_w"
        help = "Rayleigh sponge coefficient for vertical velocity"
        arg_type = Float64
        default = Float64(1)
        "--zd_viscous"
        help = "Viscous sponge height"
        arg_type = Float64
        default = Float64(15e3)
        "--kappa_2_sponge"
        help = "Viscous sponge coefficient"
        arg_type = Float64
        default = Float64(1e6)
        "--start_date"
        help = "Start date of the simulation"
        arg_type = String
        default = "19790101"
        "--topography"
        help = "Define the surface elevation profile [`NoWarp`,`Earth`,`DCMIP200`]"
        arg_type = String
        default = "NoWarp"
        "--apply_limiter"
        help = "Apply a horizontal limiter to every tracer [`true` (default), `false`]"
        arg_type = Bool
        default = true
        "--imex_edmf_turbconv"
        help = "Whether to split EDMF's `compute_turbconv_tendencies!` into implicit and explicit components"
        arg_type = Bool
        default = false
        "--imex_edmf_gm"
        help = "Whether to split EDMF's `compute_gm_tendencies!` into implicit and explicit components"
        arg_type = Bool
        default = false
        "--edmf_entr_closure"
        help = "EDMF entrainment closure. [`MoistureDeficit` (default), `Constant`, `PiDetrainment`]"
        arg_type = String
        default = "MoistureDeficit"
        "--debugging_tc"
        help = "Save most of the tc aux state to HDF5 file [`false` (default), `true`]"
        arg_type = Bool
        default = false
        "--test_edmf_consistency"
        help = "Test edmf equation consistency [`false` (default), `true`]"
        arg_type = Bool
        default = false
        "--test_dycore_consistency"
        help = "Test dycore consistency [`false` (default), `true`]"
        arg_type = Bool
        default = false
        "--use_reference_state"
        help = "Subtract a reference state from the dycore equations [`false`, `true` (default)]"
        arg_type = Bool
        default = true
        "--check_conservation"
        help = "Check conservation of mass and energy [`false` (default), `true`]"
        arg_type = Bool
        default = false
        "--non_orographic_gravity_wave"
        help = "Apply parameterization for convective gravity wave forcing on horizontal mean flow [`false` (default), `true`]"
        arg_type = Bool
        default = false
        "--orographic_gravity_wave"
        help = "Apply parameterization for orographic drag on horizontal mean flow"
        arg_type = Bool
        default = false
        "--perf_summary"
        help = "Flag for collecting performance summary information"
        arg_type = Bool
        default = false
        "--perf_mode"
        help = "A flag for analyzing performance [`PerfStandard` (default), `PerfExperimental`]"
        arg_type = String
        default = "PerfStandard"
        "--target_job"
        help = "An (optional) job to target for analyzing performance"
        arg_type = String
        "--toml"
        help = "A toml file used to override model parameters and configurations. In the case of conflicts, CLI arguments take priority over the toml"
        arg_type = String
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    return (s, parsed_args)
end

function cli_defaults(s::ArgParse.ArgParseSettings)
    defaults = Dict()
    # TODO: Don't use ArgParse internals
    for arg in s.args_table.fields
        defaults[arg.dest_name] = arg.default
    end
    return defaults
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
    # Use only keys from the default ArgParseSettings
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
        elseif _parsed_args[k] isa AbstractFloat
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


"""
    print_repl_script(str::String)

Generate a block of code to run a particular
buildkite job given the `command:` string.

Example:

"""
function print_repl_script(str)
    ib = """"""
    ib *= """\n"""
    ib *= """using Revise; include("examples/hybrid/cli_options.jl");\n"""
    ib *= """\n"""
    ib *= """(s, parsed_args) = parse_commandline();\n"""
    parsed_args = parsed_args_from_command_line_flags(str)
    for (flag, val) in parsed_args
        if val isa AbstractString
            ib *= "parsed_args[\"$flag\"] = \"$val\";\n"
        else
            ib *= "parsed_args[\"$flag\"] = $val;\n"
        end
    end
    ib *= """\n"""
    ib *= """include("examples/hybrid/driver.jl")\n"""
    println(ib)
end

parsed_args_from_ARGS(ARGS, parsed_args = Dict()) =
    parsed_args_from_ARGS_string(strip(join(ARGS, " ")), parsed_args)

parsed_args_from_command_line_flags(str, parsed_args = Dict()) =
    parsed_args_from_ARGS_string(strip(last(split(str, ".jl"))), parsed_args)

function parsed_args_from_ARGS_string(str, parsed_args = Dict())
    str = replace(str, "    " => " ", "   " => " ", "  " => " ")
    parsed_args_list = split(str, " ")
    parsed_args_list == [""] && return parsed_args
    @assert iseven(length(parsed_args_list))
    parsed_arg_pairs = map(1:2:(length(parsed_args_list) - 1)) do i
        Pair(parsed_args_list[i], strip(parsed_args_list[i + 1], '\"'))
    end
    function parse_arg(val)
        for T in (Bool, Int, Float32, Float64)
            try
                return parse(T, val)
            catch
            end
        end
        return String(val) # string
    end
    for (flag, val) in parsed_arg_pairs
        parsed_args[replace(flag, "--" => "")] = parse_arg(val)
    end
    return parsed_args
end

"""
    parsed_args_per_job_id()
    parsed_args_per_job_id(buildkite_yaml)

A dict of `parsed_args` to run the ClimaAtmos driver
whose keys are the `job_id`s from buildkite yaml.

# Example

To run the `sphere_aquaplanet_rhoe_equilmoist_allsky`
buildkite job from the standard buildkite pipeline, use:
```
using Revise; include("examples/hybrid/cli_options.jl");
dict = parsed_args_per_job_id();
parsed_args = dict["sphere_aquaplanet_rhoe_equilmoist_allsky"];
include("examples/hybrid/driver.jl")
```
"""
function parsed_args_per_job_id(; trigger = "driver.jl")
    ca_dir = joinpath(@__DIR__, "..", "..")
    buildkite_yaml = joinpath(ca_dir, ".buildkite", "pipeline.yml")
    parsed_args_per_job_id(buildkite_yaml; trigger)
end

function parsed_args_per_job_id(buildkite_yaml; trigger = "driver.jl")
    buildkite_commands = readlines(buildkite_yaml)
    filter!(x -> occursin(trigger, x), buildkite_commands)

    @assert length(buildkite_commands) > 0 # sanity check
    result = Dict()
    for bkcs in buildkite_commands
        (s, default_parsed_args) = parse_commandline()
        job_id = first(split(last(split(bkcs, "--job_id ")), " "))
        job_id = strip(job_id, '\"')
        result[job_id] =
            parsed_args_from_command_line_flags(bkcs, default_parsed_args)
    end
    return result
end

function non_default_command_line_flags_parsed_args(parsed_args)
    (s, default_parsed_args) = parse_commandline()
    s = ""
    for k in keys(parsed_args)
        default_parsed_args[k] == parsed_args[k] && continue
        s *= "--$k $(parsed_args[k]) "
    end
    return rstrip(s)
end
