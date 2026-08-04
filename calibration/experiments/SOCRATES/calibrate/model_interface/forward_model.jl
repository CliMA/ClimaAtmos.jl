"""
Helpers shared with perfect_scm for AtmosConfig toml / diagnostics mutation.
"""

function add_parameter_filepath!(config_dict, new_parameter_file)
    tomls = get!(config_dict, "toml", String[])
    # YAML may load as relative paths; normalize to strings
    tomls = String[string(t) for t in tomls]
    config_dict["toml"] = tomls
    push!(tomls, string(new_parameter_file))
    return nothing
end

function replace_diagnostic_dicts!(config_dict, diagnostic_dicts::Vector)
    config_dict["diagnostics"] = deepcopy(diagnostic_dicts)
    return nothing
end

"""Resolve toml paths relative to ClimaAtmos package root when needed."""
function resolve_toml_paths!(config_dict)
    ca_root = pkgdir(ClimaAtmos)
    tomls = get(config_dict, "toml", String[])
    resolved = String[]
    for t in tomls
        ts = string(t)
        if isfile(ts)
            push!(resolved, abspath(ts))
        elseif isfile(joinpath(ca_root, ts))
            push!(resolved, joinpath(ca_root, ts))
        else
            push!(resolved, ts) # leave as-is; Atmos will error if missing
        end
    end
    config_dict["toml"] = resolved
    return nothing
end

"""
Build YAML config dict for one case (physics/grid/output). SocratesSetup is
built separately and injected via `get_simulation_with_setup`.
"""
function build_case_config_dict(
    interface::SOCRATESAtmosModelInterface,
    case::SOCRATESCase,
    member_case_dir::AbstractString;
    forcing_nt::NamedTuple = generate_socrates_forcing(case),
)
    config_dict = ClimaAtmos.load_yaml_file(interface.config)
    config_dict["output_dir"] = member_case_dir
    config_dict["output_default_diagnostics"] = false
    config_dict["dt_save_state_to_disk"] = "Inf"
    replace_diagnostic_dicts!(config_dict, interface.diagnostic_dicts)

    config_dict["z_max"] = z_max_meters(case)
    config_dict["rayleigh_sponge"] = false
    config_dict["external_forcing"] = nothing
    config_dict["start_date"] = start_date_string(case)
    config_dict["t_end"] = t_end_string(case)
    config_dict["site_latitude"] = Float64(forcing_nt.lat)
    config_dict["site_longitude"] = Float64(forcing_nt.lon)

    resolve_toml_paths!(config_dict)
    n_ccn_toml = joinpath(member_case_dir, "n_ccn.toml")
    mkpath(member_case_dir)
    write_n_ccn_toml(case, n_ccn_toml)
    add_parameter_filepath!(config_dict, n_ccn_toml)
    return config_dict
end

function build_socrates_setup(case::SOCRATESCase; dt_sec::Float64 = 300.0)
    return SocratesSetup(generate_socrates_forcing(case; dt_sec))
end

"""
    ClimaCalibrate.forward_model(interface::SOCRATESAtmosModelInterface, iter, member)

Run all SOCRATES cases for this ensemble member (no minibatch subsetting).
"""
function ClimaCalibrate.forward_model(interface::SOCRATESAtmosModelInterface, iter, member)
    Random.seed!(1234 + member)
    (; output_dir, cases) = interface

    member_dir = ClimaCalibrate.path_to_ensemble_member(output_dir, iter, member)
    mkpath(member_dir)
    sampled_parameter_file = ClimaCalibrate.parameter_path(output_dir, iter, member)

    for case in cases
        case_dir = joinpath(member_dir, string(case.name))
        mkpath(case_dir)
        forcing_nt = generate_socrates_forcing(case)
        setup = SocratesSetup(forcing_nt)
        config_dict = build_case_config_dict(interface, case, case_dir; forcing_nt)
        add_parameter_filepath!(config_dict, sampled_parameter_file)

        @info "Forward model" iter member case = case.name
        atmos_config = ClimaAtmos.AtmosConfig(config_dict)
        simulation = get_simulation_with_setup(atmos_config, setup)
        ClimaAtmos.solve_atmos!(simulation)
    end
    return nothing
end

"""Run a single named case outside EKI (smoke / debug)."""
function run_single_case!(
    interface::SOCRATESAtmosModelInterface,
    case::SOCRATESCase;
    name::AbstractString = string(name(case)),
    output_dir = nothing,
    t_end = nothing,
)
    out = something(output_dir, joinpath(interface.output_dir, "smoke", name))
    mkpath(out)
    empty_toml = joinpath(out, "empty_params.toml")
    isfile(empty_toml) || write(empty_toml, "")
    forcing_nt = generate_socrates_forcing(case)
    setup = SocratesSetup(forcing_nt)
    config_dict = build_case_config_dict(interface, case, out; forcing_nt)
    if !isnothing(t_end)
        config_dict["t_end"] = t_end
    end
    add_parameter_filepath!(config_dict, empty_toml)
    atmos_config = ClimaAtmos.AtmosConfig(config_dict)
    simulation = get_simulation_with_setup(atmos_config, setup)
    ClimaAtmos.solve_atmos!(simulation)
    return simulation.output_dir
end
