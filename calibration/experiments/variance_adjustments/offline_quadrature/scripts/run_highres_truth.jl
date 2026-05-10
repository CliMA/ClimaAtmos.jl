# High-resolution truth run utilities for offline quadrature workflow.

using Dates
using NCDatasets
using JLD2: @save
import ClimaAnalysis: SimDir, get, times, slice, altitudes, average_xy
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD
import ClimaCore: Spaces, Fields

if !isdefined(@__MODULE__, :va_load_experiment_config)
    include(joinpath(@__DIR__, "..", "..", "lib", "experiment_common.jl"))
end

function _read_last_profile(var)
    nd = ndims(var)
    if nd == 1
        return var[:]
    elseif nd == 2
        return var[:, end]
    else
        error("Variable $(name(var)) has unsupported rank $nd (expected 1D or 2D z/time profile).")
    end
end

"""Write a single 1D profile variable on the `z` dimension.

This helper keeps the NetCDF write path explicit: the caller must provide the exact
field that should be written, and the function will fail immediately if the target
dimension or array shape is inconsistent.
"""
function _write_profile_var!(ds, name::AbstractString, data)
    var = defVar(ds, name, eltype(data), ("z",))
    var[:] = data
    return var
end

function compute_highres_truth_profile_from_case_layers(
    case_yaml_layers::Vector{String},
    scm_toml::AbstractString;
    case_name::String,
    quadrature_order::Int = 3,
    truth_sgs_distribution::Union{Nothing, String} = nothing,
    experiment_dir::String = normpath(joinpath(@__DIR__, "..", "..")),
    output_root::Union{Nothing, String} = nothing,
)
    case_dict = va_load_merged_case_yaml_dict(experiment_dir, case_yaml_layers)
    expc = Dict{String, Any}(
        "case_name" => case_name,
        "quadrature_order" => quadrature_order,
        "model_config_path" => case_yaml_layers,
        "scm_toml" => Any[scm_toml],
    )

    # Default output location for these helper-driven high-res truth runs: keep them
    # self-contained under the offline_quadrature/outputs folder to avoid clobbering
    # top-level experiment `output` trees.
    default_out = joinpath(experiment_dir, "offline_quadrature", "outputs")
    out_root = output_root === nothing ? abspath(default_out) : abspath(output_root)
    mkpath(out_root)
    merged_path = joinpath(out_root, VA_MERGED_SCM_BASELINE_BASENAME)
    va_write_merged_scm_baseline_file!(experiment_dir, Any[scm_toml], merged_path)

    cfg = copy(case_dict)
    cfg["quadrature_order"] = quadrature_order
    truth_sgs_distribution !== nothing && (cfg["sgs_distribution"] = truth_sgs_distribution)
    # Enforce writing of diagnostics required by the offline quadrature workflow.
    # This ensures the forward simulation explicitly outputs the canonical fields we
    # later read. Override any model defaults here so the run produces the exact variables we need.
    cfg["output_default_diagnostics"] = false
    cfg["diagnostics"] = Any[
        Dict(
            "short_name" => ["ta", "thetaa", "hus", "pfull", "rhoa", "clw", "cli",
                            "env_q_tot_variance", "env_temperature_variance",
                            "env_q_tot_temperature_correlation"],
            "period" => "1mins",
            "reduction" => "inst"
        )
    ]
    cfg["toml"] = [merged_path]
    cfg["output_dir"] = out_root

    atmos_config = CA.AtmosConfig(
        cfg;
        comms_ctx = va_comms_ctx(),
        job_id = "va_offline_truth_" * replace(case_name, r"[^A-Za-z0-9_.-]+" => "_") * "_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"),
    )
    sim = CA.get_simulation(atmos_config)
    thp = CAP.thermodynamics_params(sim.integrator.p.params)
    sol_res = CA.solve_atmos!(sim)
    if sol_res.ret_code == :simulation_crashed
        error("High-res truth run crashed. Check output under " * string(cfg["output_dir"]) * ".")
    end

    # Extract face z coordinates directly from the simulation's ClimaCore grid.
    # This is the authoritative source — no reconstruction needed.
    _z_face_raw = parent(Fields.coordinate_field(sim.integrator.u.f).z)
    # Guarded extraction: some grid implementations expose `AbstractPoint` objects
    # while others expose plain numeric coordinates. Extract the numeric
    # coordinate for point types, otherwise assume the value is already numeric.
    z_faces_prof = sort(unique([isa(v, ClimaCore.Geometry.AbstractPoint) ? ClimaCore.Geometry.component(v, 1) : v for v in vec(_z_face_raw)]))

    outdir = hasproperty(sim, :output_dir) ? getproperty(sim, :output_dir) : nothing
    if outdir === nothing || !isdir(outdir)
        error("Simulation did not produce an output directory: " * string(outdir))
    end
    active = isdir(joinpath(outdir, "output_active")) ? joinpath(outdir, "output_active") : outdir

    simdir = SimDir(active)

    # Read the canonical fields directly using ClimaAnalysis, and average horizontally
    # so we get a pure 1D vertical profile mapped over time.
    ta_var = average_xy(get(simdir, short_name="ta", reduction="inst", period="1m"))
    hus_var = average_xy(get(simdir, short_name="hus", reduction="inst", period="1m"))
    pfull_var = average_xy(get(simdir, short_name="pfull", reduction="inst", period="1m"))
    rhoa_var = average_xy(get(simdir, short_name="rhoa", reduction="inst", period="1m"))
    clw_var = average_xy(get(simdir, short_name="clw", reduction="inst", period="1m"))
    cli_var = average_xy(get(simdir, short_name="cli", reduction="inst", period="1m"))
    thetaa_var = average_xy(get(simdir, short_name="thetaa", reduction="inst", period="1m"))
    q_var = average_xy(get(simdir, short_name="env_q_tot_variance", reduction="inst", period="1m"))
    t_var = average_xy(get(simdir, short_name="env_temperature_variance", reduction="inst", period="1m"))
    corr_tq = average_xy(get(simdir, short_name="env_q_tot_temperature_correlation", reduction="inst", period="1m"))

    z_coord = altitudes(ta_var)
    t_coord = times(ta_var)
    if isempty(t_coord)
        error("No time slices found for canonical fields.")
    end

    last_t = t_coord[end]
    
    # We slice at the last time and extract the 1D profile data. Since the problem is 1D column, data is just 1D.
    z_prof = z_coord
    T_prof = vec(slice(ta_var, time=last_t).data)
    qv_prof = vec(slice(hus_var, time=last_t).data)
    p_prof = vec(slice(pfull_var, time=last_t).data)
    rho_prof = vec(slice(rhoa_var, time=last_t).data)
    qliq_prof = vec(slice(clw_var, time=last_t).data)
    qice_prof = vec(slice(cli_var, time=last_t).data)
    
    # Explicitly compute theta_li using the same formula as the simulation.
    # Note: hus (hus_var) is the total specific humidity (q_tot).
    theta_li_prof = zeros(length(z_prof))
    for i in 1:length(z_prof)
        theta_li_prof[i] = TD.liquid_ice_pottemp(
            thp, T_prof[i], rho_prof[i], qv_prof[i], qliq_prof[i], qice_prof[i]
        )
    end
    
    qvar_prof = vec(slice(q_var, time=last_t).data)
    tvar_prof = vec(slice(t_var, time=last_t).data)
    corr_prof = vec(slice(corr_tq, time=last_t).data)

    prof = Dict(
        "z" => z_prof,
        "z_faces" => z_faces_prof,
        "T" => T_prof,
        "qv" => qv_prof,
        "p" => p_prof,
        "rho" => rho_prof,
        "q_liq" => qliq_prof,
        "q_ice" => qice_prof,
        "theta_li" => theta_li_prof,
        "q_var_sgs" => qvar_prof,
        "T_var_sgs" => tvar_prof,
        "corr_Tq" => corr_prof,
    )

    return prof, thp
    experiment_dir = normpath(joinpath(@__DIR__, "..", ".."))
    expc = va_load_experiment_config(experiment_dir, experiment_config_path)
    return compute_highres_truth_profile_from_case_layers(
        String.(expc["model_config_path"]),
        String(expc["scm_toml"][1]);
        case_name = string(expc["case_name"]),
        quadrature_order = Int(get(expc, "quadrature_order", 3)),
        truth_sgs_distribution = get(expc, "truth_sgs_distribution", nothing) === nothing ? nothing : string(get(expc, "truth_sgs_distribution", nothing)),
        experiment_dir = experiment_dir,
        output_root = output_root,
    )
end

function write_highres_truth_profile_netcdf!(profile::Dict, outpath::String)
    mkpath(dirname(outpath))
    NCDataset(outpath, "c") do ds
        z = profile["z"]
        z_faces = profile["z_faces"]
        defDim(ds, "z", length(z))
        defDim(ds, "z_face", length(z_faces))
        _write_profile_var!(ds, "z", z)
        # Write face coordinates on their own dimension
        zf_var = defVar(ds, "z_faces", eltype(z_faces), ("z_face",))
        zf_var[:] = z_faces
        _write_profile_var!(ds, "T", profile["T"])
        _write_profile_var!(ds, "qv", profile["qv"])
        _write_profile_var!(ds, "p", profile["p"])
        _write_profile_var!(ds, "rho", profile["rho"])
        _write_profile_var!(ds, "q_liq", profile["q_liq"])
        _write_profile_var!(ds, "q_ice", profile["q_ice"])
        _write_profile_var!(ds, "theta_li", profile["theta_li"])
        _write_profile_var!(ds, "q_var_sgs", profile["q_var_sgs"])
        _write_profile_var!(ds, "T_var_sgs", profile["T_var_sgs"])
        _write_profile_var!(ds, "corr_Tq", profile["corr_Tq"])
    end
    println("Saved high-res profile to ", outpath)
end

function run_highres_truth_profile!(; outpath::String="outputs/highres_profile.nc", case::String="default", res::String="high", experiment_config_path::String="config/experiment_config.yml")
    prof = compute_highres_truth_profile(case; res, experiment_config_path)
    write_highres_truth_profile_netcdf!(prof, outpath)
    return prof
end

function run_highres_truth_profile_from_case_layers!(
    case_yaml_layers::Vector{String},
    scm_toml::AbstractString;
    case_name::String,
    outpath::String,
    quadrature_order::Int = 5,
    truth_sgs_distribution::Union{Nothing, String} = "lognormal_vertical_profile_full_cubature",
    experiment_dir::String = normpath(joinpath(@__DIR__, "..", "..")),
    output_root::Union{Nothing, String} = nothing,
)
    prof, thp = compute_highres_truth_profile_from_case_layers(
        case_yaml_layers,
        scm_toml;
        case_name = case_name,
        quadrature_order = quadrature_order,
        truth_sgs_distribution = truth_sgs_distribution,
        experiment_dir = experiment_dir,
        output_root = output_root,
    )
    write_highres_truth_profile_netcdf!(prof, outpath)
    # Persist the canonical thermodynamics parameters alongside the profile
    # so downstream offline workflows can load the exact params deterministically.
    jld2_path = joinpath(dirname(outpath), "thermo_params.jld2")
    @save jld2_path thp
    return prof
end
