using Distributed
import ClimaAtmos as CA
import ClimaCalibrate as CAL
import EnsembleKalmanProcesses as EKP
import JLD2
import SOCRATESSingleColumnForcings as SSCF
import YAML

include(joinpath(@__DIR__, "preprocess_reference.jl"))
include(joinpath(@__DIR__, "observation_map.jl"))
include(joinpath(@__DIR__, "model_interface.jl"))
include(joinpath(@__DIR__, "helper_funcs.jl"))

"""
    bootstrap_workers()

Bootstrap any existing Distributed workers with the SOCRATES project environment
and the model interface definitions needed to run `forward_model`.

Workers must be added before calling this (e.g. via `addprocs`). The workers
need either `--project=<SOCRATES_dir>` in their `exeflags`, or this function
will activate the project for them via `Pkg.activate`.
"""
function bootstrap_workers()
    extra = filter(!=(1), workers())
    isempty(extra) && return
    @info "Bootstrapping $(length(extra)) distributed workers..."
    experiment_dir = @__DIR__
	worker_setup_expr = quote
		import Pkg
		Pkg.activate($experiment_dir)
		import ClimaAtmos as CA
		import ClimaCalibrate as CAL
		import ClimaComms
		import Dates
		import EnsembleKalmanProcesses as EKP
		import JLD2
		import NCDatasets as NC
		import SOCRATESSingleColumnForcings as SSCF
		import YAML
		include(joinpath($experiment_dir, "helper_funcs.jl"))
		include(joinpath($experiment_dir, "model_interface.jl"))
	end
	for wid in extra
		Distributed.remotecall_eval(Main, wid, worker_setup_expr)
	end
    @info "Worker bootstrap complete."
end

const EXPERIMENT_CONFIG = YAML.load_file(joinpath(@__DIR__, "experiment_config.yml"))

restart_mode() = get(ENV, "SOCRATES_RESTART", "0") in ("1", "true", "TRUE", "yes", "YES")

smoke_test_mode() = get(ENV, "SOCRATES_SMOKE_TEST", "0") in ("1", "true", "TRUE", "yes", "YES")

forcing_symbol(case::Dict) = Symbol(case["forcing_type"])

case_name(case::Dict) = haskey(case, "name") ? case["name"] :
    string("RF", lpad(Int(case["flight_number"]), 2, "0"), "_", case["forcing_type"])

function get_case_target_z(case::Dict; z_max = nothing)
	z_case = Float64.(SSCF.get_default_new_z(Int(case["flight_number"])))
	if !isnothing(z_max)
		z_case = filter(z -> z <= z_max, z_case)
	end
	return z_case
end

function build_case_target_z(cfg::Dict)
	return [get_case_target_z(case; z_max = get(cfg, "z_max", nothing)) for case in cfg["cases"]]
end

function get_socrates_reference_file(case::Dict)
	flight_number = Int(case["flight_number"])
	forcing_type = forcing_symbol(case)
	case_name = "RF$(lpad(flight_number, 2, "0"))_$(forcing_type)"
	
	# Use preprocessed reference file (TC.jl format)
	ref_file = joinpath(@__DIR__, "Reference", "Atlas_LES", case_name, "stats", case_name * ".nc")
	
	if !isfile(ref_file)
		error("Preprocessed reference file not found: $ref_file. Run preprocessing first.")
	end
	
	return ref_file
end

function copy_experiment_inputs(cfg::Dict)
	output_dir = cfg["output_dir"]
	mkpath(output_dir)
	mkpath(joinpath(output_dir, "configs"))

	cp(
		joinpath(@__DIR__, "experiment_config.yml"),
		joinpath(output_dir, "configs", "experiment_config.yml"),
		force = true,
	)
	cp(
		joinpath(@__DIR__, cfg["model_config"]),
		joinpath(output_dir, "configs", basename(cfg["model_config"])),
		force = true,
	)
	cp(
		joinpath(@__DIR__, cfg["prior_path"]),
		joinpath(output_dir, "configs", basename(cfg["prior_path"])),
		force = true,
	)

	return nothing
end

function resolve_model_config_paths!(config::Dict)
	if haskey(config, "toml")
		config["toml"] = map(config["toml"]) do toml_path
			isabspath(toml_path) ? toml_path : abspath(joinpath(@__DIR__, toml_path))
		end
	end
	return config
end

function build_observations(cfg::Dict, z_model_by_case::Vector{Vector{Float64}})
"""
    compute_pooled_norms(cfg, z_model_by_case) -> Dict{String, Float64}

Compute per-variable characteristic scales by reading the LES reference data for
each case and taking the mean of the absolute nonzero values over the calibration
time window.  Mirrors CalibrateEDMF's `pooled_nonzero_mean_to_value` normalization:
after dividing by this scale, the mean of the non-zero elements in each variable
is approximately 1, regardless of physical units.

The scales are averaged across all cases and saved to `obs_metadata.jld2` so
that `observation_map` can apply the exact same normalization to model outputs.
"""
function compute_pooled_norms(
	cfg::Dict,
	z_model_by_case::Vector{Vector{Float64}},
)
	y_names = cfg["y_var_names"]
	norms_accum = Dict(name => Float64[] for name in y_names)

	for (case_idx, case) in pairs(cfg["cases"])
		y_ti = get(case, "y_t_start_sec", cfg["y_t_start_sec"])
		y_tf = get(case, "y_t_end_sec", cfg["y_t_end_sec"])
		reference_file = haskey(case, "reference_file") ? case["reference_file"] : get_socrates_reference_file(case)
		z_model = z_model_by_case[case_idx]

		for y_name in y_names
			try
				les_var_name = get(CLIMADIAGNOSTICS_TO_LES_NAME_MAP, y_name, y_name)
				var_data = fetch_interpolate_transform(les_var_name, reference_file, z_model)

				if ndims(var_data) == 2
					t_all = nc_fetch(reference_file, "time")
					ti = argmin(abs.(t_all .- y_ti))
					tf = argmin(abs.(t_all .- y_tf))
					window_data = vec(var_data[:, ti:tf])
				else
					window_data = vec(var_data)
				end

				# Mean of nonzero absolute values: characteristic scale for this variable.
				nonzero_vals = abs.(filter(v -> !iszero(v), window_data))
				if !isempty(nonzero_vals)
					push!(norms_accum[y_name], mean(nonzero_vals))
				end
			catch e
				@warn "compute_pooled_norms: failed for y_name=$y_name, case_idx=$case_idx: $e"
			end
		end
	end

	# Average characteristic scale over all cases; fall back to 1.0 if no data.
	return Dict(
		name => isempty(v) ? 1.0 : mean(v)
		for (name, v) in norms_accum
	)
end

function build_observations(cfg::Dict, z_model_by_case::Vector{Vector{Float64}}, pooled_norms_by_var::Dict)
	y_names = cfg["y_var_names"]
	const_noise = get(cfg, "const_noise_by_var", nothing)

	obs_vec = EKP.Observation[]
	for (case_idx, case) in pairs(cfg["cases"])
		y_ti = get(case, "y_t_start_sec", cfg["y_t_start_sec"])
		y_tf = get(case, "y_t_end_sec", cfg["y_t_end_sec"])
		reference_file = haskey(case, "reference_file") ? case["reference_file"] : get_socrates_reference_file(case)
		name_i = case_name(case)
		z_model = z_model_by_case[case_idx]

		# Read observation data from preprocessed reference file
		y_obs = Float64[]
		Σ_diag = Float64[]
		
		for y_name in y_names
			try
				y_var_i = process_observation_variable(
					reference_file,
					y_name,
					z_model;
					t_start = y_ti,
					t_end = y_tf,
					pooled_norms_dict = pooled_norms_by_var,
				)
				append!(y_obs, y_var_i)
				
				# Add observational noise
				if !isnothing(const_noise)
					append!(Σ_diag, const_noise[y_name] * ones(length(y_var_i)))
				else
					append!(Σ_diag, 1e-6 * ones(length(y_var_i)))
				end
			catch err
				@warn "Error reading observation for $y_name in case $name_i: $err"
				append!(y_obs, NaN * ones(length(z_model)))
				append!(Σ_diag, NaN * ones(length(z_model)))
			end
		end

		push!(
			obs_vec,
			EKP.Observation(
				Dict(
					"samples" => y_obs,
					"covariances" => Diagonal(Σ_diag),
					"names" => name_i,
				),
			),
		)
	end

	minibatcher = EKP.FixedMinibatcher(collect(1:cfg["batch_size"]))
	series_names = [case_name(case) for case in cfg["cases"]]
	return EKP.ObservationSeries(obs_vec, minibatcher, series_names)
end

function main(; restart::Bool = restart_mode())
	cfg = deepcopy(EXPERIMENT_CONFIG)
	n_cases = length(cfg["cases"])
	@assert n_cases >= cfg["batch_size"] "batch_size must be <= number of configured cases."
	if smoke_test_mode()
		@info "SOCRATES smoke-test mode enabled; script will stop before CAL.calibrate."
	end

	if restart
		output_dir = cfg["output_dir"]
		if isdir(output_dir)
			@info "Restart requested: removing existing output directory" output_dir
			rm(output_dir, recursive = true, force = true)
		else
			@info "Restart requested: output directory does not exist, nothing to clear" output_dir
		end
	end

	# Bootstrap any Distributed workers with the packages/definitions they need
	bootstrap_workers()

	# Preprocess reference data before anything else
	@info "Preprocessing SOCRATES reference data..."
	ref_output_dir = joinpath(@__DIR__, "Reference", "Atlas_LES")
	preprocess_socrates_reference(;
		flight_numbers = [Int(case["flight_number"]) for case in cfg["cases"]],
		forcing_types = [Symbol(case["forcing_type"]) for case in cfg["cases"]],
		output_dir = ref_output_dir,
		overwrite = false,  # skip if already processed
	)

	model_cfg = resolve_model_config_paths!(YAML.load_file(joinpath(@__DIR__, cfg["model_config"])))
	z_model_by_case = build_case_target_z(cfg)

	copy_experiment_inputs(cfg)
	pooled_norms_by_var = compute_pooled_norms(cfg, z_model_by_case)
	observations = build_observations(cfg, z_model_by_case, pooled_norms_by_var)
	prior = CAL.get_prior(joinpath(@__DIR__, cfg["prior_path"]))

	JLD2.jldsave(
		joinpath(cfg["output_dir"], "obs_metadata.jld2");
		z_model_by_case,
		z_model = z_model_by_case[1],
		dims_per_var = length(z_model_by_case[1]),
		pooled_norms_by_var,
	)

	if smoke_test_mode()
		@info "Smoke test completed successfully." n_cases batch_size = cfg["batch_size"]
		return nothing
	end

	eki_cfg = get(cfg, "eki", Dict())
	scheduler = EKP.DataMisfitController(
		on_terminate = get(eki_cfg, "scheduler_on_terminate", "continue"),
	)
	localization = if get(eki_cfg, "localization", "none") == "none"
		EKP.Localizers.NoLocalization()
	else
		EKP.Localizers.NoLocalization()
	end
	failure_handler = if get(eki_cfg, "failure_handler", "sample_succ_gauss") == "sample_succ_gauss"
		EKP.SampleSuccGauss()
	else
		EKP.SampleSuccGauss()
	end
	accelerator = EKP.DefaultAccelerator()

	backend = nworkers() > 1 ? CAL.WorkerBackend() : CAL.JuliaBackend()
	@info "Starting SOCRATES calibration" output_dir = cfg["output_dir"] backend = nameof(typeof(backend))
	CAL.calibrate(
		backend,
		cfg["ensemble_size"],
		cfg["n_iterations"],
		observations,
		nothing,
		prior,
		cfg["output_dir"];
		scheduler = scheduler,
		localization_method = localization,
		failure_handler_method = failure_handler,
		accelerator = accelerator,
	)
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end
