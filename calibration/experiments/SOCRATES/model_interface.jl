import ClimaAtmos as CA
import ClimaCalibrate as CAL
import ClimaComms
import Dates
import EnsembleKalmanProcesses as EKP
import JLD2
import NCDatasets as NC
import SOCRATESSingleColumnForcings as SSCF
import YAML
using ClimaCalibrate: path_to_ensemble_member

include(joinpath(@__DIR__, "debug_climacore_hooks.jl"))

@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends

const EXPERIMENT_CONFIG_PATH = joinpath(@__DIR__, "experiment_config.yml")
const DEBUG_INPUT_DUMP_ENABLED =
	get(ENV, "SOCRATES_DEBUG_DUMP_INPUTS", "1") in ("1", "true", "TRUE", "yes", "YES")

function load_experiment_config()
	return YAML.load_file(EXPERIMENT_CONFIG_PATH)
end

function resolve_model_config_paths!(config::Dict)
	if haskey(config, "toml")
		config["toml"] = map(config["toml"]) do toml_path
			isabspath(toml_path) ? toml_path : abspath(joinpath(@__DIR__, toml_path))
		end
	end
	return config
end

forcing_symbol(case::Dict) = Symbol(case["forcing_type"])

mixing_ratio_to_specific_humidity(q_mixing_ratio) = q_mixing_ratio ./ (1 .+ q_mixing_ratio)

function pressure_to_height(
	pressure_levels, temperature_profile, surface_pressure;
	R_d = CA.Parameters.R_d(CA.ClimaAtmosParameters(Float64)),
	grav = CA.Parameters.grav(CA.ClimaAtmosParameters(Float64)),
)
	p = Float64.(pressure_levels)
	T = Float64.(temperature_profile)
	length(p) == length(T) || error("pressure_levels and temperature_profile must have same length")
	all(isfinite, p) || error("pressure_levels contains non-finite values")
	all(isfinite, T) || error("temperature_profile contains non-finite values")

	# Integrate from highest pressure (near surface) to lowest pressure (aloft),
	# and enforce z = 0 at the surface reference pressure.
	sort_idx_desc = sortperm(p; rev = true)
	p_desc = p[sort_idx_desc]
	T_desc = T[sort_idx_desc]

	z_desc = zeros(Float64, length(p_desc))
	if surface_pressure > p_desc[1]
		z_desc[1] = (R_d * T_desc[1] / grav) * log(surface_pressure / p_desc[1])
	end
	for level_idx in 2:length(p_desc)
		p_lower = p_desc[level_idx - 1]
		p_upper = p_desc[level_idx]
		mean_temperature = 0.5 * (T_desc[level_idx - 1] + T_desc[level_idx])
		z_desc[level_idx] = z_desc[level_idx - 1] +
			(R_d * mean_temperature / grav) * log(p_lower / p_upper)
	end

	# Map back to original pressure-level ordering.
	z = similar(z_desc)
	z[sort_idx_desc] = z_desc
	return z
end

function build_time_axis(start_date::String, tsec)
	reference_time = Dates.DateTime(start_date, "yyyymmdd")
	return [reference_time + Dates.Second(round(Int, seconds)) for seconds in tsec]
end

function parse_duration_seconds(value)
	value isa Number && return Float64(value)
	value isa AbstractString || error("Unsupported duration value type: $(typeof(value))")

	m = match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z_]+)\s*$", value)
	isnothing(m) && error("Could not parse duration string: $value")

	amount = parse(Float64, m.captures[1])
	unit = lowercase(m.captures[2])

	factor = if unit in ("s", "sec", "secs", "second", "seconds")
		1.0
	elseif unit in ("m", "min", "mins", "minute", "minutes")
		60.0
	elseif unit in ("h", "hr", "hrs", "hour", "hours")
		3600.0
	elseif unit in ("d", "day", "days")
		86400.0
	else
		error("Unsupported duration unit in '$value'")
	end

	return amount * factor
end

function parse_datetime_flexible(value::AbstractString)
	value_str = String(value)
	for fmt in (
		Dates.dateformat"yyyy-mm-ddTHH:MM:SS",
		Dates.dateformat"yyyy-mm-dd HH:MM:SS",
		Dates.dateformat"yyyy-mm-ddTHH:MM",
		Dates.dateformat"yyyy-mm-dd HH:MM",
		Dates.dateformat"yyyy-mm-dd",
	)
		try
			return Dates.DateTime(value_str, fmt)
		catch
		end
	end
	error("Could not parse datetime string '$value_str'")
end

function forcing_time_unit_to_seconds(unit::AbstractString)
	u = lowercase(String(unit))
	if u in ("day", "days")
		return 86400.0
	elseif u in ("hour", "hours", "hr", "hrs")
		return 3600.0
	elseif u in ("minute", "minutes", "min", "mins")
		return 60.0
	elseif u in ("second", "seconds", "sec", "secs", "s")
		return 1.0
	end
	error("Unsupported forcing time unit '$unit'")
end

function validate_forcing_time_support(config::Dict)
	forcing_file = config["external_forcing_file"]
	t_start = parse_duration_seconds(get(config, "t_start", "0secs"))
	t_end = parse_duration_seconds(get(config, "t_end", "0secs"))
	start_date = get(config, "start_date", "20100101")
	model_start_dt = Dates.DateTime(start_date, "yyyymmdd")

	NC.NCDataset(forcing_file, "r") do ds
		haskey(ds, "time") || error("Forcing file missing time variable: $forcing_file")
		# Use .var[:] to get raw numeric values; ds["time"][:] auto-decodes to DateTimeStandard.
		time_values = Float64.(vec(ds["time"].var[:]))
		all(isfinite, time_values) || error("Forcing file time axis contains non-finite values: $forcing_file")
		length(time_values) >= 2 || error("Forcing file must contain at least two time points: $forcing_file")
		all(diff(time_values) .> 0) || error("Forcing file time axis must be strictly increasing: $forcing_file")

		time_units = haskey(ds["time"].attrib, "units") ? String(ds["time"].attrib["units"]) : ""
		m = match(r"^\s*([A-Za-z_]+)\s+since\s+(.+?)\s*$", time_units)
		isnothing(m) && error("Unsupported forcing time units '$time_units' in $forcing_file")

		unit_seconds = forcing_time_unit_to_seconds(m.captures[1])
		ref_dt = parse_datetime_flexible(strip(m.captures[2]))
		model_offset_seconds = Dates.value(model_start_dt - ref_dt) / 1000
		model_start_time_value = model_offset_seconds / unit_seconds + t_start / unit_seconds
		model_end_time_value = model_offset_seconds / unit_seconds + t_end / unit_seconds

		time_min = minimum(time_values)
		time_max = maximum(time_values)
		tol = 1e-8
		if model_start_time_value < time_min - tol || model_end_time_value > time_max + tol
			error(
				"Model run window [$model_start_time_value, $model_end_time_value] is outside forcing time support " *
				"[$time_min, $time_max] for $forcing_file",
			)
		end
	end

	return nothing
end

function write_repeated_4d_variable!(ds, variable_name, data_zt)
	FT = Float32
	NC.defVar(ds, variable_name, FT, ("x", "y", "z", "time"))
	for x_idx in 1:ds.dim["x"], y_idx in 1:ds.dim["y"]
		ds[variable_name][x_idx, y_idx, :, :] .= FT.(data_zt)
	end
	return nothing
end

function write_repeated_surface_variable!(ds, variable_name, data_t)
	FT = Float32
	NC.defVar(ds, variable_name, FT, ("x", "y", "z", "time"))
	for time_idx in 1:ds.dim["time"], z_idx in 1:ds.dim["z"], x_idx in 1:ds.dim["x"], y_idx in 1:ds.dim["y"]
		ds[variable_name][x_idx, y_idx, z_idx, time_idx] = FT(data_t[time_idx])
	end
	return nothing
end

function is_valid_converted_forcing_file(path::String)
	if !isfile(path)
		return false
	end
	try
		return NC.NCDataset(path, "r") do ds
			required_dims = Dict("x" => 2, "y" => 2)
			for (dim_name, expected_length) in required_dims
				haskey(ds.dim, dim_name) || return false
				ds.dim[dim_name] == expected_length || return false
			end

			haskey(ds.dim, "z") || return false
			haskey(ds.dim, "time") || return false
			n_z = ds.dim["z"]
			n_t = ds.dim["time"]
			n_z > 1 || return false
			n_t > 0 || return false

			haskey(ds, "z") || return false
			z = vec(ds["z"][:])
			length(z) == n_z || return false
			all(isfinite, z) || return false
			all(diff(z) .> 0) || return false

			haskey(ds, "time") || return false
			length(vec(ds["time"][:])) == n_t || return false

			required_profile_vars = (
				"ta",
				"ua",
				"va",
				"hus",
				"rho",
				"wap",
				"wa",
				"tntha",
				"tnhusha",
				"tntva",
				"tnhusva",
			)
			for var_name in required_profile_vars
				haskey(ds, var_name) || return false
				var = ds[var_name]
				Tuple(String.(NC.dimnames(var))) == ("x", "y", "z", "time") || return false
				size(var) == (2, 2, n_z, n_t) || return false
				all(isfinite, vec(var[:])) || return false
			end

			haskey(ds, "pressure_level") || return false
			pressure_level = ds["pressure_level"]
			Tuple(String.(NC.dimnames(pressure_level))) == ("x", "y", "z") || return false
			size(pressure_level) == (2, 2, n_z) || return false
			all(isfinite, vec(pressure_level[:])) || return false

			required_surface_vars = ("ts", "hfls", "hfss", "coszen", "rsdt")
			for var_name in required_surface_vars
				haskey(ds, var_name) || return false
				var = ds[var_name]
				Tuple(String.(NC.dimnames(var))) == ("x", "y", "z", "time") || return false
				size(var) == (2, 2, n_z, n_t) || return false
				all(isfinite, vec(var[:])) || return false
			end

			true
		end
	catch
		return false
	end
end

function drop_singleton_horizontal_dims(data, dim_names::Vector{String})
	horizontal_dims = Set(["lon", "lat", "x", "y"])
	for dim_idx in reverse(eachindex(dim_names))
		if (dim_names[dim_idx] in horizontal_dims) && size(data, dim_idx) == 1
			data = selectdim(data, dim_idx, 1)
			deleteat!(dim_names, dim_idx)
		end
	end
	return data, dim_names
end

function read_input_profile_zt(input_ds, variable_name::String)
	variable = input_ds[variable_name]
	dim_names = String.(collect(NC.dimnames(variable)))
	data = if length(dim_names) == 4
		Array(variable[:, :, :, :])
	elseif length(dim_names) == 2
		Array(variable[:, :])
	else
		Array(variable[:])
	end

	data_squeezed, squeezed_dim_names = drop_singleton_horizontal_dims(data, dim_names)
	ndims(data_squeezed) == 2 ||
		error("Unexpected dimensions for $variable_name after squeezing horizontals: ndims=$(ndims(data_squeezed)) dims=$squeezed_dim_names")

	time_position = findfirst(==("time"), squeezed_dim_names)
	lev_position = findfirst(==("lev"), squeezed_dim_names)
	if isnothing(time_position) || isnothing(lev_position)
		error("$variable_name missing required time/lev dimensions; dims=$squeezed_dim_names")
	end

	return time_position < lev_position ? permutedims(data_squeezed, (2, 1)) : data_squeezed
end

function read_input_timeseries_t(input_ds, variable_name::String)
	variable = input_ds[variable_name]
	dim_names = String.(collect(NC.dimnames(variable)))
	data = if length(dim_names) == 3
		Array(variable[:, :, :])
	elseif length(dim_names) == 1
		Array(variable[:])
	else
		Array(variable[:])
	end

	data_squeezed, squeezed_dim_names = drop_singleton_horizontal_dims(data, dim_names)
	ndims(data_squeezed) == 1 ||
		error("Unexpected dimensions for $variable_name after squeezing horizontals: ndims=$(ndims(data_squeezed)) dims=$squeezed_dim_names")

	time_position = findfirst(==("time"), squeezed_dim_names)
	isnothing(time_position) &&
		error("$variable_name missing required time dimension; dims=$squeezed_dim_names")

	return vec(data_squeezed)
end

function convert_socrates_forcing_file(raw_forcing_file::String, output_file::String, start_date::String)
	mkpath(dirname(output_file))
	is_valid_converted_forcing_file(output_file) && return output_file
	tmp_file = output_file * ".tmp.$(getpid())"
	try

	# Derive physical constants from ClimaAtmos parameter set, not hardcoded values.
	params = CA.ClimaAtmosParameters(Float64)
	R_d = CA.Parameters.R_d(params)
	grav = CA.Parameters.grav(params)

	NC.NCDataset(raw_forcing_file, "r") do input_ds
		lev = vec(input_ds["lev"][:])
		tsec = vec(input_ds["tsec"][:])
		time_axis = build_time_axis(start_date, tsec)

		temperature = read_input_profile_zt(input_ds, "T")
		u_wind = read_input_profile_zt(input_ds, "u")
		v_wind = read_input_profile_zt(input_ds, "v")
		q_mixing_ratio = read_input_profile_zt(input_ds, "q")
		hus = mixing_ratio_to_specific_humidity.(q_mixing_ratio)
		omega = read_input_profile_zt(input_ds, "omega")
		tntha = read_input_profile_zt(input_ds, "divT")
		tnhusha = read_input_profile_zt(input_ds, "divq")
		surface_pressure = read_input_timeseries_t(input_ds, "Ps")
		surface_temperature = read_input_timeseries_t(input_ds, "Tg")

		z_profile = pressure_to_height(lev, temperature[:, 1], surface_pressure[1]; R_d, grav)
		sort_index = sortperm(z_profile)
		z_profile = z_profile[sort_index]
		lev = lev[sort_index]
		temperature = temperature[sort_index, :]
		u_wind = u_wind[sort_index, :]
		v_wind = v_wind[sort_index, :]
		hus = hus[sort_index, :]
		omega = omega[sort_index, :]
		tntha = tntha[sort_index, :]
		tnhusha = tnhusha[sort_index, :]

		rho = lev ./ (R_d .* temperature .* (1 .+ 0.61 .* hus))
		wa = -omega ./ (rho .* grav)
		zeros_zt = zeros(size(temperature))
		zeros_t = zeros(length(time_axis))

		NC.NCDataset(tmp_file, "c") do output_ds
			NC.defDim(output_ds, "time", length(time_axis))
			NC.defDim(output_ds, "z", length(z_profile))
			NC.defDim(output_ds, "x", 2)
			NC.defDim(output_ds, "y", 2)

			NC.defVar(output_ds, "x", Float32, ("x",))
			output_ds["x"][:] = Float32[0, 1]
			NC.defVar(output_ds, "y", Float32, ("y",))
			output_ds["y"][:] = Float32[0, 1]
			NC.defVar(output_ds, "z", Float32, ("z",))
			output_ds["z"][:] = Float32.(z_profile)
			NC.defVar(output_ds, "time", time_axis, ("time",))

			NC.defVar(output_ds, "pressure_level", Float32, ("x", "y", "z"))
			for z_idx in 1:length(z_profile), x_idx in 1:2, y_idx in 1:2
				output_ds["pressure_level"][x_idx, y_idx, z_idx] = Float32(lev[z_idx])
			end

			write_repeated_4d_variable!(output_ds, "ta", temperature)
			write_repeated_4d_variable!(output_ds, "ua", u_wind)
			write_repeated_4d_variable!(output_ds, "va", v_wind)
			write_repeated_4d_variable!(output_ds, "hus", hus)
			write_repeated_4d_variable!(output_ds, "rho", rho)
			write_repeated_4d_variable!(output_ds, "wap", omega)
			write_repeated_4d_variable!(output_ds, "wa", wa)
			write_repeated_4d_variable!(output_ds, "tntha", tntha)
			write_repeated_4d_variable!(output_ds, "tnhusha", tnhusha)
			write_repeated_4d_variable!(output_ds, "tntva", zeros_zt)
			write_repeated_4d_variable!(output_ds, "tnhusva", zeros_zt)

			write_repeated_surface_variable!(output_ds, "ts", surface_temperature)
			write_repeated_surface_variable!(output_ds, "hfls", zeros_t)
			write_repeated_surface_variable!(output_ds, "hfss", zeros_t)
			write_repeated_surface_variable!(output_ds, "coszen", zeros_t)
			write_repeated_surface_variable!(output_ds, "rsdt", zeros_t)
		end
	end
	if is_valid_converted_forcing_file(output_file)
		rm(tmp_file; force = true)
	else
		mv(tmp_file, output_file; force = true)
	end
	catch
		rm(tmp_file; force = true)
		rethrow()
	end
	rm(tmp_file; force = true)
	return output_file
end

function get_socrates_forcing_file(case::Dict, start_date::String)
	forcing_files = SSCF.open_atlas_les_input(
		Int(case["flight_number"]),
		forcing_symbol(case);
		open_files = false,
		include_grid = false,
	)
	raw_forcing_file = getproperty(forcing_files, forcing_symbol(case))
	converted_dir = joinpath(@__DIR__, "Forcing", "ClimaAtmos")
	converted_file = joinpath(
		converted_dir,
		"RF$(lpad(Int(case["flight_number"]), 2, "0"))_$(case["forcing_type"])_tv_forcing.nc",
	)
	convert_socrates_forcing_file(raw_forcing_file, converted_file, start_date)
	return converted_file
end

function linear_interp_profile(z_src::AbstractVector{<:Real}, y_src::AbstractVector{<:Real}, z_tgt::Float64)
	if z_tgt <= z_src[1]
		return y_src[1]
	elseif z_tgt >= z_src[end]
		return y_src[end]
	end

	hi = searchsortedfirst(z_src, z_tgt)
	lo = hi - 1
	z_lo = z_src[lo]
	z_hi = z_src[hi]
	y_lo = y_src[lo]
	y_hi = y_src[hi]
	w = (z_tgt - z_lo) / (z_hi - z_lo)
	return (1 - w) * y_lo + w * y_hi
end

function read_forcing_profile_series(ds, varname::String)
	v = ds[varname]
	A = if ndims(v) == 4
		Array(v[:, :, :, :])
	elseif ndims(v) == 2
		Array(v[:, :])
	else
		Array(v[:])
	end
	ndims(A) == 4 || error("Expected 4D forcing variable $varname; got ndims=$(ndims(A))")
	# Stored as (x, y, z, time); forcing is horizontally repeated, so use a single column.
	return Float64.(A[1, 1, :, :])
end

function read_forcing_surface_series(ds, varname::String)
	v = ds[varname]
	A = if ndims(v) == 4
		Array(v[:, :, :, :])
	elseif ndims(v) == 1
		Array(v[:])
	else
		Array(v[:])
	end
	ndims(A) == 4 || error("Expected 4D forcing variable $varname; got ndims=$(ndims(A))")
	# Stored as (x, y, z, time) with values repeated over z.
	return Float64.(vec(A[1, 1, 1, :]))
end

function write_profile_var!(ods, varname::String, z_src::Vector{Float64}, values_src::Matrix{Float64}, z_tgt::Vector{Float64})
	nt = size(values_src, 2)
	vals = Matrix{Float32}(undef, length(z_tgt), nt)
	for t_idx in 1:nt
		profile_t = view(values_src, :, t_idx)
		for z_idx in eachindex(z_tgt)
			vals[z_idx, t_idx] = Float32(linear_interp_profile(z_src, profile_t, z_tgt[z_idx]))
		end
	end
	NC.defVar(ods, varname, Float32, ("z", "time"))
	ods[varname][:, :] = vals
	return nothing
end

function write_input_summary(summary_path::String, forcing_file::String, z_forcing::Vector{Float64}, z_model::Vector{Float64})
	open(summary_path, "w") do io
		println(io, "SOCRATES forcing input summary")
		println(io, "forcing_file: ", forcing_file)
		println(io, "z_forcing_min_max: ", extrema(z_forcing))
		println(io, "z_model_min_max: ", extrema(z_model))

		NC.NCDataset(forcing_file, "r") do ds
			for profile_name in ("ta", "hus", "ua", "va", "rho", "wa", "wap", "tntha", "tnhusha")
				if haskey(ds, profile_name)
					vals = read_forcing_profile_series(ds, profile_name)
					println(io, profile_name, "_min_max: ", extrema(vals))
				end
			end
			for surface_name in ("ts", "hfls", "hfss", "coszen", "rsdt")
				if haskey(ds, surface_name)
					vals = read_forcing_surface_series(ds, surface_name)
					println(io, surface_name, "_min_max: ", extrema(vals))
				end
			end
		end
	end
	return nothing
end

function dump_evaluated_forcing_inputs(config::Dict, case::Dict, z_faces::Vector{Float64})
	DEBUG_INPUT_DUMP_ENABLED || return nothing
	forcing_file = config["external_forcing_file"]
	debug_dir = joinpath(config["output_dir"], "debug_inputs")
	mkpath(debug_dir)

	z_model = 0.5 .* (z_faces[1:(end - 1)] .+ z_faces[2:end])
	out_file = joinpath(debug_dir, "forcing_inputs_on_model_z.nc")
	summary_file = joinpath(debug_dir, "forcing_inputs_summary.txt")

	NC.NCDataset(forcing_file, "r") do ids
		z_src = Float64.(vec(ids["z"][:]))
		times = ids["time"][:]

		NC.NCDataset(out_file, "c") do ods
			NC.defDim(ods, "z", length(z_model))
			NC.defDim(ods, "time", length(times))
			NC.defVar(ods, "z", Float32, ("z",))
			ods["z"][:] = Float32.(z_model)
			NC.defVar(ods, "time", times, ("time",))

			for profile_name in ("ta", "hus", "ua", "va", "rho", "wa", "wap", "tntha", "tnhusha", "tntva", "tnhusva")
				haskey(ids, profile_name) || continue
				values_src = read_forcing_profile_series(ids, profile_name)
				write_profile_var!(ods, profile_name, z_src, values_src, z_model)
			end

			for surface_name in ("ts", "hfls", "hfss", "coszen", "rsdt")
				haskey(ids, surface_name) || continue
				values = read_forcing_surface_series(ids, surface_name)
				NC.defVar(ods, surface_name, Float32, ("time",))
				ods[surface_name][:] = Float32.(values)
			end

			ods.attrib["flight_number"] = string(case["flight_number"])
			ods.attrib["forcing_type"] = string(case["forcing_type"])
			ods.attrib["source_forcing_file"] = forcing_file
		end

		write_input_summary(summary_file, forcing_file, z_src, z_model)
	end

	return nothing
end

"""
    sscf_centers_to_faces(z_centers)

Convert Atlas LES cell-center heights to cell-face heights.
Uses the same recurrence as TurbulenceConvection.jl:
    z_f[0] = 0.0,  z_f[k] = 2*z_c[k] - z_f[k-1]
"""
function sscf_centers_to_faces(z_centers::Vector{Float64})
	z_faces = Vector{Float64}(undef, length(z_centers) + 1)
	z_faces[1] = 0.0
	for k in eachindex(z_centers)
		z_faces[k + 1] = 2 * z_centers[k] - z_faces[k]
	end
	return z_faces
end

function configure_member_case(base_config::Dict, member_path::String, case::Dict, case_idx::Int)
	config = resolve_model_config_paths!(deepcopy(base_config))
	parameter_path = joinpath(member_path, "parameters.toml")
	start_date = get(config, "start_date", "20100101")

	config["output_dir"] = joinpath(member_path, "config_$(case_idx)")
	config["external_forcing_file"] = get_socrates_forcing_file(case, start_date)
	validate_forcing_time_support(config)
	config["output_default_diagnostics"] = false
	# Distributed runs can become unreadable when every worker emits progress logs.
	config["log_progress"] = false

	# Set vertical grid to match Atlas LES z-levels for this case.
	z_centers = Float64.(SSCF.get_default_new_z(Int(case["flight_number"])))
	z_faces = sscf_centers_to_faces(z_centers)
	config["z_faces"] = z_faces
	config["netcdf_interpolation_num_points"] = [2, 2, length(z_centers)]

	if haskey(config, "toml")
		push!(config["toml"], parameter_path)
	else
		config["toml"] = [parameter_path]
	end

	if haskey(case, "forcing_toml")
		forcing_toml = case["forcing_toml"]
		forcing_toml_path = isabspath(forcing_toml) ? forcing_toml : joinpath(@__DIR__, forcing_toml)
		push!(config["toml"], abspath(forcing_toml_path))
	end

	dump_evaluated_forcing_inputs(config, case, z_faces)

	return config
end

function run_single_case(config::Dict)
	setup_climacore_post_op_nonfinite_debug!()
	comms_ctx = ClimaComms.SingletonCommsContext()
	atmos_config = CA.AtmosConfig(config; comms_ctx)
	simulation = CA.get_simulation(atmos_config)
	sol_res = CA.solve_atmos!(simulation)
	if sol_res.ret_code == :simulation_crashed
		error("The ClimaAtmos simulation crashed for output_dir = $(config["output_dir"]).")
	end
	return nothing
end

function CAL.forward_model(iteration, member)
	experiment_config = load_experiment_config()
	base_model_config = YAML.load_file(joinpath(@__DIR__, experiment_config["model_config"]))

	output_dir = experiment_config["output_dir"]
	member_path = path_to_ensemble_member(output_dir, iteration, member)
	iter_path = CAL.path_to_iteration(output_dir, iteration)
	eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
	case_indices = EKP.get_current_minibatch(eki)
	cases = experiment_config["cases"]

	for case_idx in case_indices
		case = cases[case_idx]
		case_config = configure_member_case(base_model_config, member_path, case, case_idx)
		run_single_case(case_config)
	end

	return nothing
end
