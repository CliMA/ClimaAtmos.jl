"""
Preprocess SOCRATES Atlas LES reference data into a layout compatible with the
existing ClimaAtmos calibration helper utilities.

Converts:
    Raw Atlas LES variables (THETAL, QT, TABS, etc.) -> ClimaAtmos-facing
    reference variables
    Units: weight fraction, grams -> specific humidity, kg
    Time: days -> seconds
    Vertical grid: z -> zc, zf cell centers and faces

Usage:
    julia --project=. preprocess_reference.jl
"""

import NCDatasets as NC
import SOCRATESSingleColumnForcings as SSCF

# =========================================================================================
# Scaling Functions (from CalibrateEDMF.jl)
# =========================================================================================

"Convert grams to kilograms"
g_to_kg(x) = x ./ 1000.0

"Convert per-day to per-second"
perday_to_persec(x) = x ./ (24.0 * 3600.0)

"Convert weight fraction (g/kg) to specific humidity (kg/kg) then to mixing ratio"
function wt_to_qt(x)
    # Input: weight fraction in g/kg (e.g., 10 g/kg of water)
    # Output: specific humidity in kg/kg
    # q_t = w_t / (1 + w_t) where w_t is in kg/kg
    w_t = x ./ 1000.0  # g/kg to kg/kg
    q_t = w_t ./ (1.0 .+ w_t)
    return q_t
end

"Convert cloud water mixing ratio to specific humidity using total water"
function wc_to_qc(q_tot, w_c)
    # Input: total specific humidity (kg/kg), cloud water mixing ratio (kg/kg)
    # Output: cloud water specific humidity (kg/kg)
    q_c = w_c ./ (1.0 .+ w_c) .- q_tot
    return q_c
end

# =========================================================================================
# Variable Mapping: Raw Atlas LES → ClimaAtmos Diagnostic format
# =========================================================================================

const ATLAS_TO_CLIMA_VARS = Dict{Tuple{String, String}, Union{String, Nothing}}(
    # (reference_name, layout_tag) => Atlas_name
    ("thetal", "profiles") => "THETAL",
    ("ta", "profiles") => "TABS",
    ("hus", "profiles") => "QT",
    ("clw", "profiles") => "QCL",
    ("cli", "profiles") => "QCI",
    ("clwc", "profiles") => "QN",
    ("prw", "profiles") => "QR",
    ("snw", "profiles") => "QS",
    ("gwc", "profiles") => "QG",
    ("tp", "profiles") => "QP",
    ("time", "profiles") => "time",
    ("zc", "profiles") => "z",
)

const ATLAS_TO_CLIMA_SCALINGS = Dict{Tuple{String, String}, Function}(
    ("thetal", "profiles") => Base.identity,
    ("ta", "profiles") => Base.identity,
    ("hus", "profiles") => wt_to_qt ∘ g_to_kg,
    ("clw", "profiles") => g_to_kg,
    ("cli", "profiles") => g_to_kg,
    ("clwc", "profiles") => g_to_kg,
    ("prw", "profiles") => g_to_kg,
    ("snw", "profiles") => g_to_kg,
    ("gwc", "profiles") => g_to_kg,
    ("tp", "profiles") => g_to_kg,
    ("time", "profiles") => Base.identity,
    ("zc", "profiles") => Base.identity,
)

# =========================================================================================
# Helper Functions
# =========================================================================================

"""
    process_time_axis(t_raw::Vector; t_start_hour=10.0)

Convert time from days (decimal) to seconds, starting from a given hour.
Adjusts to nearest hour boundary at the end.

Input: t_raw in days (e.g., [0.0, 0.01, 0.02, ...])
Output: t_seconds starting from t_start_hour and adjusted to nearest hour
"""
function process_time_axis(t_raw::Vector; t_start_hour=10.0)
    # Convert from days to seconds relative to start of day
    t_from_midnight = (t_raw .- t_raw[1]) .* (24.0 * 3600.0)
    
    # Add offset to start from desired hour
    t_seconds = t_from_midnight .+ (t_start_hour * 3600.0)
    
    # Adjust final time to nearest hour
    t_end_adjusted = ceil(t_seconds[end] / 3600.0) * 3600.0
    t_seconds = t_seconds .+ (t_end_adjusted - t_seconds[end])
    
    return t_seconds
end

"""
    create_vertical_grid(z_center::Vector; grid_type="c_to_f")

Create cell-face grid from cell-center grid or vice versa.

If grid_type="c_to_f": input is cell centers, output is cell faces
If grid_type="f_to_c": input is cell faces, output is cell centers
"""
function create_vertical_grid(z_grid::Vector; grid_type="c_to_f")
    if grid_type == "c_to_f"
        # Cell centers → cell faces
        zf = [0.0]  # Start at surface
        for i in 1:(length(z_grid)-1)
            append!(zf, 2.0 * z_grid[i] - zf[end])
        end
        # Final face is above the last center
        append!(zf, 2.0 * z_grid[end] - zf[end])
        return zf
    elseif grid_type == "f_to_c"
        # Cell faces → cell centers
        zc = (z_grid[1:(end-1)] .+ z_grid[2:end]) ./ 2.0
        return zc
    else
        error("Unknown grid_type: $grid_type")
    end
end

# =========================================================================================
# Main Processing Function
# =========================================================================================

"""
    preprocess_socrates_reference(; kwargs...)

Preprocess SOCRATES Atlas LES reference files.

Keyword arguments:
  - flight_numbers: Vector of flight numbers to process (default: [1, 9])
  - forcing_types: Vector of forcing types to process (default: [:obs_data, :ERA5_data])
  - output_dir: Directory to store preprocessed files (default: Reference/Atlas_LES)
  - overwrite: Whether to overwrite existing files (default: true)
"""
function preprocess_socrates_reference(;
    flight_numbers::Vector{Int} = [1, 9],
    forcing_types::Vector{Symbol} = [:obs_data, :ERA5_data],
    output_dir::String = joinpath(@__DIR__, "Reference", "Atlas_LES"),
    overwrite::Bool = true,
)
    @info "Starting SOCRATES reference preprocessing..."
    @info "  Flight numbers: $flight_numbers"
    @info "  Forcing types: $forcing_types"
    @info "  Output directory: $output_dir"

    mkpath(output_dir)

    processed_files = Dict{Tuple{Int, Symbol}, Union{String, Nothing}}()

    for forcing_type in forcing_types
        for flight_number in flight_numbers
            @info "Processing RF $(lpad(flight_number, 2, "0")) with forcing type $forcing_type..."

            # Resolve input file via SSCF
            input_file = nothing
            try
                forcing_files = SSCF.open_atlas_les_output(
                    flight_number,
                    forcing_type;
                    open_files = false,
                    include_grid = false,
                )
                input_file = getproperty(forcing_files, forcing_type)
            catch err
                @warn "Could not resolve input file for RF $(lpad(flight_number, 2, "0")) with $forcing_type: $err"
                processed_files[(flight_number, forcing_type)] = nothing
                continue
            end

            if isnothing(input_file) || !isfile(input_file)
                @warn "Input file does not exist: $input_file"
                processed_files[(flight_number, forcing_type)] = nothing
                continue
            end

            # Store one processed file per SOCRATES case.
            case_name = "RF$(lpad(flight_number, 2, "0"))_$(forcing_type)"
            output_path = joinpath(output_dir, case_name, "stats", case_name * ".nc")

            if isfile(output_path) && !overwrite
                @info "Output file already exists and overwrite=false: $output_path"
                processed_files[(flight_number, forcing_type)] = output_path
                continue
            end

            mkpath(dirname(output_path))
            rm(output_path, force=true)

            # Load raw data
            raw_data = nothing
            try
                raw_data = NC.Dataset(input_file, "r")
            catch err
                @warn "Failed to read input file $input_file: $err"
                processed_files[(flight_number, forcing_type)] = nothing
                continue
            end

            try
                # Create output dataset
                out_data = NC.Dataset(output_path, "c")

                # Copy dimensions, renaming the Atlas vertical dimension `z` to `zc`
                # so existing helper interpolation functions can discover it.
                for dim_name in keys(raw_data.dim)
                    out_dim_name = dim_name == "z" ? "zc" : dim_name
                    NC.defDim(out_data, out_dim_name, raw_data.dim[dim_name])
                end

                # Copy global attributes
                for attr_name in keys(raw_data.attrib)
                    out_data.attrib[attr_name] = raw_data.attrib[attr_name]
                end

                # Process and copy variables in a flat layout.
                for ((clima_name, group), atlas_name) in ATLAS_TO_CLIMA_VARS
                    if isnothing(atlas_name) || !haskey(raw_data, atlas_name)
                        continue
                    end

                    atlas_var = raw_data[atlas_name]
                    raw_data_values = Array(atlas_var)

                    # Apply scaling if available
                    if haskey(ATLAS_TO_CLIMA_SCALINGS, (clima_name, group))
                        scaling_func = ATLAS_TO_CLIMA_SCALINGS[(clima_name, group)]
                        raw_data_values = scaling_func(raw_data_values)
                    end

                    # Rename the vertical profile dimension from z -> zc.
                    var_dims = [dim == "z" ? "zc" : dim for dim in NC.dimnames(atlas_var)]

                    # Create variable in output (no groups, flat structure)
                    NC.defVar(
                        out_data,
                        clima_name,
                        raw_data_values,
                        var_dims;
                        attrib=atlas_var.attrib
                    )
                end

                # Add face coordinates so helper_funcs.jl can interpolate both
                # center- and face-defined quantities using its standard logic.
                if haskey(out_data, "zc")
                    zc = Array(out_data["zc"])
                    zf = create_vertical_grid(vec(zc); grid_type = "c_to_f")
                    NC.defDim(out_data, "zf", length(zf))
                    NC.defVar(out_data, "zf", zf, ["zf"])
                end

                # Process time dimension
                if haskey(out_data, "time")
                    t_raw = Array(out_data["time"])
                    t_seconds = process_time_axis(t_raw)
                    out_data["time"][:] = t_seconds
                end

                close(out_data)
                @info "Successfully processed: $output_path"
                processed_files[(flight_number, forcing_type)] = output_path

            catch err
                @warn "Error processing RF $(lpad(flight_number, 2, "0")) with $forcing_type: $err"
                processed_files[(flight_number, forcing_type)] = nothing
                rm(output_path, force=true)
            finally
                if !isnothing(raw_data)
                    close(raw_data)
                end
            end
        end
    end

    @info "Reference preprocessing complete."
    return processed_files
end

# =========================================================================================
# Run Preprocessing
# =========================================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Running SOCRATES reference preprocessing from command line..."
    preprocess_socrates_reference()
end
