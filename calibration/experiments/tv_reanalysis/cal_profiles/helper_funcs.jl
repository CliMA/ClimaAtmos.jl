using NCDatasets
using Statistics
using LinearAlgebra
import ClimaAtmos as CA
import ClimaCalibrate as CAL
import Interpolations
using JLD2
using Dates

FT = Float64
# batch over locations
lats = [
    -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -18.5, -17.0,
    -15.5, -14.0, -12.5, -11.0, -9.5, -8.0, 38.09999847, 35.0, 32.0,
    29.0, 26.0, 23.0, 20.0, 17.0
]

lons = [
    -72.5, -75.0, -77.5, -80.0, -82.5, -85.0, -90.0, -95.0,
    -100.0, -105.0, -110.0, -115.0, -120.0, -125.1000061, -123.0,
    -125.0, -129.0, -133.0, -137.0, -141.0, -145.0, -149.0
]

start_dates = fill("20070701", length(lats))
convection_type = fill("shallow", length(lats))

function get_era5_calibration_library(lats = lats, lons = lons, convection_type = convection_type, start_dates=start_dates)
    # test if forcing files have already been created 
    for i in 1:length(lats)
        site_info = Dict(
            "start_date" => get_startdate(i, start_dates),
            "site_latitude" => get_latitude(i, lats),
            "site_longitude" => get_longitude(i, lons),
        )
        fpath = CA.get_external_forcing_file_path(site_info)

        if !isfile(fpath)
            @info "Creating forcing file at $fpath ..."
            CA.generate_external_era5_forcing_file(site_info["site_latitude"], 
                                                site_info["site_longitude"], 
                                                site_info["start_date"], 
                                                fpath,
                                                FT,
            )
        end
    end
    
    start_dates, lats, lons, convection_type, length(lats) # also return number of locations
end

function get_latitude(i, lats)
    return lats[i]
end

function get_longitude(i, lons)
    return lons[i]
end

function get_startdate(i, dates)
    return "20070701"
end

function get_forcing_type(i, convection_types)
    return convection_types[i]
end

function get_batch_indicies_in_iteration(iteration, output_dir::AbstractString)
    iter_path = CAL.path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    return EKP.get_current_minibatch(eki)
end

#norm_factors_dict = JLD2.load("../data/norm_factors_dict.jld2")["norm_factors_dict"]

function get_obs(
    filename::String,
    y_names::Vector{String},
    obs_start::DateTime,
    obs_end::DateTime;
    normalize::Bool = true,
    norm_factors_dict = norm_factors_dict,
    z_scm::Vector{FT} = z_scm,
    log_vars::Vector{String} = [""],
) where {FT <: AbstractFloat}
    y = []
    for var_name in y_names
        var_ = vertical_interpolation(obs_start, obs_end, var_name, filename, z_scm)
        if var_name in log_vars
            var_ = log10.(var_ .+ 1e-12)
        end
        if normalize
            # normalize
            var_ = (var_ .- norm_factors_dict[var_name][1]) ./ norm_factors_dict[var_name][2]
            push!(y, var_)
        else
            push!(y, var_)
        end
    end
    obs_data = vcat(y...)
    return obs_data
end


function vertical_interpolation(
    obs_start::DateTime,
    obs_end::DateTime,
    var_name::String,
    filename::String,
    z_scm::Vector{FT};
) where {FT <: AbstractFloat}
    # get height of ERA5 data
    z_ref =get_height(filename)
    # get ERA5 variable
    var_ = nc_fetch(filename, var_name, obs_start, obs_end)
    if ndims(var_) == 2
        # Create interpolant
        nodes = (z_ref, 1:size(var_, 2))
        var_itp = Interpolations.extrapolate(
            Interpolations.interpolate(
                nodes,
                var_,
                (
                    Interpolations.Gridded(Interpolations.Linear()),
                    Interpolations.NoInterp(),
                ),
            ),
            Interpolations.Line(),
        )
        # Return interpolated vector
        return var_itp(z_scm, 1:size(var_, 2))
    elseif ndims(var_) == 1
        # Create interpolant
        nodes = (z_ref,)
        if var_name in ["clw"]
            # prevent from going negative at the bottom boundary
            var_itp = Interpolations.LinearInterpolation(nodes,var_;
                extrapolation_bc = Interpolations.Flat(),
            )
        else
            var_itp = Interpolations.LinearInterpolation(nodes,var_;
                extrapolation_bc = Interpolations.Line(),
            )
        end
        # Return interpolated vector
        return var_itp(z_scm)
    end
end


function nc_fetch(filename::String, month::Integer, site::String, var_name::String)
    NCDataset(filename, "r") do ds
        # carefully select the correct month index
        month_idx = findfirst(==(month), ds.group[site]["month"][:])
        return ds.group[site][var_name][:, month_idx]
    end
end

function nc_fetch(filename::String, var::String, obs_start::DateTime, obs_end::DateTime)
    NCDataset(filename, "r") do ds
        time = ds["time"][:]
        time_idx = findall(x -> x >= obs_start && x <= obs_end, time)
        return mean(ds[var][1, 1, :, time_idx], dims = 2)[:]
    end

end

# get height
function get_height(filename::String, month::Integer, site::String)
    NCDataset(filename, "r") do ds
        month_idx = findfirst(==(month), ds.group[site]["month"][:])
        # divide by gravity
        return ds.group[site]["z"][:, month_idx] ./ 9.807
    end
end

function get_height(filename::String)
    NCDataset(filename, "r") do ds
        return ds["z"][:]
    end
end


function get_z_grid(atmos_config; z_max = nothing)
    params = CA.ClimaAtmosParameters(atmos_config)
    spaces =
        CA.get_spaces(atmos_config.parsed_args, params, atmos_config.comms_ctx)
    coord = CA.Fields.coordinate_field(spaces.center_space)
    z_vec = convert(Vector{Float64}, parent(coord.z)[:])
    if !isnothing(z_max)
        z_vec = filter(x -> x <= z_max, z_vec)
    end
    return z_vec
end



"""Get minimum loss (RMSE) from EKI obj for a given iteration."""
function get_loss_min(output_dir, iteration; n_lowest = 10, return_rmse = true)
    iter_path = CAL.path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))

    iter_path_p1 = CAL.path_to_iteration(output_dir, iteration + 1)
    eki_p1 = JLD2.load_object(joinpath(iter_path_p1, "eki_file.jld2"))

    y = EKP.get_obs(eki)
    g = EKP.get_g_final(eki_p1)

    # Find successful simulations
    non_nan_columns_indices = findall(x -> !x, vec(any(isnan, g, dims = 1)))
    g = g[:, non_nan_columns_indices]

    return lowest_loss_rmse(y, g; n_lowest, return_rmse)
end

function lowest_loss_rmse(
    y::Vector,
    g::Matrix;
    n_lowest::Int = 1,
    return_rmse = true,
)
    @assert length(y) == size(g, 1)
    y_diff = y .- g
    rmse = sqrt.(sum(y_diff .^ 2, dims = 1) ./ size(y_diff, 1))
    sorted_indices = sortperm(vec(rmse); rev = false)

    if return_rmse
        return sorted_indices[1:n_lowest], rmse[sorted_indices[1:n_lowest]]
    else
        return sorted_indices[1:n_lowest]
    end
end

"""Given calibration output directory, find iterations that contain given config."""
function get_iters_with_config(config_i::Int, config_dict::Dict)
    config_i_dir = "config_$config_i"
    iters_with_config = []
    for iter in 0:config_dict["n_iterations"]
        for m in 1:config_dict["ensemble_size"]
            member_path =
                TOMLInterface.path_to_ensemble_member(output_dir, iter, m)

            if isdir(member_path)
                dirs = filter(
                    entry -> isdir(joinpath(member_path, entry)),
                    readdir(member_path),
                )

                if (config_i_dir in dirs) & ~(iter in iters_with_config)
                    push!(iters_with_config, iter)
                end

            else
                @info "Iteration not reached: $iter"
            end

        end
    end
    if length(iters_with_config) == 0
        @info "No iterations found for config $config_i"
    end
    return iters_with_config
end

"""
    ensemble_data(
        process_profile_func,
        iteration,
        config_i::Int,
        config_dict;
        var_name = "hus",
        reduction = "inst",
        output_dir = nothing,
        z_max = nothing,
        n_vert_levels,
    )

Fetch output vectors `y` for a specific variable across ensemble members.
    Fills missing data from crash simulations with NaN.

# Arguments
 - `process_profile_func` :: Function to process profile data (typically the observation map)
 - `iteration`   :: Iteration number for the ensemble data
 - `config_i`    :: Configuration id
 - `config_dict` :: Configuration dictionary

# Keywords
 - `var_name`    :: Name of the variable to extract data for
 - `reduction`   :: Type of ClimaDiagnostics data reduction to apply
 - `output_dir`  :: Calibration output dir
 - `z_max`       :: maximum z
 - `n_vert_levels` :: Number of vertical levels in the data

# Returns
 - `G_ensemble::Array{Float64}` :: Array containing the data for the specified variable across all ensemble members, with shape (n_vert_levels, ensemble_size).
"""
function ensemble_data(
    process_profile_func,
    iteration,
    config_i::Int,
    config_dict;
    var_name = "hus",
    reduction = "inst",
    output_dir = nothing,
    z_max = nothing,
    n_vert_levels,
)

    G_ensemble =
        Array{Float64}(undef, n_vert_levels, config_dict["ensemble_size"])

    for m in 1:config_dict["ensemble_size"]

        try
            member_path =
                TOMLInterface.path_to_ensemble_member(output_dir, iteration, m)
            simdir =
                SimDir(joinpath(member_path, "config_$config_i", "output_0000"))

            G_ensemble[:, m] .= process_profile_func(
                simdir,
                var_name;
                reduction = reduction,
                t_start = config_dict["g_t_start_sec"],
                t_end = config_dict["g_t_end_sec"],
                z_max = z_max,
            )
        catch err
            @info "Error during observation map for ensemble member $m" err
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end
