using NCDatasets
using Statistics
using LinearAlgebra
import ClimaAtmos as CA
import ClimaCalibrate as CAL
import Interpolations


function get_era5_calibration_library()
    ref_paths, months, sites = [], [], []
    for month in [1,4,7,10]
        for cfsite in 2:23
            filename = "../data/era5_cfsite_obs_data.nc"
            site = "site$cfsite"
            push!(ref_paths, filename)
            push!(months, month)
            push!(sites, site)
        end
    end
    ref_paths, months, sites
end

function get_forcing_file(i, months)
    return "../data/era5_monthly_forcing_$(months[i]).nc"
end

function get_cfsite_id(i, cfsites)
    return cfsites[i]
end

function get_batch_indicies_in_iteration(iteration, output_dir::AbstractString)
    iter_path = CAL.path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    return EKP.get_current_minibatch(eki)
end

norm_factors_dict = JLD2.load("../data/norm_factors_dict.jld2")["norm_factors_dict"]

function get_obs(
    filename::String,
    month::Integer,
    site::String,
    y_names::Vector{String};
    normalize::Bool = true,
    norm_factors_dict = norm_factors_dict,
    z_scm::Vector{FT} = z_scm,
    log_vars::Vector{String} = ["clw"],
) where {FT <: AbstractFloat}
    y = []
    for var_name in y_names
        var_ = vertical_interpolation(month, site, var_name, filename, z_scm)
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
    month::Integer,
    site::String,
    var_name::String,
    filename::String,
    z_scm::Vector{FT};
) where {FT <: AbstractFloat}
    # get height of ERA5 data 
    z_ref =get_height(filename, month, site)
    # get ERA5 variable
    var_ = nc_fetch(filename, month, site, var_name)
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

# get height 
function get_height(filename::String, month::Integer, site::String)
    NCDataset(filename, "r") do ds
        month_idx = findfirst(==(month), ds.group[site]["month"][:])
        # divide by gravity 
        return ds.group[site]["z"][:, month_idx] ./ 9.807
    end
end


function get_z_grid(atmos_config; z_max = nothing)
    params = CA.create_parameter_set(atmos_config)
    spaces =
        CA.get_spaces(atmos_config.parsed_args, params, atmos_config.comms_ctx)
    coord = CA.Fields.coordinate_field(spaces.center_space)
    z_vec = convert(Vector{Float64}, parent(coord.z)[:])
    if !isnothing(z_max)
        z_vec = filter(x -> x <= z_max, z_vec)
    end
    return z_vec
end