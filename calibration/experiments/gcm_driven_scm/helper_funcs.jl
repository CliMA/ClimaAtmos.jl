
using NCDatasets
using Statistics
using LinearAlgebra
import ClimaAtmos as CA
import ClimaCalibrate as CAL
using Interpolations
import EnsembleKalmanProcesses as EKP
using Logging
using TOML
using Flux
using JLD2

include("nn_helpers.jl")

import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends


"""Suppress Info and Warnings for any function"""
function suppress_logs(f, args...; kwargs...)
    Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
        f(args...; kwargs...)
    end
end

"Optional vector"
const OptVec{T} = Union{Nothing, Vector{T}}

"Optional real"
const OptReal = Union{Real, Nothing}

"Optional dictionary"
const OptDict = Union{Nothing, Dict}

CLIMADIAGNOSTICS_LES_NAME_MAP = Dict(
    "thetaa" => "theta_mean",
    "hus" => "qt_mean",
    "clw" => "ql_mean",
    "cli" => "qi_mean",
    "wap " => "w_core",
)


"""Get z cell centers coordinates for CA run, given config. """
function get_z_grid(atmos_config; z_max = nothing)
    params = CA.ClimaAtmosParameters(atmos_config)
end

function get_z_grid(atmos_config::CA.AtmosConfig; z_max = nothing)
    params = CA.ClimaAtmosParameters(atmos_config)
    grid = CA.get_grid(atmos_config.parsed_args, params, atmos_config.comms_ctx)
    spaces = CA.get_spaces(grid)
    coord = CA.Fields.coordinate_field(spaces.center_space)
    # Use unique() to handle box grids with multiple horizontal points
    z_vec = convert(Vector{Float64}, unique(parent(coord.z)[:]))
    if !isnothing(z_max)
        z_vec = filter(x -> x <= z_max, z_vec)
    end
    return z_vec
end

"""Creates stretched vertical grid using ClimaCore utils, given `z_max`, `z_elem`, and `dz_bottom`.

Output:
 - `z_vec` :: Vector of `z` coordinates.
"""
function create_z_stretch(
    atmos_config;
    z_max = nothing,
    z_elem = nothing,
    dz_bottom = nothing,
)

    config_tmp = deepcopy(atmos_config)
    params = CA.ClimaAtmosParameters(config_tmp)

    !isnothing(z_max) ? config_tmp.parsed_args["z_max"] = z_max : nothing
    !isnothing(z_elem) ? config_tmp.parsed_args["z_elem"] = z_elem : nothing
    !isnothing(dz_bottom) ? config_tmp.parsed_args["dz_bottom"] = dz_bottom :
    nothing

    grid = CA.get_grid(config_tmp.parsed_args, params, config_tmp.comms_ctx)
    spaces = CA.get_spaces(grid)

    coord = CA.Fields.coordinate_field(spaces.center_space);
    # Use unique() to handle box grids with multiple horizontal points
    z_vec = convert(Vector{Float64}, unique(parent(coord.z)[:]))
    return z_vec
end

function get_cal_z_grid(atmos_config, z_cal_grid::Dict, forcing_type::String)
    if forcing_type in keys(z_cal_grid)
        return create_z_stretch(
            atmos_config;
            z_max = z_cal_grid[forcing_type]["z_max"],
            z_elem = z_cal_grid[forcing_type]["z_elem"],
            dz_bottom = z_cal_grid[forcing_type]["dz_bottom"],
        )
    else
        return get_z_grid(atmos_config)
    end
end


"""
    is_timeseries(filename::String, var_name::String)

A `Bool` indicating whether the given variable is a timeseries.
"""
function is_timeseries(filename::String, var_name::String)
    NCDataset(filename) do ds
        if haskey(ds.group, "timeseries")
            if haskey(ds.group["timeseries"], var_name)
                return true
            else
                return false
            end
        else
            return false
        end
    end
end

"""
    is_face_variable(filename::String, var_name::String)

A `Bool` indicating whether the given variable is defined in a face,
or not (cell center).

TurbulenceConvection data are consistent, meaning that variables at
cell faces (resp. centers) have as dim `zf` (resp., `zc`).

PyCLES variables are inconsistent. All variables have as a dim `z`,
the cell face locations, but most of them (except for the statistical
moments of w) are actually defined at cell centers (`z_half`).
"""
function is_face_variable(filename::String, var_name::String)
    # PyCLES cell face variables
    pycles_face_vars = ["w_mean", "w_mean2", "w_mean3"]

    NCDataset(filename) do ds
        for group_option in ["profiles", "reference"]
            haskey(ds.group, group_option) || continue
            if haskey(ds.group[group_option], var_name)
                var_dims = dimnames(ds.group[group_option][var_name])
                if ("zc" in var_dims) | ("z_half" in var_dims)
                    return false
                elseif ("zf" in var_dims) | (var_name in pycles_face_vars)
                    return true
                elseif ("z" in var_dims) # "Inconsistent" PyCLES variables, defined at cell centers.
                    return false
                else
                    throw(
                        ArgumentError(
                            "Variable $var_name does not contain a vertical coordinate.",
                        ),
                    )
                end
            end
        end
    end
end

"""
    get_height(filename::String; get_faces::Bool = false)

Returns the vertical cell centers or faces of the given configuration.

Inputs:
 - filename :: nc filename.
 - get_faces :: If true, returns the coordinates of cell faces. Otherwise,
    returns the coordinates of cell centers.
Output:
 - z: Vertical level coordinates.
"""
function get_height(filename::String; get_faces::Bool = false)
    return get_faces ? nc_fetch(filename, ("zf", "z")) :
           nc_fetch(filename, ("zc", "z_half"))
end

"""
    nc_fetch(filename::String, var_names::NTuple{N, Tuple}) where {N}
    nc_fetch(filename::String, var_name::String)

Returns the data for a variable `var_name` (or
tuple of strings, `varnames`), looping through
all dataset groups.
"""
function nc_fetch(filename::String, var_names::Tuple)
    NCDataset(filename) do ds
        for var_name in var_names
            if haskey(ds, var_name)
                return Array(ds[var_name])
            else
                for group_option in ["profiles", "reference", "timeseries"]
                    haskey(ds.group, group_option) || continue
                    if haskey(ds.group[group_option], var_name)
                        return Array(ds.group[group_option][var_name])
                    end
                end
            end
        end
        error(
            "Variables $var_names not found in the output netCDF file $filename.",
        )
    end
end
nc_fetch(filename::String, var_name::String) = nc_fetch(filename, (var_name,))



"""
    nc_fetch_interpolate(var_name::String, filename::String, z_scm::OptVec{<:Real})

Returns the netcdf variable `var_name`, possibly interpolated to heights `z_scm`.

Inputs:
 - `var_name` :: Name of variable in the netcdf dataset.
 - `filename` :: nc filename
 - `z_scm` :: Vertical coordinate vector onto which var_name is interpolated.
Output:
 - The interpolated vector.
"""
function nc_fetch_interpolate(
    var_name::String,
    filename::String,
    z_scm::OptVec{<:Real},
)
    if !is_timeseries(filename, var_name) && !isnothing(z_scm)
        return vertical_interpolation(var_name, filename, z_scm)
    else
        return nc_fetch(filename, var_name)
    end
end

"""
    fetch_interpolate_transform(var_name::String, filename::String, z_scm::OptVec{<:Real})

Returns the netcdf variable `var_name`, possibly interpolated to heights `z_scm`. If the
variable needs to be transformed to be equivalent to an SCM variable, applies the
transformation as well.

Inputs:
 - `var_name` :: Name of variable in the netcdf dataset.
 - `filename` :: nc filename
 - `z_scm` :: Vertical coordinate vector onto which var_name is interpolated.
Output:
 - The interpolated and transformed vector.

### PyCLES variables that require transformations:

- PyCLES diagnostic vertical fluxes (defined in [AuxiliaryStatistics.pyx](https://github.com/CliMA/pycles/blob/master/AuxiliaryStatistics.pyx#L845)) are specific quantities,
    not multiplied by density, and written at cell centers. These include all `resolved_z_flux_(...)` and
    `sgs_z_flux_(...)` diagnostics. For instance the `resolved_z_flux_theta` is ``\\langle{w^*\\theta^*}\\rangle``.
    In contrast, all `massflux_(...)`, `diffusive_flux_(...)` and `total_flux_(...)` outputs from
    TC.jl are already multiplied by density and written at cell faces; e.g. `total_flux_h` is ``\\rho\\langle{w^*\\theta^*}\\rangle``.
    The location mismatch is handled through `is_face_variable` and interpolation. Another difference
    is that the `total_flux_(...)` in TC.jl simulations includes the full flux, whereas the PyCLES `resolved`
    definitions only include the resolved flux. We must add the `sgs_z_flux_(...)` component here.


- PyCLES prognostic vertical fluxes (defined in [ScalarAdvection.pyx](https://github.com/CliMA/pycles/blob/master/ScalarAdvection.pyx#L136),
    [ScalarDiffusion.pyx](https://github.com/CliMA/pycles/blob/master/ScalarDiffusion.pyx#L174), MomentumAdvection.pyx,
    MomentumDiffusion.pyx) are defined at cell centers and have already been multiplied by density. They are computed
    at cell faces in the low-level functions in [`scalar_advection.h`](https://github.com/CliMA/pycles/blob/master/Csrc/scalar_diffusion.h#L31) and `scalar_diffusion.h`,
    and then [interpolated](https://github.com/CliMA/pycles/blob/master/ScalarDiffusion.pyx#L173)
    in the `.pyx` files before they are written to file. These include all
    `(...)_flux_z` and `(...)__sgs_flux_z` fluxes. In contrast, flux diagnostics from TC.jl are defined
    at cell faces. This mismatch is handled through `is_face_variable`. Another difference is that the `total_flux_(...)`
    in TC.jl simulations includes the full flux, whereas the PyCLES `(...)_flux_z` definition only includes the resolved
    flux. We must add the `(...)__sgs_flux_z` component here.
    PyCLES `sgs_z_flux` and `resolved` flux fields do not include a contribution from the surface flux,
    so the bottom cell is set to the diagnosed surface flux.
"""
function fetch_interpolate_transform(
    var_name::String,
    filename::String,
    z_scm::OptVec{<:Real},
)
    # Multiply by density, add sgs flux
    if occursin("resolved_z_flux", var_name)
        resolved_flux = nc_fetch_interpolate(var_name, filename, z_scm)
        sgs_flux_name =
            string("sgs_z_flux", last(split(var_name, "resolved_z_flux")))
        sgs_flux = nc_fetch_interpolate(sgs_flux_name, filename, z_scm)
        rho_half = nc_fetch_interpolate("rho0_half", filename, z_scm)
        total_flux = rho_half .* (resolved_flux .+ sgs_flux)
        total_flux = rectify_surface_flux(total_flux, var_name, filename, z_scm)
        var_ = total_flux

        # Add sgs flux
    elseif occursin("_flux_z", var_name)
        resolved_flux = nc_fetch_interpolate(var_name, filename, z_scm)
        sgs_flux_name = string(first(split(var_name, "flux_z")), "sgs_flux_z")
        sgs_flux = nc_fetch_interpolate(sgs_flux_name, filename, z_scm)
        total_flux = resolved_flux .+ sgs_flux
        total_flux = rectify_surface_flux(total_flux, var_name, filename, z_scm)
        var_ = total_flux

        # Combine horizontal velocities
    elseif var_name == "horizontal_vel"
        u_ = nc_fetch_interpolate("u_mean", filename, z_scm)
        v_ = nc_fetch_interpolate("v_mean", filename, z_scm)
        var_ = sqrt.(u_ .^ 2 + v_ .^ 2)

    else
        var_ = nc_fetch_interpolate(var_name, filename, z_scm)
    end

    if var_name == "ql_mean" || var_name == "qi_mean"
        var_ = max.(var_, 0.0)
    end

    return var_
end

"""
    rectify_surface_flux(interpolated_var::Vector{FT}, var_name::String, filename::String, z_scm::OptVec{<:Real})

Sets bottom cell in interpolated flux profile equal to surface flux. This is needed for
LES profiles since neither the resolved nor the SGS fluxes include contributions
from the surface flux (otherwise flux goes to zero at the surface).

Inputs:
 - `interpolated_var` :: Interpolated variable vector.
 - `var_name` :: Name of variable in the netcdf dataset.
 - `filename` :: nc filename
 - `z_scm` :: Vertical coordinate vector onto which var_name is interpolated.
Output:
 - Flux profile with bottom cell set equal to surface flux.
 """

function rectify_surface_flux(
    interpolated_var::Array{FT},
    var_name::String,
    filename::String,
    z_scm::OptVec{<:Real},
) where {FT}
    if z_scm != nothing
        min_z_index = argmin(z_scm)
    else
        min_z_index = 1
    end

    if occursin("qt", var_name)
        lhf_surface = nc_fetch_interpolate("lhf_surface_mean", filename, z_scm)
        t_profile = nc_fetch_interpolate("temperature_mean", filename, z_scm)
        surf_qt_flux =
            lhf_surface ./
            TD.latent_heat_vapor.(thermo_param_set, t_profile[min_z_index])
        interpolated_var[min_z_index, :] .= surf_qt_flux
        return interpolated_var
    elseif occursin("s", var_name)
        surf_s_flux =
            nc_fetch_interpolate("s_flux_surface_mean", filename, z_scm)
        interpolated_var[min_z_index, :] .= surf_s_flux
        return interpolated_var
    else
        @warn "Surface flux correcton not implemented for $(var_name). Check consistency of flux definitions."
    end
end


"""
    normalize_profile(
        y::Array{FT},
        y_names,
        prof_dof::IT,
        prof_indices::OptVec{Bool} = nothing;
        norm_factors_dict = nothing,
        z_score_norm::Bool = true,
    ) where {FT <: Real, IT <: Integer}

Perform normalization of the aggregate observation vector `y` using separate
normalization factors (μ, σ) for each variable, returned as `norm_vec`.

Inputs:
 - `y` :: Aggregate observation vector (containing several variables).
 - `y_names` :: Vector of squares of normalization factors.
 - `prof_dof` :: Degrees of freedom of vertical profiles contained in `y`.
 - `prof_indices` :: Vector of booleans specifying which variables are profiles, and which
    are timeseries.
- `norm_factors_dict` :: Dict of precomputed normalization factors Dict(var_name => (μ, σ)) for each variable.
- `z_score_norm` :: z-score normalization
Output:
 - `y_` : normalized `y` vector.
 - `norm_vec` : normalization factors (μ, σ) for each variable.
 The normalized aggregate observation vector.
"""
function normalize_profile(
    y::Array{FT},
    y_names,
    prof_dof::IT,
    prof_indices::OptVec{Bool} = nothing;
    norm_factors_dict = nothing,
    z_score_norm::Bool = true,
    log_vars = [],
    Σ_const::Dict = nothing,
    Σ_scaling::String = nothing,
) where {FT <: Real, IT <: Integer}
    y_ = deepcopy(y)
    n_vars = length(y_names)
    noise = zeros(0)
    prof_indices =
        isnothing(prof_indices) ? repeat([true], n_vars) : prof_indices
    norm_vec = Array{Float64}(undef, n_vars, 2)
    loc_start = 1
    for i in 1:n_vars
        var_name = y_names[i]
        loc_end = prof_indices[i] ? loc_start + prof_dof - 1 : loc_start

        if z_score_norm
            y_i = y_[loc_start:loc_end]
            if var_name in log_vars
                y_i = log10.(y_i .+ 1e-12)
            end

            if !isnothing(norm_factors_dict)
                norm_i = norm_factors_dict[var_name]
                y_μ, y_σ = norm_i
            else
                y_μ, y_σ = mean(y_i), std(y_i)
            end
            y_[loc_start:loc_end] = (y_i .- y_μ) ./ y_σ
            norm_vec[i, :] = [y_μ, y_σ]
        else
            y_[loc_start:loc_end] = y_[loc_start:loc_end]
        end


        if Σ_scaling == "prop"
            append!(noise, Σ_const[var_name] * abs.(y_[loc_start:loc_end]))
        elseif Σ_scaling == "const"
            append!(
                noise,
                Σ_const[var_name] * ones(length(y_[loc_start:loc_end])),
            )
        end

        loc_start = loc_end + 1
    end
    return y_, norm_vec, Diagonal(noise)
end

"""
    vertical_interpolation(
        var_name::String,
        filename::String,
        z_scm::Vector{FT};
    ) where {FT <: AbstractFloat}

Returns the netcdf variable `var_name` interpolated to heights `z_scm`.

Inputs:
 - `var_name` :: Name of variable in the netcdf dataset.
 - `filename` :: nc filename
 - `z_scm` :: Vertical coordinate vector onto which var_name is interpolated.
Output:
 - The interpolated vector.
"""
function vertical_interpolation(
    var_name::String,
    filename::String,
    z_scm::Vector{FT};
) where {FT <: AbstractFloat}
    z_ref =
        get_height(filename, get_faces = is_face_variable(filename, var_name))
    var_ = nc_fetch(filename, var_name)
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
        var_itp = Interpolations.LinearInterpolation(
            nodes,
            var_;
            extrapolation_bc = Interpolations.Line(),
        )
        # Return interpolated vector
        return var_itp(z_scm)
    end
end

function interp_prof_1D(var_data, z_ref, z_interp)
    nodes = (z_ref,)
    var_itp = LinearInterpolation(nodes, var_data; extrapolation_bc = Line())
    return var_itp(z_interp)
end

"""
    get_profile(
        filename::String,
        y_names::Vector{String};
        ti::Real = 0.0,
        tf::OptReal = nothing,
        z_scm::OptVec{T} = nothing,
        prof_ind::Bool = false,
    ) where {T}

Get time-averaged profiles for variables `y_names`, interpolated to
`z_scm` (if given), and concatenated into a single output vector.

Inputs:

 - `filename`    :: nc filename
 - `y_names`     :: Names of variables to be retrieved.
 - `ti`          :: Initial time of averaging window [s].
 - `tf`          :: Final time of averaging window [s].
 - `z_scm`       :: If given, interpolate LES observations to given levels.
 - `prof_ind`    :: Whether to return a boolean array indicating the variables that are profiles (i.e., not scalars). 

Outputs:

 - `y` :: Output vector used in the inverse problem, which concatenates the requested profiles.
"""
function get_profile(
    filename::String,
    y_names::Vector{String};
    ti::Real = 0.0,
    tf::OptReal = nothing,
    z_scm::OptVec{T} = nothing,
    prof_ind::Bool = false,
) where {T}

    t = nc_fetch(filename, "t")
    dt = length(t) > 1 ? mean(diff(t)) : 0.0
    y = zeros(0)
    is_profile = Bool[]

    # Check that times are contained in simulation output
    Δt_start, ti_index = findmin(broadcast(abs, t .- ti))
    # If simulation does not contain values for ti or tf, return high value (penalization)
    if t[end] < ti
        @warn string(
            "Note: t_end < ti, which means that simulation stopped before reaching the requested t_start.",
            "Requested t_start = $ti s. However, the last time available is $(t[end]) s.",
            "Defaulting to penalized profiles...",
        )
        for i in 1:length(y_names)
            var_ = isnothing(z_scm) ? get_height(filename) : z_scm
            append!(y, 1.0e5 * ones(length(var_[:])))
        end
        return prof_ind ? (y, repeat([true], length(y_names))) : y
    end
    if !isnothing(tf)
        Δt_end, tf_index = findmin(broadcast(abs, t .- tf))
        if t[end] < tf - dt
            @warn string(
                "Note: t_end < tf - dt, which means that simulation stopped before reaching the requested t_end.",
                "Requested t_end = $tf s. However, the last time available is $(t[end]) s.",
                "Defaulting to penalized profiles...",
            )
            for i in 1:length(y_names)
                var_ = isnothing(z_scm) ? get_height(filename) : z_scm
                append!(y, 1.0e5 * ones(length(var_[:])))
            end
            return prof_ind ? (y, repeat([true], length(y_names))) : y
        end
    end

    # Return time average for non-degenerate cases
    for var_name in y_names
        var_ = fetch_interpolate_transform(var_name, filename, z_scm)
        if ndims(var_) == 2
            var_mean =
                !isnothing(tf) ? mean(var_[:, ti_index:tf_index], dims = 2) :
                var_[:, ti_index]
            append!(is_profile, true)
        elseif ndims(var_) == 1
            var_mean =
                !isnothing(tf) ? mean(var_[ti_index:tf_index]) : var_[ti_index]
            append!(is_profile, false)
        end
        append!(y, var_mean)
    end
    return prof_ind ? (y, is_profile) : y
end



"""
    get_obs(
        filename::String,
        y_names::Vector{String},
        z_scm::OptVec{FT} = nothing;
        kwargs...)

Get observational mean `y` and time covariance `Σ`.

The keyword `z_score_norm` specifies whether observations are to be normalized or not. 
See [`normalize_profile`](@ref) for details. The normalization vector is returned along with `y` and `Σ`.

If `z_scm` is given, interpolate observations to the given levels.

# Arguments
 - `filename`    :: nc filename
 - `y_names`      :: Names of variables to include.
 - `z_scm`        :: If given, interpolate LES observations to given array of vertical levels.

# Keywords
 - `model_error`  :: Model error per variable, added to the internal variability noise, and
                    normalized by the pooled variance of the variable.
 - `ti`          :: Initial time of averaging window [s].
 - `tf`          :: Final time of averaging window [s].
 - `norm_factors_dict` :: Dict of precomputed normalization factors Dict(var_name => (μ, σ)) for each variable.
 - `z_score_norm` :: z-score normalization
 - `Σ_const` :: If given, sets constant diagonal σ^2 to use for noise cov Σ
 -  `Σ_scaling` ::String = {"const", "prop"}

# Returns
 - `y::Vector`           :: Mean of observations `y`, possibly interpolated to `z_scm` levels.
 - `Σ::Matrix`           :: Observational covariance matrix `Σ`
- `norm_vec::Matrix`    :: Normalization mean & std, one row for each variable
"""
function get_obs(
    filename::String,
    y_names::Vector{String},
    z_scm::OptVec{FT} = nothing;
    ti::FT = 5.5 * 3600 * 24,
    tf::FT = 6.0 * 3600 * 24,
    model_error::OptVec{FT} = nothing,
    norm_factors_dict = nothing,
    z_score_norm = true,
    log_vars = [],
    Σ_const::Dict = nothing,
    Σ_scaling::String = "const",
) where {FT <: Real}

    # map to CA names to LES names 
    y_names = [CLIMADIAGNOSTICS_LES_NAME_MAP[var_i] for var_i in y_names]
    if !isnothing(log_vars)
        log_vars =
            [CLIMADIAGNOSTICS_LES_NAME_MAP[log_var_i] for log_var_i in log_vars]
    end

    if !isnothing(norm_factors_dict)
        norm_factors_dict = Dict(
            CLIMADIAGNOSTICS_LES_NAME_MAP[var_i] => value for
            (var_i, value) in norm_factors_dict
        )
    end

    if !isnothing(Σ_const)
        Σ_const = Dict(
            CLIMADIAGNOSTICS_LES_NAME_MAP[var_i] => value for
            (var_i, value) in Σ_const
        )
    end

    # Get true observables
    y, prof_indices = get_profile(
        filename,
        y_names;
        z_scm = z_scm,
        ti = ti,
        tf = tf,
        prof_ind = true,
    )
    # normalize
    y, norm_vec, Σ = normalize_profile(
        y,
        y_names,
        length(z_scm),
        prof_indices;
        norm_factors_dict = norm_factors_dict,
        z_score_norm = z_score_norm,
        log_vars = log_vars,
        Σ_const,
        Σ_scaling,
    )

    if isnothing(Σ_const) & isnothing(Σ_scaling)
        # time covariance
        Σ, pool_var = get_time_covariance(
            filename,
            y_names,
            z_scm,
            model_error = model_error,
            pooled_var_dict = pooled_var_dict,
        )
    end

    return y, Σ, norm_vec
end


"""
    get_time_covariance(
        filename::String,
        y_names::Vector{String},
        z_scm::Vector{FT};
        ti::FT = 72000.0,
        normalize::Bool = false,
        model_error::OptVec{FT} = nothing,
        pooled_var_dict::Union{Dict, Nothing} = nothing,
    ) where {FT <: Real}

Obtain the covariance matrix of a group of profiles, where the covariance
is obtained in time.

Inputs:
 - `filename`    :: nc filename
 - `y_names`      :: Names of variables to include.
 - `z_scm`        :: If given, interpolates covariance matrix to this locations.
 - `ti`          :: Initial time, after which covariance is computed [s].
 - `normalize`    :: Whether to normalize the time series before computing the covariance, or not.
 - `model_error`  :: Model error per variable, added to the internal variability noise, and
                    normalized by the pooled variance of the variable.
 - `pooled_var_dict` :: Dict of precomputed pooled variances
"""
function get_time_covariance(
    filename::String,
    y_names::Vector{String},
    z_scm::Vector{FT};
    ti::FT = 72000.0,
    normalize::Bool = false,
    model_error::OptVec{FT} = nothing,
    pooled_var_dict::Union{Dict, Nothing} = nothing,
) where {FT <: Real}
    t = nc_fetch(filename, "t")
    # Find closest interval in data
    ti_index = argmin(broadcast(abs, t .- ti))

    tf_index = length(t)
    N_samples = length(ti_index:tf_index)
    ts_vec = zeros(0, N_samples)
    num_outputs = length(y_names)
    pool_var = zeros(num_outputs)
    model_error_expanded = Vector{FT}[]

    for (i, var_name) in enumerate(y_names)

        var_ = fetch_interpolate_transform(var_name, filename, z_scm)

        if ndims(var_) == 2
            if !isnothing(pooled_var_dict)
                if haskey(pooled_var_dict, var_name)
                    pool_var[i] = pooled_var_dict[var_name]
                end
            else
                # Store pooled variance
                pool_var[i] =
                    mean(var(var_[:, ti_index:tf_index], dims = 2)) + eps(FT) # vertically averaged time-variance of variable
            end

            # Normalize time series
            ts_var_i =
                normalize ? var_[:, ti_index:tf_index] ./ sqrt(pool_var[i]) :
                var_[:, ti_index:tf_index] # dims: (Nz, Nt)
        elseif ndims(var_) == 1
            if !isnothing(pooled_var_dict)
                if haskey(pooled_var_dict, var_name) &
                   !isnothing(pooled_var_dict)
                    pool_var[i] = pooled_var_dict[var_name]
                end
            else
                # Store pooled variance
                pool_var[i] = var(var_[ti_index:tf_index]) + eps(FT) # time-variance of variable
            end
            # Normalize time series
            ts_var_i =
                normalize ?
                Array(var_[ti_index:tf_index]') ./ sqrt(pool_var[i]) :
                Array(var_[ti_index:tf_index]') # dims: (1, Nt)
        else
            throw(
                ArgumentError(
                    "Variable `$var_name` has more than 2 dimensions, 1 or 2 were expected.",
                ),
            )
        end
        ts_vec = cat(ts_vec, ts_var_i, dims = 1)  # final dims: (Nz*num_profiles + num_timeseries, Nt)

        # Add structural model error
        if !isnothing(model_error)
            var_model_error =
                normalize ? model_error[i] : model_error[i] * pool_var[i]
            model_error_expanded = cat(
                model_error_expanded,
                repeat([var_model_error], size(ts_var_i, 1)),
                dims = 1,
            )
        end
    end
    cov_mat = cov(ts_vec, dims = 2)  # covariance, w/ samples across time dimension (t_inds).
    cov_mat =
        !isnothing(model_error) ?
        cov_mat + Diagonal(FT.(model_error_expanded)) : cov_mat
    return cov_mat, pool_var
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
    return_z_interp = false,
)

    G_ensemble =
        Array{Float64}(undef, n_vert_levels, config_dict["ensemble_size"])
    z_interp = nothing

    for m in 1:config_dict["ensemble_size"]
        try
            member_path =
                TOMLInterface.path_to_ensemble_member(output_dir, iteration, m)
            simulation_dir =
                joinpath(member_path, "config_$config_i", "output_active")

            model_config_dict = YAML.load_file(joinpath(simulation_dir, ".yml"))
            # suppress logs when creating model config, z grids to avoid cluttering output
            model_config = suppress_logs(
                CA.AtmosConfig,
                model_config_dict;
                comms_ctx = ClimaComms.SingletonCommsContext(),
            )
            if !isnothing(config_dict["z_cal_grid"])
                z_interp = suppress_logs(
                    get_cal_z_grid,
                    model_config,
                    config_dict["z_cal_grid"],
                )
            end

            simdir = SimDir(simulation_dir)

            G_ensemble[:, m] .= process_profile_func(
                simdir,
                var_name;
                reduction = reduction,
                t_start = config_dict["g_t_start_sec"],
                t_end = config_dict["g_t_end_sec"],
                z_max = z_max,
                z_interp = z_interp,
            )

            # catch file i/o errors -> ensemble member crashed
        catch err
            err_str = string(err)
            if occursin("Simulation failed at:", err_str) ||
               occursin("opening file", err_str)
                @info "Simulation failed at a specific time for ensemble member $m" err
                G_ensemble[:, m] .= NaN
            elseif occursin("HDF error", err_str)
                @info "NetCDF HDF error encountered for ensemble member $m" err
                G_ensemble[:, m] .= NaN
            else
                rethrow(err)
            end
        end

    end
    return return_z_interp ? (G_ensemble, z_interp) : G_ensemble
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

function get_forcing_file(i, ref_paths)
    ref_path = ref_paths[i]
    cfsite_info = get_cfsite_info_from_path(ref_path)

    forcing_model = cfsite_info["forcing_model"]
    experiment = cfsite_info["experiment"]
    month = cfsite_info["month"]

    forcing_file_path = "/resnick/groups/esm/zhaoyi/GCMForcedLES/forcing/corrected/$(forcing_model)_$(experiment).2004-2008.$(month).nc"

    return forcing_file_path
end

function get_cfsite_id(i, cfsite_numbers)
    return string("site", cfsite_numbers[i])
end

function get_cfsite_info_from_path(input_string::String)
    pattern =
        r"cfsite/(\d+)/([^/]+)/([^/]+)/.*cfsite(\d+)_([^_]+)_([^_]+)_.*\.(\d{2})\..*nc"
    m = match(pattern, input_string)
    if m !== nothing
        return Dict(
            "forcing_model" => m.captures[2],
            "cfsite_number" => m.captures[4],
            "month" => m.captures[7],
            "experiment" => m.captures[3],
        )
    else
        return Dict{String, String}()
    end
end


function get_batch_indicies_in_iteration(iteration, output_dir::AbstractString)
    iter_path = CAL.path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    return EKP.get_current_minibatch(eki)
end




function create_prior_with_nn(
    prior_path,
    pretrained_nn_path;
    arc = [8, 20, 15, 10, 1],
)

    prior_dict = TOML.parsefile(prior_path)
    parameter_names = keys(prior_dict)

    prior_vec =
        Vector{EKP.ParameterDistribution}(undef, length(parameter_names))
    for (i, n) in enumerate(parameter_names)
        prior_vec[i] = CAL.get_parameter_distribution(prior_dict, n)
    end

    @load pretrained_nn_path serialized_weights
    num_nn_params = length(serialized_weights)


    # nn_model = construct_fully_connected_nn(arc, deepcopy(serialized_weights); biases_bool = true, output_layer_activation_function = Flux.identity)
    nn_model = construct_fully_connected_nn(
        arc,
        deepcopy(serialized_weights);
        biases_bool = true,
        activation_function = Flux.leakyrelu,
        output_layer_activation_function = Flux.identity,
    )

    # serialized_stds = serialize_std_model(nn_model; std_weight = 0.05, std_bias = 0.005)
    serialized_stds =
        serialize_std_model(nn_model; std_weight = 0.1, std_bias = 0.00001)

    nn_mean_std = EKP.VectorOfParameterized([
        Normal(serialized_weights[ii], serialized_stds[ii]) for
        ii in 1:num_nn_params
    ])
    nn_constraint = repeat([EKP.no_constraint()], num_nn_params)
    nn_prior = EKP.ParameterDistribution(
        nn_mean_std,
        nn_constraint,
        "mixing_length_param_vec",
    )
    push!(prior_vec, nn_prior)

    prior = EKP.combine_distributions(prior_vec)
    return prior
end
