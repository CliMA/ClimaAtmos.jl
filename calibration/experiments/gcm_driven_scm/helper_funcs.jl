
using NCDatasets
using Interpolations
using Statistics
using LinearAlgebra
import ClimaAtmos as CA

"Optional vector"
const OptVec{T} = Union{Nothing, Vector{T}}

"Optional real"
const OptReal = Union{Real, Nothing}

"Optional real"
const OptDict = Union{Nothing, Dict}

CLIMADIAGNOSTICS_LES_NAME_MAP =
    Dict("thetaa" => "theta_mean", "hus" => "qt_mean", "clw" => "ql_mean")



"""Get z coordinates of CA run, given config. """
function get_z_grid(atmos_config)
    params = CA.create_parameter_set(atmos_config)
    spaces =
        CA.get_spaces(atmos_config.parsed_args, params, atmos_config.comms_ctx)
    coord = CA.Fields.coordinate_field(spaces.center_space)
    return convert(Vector{Float64}, parent(coord.z)[:])
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
        norm_vec::Array{FT},
        prof_dof::IT,
        prof_indices::OptVec{Bool} = nothing,
    ) where {FT <: Real, IT <: Integer}

Perform normalization of the aggregate observation vector `y` using separate
normalization constants for each variable, contained in `norm_vec`.

Inputs:
 - `y` :: Aggregate observation vector.
 - `norm_vec` :: Vector of squares of normalization factors.
 - `prof_dof` :: Degrees of freedom of vertical profiles contained in `y`.
 - `prof_indices` :: Vector of booleans specifying which variables are profiles, and which
    are timeseries.
- `z_score_norm` :: z-score normalization
Output:
 - The normalized aggregate observation vector.
"""
function normalize_profile(
    y::Array{FT},
    y_names,
    prof_dof::IT,
    prof_indices::OptVec{Bool} = nothing;
    norm_factors_dict = nothing,
    z_score_norm::Bool = true,
) where {FT <: Real, IT <: Integer}
    y_ = deepcopy(y)
    n_vars = length(y_names)
    prof_indices =
        isnothing(prof_indices) ? repeat([true], n_vars) : prof_indices
    norm_vec = Array{Float64}(undef, n_vars, 2)
    loc_start = 1
    for i in 1:n_vars
        loc_end = prof_indices[i] ? loc_start + prof_dof - 1 : loc_start

        if z_score_norm
            y_i = y_[loc_start:loc_end]

            if !isnothing(norm_factors_dict)
                norm_i = norm_factors_dict[y_names[i]]
                y_μ, y_σ = norm_i
            else
                y_μ, y_σ = mean(y_i), std(y_i)
            end
            y_[loc_start:loc_end] = (y_i .- y_μ) ./ y_σ
            norm_vec[i, :] = [y_μ, y_σ]
        else
            y_[loc_start:loc_end] = y_[loc_start:loc_end]
        end
        loc_start = loc_end + 1
    end
    return y_, norm_vec
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
        var_itp = extrapolate(
            interpolate(nodes, var_, (Gridded(Linear()), NoInterp())),
            Line(),
        )
        # Return interpolated vector
        return var_itp(z_scm, 1:size(var_, 2))
    elseif ndims(var_) == 1
        # Create interpolant
        nodes = (z_ref,)
        var_itp = LinearInterpolation(nodes, var_; extrapolation_bc = Line())
        # Return interpolated vector
        return var_itp(z_scm)
    end
end


"""
    get_profile(
        filename::String,
        y_names::Vector{String};
        ti::Real = 0.0,
        tf::OptReal = nothing,
        z_scm::Union{Vector{T}, T} = nothing,
        prof_ind::Bool = false,
    ) where {T}

Get time-averaged profiles for variables `y_names`, interpolated to
`z_scm` (if given), and concatenated into a single output vector.

Inputs:

 - `filename`    :: nc filename
 - `y_names`     :: Names of variables to be retrieved.
 - `ti`          :: Initial time of averaging window.
 - `tf`          :: Final time of averaging window.
 - `z_scm`       :: If given, interpolate LES observations to given levels.
 - `m`           :: ReferenceModel from which to fetch profiles, implicitly defines `ti` and `tf`.
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
    get_obs(filename::String,, y_names, normalize; [z_scm])

Get observational mean `y` and empirical time covariance `Σ`.

The keyword `normalize` specifies whether observations are to be normalized with respect to the 
per-quantity pooled variance or not. See [`normalize_profile`](@ref) for details.
The normalization vector is returned along with `y` and `Σ`.

If `z_scm` is given, interpolate observations to the given levels.

# Arguments
- `m`            :: A `ReferenceModel`](@ref)
- `y_names`      :: Names of observed fields from the [`ReferenceModel`](@ref) `m`.

# Keywords
- `z_scm`        :: If given, interpolate LES observations to given array of vertical levels.
- `Σ_const`      :: Constant diagonal noise  
- `model_error`  :: Model error per variable, added to the internal variability noise, and
                    normalized by the pooled variance of the variable.

# Returns
- `y::Vector`           :: Mean of observations `y`, possibly interpolated to `z_scm` levels.
- `Σ::Matrix`           :: Observational covariance matrix `Σ`, possibly pool-normalized.
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
    Σ_const::FT = nothing,
) where {FT <: Real}

    # map to CA names to LES names 
    y_names = [CLIMADIAGNOSTICS_LES_NAME_MAP[var_i] for var_i in y_names]
    if !isnothing(norm_factors_dict)
        norm_factors_dict = Dict(
            CLIMADIAGNOSTICS_LES_NAME_MAP[var_i] => value for
            (var_i, value) in norm_factors_dict
        )
    end

    # Get true observables
    y, prof_indices = get_profile(
        filename,
        y_names,
        z_scm = z_scm,
        ti = ti,
        tf = tf,
        prof_ind = true,
    )
    # normalize
    y, norm_vec = normalize_profile(y, y_names, length(z_scm), prof_indices, norm_factors_dict = norm_factors_dict, z_score_norm = z_score_norm)


    if !isnothing(Σ_const)
        Σ = collect(Diagonal(Σ_const * ones(length(y))))
    else
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
        y_names::Vector{String},
        z_scm::Vector{FT};
        normalize::Bool = true,
        model_error::OptVec{FT} = nothing,
        pooled_var_dict::Union{Dict, Nothing} = nothing,
    ) where {FT <: Real}

Obtain the covariance matrix of a group of profiles, where the covariance
is obtained in time.

Inputs:
 - `y_names`      :: List of variable names to be included.
 - `z_scm`        :: If given, interpolates covariance matrix to this locations.
 - `normalize`    :: Whether to normalize the time series with the pooled variance
        before computing the covariance, or not.
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

            # Normalize timeseries
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
            # Normalize timeseries
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
