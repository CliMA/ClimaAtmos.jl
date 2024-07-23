# Utilities for reading NetCDF data for GCM-driven single-column simulations.
import NCDatasets as NC
import StatsBase: mean

"""
    gcm_driven_profile_tmean(ds, varname)

Extract time-averaged data for `varname` from the "profile" group in the GCM-driven dataset `ds`

Returns a 1D ("z",) `Vector{FT}` of the time-averaged data.

!!! note

    This method currently assumes the underlying data is `Float64`. 
    If this is not the case, "garbage" data may be returned with no warning.
"""
function gcm_driven_profile_tmean(ds, varname)
    # vec(mean(gcm_driven_profile(ds, varname)[:, imin:end]; dims = 2))
    vec(mean(gcm_driven_profile(ds, varname), dims = 2))
end

"""
    gcm_driven_profile(ds, varname)

Extract `varname` from the "profile" group in the GCM-driven dataset `ds`

Returns a 2D ("z", "t") `NCDatasets.Variable` object.

!!! note

    This method currently assumes the underlying data is `Float64`. 
    If this is not the case, "garbage" data may be returned with no warning.
"""
# function gcm_driven_profile(ds, varname)
#     varname ∈ ("z", "z_half", "t") &&
#         throw(ArgumentError("This method does not support access to $varname"))
#     _gcm_driven_variable(ds.group["profiles"], varname, ("z", "t"))
# end

function gcm_driven_profile(ds, varname)
    ds[varname][:,:]
end

"""
    gcm_driven_reference(ds, varname)

Extract `varname` from the "reference" group in the GCM-driven dataset `ds`

Returns a 1D ("z",) `NCDatasets.Variable` object.

!!! note

    This method currently assumes the underlying data is `Float64`. 
    If this is not the case, "garbage" data may be returned with no warning.
"""
# function gcm_driven_reference(ds, varname)
#     dimnames = endswith(varname, "_full") ? ("z_full",) : ("z",)
#     _gcm_driven_variable(ds.group["reference"], varname, dimnames)
# end
function gcm_height(ds)
    vec(mean(ds["zg"][:,:], dims=2))
end
"""
#    gcm_driven_timeseries(ds, varname)

# Get `varname` from the dataset `ds` and return values

"""
function gcm_driven_timeseries(ds, varname)
    ds[varname][:]
end

# """
#     gcm_driven_timeseries(ds, varname)

# Extract `varname` from the "timeseries" group in the GCM-driven dataset `ds`

# Returns a 1D ("t",) `NCDatasets.Variable` object.

# !!! note

#     This method currently assumes the underlying data is `Float64`. 
#     If this is not the case, "garbage" data may be returned with no warning.
# """
# gcm_driven_timeseries(ds, varname) =
#     _gcm_driven_variable(ds.group["timeseries"], varname, ("t",))

# """
#     _gcm_driven_variable(ds, varname, dimnames, [FT=Float64])

# Fetch a variable (`varname`) from a GCM-driven SCM NetCDF dataset (or subgroup) `ds`.

# Returns an `NCDatasets.Variable` object.

# !!! note

#     It is critical to provide _correct_ data type `FT` (e.g. `Float64`) and dimension names `dimnames`
#     associated with the variable in the dataset,
#     otherwise "garbage" data is returned with no warning (wrong `FT`), or an error is thrown (wrong `dimnames`).
# """
# function _gcm_driven_variable(
#     ds,
#     varname,
#     dimnames::NTuple{N, String},
#     FT = Float64,
# ) where {N}
#     dimids = NC.nc_inq_dimid.(ds.ncid, dimnames)
#     varid = NC.nc_inq_varid(ds.ncid, varname)
#     NC.Variable{FT, N, typeof(ds)}(ds, varid, dimids)
# end
