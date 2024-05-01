# Utilities for reading NetCDF data for GCM-driven single-column simulations.
import NCDatasets as NC
import StatsBase: mean

"""
    gcm_driven_profile(FT, ds, varname; imin = 100)

Extract time-averaged data (`imin:end`) for `varname` from the "profile" group in the GCM-driven dataset `ds`

Returns a `Vector{FT}` of the time-averaged data.

!!! note

    It is critical to provide the `FT` (e.g. `Float64`) associated with the variable in the dataset.
    Otherwise, "garbage" data is returned with no warning.
"""
function gcm_driven_profile_tmean(FT, ds, varname; imin = 100)
    vec(mean(gcm_driven_profile(FT, ds, varname)[:, imin:end]; dims = 2))
end

"""
    gcm_driven_profile(FT, ds, varname)

Extract `varname` from the "profile" group in the GCM-driven dataset `ds`

Returns an `NCDatasets.Variable` object.

!!! note

    It is critical to provide the `FT` (e.g. `Float64`) associated with the variable in the dataset.
    Otherwise, "garbage" data is returned with no warning.
"""
function gcm_driven_profile(FT, ds, varname)
    varname ∈ ("z", "z_half", "t") &&
        throw(ArgumentError("This method does not support access to $varname"))
    _gcm_driven_variable(FT, ds.group["profiles"], varname, ("z", "t"))
end

"""
    gcm_driven_reference(FT, ds, varname)

Extract `varname` from the "reference" group in the GCM-driven dataset `ds`

Returns an `NCDatasets.Variable` object.

!!! note

    It is critical to provide the `FT` (e.g. `Float64`) associated with the variable in the dataset.
    Otherwise, "garbage" data is returned with no warning.
"""
function gcm_driven_reference(FT, ds, varname)
    dimnames = endswith(varname, "_full") ? ("z_full",) : ("z",)
    _gcm_driven_variable(FT, ds.group["reference"], varname, dimnames)
end

"""
    gcm_driven_timeseries(FT, ds, varname)

Extract `varname` from the "timeseries" group in the GCM-driven dataset `ds`

Returns an `NCDatasets.Variable` object.

!!! note

    It is critical to provide the `FT` (e.g. `Float64`) associated with the variable in the dataset.
    Otherwise, "garbage" data is returned with no warning.
"""
gcm_driven_timeseries(FT, ds, varname) =
    _gcm_driven_variable(FT, ds.group["timeseries"], varname, ("t",))

"""
    _gcm_driven_variable(FT, ds, varname, dimnames)

Fetch a variable (`varname`) from a GCM-driven SCM NetCDF dataset (or subgroup) `ds`.

Returns an `NCDatasets.Variable` object.

!!! note

    It is critical to provide _correct_ data type `FT` (e.g. `Float64`) and dimension names `dimnames`
    associated with the variable in the dataset,
    otherwise "garbage" data is returned with no warning (wrong `FT`), or an error is thrown (wrong `dimnames`).
"""
function _gcm_driven_variable(
    FT,
    ds,
    varname,
    dimnames::NTuple{N, String},
) where {N}
    dimids = NC.nc_inq_dimid.(ds.ncid, dimnames)
    varid = NC.nc_inq_varid(ds.ncid, varname)
    NC.Variable{FT, N, typeof(ds)}(ds, varid, dimids)
end
