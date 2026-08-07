# Utilities for reading NetCDF data for GCM-driven single-column simulations.
import NCDatasets as NC
import Statistics: mean

"""
    gcm_driven_profile_tmean(ds, varname)

Extract `varname` from the GCM-driven dataset `ds` and average it over time.

# Returns

A 1D `Vector` over `z`.

!!! note

    This method currently assumes the underlying data is `Float64`.
    If this is not the case, "garbage" data may be returned with no warning.
"""
function gcm_driven_profile_tmean(ds, varname)
    vec(mean(gcm_driven_profile(ds, varname), dims = 2))
end

"""
    gcm_driven_profile(ds, varname)

Extract the profile variable `varname` from the GCM-driven dataset `ds`.

# Returns

A 2D `Matrix` indexed by `("z", "t")`.

!!! note

    This method currently assumes the underlying data is `Float64`.
    If this is not the case, "garbage" data may be returned with no warning.
"""
function gcm_driven_profile(ds, varname)
    ds[varname][:, :]
end

"""
    gcm_height(ds)

Extract the geopotential height `zg` from the GCM-driven dataset `ds` and
average it over time.

# Returns

A 1D `Vector` over `z` [m].

!!! note

    This method currently assumes the underlying data is `Float64`.
    If this is not the case, "garbage" data may be returned with no warning.
"""
function gcm_height(ds)
    vec(mean(ds["zg"][:, :], dims = 2))
end
"""
    gcm_driven_timeseries(ds, varname)

Extract the time series `varname` from the GCM-driven dataset `ds` as a 1D
`Vector` over time.

!!! note

    This method currently assumes the underlying data is `Float64`.
    If this is not the case, "garbage" data may be returned with no warning.
"""
function gcm_driven_timeseries(ds, varname)
    ds[varname][:]
end
