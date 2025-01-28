# Utilities for reading NetCDF data for GCM-driven single-column simulations.
import NCDatasets as NC
import Statistics: mean

"""
    gcm_driven_profile_tmean(ds, varname)

Extract time-averaged data for `varname` from the "profile" group in the GCM-driven dataset `ds`

Returns a 1D ("z",) `Vector{FT}` of the time-averaged data.

!!! note

    This method currently assumes the underlying data is `Float64`. 
    If this is not the case, "garbage" data may be returned with no warning.
"""
function gcm_driven_profile_tmean(ds, varname)
    vec(mean(gcm_driven_profile(ds, varname), dims = 2))
end

"""
    gcm_driven_profile(ds, varname)

Extract `varname` from the "profile" group in the GCM-driven dataset `ds`

Returns a 2D ("z", "t") `Matrix` object.

!!! note

    This method currently assumes the underlying data is `Float64`. 
    If this is not the case, "garbage" data may be returned with no warning.
"""
function gcm_driven_profile(ds, varname)
    ds[varname][:, :]
end

"""
    era5_driven_profile_tmean(ds, varname)
Extract `varname` from the "profile" group in the ERA5-driven dataset `ds`
Returns a 1D ("z",) `Vector{FT}` of the time-averaged data.
"""
function era5_driven_profile(ds, varname)
    ds[varname][:]
end

"""
    gcm_driven_reference(ds, varname)

Extract `height` from the GCM-driven dataset `ds`

Returns a 1D ("z",) `vec` object.

!!! note

    This method currently assumes the underlying data is `Float64`. 
    If this is not the case, "garbage" data may be returned with no warning.
"""
function gcm_height(ds)
    vec(mean(ds["zg"][:, :], dims = 2))
end

"""
    era5_height(ds)
Extract time-averaged `height` from the ERA5-driven dataset `ds`
"""
function era5_height(ds)
    vec(mean(ds["z"][:], dims = 2))
end

"""
#    gcm_driven_timeseries(ds, varname)

# Get `varname` from the dataset `ds` and return values

!!! note

    This method currently assumes the underlying data is `Float64`. 
    If this is not the case, "garbage" data may be returned with no warning.
"""
function gcm_driven_timeseries(ds, varname)
    ds[varname][:]
end

"""
    era5_driven_timeseries(ds, varname)

# Get `varname` from the dataset `ds` and return values
"""
function era5_driven_timeseries(ds, varname)
    ds[varname][:]
end
