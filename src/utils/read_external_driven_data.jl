# Utilities for reading NetCDF data for GCM- and ERA5-driven single-column simulations.
import NCDatasets as NC
import Statistics: mean

"""
    gcm_driven_reference(ds, varname)

Extract `height` from the GCM-driven dataset `ds`

Returns a 1D ("z",) `vec` object.

!!! note

    This method currently assumes the underlying data is `Float64`. 
    If this is not the case, "garbage" data may be returned with no warning.
"""

function era5_height(ds)
    vec(mean(ds["z"][:], dims = 2))
end

function gcm_height(ds)
    vec(mean(ds["zg"][:, :], dims = 2))
end


"""
    gcm_driven_profile_tmean(ds, varname)

Extract time-averaged data for `varname` from the "profile" group in the GCM-driven dataset `ds`

Returns a 1D ("z",) `Vector{FT}` of the time-averaged data.

!!! note

    This method currently assumes the underlying data is `Float64`. 
    If this is not the case, "garbage" data may be returned with no warning.
"""
function gcm_driven_profile_tmean(ds, varname)
    vec(mean(ds[varname][:, :], dims = 2))
end

"""
    era5_driven_profile_tmean(ds, varname)

Extract time-averaged data for `varname` from the "profile" group in the ERA5-driven dataset `ds`

Returns a 1D ("z",) `Vector{FT}` of the time-averaged data.

!!! note

    This method currently assumes the underlying data is `Float64`. 
    If this is not the case, "garbage" data may be returned with no warning.
"""

function era5_driven_profile_tmean(ds, varname)
    ds[varname][:]
end


"""
    {gcm,era5}_driven_rho_profile_tmean(ds)

Extract time-averaged density profile, ρ, from the GCM- or ERA5-driven dataset `ds`

Returns a 1D ("z",) `Vector{FT}` of the time-averaged ρ.
    
"""
function gcm_driven_rho_profile_tmean(ds)
    vec(mean(1 ./ ds["alpha"][:, :], dims = 2)) # convert alpha to rho using rho=1/alpha, take average profile
end

function era5_driven_rho_profile_tmean(ds)
    ds["rho"][:] # for ERA5 rho is already available and is stored as a time average already
end

"""
    external_driven_timeseries(ds, varname)

Get `varname` from the dataset `ds` and return values

!!! note

    This method currently assumes the underlying data is `Float64`. 
    If this is not the case, "garbage" data may be returned with no warning.
"""
function external_driven_timeseries(ds, varname)
    ds[varname][:]
end
