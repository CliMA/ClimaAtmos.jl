# Utilities for reading NetCDF data for GCM-driven single-column simulations.
import NCDatasets as NC
import StatsBase: mean

read_gcm_driven_initial_profile_mean(FT, ds, group, varname; imin = 100) = vec(
    mean(
        _gcm_driven_variable_zt(FT, ds, group, varname)[:, imin:end],
        dims = 2,
    ),
)

# TODO: Cast to CuVector for GPU compatibility
read_gcm_driven_initial_profile(FT, ds, group, varname) =
    _gcm_driven_variable_zt(FT, ds, group, varname)[:, 1]  # 1 is initial time index

read_gcm_driven_reference_profile(FT, ds, group, varname) =
    _gcm_driven_variable_z(FT, ds, group, varname)[:]

_gcm_driven_nz(ds) = ds.group["reference"].dim["z"]

_gcm_driven_variable_zt(FT, ds, group, varname) =
    _gcm_driven_variable(FT, ds, group, varname, ("z", "t"))

_gcm_driven_variable_z(FT, ds, group, varname) =
    _gcm_driven_variable(FT, ds, group, varname, ("z",))


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

function _gcm_driven_variable(
    FT,
    ds,
    group,
    varname,
    dimnames::NTuple{N, String},
) where {N}
    grp = ds.group[group]
    _gcm_driven_variable(FT, grp, varname, dimnames)
end
