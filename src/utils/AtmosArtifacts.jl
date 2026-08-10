"""
    AtmosArtifacts

Paths to the input datasets that ClimaAtmos reads from CliMA artifacts.

Each function returns the path of one file (or directory) inside its artifact,
downloading the artifact on first use. Several datasets ship in a high- and a
low-resolution version; `res_file_path` prefers the high-resolution one and
falls back to the low-resolution one, which can always be downloaded.

All functions take an optional `context` keyword, the `ClimaComms` context,
which lazy artifacts need in MPI runs so that only one rank downloads.
"""
module AtmosArtifacts

import Artifacts
import LazyArtifacts
import ClimaUtilities.ClimaArtifacts: @clima_artifact

# There seems to be no easy way to determine if an artifact exists from the name
# only...
"""
    _artifact_exists(name)

Return whether an artifact named `name` is available, by trying to resolve it
and catching the failure.
"""
function _artifact_exists(name)
    return try
        Artifacts.@artifact_str(name)
        true
    catch error
        false
    end
end


"""
    res_file_path(name; context = nothing)

Construct the path of the NetCDF file `<name>.nc` inside the artifact `name`,
falling back to `<name>_lowres` when `name` is unavailable.

The high-resolution artifact is assumed to be present locally but not
downloadable, while the low-resolution one can always be downloaded; the
fallback warns once per artifact name.

# Arguments

  - `name`: Artifact name; also the basename of the NetCDF file it contains.

# Keyword Arguments

  - `context = nothing`: `ClimaComms` context, needed for lazy artifacts in MPI
    runs.

# Returns

The full path to the NetCDF file.
"""
function res_file_path(name; context = nothing)
    if _artifact_exists(name)
        full_name = name
    else
        @warn "Higher resolution $name is not available. Using low-res version. Consult ClimaArtifacts to acquire the higher resolution version." _id =
            Symbol(name) maxlog = 1
        full_name = "$(name)_lowres"
    end
    return joinpath(@clima_artifact(full_name, context), "$(full_name).nc")
end

"""
    ozone_concentration_file_path(; context = nothing)

Construct the path of the ozone-concentration NetCDF file.

Uses the high-resolution `ozone_concentrations` artifact when available and the
low-resolution one otherwise.
"""
function ozone_concentration_file_path(; context = nothing)
    return res_file_path("ozone_concentrations"; context)
end

"""
    aerosol_concentration_file_path(; context = nothing)

Construct the path of the MERRA-2 aerosol-concentration NetCDF file.

Uses the high-resolution `merra2_aerosols` artifact when available and the
low-resolution one otherwise.
"""
function aerosol_concentration_file_path(; context = nothing)
    return res_file_path("merra2_aerosols"; context)
end

"""
    era5_cloud_file_path(; context = nothing)

Construct the path of the ERA5 cloud-properties NetCDF file.

Uses the high-resolution `era5_cloud` artifact when available and the
low-resolution one otherwise.
"""
function era5_cloud_file_path(; context = nothing)
    return res_file_path("era5_cloud"; context)
end

"""
    earth_orography_file_path(; context = nothing)

Construct the path of the 60 arc-second ETOPO 2022 surface-orography NetCDF
file.
"""
function earth_orography_file_path(; context = nothing)
    filename = "ETOPO_2022_v1_60s_N90W180_surface.nc"
    return joinpath(
        @clima_artifact("earth_orography_60arcseconds", context),
        filename,
    )
end

"""
    earth_orography_30arcsecond_file_path(; context = nothing)

Construct the path of the 30 arc-second ETOPO 2022 surface-orography NetCDF
file.
"""
function earth_orography_30arcsecond_file_path(; context = nothing)
    filename = "ETOPO_2022_v1_30s_N90W180_surface.nc"
    return joinpath(
        @clima_artifact("earth_orography_30arcseconds", context),
        filename,
    )
end

"""
    co2_concentration_file_path(; context = nothing)

Construct the path of the Mauna Loa monthly CO2-concentration text file.
"""
function co2_concentration_file_path(; context = nothing)
    return joinpath(@clima_artifact("co2_dataset", context), "co2_mm_mlo.txt")
end

"""
    ogw_computed_drag_file_path(; h_elem::Int, context = nothing)

Construct the path of the precomputed orographic-gravity-wave drag HDF5 file
for a given horizontal resolution.

The file holds the drag tensor fields `t11`, `t12`, `t21`, `t22` and the
mountain heights `hmax` and `hmin`.

# Keyword Arguments

  - `h_elem`: Number of horizontal elements per cubed-sphere panel edge; artifacts
    exist for 6, 8, 12, and 16.
  - `context = nothing`: `ClimaComms` context.
"""
function ogw_computed_drag_file_path(; h_elem::Int, context = nothing)
    artifact_name = "ogw_computed_drag_h$(h_elem)"
    filename = "computed_drag_Earth_false_1_$(h_elem).hdf5"
    return joinpath(@clima_artifact(artifact_name, context), filename)
end

# ARM SGP VARANAL single-column forcing and validation obs (see README_armvaranal.md)
const ARM_SGP_VARANAL_FORCING_FILENAME = "sgp60varanarucC1.c1.20100901.000000.cdf"

"""
    arm_sgp_varanal_forcing_file_path(; context = nothing)

Construct the path of the default ARM VARANAL monthly forcing file, from the
`arm_sgp_varanal_forcing` artifact.

This is the SGP site for September 2010, used by
`prognostic_edmfx_armvaranal_column.yml`.
"""
function arm_sgp_varanal_forcing_file_path(; context = nothing)
    return joinpath(
        @clima_artifact("arm_sgp_varanal_forcing", context),
        ARM_SGP_VARANAL_FORCING_FILENAME,
    )
end

const _ARM_VARANAL_OBS_PRODUCT_DIRS = Dict(
    "sonde" => "sgpinterpolatedsondeC1.c1",
    "beatm" => "sgparmbeatmC1.c1",
    "cldrad" => "sgparmbecldradC1.c1",
)

"""
    arm_sgp_varanal_obs_dir(product; context = nothing)

Construct the root directory of an ARM observation product, from the
`arm_sgp_varanal_obs` artifact.

The artifact holds one subdirectory per ARM product, e.g.
`sgpinterpolatedsondeC1.c1/`. Used by `plot_varanal.jl`. Throws for an unknown
product.

# Arguments

  - `product`: One of `"sonde"`, `"beatm"`, or `"cldrad"`.
"""
function arm_sgp_varanal_obs_dir(product::AbstractString; context = nothing)
    subdir = get(_ARM_VARANAL_OBS_PRODUCT_DIRS, product, nothing)
    isnothing(subdir) &&
        error("Unknown ARM VARANAL obs product `$product`")
    return joinpath(@clima_artifact("arm_sgp_varanal_obs", context), subdir)
end

end
