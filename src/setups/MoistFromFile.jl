"""
    MoistFromFile(file_path)

File-based initial condition that reads thermodynamic and kinematic state
from a NetCDF file and regrids it onto the model grid.

Assigns NaN placeholders during pointwise construction, then overwrites
the full prognostic state with data regridded from the given file via
`overwrite_from_file!`.

## Fields
- `file_path`: Path to the NetCDF file containing initial condition data.

## Expected variables in the file
- `p`: pressure (2D surface, broadcast in z)
- `t`: temperature (3D)
- `q`: specific humidity (3D)
- `u, v, w`: velocity (3D)
- `cswc, crwc`: snow and rain water content (optional)
- `z_sfc`: surface altitude (optional, for topographic pressure correction)
"""
struct MoistFromFile
    file_path::String
end

function center_initial_condition(setup::MoistFromFile, local_geometry, params)
    FT = eltype(params)
    return physical_state(; T = FT(NaN), p = FT(NaN))
end

function overwrite_initial_state!(setup::MoistFromFile, Y, thermo_params)
    return overwrite_from_file!(setup.file_path, nothing, Y, thermo_params)
end
