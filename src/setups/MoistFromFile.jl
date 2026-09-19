"""
    MoistFromFile(file_path)

File-based initial condition that reads thermodynamic and kinematic state
from a NetCDF file and regrids it onto the model grid.

Assigns NaN placeholders during pointwise construction, then overwrites
the full prognostic state with data regridded from the given file via
`overwrite_from_file!`.

# Fields

  - `file_path`: Path to the NetCDF file holding the initial condition.

# Notes

The file is expected to carry:

  - `p`: Surface pressure, 2D and broadcast in `z` [Pa].
  - `t`: Temperature, 3D [K].
  - `q`: Specific humidity, 3D [kg/kg].
  - `u`, `v`, `w`: Velocity components, 3D [m/s].
  - `cswc`, `crwc`: Snow and rain water contents, optional [kg/kg].
  - `z_sfc`: Surface altitude, optional; enables the topographic pressure
    correction [m].

A second field, `hydrostatic_rebalance`, selects the discrete rebalance of the
column pressure against the model's vertical PGF operator (see
`rebalance_hydrostatic_pressure`).
"""
struct MoistFromFile
    file_path::String
    hydrostatic_rebalance::Bool
end

MoistFromFile(file_path::String) = MoistFromFile(file_path, false)

function center_initial_condition(setup::MoistFromFile, local_geometry, params)
    FT = eltype(params)
    return physical_state(; T = FT(NaN), p = FT(NaN))
end

function overwrite_initial_state!(setup::MoistFromFile, Y, thermo_params)
    return overwrite_from_file!(
        setup.file_path, nothing, Y, thermo_params;
        hydrostatic_rebalance = setup.hydrostatic_rebalance,
    )
end
