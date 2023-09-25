# Writers.jl
#
# This file defines function-generating functions for output_writers for diagnostics. The
# writers come with opinionated defaults.

import ClimaCore.Remapping: interpolate_array
import NCDatasets

"""
    HDF5Writer()


Save a `ScheduledDiagnostic` to a HDF5 file inside the `output_dir` of the simulation.


TODO: This is a very barebone HDF5Writer. Do not consider this implementation as the "final
word".

We need to implement the following features/options:
- Toggle for write new files/append
- Checks for existing files
- Check for new subfolders that have to be created
- More meaningful naming conventions (keeping in mind that we can have multiple variables
  with different reductions)
- All variables in one file/each variable in its own file
- All timesteps in one file/each timestep in its own file
- Writing the correct attributes
- Overriding simulation.output_dir (e.g., if the path starts with /)
- ...more features/options

"""
function HDF5Writer()
    # output_drivers are called with the three arguments: the value, the ScheduledDiagnostic,
    # and the integrator
    function write_to_hdf5(value, diagnostic, integrator)
        var = diagnostic.variable
        time = integrator.t

        # diagnostic here is a ScheduledDiagnosticIteration. If we want to obtain a
        # descriptive name (e.g., something with "daily"), we have to pass the timestep as
        # well

        output_path = joinpath(
            integrator.p.simulation.output_dir,
            "$(diagnostic.output_short_name)_$(time).h5",
        )

        hdfwriter = InputOutput.HDF5Writer(output_path, integrator.p.comms_ctx)
        InputOutput.write!(hdfwriter, value, "$(diagnostic.output_short_name)")
        attributes = Dict(
            "time" => time,
            "long_name" => diagnostic.output_long_name,
            "variable_units" => var.units,
            "standard_variable_name" => var.standard_name,
        )

        # TODO: Use directly InputOutput functions
        InputOutput.HDF5.h5writeattr(
            hdfwriter.file.filename,
            "fields/$(diagnostic.output_short_name)",
            attributes,
        )

        Base.close(hdfwriter)
        return nothing
    end
end


"""
    NetCDFWriter()


Save a `ScheduledDiagnostic` to a NetCDF file inside the `output_dir` of the simulation by
performing a pointwise (non-conservative) remapping first. This writer does not work on
distributed simulations.


TODO: This is a very barebone NetCDFWriter. This writer only supports 3D fields on cubed
spheres at the moment. Do not consider this implementation as the "final word".

We need to implement the following features/options:
- Toggle for write new files/append
- Checks for existing files
- Check for new subfolders that have to be created
- More meaningful naming conventions (keeping in mind that we can have multiple variables
  with different reductions)
- All variables in one file/each variable in its own file
- All timesteps in one file/each timestep in its own file
- Writing the correct attributes
- Overriding simulation.output_dir (e.g., if the path starts with /)
- ...more features/options

"""
function NetCDFWriter(;
    num_points_longitude = 100,
    num_points_latitude = 100,
    num_points_altitude = 100,
    compression_level = 9,
)
    function write_to_netcdf(field, diagnostic, integrator)
        var = diagnostic.variable
        time = integrator.t

        # TODO: Automatically figure out details on the remapping depending on the given
        # field

        # Let's support only cubed spheres for the moment
        typeof(axes(field).horizontal_space.topology.mesh) <:
        Meshes.AbstractCubedSphere ||
            error("NetCDF writer supports only cubed sphere at the moment")

        # diagnostic here is a ScheduledDiagnosticIteration. If we want to obtain a
        # descriptive name (e.g., something with "daily"), we have to pass the timestep as
        # well
        output_path = joinpath(
            integrator.p.simulation.output_dir,
            "$(diagnostic.output_short_name)_$time.nc",
        )

        vert_domain = axes(field).vertical_topology.mesh.domain
        z_min, z_max = vert_domain.coord_min.z, vert_domain.coord_max.z

        FT = Spaces.undertype(axes(field))

        longpts = range(
            Geometry.LongPoint(-FT(180.0)),
            Geometry.LongPoint(FT(180.0)),
            length = num_points_longitude,
        )
        latpts = range(
            Geometry.LatPoint(-FT(80.0)),
            Geometry.LatPoint(FT(80.0)),
            length = num_points_latitude,
        )
        zpts = range(
            Geometry.ZPoint(FT(z_min)),
            Geometry.ZPoint(FT(z_max)),
            length = num_points_altitude,
        )

        nc = NCDatasets.Dataset(output_path, "c")

        NCDatasets.defDim(nc, "lon", num_points_longitude)
        NCDatasets.defDim(nc, "lat", num_points_latitude)
        NCDatasets.defDim(nc, "z", num_points_altitude)

        nc.attrib["long_name"] = diagnostic.output_long_name
        nc.attrib["units"] = var.units
        nc.attrib["comments"] = var.comments

        v = NCDatasets.defVar(
            nc,
            "$(var.short_name)",
            FT,
            ("lon", "lat", "z"),
            deflatelevel = compression_level,
        )

        v[:, :, :] = interpolate_array(field, longpts, latpts, zpts)

        close(nc)
    end
end
