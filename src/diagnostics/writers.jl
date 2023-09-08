# Writers.jl
#
# This file defines function-generating functions for output_writers for diagnostics. The
# writers come with opinionated defaults.

"""
    HDF5Writer()


Save a `ScheduledDiagnostic` to a HDF5 file inside the `output_dir` of the simulation.


TODO: This is a very barebone HDF5Writer.

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
            "$(diagnostic.name)_$time.h5",
        )

        hdfwriter = InputOutput.HDF5Writer(output_path, integrator.p.comms_ctx)
        InputOutput.HDF5.write_attribute(hdfwriter.file, "time", time)
        InputOutput.HDF5.write_attribute(
            hdfwriter.file,
            "long_name",
            var.long_name,
        )
        InputOutput.write!(
            hdfwriter,
            Fields.FieldVector(; Symbol(var.short_name) => value),
            "diagnostics",
        )
        Base.close(hdfwriter)
        return nothing
    end
end
