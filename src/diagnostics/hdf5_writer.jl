import ClimaComms

##############
# HDF5Writer #
##############

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
struct HDF5Writer end


"""
    close(writer::HDF5Writer)

Close all the files open in `writer`. (Currently no-op.)
"""
close(writer::HDF5Writer) = nothing

function write_field!(
    writer::HDF5Writer,
    field,
    diagnostic,
    integrator,
    output_dir,
)
    var = diagnostic.variable
    time = integrator.t

    output_path =
        joinpath(output_dir, "$(diagnostic.output_short_name)_$(time).h5")

    comms_ctx = ClimaComms.context(integrator.u.c)
    hdfwriter = InputOutput.HDF5Writer(output_path, comms_ctx)
    InputOutput.write!(hdfwriter, field, "$(diagnostic.output_short_name)")
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
