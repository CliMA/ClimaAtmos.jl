# Writers.jl
#
# This file defines function-generating functions for output_writers for diagnostics. The
# writers come with opinionated defaults.

"""
    get_descriptive_name(sd_t::ScheduledDiagnosticTime)


Return a compact, unique-ish, identifier for the given `ScheduledDiagnosticTime` `sd_t`.

We split the period in seconds into days, hours, minutes, seconds. In most cases
(with standard periods), this output will look something like
`air_density_1d_max`. This function can be used for filenames.

The name is not unique to the `ScheduledDiagnosticTime` because it ignores other
parameters such as whether there is a reduction in space or the compute
frequency.

"""
get_descriptive_name(sd_t::ScheduledDiagnosticTime) = get_descriptive_name(
    sd_t.variable,
    sd_t.output_every,
    sd_t.reduction_time_func;
    units_are_seconds = true,
)

"""
    get_descriptive_name(sd_i::ScheduledDiagnosticIterations[, Δt])


Return a compact, unique-ish, identifier for the given
`ScheduledDiagnosticIterations` `sd_i`.

If the timestep `Δt` is provided, convert the steps into seconds. In this case,
the output will look like `air_density_1d_max`. Otherwise, the output will look
like `air_density_100it_max`. This function can be used for filenames.

The name is not unique to the `ScheduledDiagnosticIterations` because it ignores
other parameters such as whether there is a reduction in space or the compute
frequency.

"""
get_descriptive_name(sd_i::ScheduledDiagnosticIterations, Δt::Nothing) =
    get_descriptive_name(
        sd_t.variable,
        sd_t.output_every,
        sd_t.reduction_time_func;
        units_are_seconds = false,
    )
get_descriptive_name(sd_i::ScheduledDiagnosticIterations, Δt::T) where {T} =
    get_descriptive_name(
        sd_i.variable,
        sd_i.output_every * Δt,
        sd_i.reduction_time_func;
        units_are_seconds = true,
    )


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
            "$(get_descriptive_name(diagnostic, integrator.p.simulation.dt))_$time.h5",
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
