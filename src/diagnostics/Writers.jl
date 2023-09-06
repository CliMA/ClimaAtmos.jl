# Writers.jl
#
# This file defines function-generating functions for output_writers for diagnostics. The
# writers come with opinionated defaults.


"""
    descriptive_name(sd_t::ScheduledDiagnosticTime)


Return a compact, unique-ish, identifier for the given `ScheduledDiagnosticTime` `sd_t`.

We split the period in seconds into days, hours, minutes, seconds. In most cases
(with standard periods), this output will look something like
`air_density_1d_max`. This function can be used for filenames.

The name is not unique to the `ScheduledDiagnosticTime` because it ignores other
parameters such as whether there is a reduction in space or the compute
frequency.

"""
function descriptive_name(sd_t::ScheduledDiagnosticTime)
    var = "$(sd_t.variable.short_name)"

    isa_reduction = !isnothing(sd_t.reduction_time_func)

    if isa_reduction
        red = "$(sd_t.reduction_time_func)"

        # Convert period from seconds to days, hours, minutes, seconds
        period = ""

        days, rem_seconds = divrem(sd_t.output_every, 24 * 60 * 60)
        hours, rem_seconds = divrem(rem_seconds, 60 * 60)
        minutes, seconds = divrem(rem_seconds, 60)

        if days > 0
            period *= "$(days)d_"
        end
        if hours > 0
            period *= "$(hours)h_"
        end
        if minutes > 0
            period *= "$(minutes)m_"
        end
        if seconds > 0
            period *= "$(seconds)s_"
        end

        suffix = period * red
    else
        # Not a reduction
        suffix = "inst"
    end
    return "$(var)_$(suffix)"
end

"""
    descriptive_name(sd_i::ScheduledDiagnosticIterations, [Δt])


Return a compact, unique-ish, identifier for the given
`ScheduledDiagnosticIterations` `sd_i`.

If the timestep `Δt` is provided, convert the steps into seconds. In this case,
the output will look like `air_density_1d_max`. Otherwise, the output will look
like `air_density_100it_max`. This function can be used for filenames.

The name is not unique to the `ScheduledDiagnosticIterations` because it ignores
other parameters such as whether there is a reduction in space or the compute
frequency.

"""
function descriptive_name(sd_i::ScheduledDiagnosticIterations,
                          Δt = nothing)

    if !isnothing(Δt)
        # Convert iterations into time
        return descriptive_name(ScheduledDiagnosticTime(sd_i, Δt))
    else
        var = "$(sd_i.variable.short_name)"
        suffix =
            isnothing(sd_i.reduction_time_func) ? "inst" :
            "$(sd_i.output_every)it_(sd_i.reduction_time_func)"
        return "$(var)_$(suffix)"
    end
end


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
        time = lpad(integrator.t, 10, "0")
        red = isnothing(diagnostic.reduction_time_func) ? "" : diagnostic.reduction_time_func

        output_path = joinpath(
            integrator.p.simulation.output_dir,
            "$(var.short_name)_$(red)_$(time).h5",
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
