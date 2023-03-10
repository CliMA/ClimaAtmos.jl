time_to_seconds(t::Number) =
    t == Inf ? t : error("Uncaught case in computing time from given string.")

function time_to_seconds(s::String)
    factor = Dict(
        "secs" => 1,
        "mins" => 60,
        "hours" => 60 * 60,
        "days" => 60 * 60 * 24,
    )
    s == "Inf" && return Inf
    if count(occursin.(keys(factor), Ref(s))) != 1
        error(
            "Bad format for flag $s. Examples: [`10secs`, `20mins`, `30hours`, `40days`]",
        )
    end
    for match in keys(factor)
        occursin(match, s) || continue
        return parse(Float64, first(split(s, match))) * factor[match]
    end
    error("Uncaught case in computing time from given string.")
end

import ClimaComms
import ClimaCore: Spaces, Topologies
import DiffEqBase
import JLD2

function export_scaling_file(sol, output_dir, walltime, comms_ctx, nprocs)
    # replace sol.u on the root processor with the global sol.u
    if ClimaComms.iamroot(comms_ctx)
        Y = sol.u[1]
        center_space = axes(Y.c)
        horz_space = ClimaCore.Spaces.horizontal_space(center_space)
        horz_topology = horz_space.topology
        Nq = ClimaCore.Spaces.Quadratures.degrees_of_freedom(
            horz_space.quadrature_style,
        )
        nlocalelems = Topologies.nlocalelems(horz_topology)
        ncols_per_process = nlocalelems * Nq * Nq
        scaling_file =
            joinpath(output_dir, "scaling_data_$(nprocs)_processes.jld2")
        @info(
            "Writing scaling data",
            "walltime (seconds)" = walltime,
            scaling_file
        )
        JLD2.jldsave(scaling_file; nprocs, ncols_per_process, walltime)
    end
    return nothing
end

function verify_callbacks(t)
    if length(t) ≠ length(unique(t))
        @show length(t)
        @show length(unique(t))
        error(
            string(
                "Saving duplicate solutions at the same time.",
                "Please change the callbacks to not save ",
                "duplicate solutions at the same timestep.",
            ),
        )
    end
end
