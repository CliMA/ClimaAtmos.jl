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
import DiffEqBase
import JLD2

function export_scaling_file(sol, output_dir, walltime, comms_ctx, nprocs)
    # replace sol.u on the root processor with the global sol.u
    if ClimaComms.iamroot(comms_ctx)
        Y = sol.u[1]
        center_space = axes(Y.c)
        face_space = axes(Y.f)
        Yc_type =
            Fields.Field{typeof(Fields.field_values(Y.c)), typeof(center_space)}
        Yf_type =
            Fields.Field{typeof(Fields.field_values(Y.f)), typeof(face_space)}
        Y_type = Fields.FieldVector{
            FT,
            NamedTuple{(:c, :f), Tuple{Yc_type, Yf_type}},
        }
        global_sol_u = similar(sol.u, Y_type)
    end
    for i in 1:length(sol.u)
        global_Y_c =
            DataLayouts.gather(comms_ctx, Fields.field_values(sol.u[i].c))
        global_Y_f =
            DataLayouts.gather(comms_ctx, Fields.field_values(sol.u[i].f))
        if ClimaComms.iamroot(comms_ctx)
            global_sol_u[i] = Fields.FieldVector(
                c = Fields.Field(global_Y_c, center_space),
                f = Fields.Field(global_Y_f, face_space),
            )
        end
    end

    if ClimaComms.iamroot(comms_ctx)
        sol = DiffEqBase.sensitivity_solution(sol, global_sol_u, sol.t)
        @info "walltime = $walltime (seconds)"
        scaling_file =
            joinpath(output_dir, "scaling_data_$(nprocs)_processes.jld2")
        @info "writing performance data to $scaling_file"
        JLD2.jldsave(scaling_file; nprocs, walltime)
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
