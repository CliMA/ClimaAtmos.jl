import ClimaCore: Spaces, Topologies
import JLD2

function export_scaling_file(sol, output_dir, walltime, comms_ctx, nprocs)
    # replace sol.u on the root processor with the global sol.u
    if ClimaComms.iamroot(comms_ctx)
        Y = sol.u[1]
        center_space = axes(Y.c)
        horz_space = Spaces.horizontal_space(center_space)
        horz_topology = horz_space.topology
        Nq = Spaces.Quadratures.degrees_of_freedom(horz_space.quadrature_style)
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

#TODO - do we want to change anything here now?
is_baro_wave(parsed_args) = all((
    parsed_args["config"] == "sphere",
    parsed_args["forcing"] == nothing,
    parsed_args["surface_setup"] == nothing,
    parsed_args["perturb_initstate"] == true,
))

is_solid_body(parsed_args) = all((
    parsed_args["config"] == "sphere",
    parsed_args["forcing"] == nothing,
    parsed_args["rad"] == nothing,
    parsed_args["perturb_initstate"] == false,
))

is_column_without_edmfx(parsed_args) = all((
    parsed_args["config"] == "column",
    parsed_args["turbconv"] == nothing,
    parsed_args["forcing"] == nothing,
    parsed_args["turbconv"] == nothing,
))
