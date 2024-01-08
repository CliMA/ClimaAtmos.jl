struct AtmosSimulation{
    FT <: AbstractFloat,
    S1 <: AbstractString,
    S2 <: AbstractString,
    OW,
    OD,
}
    job_id::S1
    output_dir::S2
    start_date::DateTime
    t_end::FT
    output_writers::OW
    integrator::OD
end

ClimaComms.context(sim::AtmosSimulation) =
    ClimaComms.context(sim.integrator.u.c)
ClimaComms.device(sim::AtmosSimulation) = ClimaComms.device(sim.integrator.u.c)

function Base.show(io::IO, sim::AtmosSimulation)
    device_type = nameof(typeof(ClimaComms.device(sim)))
    return print(
        io,
        "Simulation $(sim.job_id)\n",
        "├── Running on: $(device_type)\n",
        "├── Output folder: $(sim.output_dir)\n",
        "├── Start date: $(sim.start_date)\n",
        "├── Current time: $(sim.integrator.t) seconds\n",
        "└── Stop time: $(sim.t_end) seconds",
    )
end
