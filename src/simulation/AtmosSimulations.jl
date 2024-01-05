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

function Base.show(io::IO, sim::AtmosSimulation)
    return print(
        io,
        "Simulation $(sim.job_id)\n",
        "├── Output folder: $(sim.output_dir)\n",
        "├── Start date: $(sim.start_date)\n",
        "├── Current time: $(sim.integrator.t) seconds\n",
        "└── Stop time: $(sim.t_end) seconds",
    )
end
