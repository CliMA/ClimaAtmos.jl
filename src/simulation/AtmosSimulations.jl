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
