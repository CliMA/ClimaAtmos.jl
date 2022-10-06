import ClimaComms, Logging

if !@isdefined comms_ctx # for compatibility with coupler
    if haskey(ENV, "CLIMACORE_DISTRIBUTED")
        if ENV["CLIMACORE_DISTRIBUTED"] == "MPI"
            import ClimaCommsMPI
            const comms_ctx = ClimaCommsMPI.MPICommsContext()
        else
            error(
                "ENV[\"CLIMACORE_DISTRIBUTED\"] only supports the \"MPI\" option",
            )
        end
    else
        const comms_ctx = ClimaComms.SingletonCommsContext()
    end
    ClimaComms.init(comms_ctx)

    if ClimaComms.iamroot(comms_ctx)
        Logging.global_logger(Logging.ConsoleLogger(stderr, Logging.Info))
    else
        Logging.global_logger(Logging.NullLogger())
    end
end
if comms_ctx isa ClimaComms.SingletonCommsContext
    @info "Setting up single-process ClimaAtmos run"
else
    @info "Setting up distributed ClimaAtmos run" nprocs =
        ClimaComms.nprocs(comms_ctx)
end
