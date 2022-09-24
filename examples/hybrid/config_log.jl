import Logging
import ClimaComms

if haskey(ENV, "CLIMACORE_DISTRIBUTED") && ENV["CLIMACORE_DISTRIBUTED"] == "MPI"
    using ClimaCommsMPI
end
function get_comms_ctx()
    if haskey(ENV, "CLIMACORE_DISTRIBUTED")
        if ENV["CLIMACORE_DISTRIBUTED"] == "MPI"
            comms_ctx = ClimaCommsMPI.MPICommsContext()
        else
            error(
                "ENV[\"CLIMACORE_DISTRIBUTED\"] only supports the \"MPI\" option",
            )
        end
    else
        comms_ctx = ClimaComms.SingletonCommsContext()
    end
    return comms_ctx
end

if !haskey(ENV, "CLIMACORE_DISTRIBUTED")
    import TerminalLoggers
end
function get_logger(comms_ctx)
    return if haskey(ENV, "CLIMACORE_DISTRIBUTED")
        if ClimaComms.iamroot(comms_ctx)
            Logging.ConsoleLogger(stderr, Logging.Info)
        else
            Logging.NullLogger()
        end
    else
        TerminalLoggers.TerminalLogger()
    end
end
