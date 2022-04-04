using Logging
usempi = get(ENV, "CLIMAATMOS_DISTRIBUTED", "") == "MPI"
#const usempi = true
if usempi
    using ClimaComms
    using ClimaCommsMPI
    const Context = ClimaCommsMPI.MPICommsContext
    const pid, nprocs = ClimaComms.init(Context)
    if pid == 1
        println("Parallel run with $nprocs processes.")
    end
    logger_stream = ClimaComms.iamroot(Context) ? stderr : devnull
    prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
    atexit() do
        global_logger(prev_logger)
    end
else
    using Logging: global_logger
    using TerminalLoggers: TerminalLogger
    global_logger(TerminalLogger())
end
