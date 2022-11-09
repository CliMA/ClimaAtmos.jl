using NVTX, Colors

if NVTX.isactive()
    NVTX.enable_gc_hooks()
    # makes output on buildkite a bit nicer
    if ClimaComms.iamroot(comms_ctx)
        atexit() do
            println("--- Saving profiler information")
        end
    end
end
# for compatibility until we move everything into src:
macro nvtx(message, args...)
    expr = args[end]
    args = args[1:(end - 1)]
    quote
        NVTX.@range(
            $message,
            domain = NVTX.Domain(ClimaAtmos),
            $(args...),
            $(esc(expr))
        )
    end
end
