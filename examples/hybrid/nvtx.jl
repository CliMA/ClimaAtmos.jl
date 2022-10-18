using NVTX, Colors

if NVTX.isactive()
    NVTX.enable_gc_hooks()
    const nvtx_domain = NVTX.Domain("ClimaAtmos")
    # makes output on buildkite a bit nicer
    if ClimaComms.iamroot(comms_ctx)
        atexit() do
            println("--- Saving profiler information")
        end
    end
end
macro nvtx(message, args...)
    block = args[end]
    kwargs = [Expr(:kw, arg.args...) for arg in args[1:(end - 1)]]
    quote
        range =
            NVTX.isactive() ?
            NVTX.range_start(nvtx_domain; message = $message, $(kwargs...)) :
            nothing
        $(esc(block))
        if !isnothing(range)
            NVTX.range_end(range)
        end
    end
end
