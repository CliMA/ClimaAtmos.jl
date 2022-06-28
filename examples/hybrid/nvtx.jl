using NVTX, Colors

if NVTX.isactive()
    nvtx_domain = NVTX.Domain("ClimaAtmos")
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
