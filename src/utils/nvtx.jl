import ClimaComms
import NVTX
using Colors

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
