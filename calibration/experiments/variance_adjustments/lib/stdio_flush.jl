# Aggressive stdio flushing when logs are redirected to a file (`> log 2>&1`), where libc often
# fully buffers. Include early in CLI drivers, then call [`va_setup_stdio_flushing!`](@ref).
import Logging

"""Wrap a logger so every `handle_message` ends with `flush(stderr)` (where `@info` / `@warn` go)."""
struct FlushingLogger{T <: Logging.AbstractLogger} <: Logging.AbstractLogger
    inner::T
end

Logging.min_enabled_level(l::FlushingLogger) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::FlushingLogger, level, _module, group, id) =
    Logging.shouldlog(l.inner, level, _module, group, id)
Logging.catch_exceptions(l::FlushingLogger) = Logging.catch_exceptions(l.inner)

function Logging.handle_message(
    l::FlushingLogger,
    level,
    message,
    _module,
    group,
    id,
    filepath,
    line;
    kwargs...,
)
    Logging.handle_message(l.inner, level, message, _module, group, id, filepath, line; kwargs...)
    flush(stderr)
    return nothing
end

"""
    va_setup_stdio_flushing!()

Install [`FlushingLogger`](@ref) around the current global logger (idempotent). Safe to call in
every subprocess (`run_full_study`, forward sweep, EKI slice, etc.).
"""
function va_setup_stdio_flushing!()
    lg = Logging.global_logger()
    lg isa FlushingLogger && return nothing
    Logging.global_logger(FlushingLogger(lg))
    return nothing
end

"""Flush both standard streams (e.g. after external `run` or heavy stdout from dependencies)."""
function va_flush_stdio()
    flush(stdout)
    flush(stderr)
    return nothing
end
