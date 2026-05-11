import ClimaCore

const _infiltrator_loaded = try
    @eval import Infiltrator
    true
catch
    false
end

# SOCRATES-local optional ClimaCore post-op nonfinite debugging.
# Enabled only when CLIMA_DEBUG_POST_OP_NONFINITE is truthy.
const _post_op_nonfinite_enabled = Ref(false)
const _post_op_nonfinite_hits = Ref(0)
const _post_op_nonfinite_configured = Ref(false)

_debug_env_bool(name::String, default::Bool = false) =
    get(ENV, name, default ? "1" : "0") in ("1", "true", "TRUE", "yes", "YES", "on", "ON")

_debug_env_int(name::String, default::Int) = try
    parse(Int, get(ENV, name, string(default)))
catch
    default
end

_has_nonfinite_number(x::Number) = !isfinite(x)
_has_nonfinite_number(::Any) = false

function _has_nonfinite_data(x)
    if x isa Number
        return _has_nonfinite_number(x)
    end
    try
        return any(_has_nonfinite_number, parent(x))
    catch
        return false
    end
end

function _structured_repr(x)
    if isdefined(ClimaCore.DebugOnly, :recursive_make_named_tuple)
        try
            return sprint(show, MIME("text/plain"), ClimaCore.DebugOnly.recursive_make_named_tuple(x))
        catch
        end
    end
    return sprint(show, MIME("text/plain"), x)
end

function _write_nonfinite_dump(path::String, hit::Int, result, args, kwargs, st)
    open(path, "w") do io
        println(io, "ClimaCore post-op nonfinite detection")
        println(io, "hit = $hit")
        println(io, "result_type = $(typeof(result))")
        println(io, "args_length = $(length(args))")
        println(io)

        println(io, "[result]")
        println(io, _structured_repr(result))
        println(io)

        println(io, "[args]")
        println(io, _structured_repr(args))
        println(io)

        println(io, "[kwargs]")
        println(io, _structured_repr(kwargs))
        println(io)

        println(io, "[stacktrace]")
        for frame in st
            println(io, frame)
        end
    end
end

if _infiltrator_loaded
    function _infiltrate_nonfinite(result, args, kwargs, st)
        Infiltrator.@infiltrate
        return nothing
    end

    function _exfiltrate_nonfinite(result, args, kwargs, st)
        Infiltrator.@exfiltrate
        return nothing
    end
else
    _infiltrate_nonfinite(result, args, kwargs, st) = nothing
    _exfiltrate_nonfinite(result, args, kwargs, st) = nothing
end

function setup_climacore_post_op_nonfinite_debug!()
    if _post_op_nonfinite_configured[]
        return nothing
    end
    _post_op_nonfinite_configured[] = true

    _post_op_nonfinite_enabled[] = _debug_env_bool("CLIMA_DEBUG_POST_OP_NONFINITE", false)
    if _post_op_nonfinite_enabled[]
        @info "SOCRATES ClimaCore post-op nonfinite debug enabled" max_hits = _debug_env_int(
            "CLIMA_DEBUG_POST_OP_MAX_HITS",
            5,
        ) dump_dir = get(ENV, "CLIMA_DEBUG_POST_OP_DUMP_DIR", "") infiltrator_loaded = _infiltrator_loaded
        if (_debug_env_bool("CLIMA_DEBUG_POST_OP_INFILTRATE", false) ||
            _debug_env_bool("CLIMA_DEBUG_POST_OP_EXFILTRATE", false)) &&
           !_infiltrator_loaded
            @warn "Infiltrator requested via env flags but package is unavailable in this environment"
        end
    end
    return nothing
end

ClimaCore.DebugOnly.call_post_op_callback() = _post_op_nonfinite_enabled[]

function ClimaCore.DebugOnly.post_op_callback(result, args...; kwargs...)
    _post_op_nonfinite_enabled[] || return nothing
    _has_nonfinite_data(result) || return nothing

    _post_op_nonfinite_hits[] += 1
    hit = _post_op_nonfinite_hits[]
    max_hits = _debug_env_int("CLIMA_DEBUG_POST_OP_MAX_HITS", 5)
    hit <= max_hits || return nothing

    @warn "ClimaCore post-op callback detected nonfinite result" hit max_hits result_type = typeof(result)

    dump_dir = get(ENV, "CLIMA_DEBUG_POST_OP_DUMP_DIR", "")
    if !isempty(dump_dir)
        mkpath(dump_dir)
        dump_path = joinpath(dump_dir, "post_op_nonfinite_$(lpad(hit, 4, '0')).txt")
        st = stacktrace()
        _write_nonfinite_dump(dump_path, hit, result, args, kwargs, st)
    end

    if _debug_env_bool("CLIMA_DEBUG_POST_OP_EXFILTRATE", false)
        st = stacktrace()
        _exfiltrate_nonfinite(result, args, kwargs, st)
    end

    if _debug_env_bool("CLIMA_DEBUG_POST_OP_INFILTRATE", false)
        st = stacktrace()
        _infiltrate_nonfinite(result, args, kwargs, st)
    end

    if _debug_env_bool("CLIMA_DEBUG_POST_OP_THROW", false)
        error("ClimaCore post-op callback detected nonfinite result (hit=$hit)")
    end
    return nothing
end
