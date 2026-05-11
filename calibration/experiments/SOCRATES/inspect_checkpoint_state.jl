import ClimaAtmos as CA
import ClimaComms

function first_bad_summary(array)
    idx = findfirst(x -> !isfinite(x), array)
    return isnothing(idx) ? nothing : (idx = idx, value = array[idx])
end

function scan_value(x, path::String, findings)
    length(findings) >= 20 && return

    if x isa Number
        isfinite(x) || push!(findings, (path = path, detail = "scalar=$(x)"))
        return
    end

    if x isa NamedTuple
        for name in propertynames(x)
            scan_value(getproperty(x, name), string(path, ".", name), findings)
        end
        return
    end

    names = propertynames(x)
    if !isempty(names)
        for name in names
            scan_value(getproperty(x, name), string(path, ".", name), findings)
        end
        return
    end

    if x isa AbstractArray
        if eltype(x) <: Number
            bad = first_bad_summary(x)
            isnothing(bad) || push!(
                findings,
                (path = path, detail = "size=$(size(x)) first_bad_idx=$(bad.idx) value=$(bad.value)"),
            )
        else
            for idx in eachindex(x)
                scan_value(x[idx], string(path, "[", idx, "]"), findings)
                length(findings) >= 20 && return
            end
        end
        return
    end

    if applicable(parent, x)
        raw = parent(x)
        scan_value(raw, string(path, ".parent"), findings)
    end
end

function inspect_checkpoint(path::String)
    reader = CA.InputOutput.HDF5Reader(path, ClimaComms.SingletonCommsContext())
    Y = CA.InputOutput.read_field(reader, "Y")
    findings = NamedTuple[]
    scan_value(Y, "Y", findings)
    close(reader)

    println("Checkpoint: ", path)
    if isempty(findings)
        println("No non-finite values found in saved state.")
    else
        println("Non-finite state components:")
        for item in findings
            println("  ", item.path, " -- ", item.detail)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    checkpoint_path = length(ARGS) >= 1 ? ARGS[1] : error("Usage: julia inspect_checkpoint_state.jl <checkpoint.hdf5>")
    inspect_checkpoint(checkpoint_path)
end
