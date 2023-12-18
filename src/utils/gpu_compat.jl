"""
    @any_reltype(::Any, t::Tuple, warn=true)

Returns a Bool (and prints warnings) if the given
data structure has an instance of any types in `t`.
"""
function any_reltype(found, obj, name, ets, pc = (); warn = true)
    for pn in propertynames(obj)
        prop = getproperty(obj, pn)
        pc_full = (pc..., ".", pn)
        pc_string = name * string(join(pc_full))
        for et in ets
            if prop isa et
                warn && @warn "$pc_string::$(typeof(prop)) is a DataType"
                found = true
            end
        end
        found = found || any_reltype(found, prop, name, ets, pc_full; warn)
    end
    return found
end
macro any_reltype(obj, ets, warn = true)
    return :(any_reltype(
        false,
        $(esc(obj)),
        $(string(obj)),
        $(esc(ets));
        warn = $(esc(warn)),
    ))
end
