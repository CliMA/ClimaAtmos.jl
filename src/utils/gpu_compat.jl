"""
    any_reltype(found, obj, name, ets, pc = (); warn = true)

Recurse through the properties of `obj` and return whether any of them is an
instance of one of the types in `ets`.

Types that are not `isbits` (`DataType`s, `String`s, closures over boxed
values, ...) cannot be moved into a GPU kernel, so this walks a cache or model
object and reports every offending field by its full property path. Used
through the `@any_reltype` macro, which supplies `found`, `name`, and `pc`.

# Arguments

  - `found`: Accumulator; pass `false` at the top level.
  - `obj`: Object whose properties are searched, recursively.
  - `name`: String used as the root of the reported property paths.
  - `ets`: Tuple of types to look for.
  - `pc = ()`: Property path accumulated so far.

# Keyword Arguments

  - `warn = true`: Emit a `@warn` for each match.
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
"""
    @any_reltype(obj, ets, warn = true)

Return whether `obj` contains, at any depth, a property that is an instance of
one of the types in the tuple `ets`, warning about each match unless
`warn = false`.

The macro exists to capture the expression text of `obj`, which is used as the
root of the reported property paths.

# Examples

```julia
@any_reltype p (DataType, UnionAll)
```
"""
macro any_reltype(obj, ets, warn = true)
    return :(any_reltype(
        false,
        $(esc(obj)),
        $(string(obj)),
        $(esc(ets));
        warn = $(esc(warn)),
    ))
end
