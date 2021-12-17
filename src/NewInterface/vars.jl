"""
    Var{symbols}

A representation of a "variable", which contains the tuple of symbols used to
access that variable.

The symbols are stored as type parameters rather than fields in order to make
them known to the compiler. This ensures that using `Base.getindex` with a `Var`
will be type-stable; i.e., if the compiler knows the types of `object` and
`var`, then it will be able to infer the type of `object[var]`.

```@repl
Var(:ρ)
Var(:atmos, :f, :ρw)
object = (atmos = (c = (ρ = 1, ρθ = 2, ρu = 3), f = (ρw = 4,)), land = ())
object[Var(:atmos, :f, :ρw)]
```
"""
struct Var{symbols} end # Like Val, but for variables!
Var(symbols::Symbol...) = Var{symbols}()

"""
    symbols(var)

Extract the tuple of symbols stored in a variable.
"""
symbols(::Var{symbols}) where {symbols} = symbols


function Base.getindex(object, ::Var{symbols}) where {symbols}
    if @generated
        expr = :(object)
        for symbol in symbols
            expr = :($expr.$symbol)
        end
        return expr
    else
        result = object
        for symbol in symbols
            result = getproperty(result, symbol)
        end
        return result
    end
end