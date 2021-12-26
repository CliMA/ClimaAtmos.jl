"""
    Var{symbs}

A representation of a "variable", which contains the tuple of symbols used to
access that variable.

The symbols are stored as type parameters rather than fields in order to make
them known to the compiler. This ensures that using `get_var` will be
type-stable; i.e., if the compiler knows the types of `object` and `var`, then
it will be able to infer the type of `get_var(object, var)`.

```@repl
var = Var(:atmos, :f, :ρw)
symbols(var)
object = (atmos = (c = (ρ = 1, ρθ = 2, ρu = 3), f = (ρw = 4,)), land = ())
get_var(object, var)
```
"""
struct Var{symbs} end # Like Val, but for variables!

Var(symbs::Symbol...) = Var{symbs}()

"""
    symbols(var)

Get the tuple of symbols stored in the variable.
"""
symbols(::Var{symbs}) where {symbs} = symbs


"""
    get_var(object, var)

Get the component of the object specified by the variable.
"""
function get_var(object, ::Var{symbs}) where {symbs}
    if @generated
        expr = :(object)
        for symb in symbs
            expr = :($expr.$symb)
        end
        return expr
    else
        result = object
        for symb in symbs
            result = getproperty(result, symb)
        end
        return result
    end
end

function Base.show(io::IO, var::Var)
    print(io, "*.")
    join(io, symbols(var), '.')
end

"""
    Vars{V <: NTuple{N, Var} where {N}}

A simple wrapper for a tuple of `Var` objects which offers a type-stable method
for `in`.

For any variable `var` and any collection of variables `vars`, the compiler will
be able to infer whether `var ∈ vars` or `var ∉ vars`. This is achieved by
checking for physical equality (`===`) rather than generic equality (`==`) when
comparing `var` to each element of `vars`.

```@repl
var = Var(:f, :ρw)
vars = Vars((Var(:c, :ρ), Var(:c, :ρθ), Var(:f, :ρw)))
var ∈ vars
```
"""
struct Vars{V <: NTuple{N, Var} where {N}}
    vars::V
end

Base.in(var::Var, vars::Vars) = var_in(var, vars.vars...)
var_in(var) = false
var_in(var, var′, vars...) = var === var′ || var_in(var, vars...)
