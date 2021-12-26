"""
    AbstractFormulaFunction

Supertype for all formula functions.
"""
abstract type AbstractFormulaFunction end

"""
    cache_reqs(formula_function, vars)

Get the cache variables required by the specified formula function, given the
model's independent variables.
"""
function cache_reqs(::AbstractFormulaFunction, vars) end

"""
    (::AbstractFormulaFunction)(vars, Y, cache, consts, t)

Compute the value of the specified formula function.

If the returned value involves dotted operations, it is recommended to use
`@lazydots` to delay the evaluation of those operations until an optimal time.
"""
function (::AbstractFormulaFunction)(vars, Y, cache, consts, t) end

"""
    Formula{V <: Var, F <: AbstractFormulaFunction}

A representation of a "formula" ``
    \\text{var} = \\text{f}(vars, Y, cache, consts, t)
``.

Provides the compiler with a way to map a variable to its formula function.

If the formula requires some boundary conditions in order to be evaluated, those
boundary conditions must be handled by the formula function `f`.
"""
struct Formula{V <: Var, F <: AbstractFormulaFunction}
    var::V
    f::F
end

cache_reqs(formula::Formula, vars) = cache_reqs(formula.f, vars)
