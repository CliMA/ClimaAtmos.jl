using UnPack
using ClimaCore: Spaces
using OrdinaryDiffEq: ODEFunction

"""
    Model{
        T <: NTuple{N₁, Tendency} where {N₁},
        F <: NTuple{N₂, Formula} where {N₂},
        D <: NTuple{N₃, Formula} where {N₃},
    }

A representation of a "model".

Contains tendencies for evolving the independent variables over time, formulas
for updating the cached values required by the tendencies, and additional
formulas for diagnostics or debugging. Since cached values may depend on other
cached values, the formulas are stored in the order in which they get evaluated.
"""
struct Model{
    T <: NTuple{N₁, Tendency} where {N₁},
    F <: NTuple{N₂, Formula} where {N₂},
    D <: NTuple{N₃, Formula} where {N₃},
}
    tendencies::T
    formulas::F
    diagnostics_formulas::D
end

"""
    Model(
        FunctionConstructor;
        [tendencies],
        [custom_formulas],
        [diagnostics_formulas],
    )

Alternative constructor for a `Model` that allows users to avoid specifying all
of the model's required formulas and ensuring that those formulas are evaluated
in a valid order.

Accepts a type argument `FunctionConstructor <: AbstractFormulaFunction` and
automatically generates a formula for each cache variable `var` by calling
`FunctionConstructor(var)`. If the automatically generated formula for a
variable should be overwritten, or if `FunctionConstructor` is not defined for a
variable, a formula for that variable must be provided in `custom_formulas`.
"""
function Model(
    ::Type{FunctionConstructor};
    tendencies = (),
    custom_formulas = (),
    diagnostics_formulas = (),
) where {FunctionConstructor}
    formulas = sorted_cache_formulas(
        tendencies,
        FunctionConstructor,
        custom_formulas,
        diagnostics_formulas,
    )
    return Model(tendencies, formulas, diagnostics_formulas)
end

"""
    variables(model)

Get the independent variables of the specified model.
"""
variables(model) = Vars(_variables(model.tendencies...))
_variables() = ()
_variables(equation, equations...) = (equation.var, _variables(equations...)...)

##
## Formula sorting
##

# Run a compile time topological sort of the cache dependency graph with a
# recursive depth-first search algorithm.
# TODO: Verify that the type-stability of this function outweighs the fact that
# it might take a long time to compile.
# TODO: Instead of generating a list of sorted formulas, consider generating a
# list of sub-lists, where each sub-list is a component of the dependency graph
# (and hence can be computed independently of the other sub-lists).
function sorted_cache_formulas(
    tendencies,
    ::Type{FunctionConstructor},
    custom_formulas,
    diagnostics_formulas,
) where {FunctionConstructor}
    return formulas_for_equations(
        (),
        Vars(_variables(tendencies...)),
        FunctionConstructor,
        custom_formulas,
        tendencies...,
        diagnostics_formulas...,
    )
end

formulas_for_equations(formulas, _, _, _) = formulas
function formulas_for_equations(
    formulas,
    vars,
    ::Type{FunctionConstructor},
    custom_formulas,
    equation,
    equations...,
) where {FunctionConstructor}
    formulas = formulas_for_reqs(
        formulas,
        (),
        vars,
        FunctionConstructor,
        custom_formulas,
        cache_reqs(equation, vars)...,
    )
    return formulas_for_equations(
        formulas,
        vars,
        FunctionConstructor,
        custom_formulas,
        equations...,
    )
end

formulas_for_reqs(formulas, _, _, _, _) = formulas
function formulas_for_reqs(
    formulas,
    visited,
    vars,
    ::Type{FunctionConstructor},
    custom_formulas,
    req_var,
    req_vars...,
) where {FunctionConstructor}
    formulas = visit_dependency_graph_vertex(
        req_var,
        formulas,
        visited,
        vars,
        FunctionConstructor,
        custom_formulas,
    )
    return formulas_for_reqs(
        formulas,
        visited,
        vars,
        FunctionConstructor,
        custom_formulas,
        req_vars...,
    )
end

function visit_dependency_graph_vertex(
    req_var,
    formulas,
    visited,
    vars,
    ::Type{FunctionConstructor},
    custom_formulas,
) where {FunctionConstructor}
    if req_var ∈ Vars(_variables(formulas...))
        return formulas
    end
    if req_var ∈ Vars(visited)
        s_cycle = join(cycle_vars(req_var, visited...), " -> ")
        throw(ArgumentError("cache dependency cycle detected: $s_cycle"))
    end
    formula = get_formula(req_var, FunctionConstructor, custom_formulas...)
    formulas = formulas_for_reqs(
        formulas,
        (visited..., req_var),
        vars,
        FunctionConstructor,
        custom_formulas,
        cache_reqs(formula, vars)...,
    )
    return (formulas..., formula)
end

cycle_vars(var, var′, vars...) =        # no need for a base case
    var === var′ ? (var, vars..., var) : cycle_vars(var, vars...)

get_formula(var, ::Type{FunctionConstructor}) where {FunctionConstructor} =
    Formula(var, FunctionConstructor(var))
function get_formula(
    var,
    ::Type{FunctionConstructor},
    formula,
    formulas...,
) where {FunctionConstructor}
    var === formula.var ? formula :
    get_formula(var, FunctionConstructor, formulas...)
end

################################################################################

struct InstantiatedModel{M <: Model, C₁, C₂}
    model::M
    consts::C₁
    cache::C₂
end

"""
    instantiate(model, consts, Y, t)

Assign a collection of constants and a cache to the specified model. Allocate
the cache by evaluating the model's formulas for the state vector `Y` at time
`t`.
"""
function instantiate(model, consts, Y, t)
    args = (variables(model), Y, consts, t)
    cache = cache_for_formulas(NamedTuple(), args, model.formulas...)
    if length(model.diagnostics_formulas) > 0
        args = (variables(model), Y, cache, consts, t)
        formulas = model.diagnostics_formulas
        diagnostics = diagnostics_for_formulas(NamedTuple(), args, formulas...)
        cache = (; cache..., diagnostics)
    end
    return InstantiatedModel(model, consts, cache)
end

cache_for_formulas(cache, args) = cache
function cache_for_formulas(cache, args, formula, formulas...)
    vars, Y, consts, t = args
    value = Base.materialize(formula.f(vars, Y, cache, consts, t))
    cache = named_tuple_insert(cache, value, formula.var)
    return cache_for_formulas(cache, args, formulas...)
end

diagnostics_for_formulas(diagnostics, args) = diagnostics
function diagnostics_for_formulas(diagnostics, args, formula, formulas...)
    value = Base.materialize(formula.f(args...))
    diagnostics = named_tuple_insert(diagnostics, value, formula.var)
    return diagnostics_for_formulas(diagnostics, args, formulas...)
end

function named_tuple_insert(nt, x, var)
    symbs = symbols(var)
    if length(symbs) == 1
        sub_nt = x
    else
        sub_nt = symbs[1] in keys(nt) ? getindex(nt, symbs[1]) : NamedTuple()
        sub_nt = named_tuple_insert(sub_nt, x, Var(symbs[2:end]...))
    end
    return (; nt..., NamedTuple{(symbs[1],)}((sub_nt,))...)
end

"""
    ode_function(instantiated_model)

Convert an instantiated model into an `ODEFunction`.
"""
function ode_function(instantiated_model)
    return ODEFunction(
        instantiated_model;
        tgrad = (∂ₜY, Y, cache, t) -> fill!(∂ₜY, zero(eltype(∂ₜY))),
    ) # TODO: Automatically determine when the tgrad optimization is valid.
end
# TODO: By default, the output of ode_function() should include a JacVecOperator
# jac_prototype that uses finite differences. This ensures that implicit solvers
# will work out of the box.

# TODO: Consider parallelizing the equation evaluations. All of the tendencies
# and diagnostics can be evaluated in parallel, and, if the dependency graph has
# multiple components, some groups of formulas can be evaluated in parallel.
function (instantiated_model::InstantiatedModel)(∂ₜY, Y, _, t)
    @unpack model, consts, cache = instantiated_model
    args = (variables(model), Y, cache, consts, t)
    evaluate_equations!(cache, args, model.formulas...)
    evaluate_equations!(∂ₜY, args, model.tendencies...)
    evaluate_equations!(cache.diagnostics, args, model.diagnostics_formulas...)
    # TODO: Run Spaces.weighted_dss! on the components of ∂ₜY when necessary.
    return ∂ₜY
end

evaluate_equations!(dest, args) = nothing
function evaluate_equations!(dest, args, equation, equations...)
    evaluate!(dest, args, equation)
    evaluate_equations!(dest, args, equations...)
end

evaluate!(dest, args, formula::Formula) =
    Base.materialize!(get_var(dest, formula.var), formula.f(args...))
function evaluate!(dest, args, tendency::Tendency)
    @unpack var, bcs, terms = tendency
    if length(terms) == 0
        tendency_bc = Base.broadcasted(zero, get_var(dest, var))
    elseif length(terms) == 1
        tendency_bc = terms[1](args...)
    else
        # TODO: Is this type stable?
        tendency_bc = Base.broadcasted(+, map(term -> term(args...), terms)...)
    end
    if bcs isa VerticalBoundaryConditions
        # TODO: Use the dotted version when #325 is merged into ClimaCore.
        # B = Operators.SetBoundaryOperator(
        #     bottom = Operators.SetValue.(bcs.bottom(args...)),
        #     top = Operators.SetValue.(bcs.top(args...)),
        # )
        B = Operators.SetBoundaryOperator(
            bottom = Operators.SetValue(bcs.bottom(args...)),
            top = Operators.SetValue(bcs.top(args...)),
        )
        tendency_bc = Base.broadcasted(B, tendency_bc)
    end
    Base.materialize!(get_var(dest, var), tendency_bc)
end

#=
Ideas for higher-level interface:
- Compressible vs. Incompressible (ρ ∈ Y vs. ρ ∈ consts)
- Conservative vs. Convective (w vs. ρw and uₕ vs. ρuₕ)
- Hydrostatic vs. Non-hydrostatic (w ∈ Y or ρw ∈ Y vs. w ∈ cache or ρw ∈ cache)
    - Do we actually want this functionality?
- Energy variable (ρθ vs. ρe_tot)
    - Do we want any other options?
- Number of horizontal dimensions (0 vs. 1 vs. 2)

struct DryFluidModel{
    IsCompressible,
    IsConservative,
    IsHydrostatic,
    EnergyVar,
    NHorzDims,
}
=#
