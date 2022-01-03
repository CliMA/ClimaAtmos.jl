using UnPack
using ClimaCore: Spaces
using OrdinaryDiffEq: ODEFunction

"""
    Model{
        T <: NTuple{N₁, Tendency} where {N₁},
        F <: NTuple{N₂, Formula} where {N₂},
        D <: NTuple{N₃, Formula} where {N₃},
    }

A representation of a "model", which describes a system of ordinary differential
equations and provides some information for solving that system.

Contains tendencies for evolving the independent variables over time, formulas
for updating the cache variables required by the tendency terms, and formulas
for updating any additional variables meant for diagnostics (and debugging). The
cache variables and diagnostics variables can be thought of as "dependent"
variables, since they are computed using the independent variables (and possibly
some constant values and the simulation time).

Cache variables may depend on other cache variables, so the order in which the
cache formulas are stored must correspond to a valid sequence of evaluations for
the cache variables (i.e., a cache variable should only be evaluated after all
of the cache variables on which it depends have been evaluated). All of the
tendency terms and diagnostics variables are independent of each other, so the
order of the tendencies and diagnostics formulas is irrelevant.
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
# TODO: Consider adding diagnostics_tendencies. This will allow users to run
# their models normally, but also track what the values of certain tendencies
# would be at each timestep if, say, some tendency terms were disabled or some
# boundary conditions were modified.
# TODO: Ensure that there are no duplicate variables in any of the Model fields.

"""
    Model(
        [formula_function_constructor];
        [tendencies],
        [custom_formulas],
        [diagnostics_formulas],
    )

Alternative constructor for a `Model` that does not require users to specify all
of the required cache formulas and ensure that those formulas are evaluated in a
valid order.

If the argument `formula_function_constructor` is provided, this `Model`
constructor automatically generates a formula for each cache variable `var` by
calling `formula_function_constructor(var)`, which must return an
`AbstractFormulaFunction`. If the automatically generated formula for a
variable should be overwritten, or if `formula_function_constructor` is not
defined for a variable, a formula for that variable must be included in
`custom_formulas`.

If `formula_function_constructor` is not provided, this constructor will verify
that all required cache formulas are included in `custom_formulas` and determine
a valid evaluation order for them.
"""
function Model(
    formula_function_constructor = nothing;
    tendencies = (),
    custom_formulas = (),
    diagnostics_formulas = (),
)
    formulas = sorted_formulas_for_equations(
        (),
        _variables(tendencies),
        formula_function_constructor,
        custom_formulas,
        tendencies...,
        diagnostics_formulas...,
    )
    return Model(tendencies, formulas, diagnostics_formulas)
end

"""
    variables(model)

Get the independent variables of the specified model.
"""
variables(model) = _variables(model.tendencies)
_variables(equations) = Vars(map(equation -> equation.var, equations))

##
## Formula construction and sorting
##

# Run a compile time topological sort of the cache dependency graph with a
# recursive depth-first search algorithm.
# TODO: Instead of generating a list of sorted formulas, consider generating a
# list of sub-lists, where each sub-list contains a component of the dependency
# graph (and can therefore be computed independently of the other sub-lists).

sorted_formulas_for_equations(formulas, _, _, _) = formulas
function sorted_formulas_for_equations(
    formulas,
    vars,
    formula_function_constructor,
    custom_formulas,
    equation,
    equations...,
)
    formulas = sorted_formulas_for_reqs(
        formulas,
        (),
        vars,
        formula_function_constructor,
        custom_formulas,
        cache_reqs(equation, vars)...,
    )
    return sorted_formulas_for_equations(
        formulas,
        vars,
        formula_function_constructor,
        custom_formulas,
        equations...,
    )
end

sorted_formulas_for_reqs(formulas, _, _, _, _) = formulas
function sorted_formulas_for_reqs(
    formulas,
    visited,
    vars,
    formula_function_constructor,
    custom_formulas,
    req_var,
    req_vars...,
)
    formulas = visit_dependency_graph_vertex(
        req_var,
        formulas,
        visited,
        vars,
        formula_function_constructor,
        custom_formulas,
    )
    return sorted_formulas_for_reqs(
        formulas,
        visited,
        vars,
        formula_function_constructor,
        custom_formulas,
        req_vars...,
    )
end

function visit_dependency_graph_vertex(
    req_var,
    formulas,
    visited,
    vars,
    formula_function_constructor,
    custom_formulas,
)
    if req_var ∈ _variables(formulas)
        return formulas
    end
    if req_var ∈ Vars(visited)
        s_cycle = join(cycle_vars(req_var, visited...), " -> ")
        throw(ArgumentError("cache dependency cycle detected: $s_cycle"))
    end
    formula =
        get_formula(req_var, formula_function_constructor, custom_formulas...)
    req_formulas = sorted_formulas_for_reqs(
        formulas,
        (visited..., req_var),
        vars,
        formula_function_constructor,
        custom_formulas,
        cache_reqs(formula, vars)...,
    )
    return (req_formulas..., formula)
end

cycle_vars(var, var′, vars...) =   # no need for a base case here
    var === var′ ? (var, vars..., var) : cycle_vars(var, vars...)

get_formula(var, ::Nothing) = throw(ArgumentError("missing formula for $var"))
get_formula(var, formula_function_constructor) =
    Formula(var, formula_function_constructor(var))
get_formula(var, formula_function_constructor, formula, formulas...) =
    var === formula.var ? formula :
    get_formula(var, formula_function_constructor, formulas...)

################################################################################

struct InstantiatedModel{M <: Model, C₁, C₂}
    model::M
    consts::C₁
    cache::C₂
end

"""
    instantiate(model, consts, Y, t)

Add a collection of constants and a cache to the specified model. Allocate the
cache by evaluating the model's formulas for the state vector `Y` at time `t`.
"""
function instantiate(model, consts, Y, t)
    vars = variables(model)
    formulas = model.formulas
    cache = cache_for_formulas(NamedTuple(), vars, Y, consts, t, formulas...)
    if length(model.diagnostics_formulas) > 0
        args = (vars, Y, cache, consts, t)
        formulas = model.diagnostics_formulas
        diagnostics = diagnostics_for_formulas(NamedTuple(), args, formulas...)
        cache = (; cache..., diagnostics)
    end
    return InstantiatedModel(model, consts, cache)
end

cache_for_formulas(cache, _, _, _, _) = cache
function cache_for_formulas(cache, vars, Y, consts, t, formula, formulas...)
    value = Base.materialize(formula.f(vars, Y, cache, consts, t))
    cache = named_tuple_insert(cache, value, formula.var)
    return cache_for_formulas(cache, vars, Y, consts, t, formulas...)
end

diagnostics_for_formulas(diagnostics, _) = diagnostics
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
    ode_function(instantiated_model; [is_autonomous])

Convert an instantiated model into an `ODEFunction`.

By default, the system of ordinary differential equation defined by the model is
assumed to be autonomous, which allows some ODE solvers to utilize certain
performance optimizations. If any of the model's tendencies or cache formulas
are time-dependent, `is_autonomous` must be set to `false`; if this is not done,
those ODE solvers may generate incorrect solutions.
"""
function ode_function(instantiated_model; is_autonomous = true)
    kwargs = NamedTuple()
    if is_autonomous
        kwargs =
            (kwargs..., tgrad = (∂ₜY, Y, _, t) -> fill!(∂ₜY, zero(eltype(∂ₜY))))
    end
    return ODEFunction{true}(instantiated_model; kwargs...) # iip isn't inferred
end
# TODO: By default, the output of ode_function() should include a JacVecOperator
# jac_prototype that uses finite differences. This ensures that implicit solvers
# will work out of the box.
# TODO: Automatically determine the value of is_autonomous.

# TODO: Don't materialize a formula if it's only used by a single tendency term.
# TODO: Consider parallelizing the equation evaluations. All of the tendencies
# and diagnostics formulas can be evaluated in parallel, and, if the dependency
# graph has multiple components, some groups of cache formulas can be evaluated
# in parallel.
function (instantiated_model::InstantiatedModel)(∂ₜY, Y, _, t)
    @unpack model, consts, cache = instantiated_model
    args = (variables(model), Y, cache, consts, t)
    evaluate_equations!(cache, args, model.formulas)
    evaluate_equations!(∂ₜY, args, model.tendencies)
    evaluate_equations!(cache.diagnostics, args, model.diagnostics_formulas)
    # TODO: Run Spaces.weighted_dss! on the components of ∂ₜY when necessary.
    return ∂ₜY
end

include("factorize_bc.jl")

evaluate_equations!(dest, args, equations) =
    foreach(equation -> evaluate_equation!(dest, args, equation), equations)
evaluate_equation!(dest, args, formula::Formula) =
    Base.materialize!(get_var(dest, formula.var), formula.f(args...))
function evaluate_equation!(dest, args, tendency::Tendency)
    @unpack var, boundary_conditions, terms = tendency
    if length(terms) == 0
        tendency_bc = Base.broadcasted(zero, get_var(dest, var))
    elseif length(terms) == 1
        tendency_bc = terms[1](args...)
    else
        tendency_bc = Base.broadcasted(+, map(term -> term(args...), terms)...)
    end
    tendency_bc = factorize_bc(tendency_bc)
    tendency_bc = boundary_conditions(factorize_bc(tendency_bc), args...)
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
