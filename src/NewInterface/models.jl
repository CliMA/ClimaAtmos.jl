"""
    Model{
        T <: NTuple{N₁, Tendency} where N₁,
        F <: NTuple{N₂, Formula} where N₂,
        D <: NTuple{N₃, Formula} where N₃,
    }

A representation of a "model". Contains a tuple of tendencies for evolving the
state vector over time, a tuple of formulas for updating the cached values
required by the tendencies, and a tuple of additional formulas for debugging.
"""
struct Model{
    T <: NTuple{N₁, Tendency} where {N₁},
    F <: NTuple{N₂, Formula} where {N₂},
    D <: NTuple{N₃, Formula} where {N₃},
}
    tendencies::T
    formulas::F
    debug_formulas::D
end

function Model(;
    tendencies = (),
    default_function_constructor,
    custom_formulas = (),
    debug_formulas = (),
)
    formulas = sorted_formulas(
        tendencies,
        default_function_constructor,
        custom_formulas,
        debug_formulas,
    )
    return Model(tendencies, formulas, debug_formulas)
end

"""
    variables(model)

Get the state variables of the given model.
"""
variables(model) = _variables(model.tendencies...)
_variables() = ()
_variables(tendency, tendencies...) =
    (tendency.var, _variables(tendencies...)...)

##
## Formula sorting
##

# Run a compile time topological sort of the cache dependency graph with a
# recursive depth-first search algorithm.
# TODO: Instead of generating a list of sorted formulas, consider generating a
# list of sub-lists, where each sub-list is a component of the dependency graph
# (and hence can be computed independently of the other sub-lists).
function sorted_formulas(
    tendencies,
    default_function_constructor,
    custom_formulas,
    debug_formulas,
)
    formulas = ()
    vars = _variables(tendencies...)
    for tendency in tendencies
        for term in tendency.terms
            for req_var in cache_reqs(term, vars)
                formulas = visit_graph_vertex(
                    req_var,
                    (),
                    formulas,
                    vars,
                    default_function_constructor,
                    custom_formulas,
                )
            end
        end
    end
    for formula in debug_formulas
        for req_var in cache_reqs(formula, vars)
            formulas = visit_graph_vertex(
                req_var,
                (),
                formulas,
                vars,
                default_function_constructor,
                custom_formulas,
            )
        end
    end
    return formulas
end

function visit_graph_vertex(
    var,
    visited,
    sorted,
    vars,
    default_function_constructor,
    custom_formulas,
)
    if not_yet_sorted(var, sorted...)
        if var in visited
            s_cycle = join(cycle_vars(var, visited...), " -> ")
            throw(ArgumentError("cache dependency cycle detected: $s_cycle"))
        end
        visited = (visited..., var)
        f = formula_function(
            var,
            default_function_constructor,
            custom_formulas...,
        )
        for req_var in cache_reqs(f, vars)
            sorted = visit_graph_vertex(
                req_var,
                visited,
                sorted,
                vars,
                default_function_constructor,
                custom_formulas,
            )
        end
        sorted = (sorted..., Formula(var, f))
    end
    return sorted
end

not_yet_sorted(var) = true
not_yet_sorted(var, formula, formulas...) =
    var !== formula.var && not_yet_sorted(var, formulas...)

cycle_vars(var, var′, vars...) =
    var === var′ ? (var, vars..., var) : cycle_vars(var, vars...)

formula_function(var, default_function_constructor) =
    default_function_constructor(var)
formula_function(var, default_function_constructor, formula, formulas...) =
    var === formula.var ? formula.f :
    formula_function(var, default_function_constructor, formulas...)

################################################################################

struct InstantiatedModel{M <: Model, C₁, C₂}
    model::M
    cache::C₁
    consts::C₂
end

"""
    instantiate(model, Y, consts, t)

Assign a cache and a set of constants to the given model. Allocate the cache by
evaluating the model's formulas for the given values of `Y` and `t`.
"""
function instantiate(model, Y, consts, t)
    vars = variables(model)
    cache = NamedTuple()
    for formula in model.formulas
        value = Base.materialize(formula.f(vars, Y, cache, consts, t))
        cache = named_tuple_insert(cache, formula.var, value)
    end
    if length(model.debug_formulas) > 0
        debug = NamedTuple()
        for formula in model.debug_formulas
            value = Base.materialize(formula.f(vars, Y, cache, consts, t))
            debug = named_tuple_insert(debug, formula.var, value)
        end
        cache = (; cache..., debug)
    end
    return InstantiatedModel(model, cache, consts)
end

function named_tuple_insert(nt, var, x)
    symbs = symbols(var)
    if length(symbs) == 1
        sub_nt = x
    else
        sub_nt = symbs[1] in keys(nt) ? getproperty(nt, symbs[1]) : NamedTuple()
        sub_var = Var(symbs[2:end]...)
        sub_nt = named_tuple_insert(sub_nt, sub_var, x)
    end
    return (; nt..., NamedTuple((symbs[1] => sub_nt,))...)
end

"""
    ode_function(instantiated_model)

Convert an instantiated model into an `ODEFunction`.
"""
function ode_function(instantiated_model)
    return ODEFunction(
        instantiated_model;
        tgrad = (∂ₜY, Y, cache, t) -> fill!(∂ₜY, zero(eltype(∂ₜY))), # XXX
    )
end

# TODO: Consider parallelizing the formula and tendency loops. All the tendency
# broadcasts can be materialized in parallel, and, if the dependency graph has
# multiple components, some groups of variables can be cached in parallel.
function (instantiated_model::InstantiatedModel)(∂ₜY, Y, _, t)
    @unpack model, cache, consts = instantiated_model
    vars = variables(model)
    for formula in formulas
        Base.materialize!(
            cache[formula.var],
            formula.f(vars, Y, cache, consts, t),
        )
    end
    for tendency in model.tendencies
        @unpack var, bcs, terms = tendency
        tendency_broadcast = Base.broadcasted(
            +,
            map(term -> term(vars, Y, cache, consts, t), terms)...,
        )
        if bcs isa VerticalBoundaryConditions
            B = Operators.SetBoundaryOperator(
                bottom = Operators.SetValue.(bcs.bottom),
                top = Operators.SetValue.(bcs.top),
            )
            tendency_broadcast = Base.broadcasted(B, tendency_broadcast)
        end
        Base.materialize!(∂ₜY[var], tendency_broadcast)
    end
    for formula in model.debug_formulas
        Base.materialize!(
            cache.debug[formula.var],
            formula.f(vars, Y, cache, consts, t),
        )
    end
    # TODO: Spaces.weighted_dss!(∂ₜY) when necessary
    return ∂ₜY
end
