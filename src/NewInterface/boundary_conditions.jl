using Base.Broadcast: broadcasted, Broadcasted
using ClimaCore: Operators

"""
    AbstractBoundaryCondition

Supertype for all boundary conditions.
"""
abstract type AbstractBoundaryCondition end

"""
    SetTendencyAtBoundary{F} <: AbstractBoundaryCondition

A boundary condition that sets the value of a tendency; serves as the simplest
way to implement a Neumann boundary condition for an independent variable.

The field `func` contains a function (or a function-like object) that is used to
compute the value on every tendency evaluation. This function must have the type
signature `func(vars, Y, cache, consts, t)`.
"""
struct SetTendencyAtBoundary{F} <: AbstractBoundaryCondition
    func::F
end

"""
    SetTendencyAtBoundary(; tendency)

Alternative constructor for `SetTendencyAtBoundary` that automatically generates
a function which always returns the specified tendency value.
"""
SetTendencyAtBoundary(; tendency) =
    SetTendencyAtBoundary(identity_func(tendency))

"""
    SetOperatorInputsAtBoundary{N, O <: NTuple{N, Type}, F <: NTuple{N}} <:
        AbstractBoundaryCondition

A boundary condition that sets the inputs for ClimaCore operators that do not
already include boundary conditions.

This serves as a way to apply a single boundary condition across all tendency
terms that share a particular outermost operator. To instead apply boundary
conditions to a specific tendency term, one should create a version of that term
that includes boundary conditions in one or more of its operators.

The field `op_types` contains a tuple of operator types. The field `funcs`
contains a tuple of functions (or function-like objects) that are used to
compute the inputs for the corresponding operator types on every tendency
evaluation. Each of these functions must have the type signature
`func(vars, Y, cache, consts, t)`.

Before this boundary condition gets applied, all ClimaCore operators that have
been used in a linear manner (i.e., added, subtracted, or multiplied/divided by
scalars) are factored out of the sum of tendency terms. The resulting factored
tendency is of the form `@. op₁(...) + op₂(...) + ... + opₙ(...) + ... + C`,
where `C` is some expression whose outermost operations are not ClimaCore
operators, and where each `opₙ` is a distinct ClimaCore operator. Then, for each
`op_type` in `op_types`, the tendency is scanned for an operator `opₙ` that is
an instance of `op_type` without boundary conditions (i.e., `opₙ == op_type()`).
If no such operator is found, a warning is printed. Otherwise, `opₙ` is replaced
with an instance of `op_type` that includes a boundary condition. This operator
boundary condition specifies that the input for the operator at the boundary is
the result of evaluating the `func` that corresponds to `op_type`.
"""
struct SetOperatorInputsAtBoundary{N, O <: NTuple{N, Type}, F <: NTuple{N}} <:
       AbstractBoundaryCondition
    op_types::O
    funcs::F
end

"""
    SetOperatorInputsAtBoundary(
        [op_types],
        [funcs];
        [left_face_tendency],
        [right_face_tendency],
        [total_face_scalar_potential],
        [total_face_flux],
    )

Alternative constructor for `SetOperatorInputsAtBoundary` that automatically
generates functions for the `LeftBiasedF2C`, `RightBiasedF2C`, `GradientF2C`,
and `DivergenceF2C` ClimaCore operators, if their respective inputs at the
boundary are specified.

For `LeftBiasedF2C` and `RightBiasedF2C`, the automatically generated function
will always return the the specified input value (`left_face_tendency` or
`right_face_tendency`, respectively). On the other hand, for `GradientF2C` and
`DivergenceF2C`, the automatically generated function will always return the
negative of the specified input value (`total_face_scalar_potential` or
`total_face_flux`, respectively). This is because the tendency due to a scalar
potential is the negative of the potential's gradient, and the tendency due to a
flux is the negative of the flux's divergence. If `total_face_scalar_potential`
or `total_face_flux` is a mutable value, its negative is recomputed on every
tendency evaluation.

Input values for any other operators (or alternative input functions for the
aformentioned operators) may be specified using the `op_types` and `funcs`
arguments.
"""
function SetOperatorInputsAtBoundary(
    op_types = (),
    funcs = ();
    left_face_tendency = nothing,
    right_face_tendency = nothing,
    total_face_scalar_potential = nothing,
    total_face_flux = nothing,
)
    if !isnothing(left_face_tendency)
        op_types = (op_types..., Operators.LeftBiasedF2C)
        funcs = (funcs..., identity_func(left_face_tendency))
    end
    if !isnothing(right_face_tendency)
        op_types = (op_types..., Operators.RightBiasedF2C)
        funcs = (funcs..., identity_func(right_face_tendency))
    end
    if !isnothing(total_face_scalar_potential)
        op_types = (op_types..., Operators.GradientF2C)
        funcs = (funcs..., negating_func(total_face_scalar_potential))
    end
    if !isnothing(total_face_flux)
        op_types = (op_types..., Operators.DivergenceF2C)
        funcs = (funcs..., negating_func(total_face_flux))
    end
    return SetOperatorInputsAtBoundary(op_types, funcs)
end
# TODO: Add face_tendency if InterpolateF2C is modified to allow boundary
# conditions. Add face_weighted_tendency if WeightedInterpolateF2C is modified
# to allow boundary conditions and if broadcast factorization is extended to
# multi-argument operators. Add total_face_vector_potential if CurlF2C is
# implemented.
# TODO: Should this constructor also have a descriptive keyword argument for
# every C2F operator that supports SetValue? Will there ever be a situation
# where someone wants to modify the values at the top or bottom cell centers?
# TODO: Should this constructor also accept functions as keyword arguments? Or
# would that be more confusing than just using the default constructor?

# TODO: Implement SetOperatorOutputsAtBoundary, or modify the implementation of
# SetOperatorInputsAtBoundary to allow such functionality. In the latter route,
# maybe also allow other operator boundary conditions, such as extrapolation.

identity_func(value) = (vars, Y, cache, consts, t) -> value

function negating_func(value)
    if ismutable(value)
        if value isa Ref
            return (vars, Y, cache, consts, t) -> -value[]
        else
            negated = similar(value) # assume a mutable non-Ref is array-like
            return (vars, Y, cache, consts, t) -> @. negated = -value
        end
    else
        negated = -value
        return (vars, Y, cache, consts, t) -> negated
    end
end

################################################################################

# TODO: Replace this whole section with `NamedTuple`s, like in ClimaCore.
# TODO: Remove the requirement that top and bottom boundary conditions must have
# the same `op_types` fields.

# const AbstractBoundaryConditionSet = NamedTuple{
#     names,
#     NTuple{N, AbstractBoundaryCondition} where {N},
# } where {names}
# empty_boundary_condition_set() = NamedTuple()

# TODO: Add a dot to every use of SetValue once #325 is merged into ClimaCore.
# TODO: Add support for horizontal and omnidirectional boundary conditions. For
# example, find a way to deal with boundary numerical fluxes.

"""
    AbstractBoundaryConditionSet

Supertype for all sets of boundary conditions.
"""
abstract type AbstractBoundaryConditionSet end

"""
    NoBoundaryConditions <: AbstractBoundaryConditionSet

An empty set of boundary conditions.

Indicates that no boundary conditions need to be applied to the corresponding
tendency. Used as a default for new tendencies.
"""
struct NoBoundaryConditions <: AbstractBoundaryConditionSet end

"""
    VerticalBoundaryConditions{
        T <: AbstractBoundaryCondition,
        B <: AbstractBoundaryCondition,
    } <: AbstractBoundaryConditionSet

A set of two boundary conditions---one for the top of the domain, and one for
the bottom.

If both boundary conditions are instances of `SetOperatorInputsAtBoundary`,
their `op_types` fields must be identical.
"""
struct VerticalBoundaryConditions{
    T <: AbstractBoundaryCondition,
    B <: AbstractBoundaryCondition,
} <: AbstractBoundaryConditionSet
    top::T
    bottom::B
end

"""
    VerticalBoundaryConditions(; top, bottom)

Keyword constructor for a set of vertical boundary conditions.
"""
VerticalBoundaryConditions(; top, bottom) =
    VerticalBoundaryConditions(top, bottom)

apply_boundary_conditions(tendency_bc, b::NoBoundaryConditions, args...) =
    tendency_bc

apply_boundary_conditions(tendency_bc, b::VerticalBoundaryConditions, args...) =
    apply_vertical_boundary_conditions(tendency_bc, b.top, b.bottom, args...)

function apply_vertical_boundary_conditions(
    tendency_bc,
    top::SetTendencyAtBoundary,
    bottom::SetTendencyAtBoundary,
    args...,
)
    boundary_op = Operators.SetBoundaryOperator(
        top = Operators.SetValue(top.func(args...)),
        bottom = Operators.SetValue(bottom.func(args...)),
    )
    return broadcasted(boundary_op, tendency_bc)
end

function apply_vertical_boundary_conditions(
    tendency_bc,
    top::SetTendencyAtBoundary,
    bottom::SetOperatorInputsAtBoundary,
    args...,
)
    tendency_bc = Base.afoldl(
        (tendency_bc, (op_type, func)) -> replace_op(
            tendency_bc,
            op_type(),
            op_type(bottom = Operators.SetValue(func(args...))),
        ),
        tendency_bc,
        zip(bottom.op_types, bottom.funcs)...,
    )
    boundary_op = Operators.SetBoundaryOperator(
        top = Operators.SetValue(top.func(args...)),
    )
    return broadcasted(boundary_op, tendency_bc)
end

function apply_vertical_boundary_conditions(
    tendency_bc,
    top::SetOperatorInputsAtBoundary,
    bottom::SetTendencyAtBoundary,
    args...,
)
    tendency_bc = Base.afoldl(
        (tendency_bc, (op_type, func)) -> replace_op(
            tendency_bc,
            op_type(),
            op_type(top = Operators.SetValue(func(args...))),
        ),
        tendency_bc,
        zip(top.op_types, top.funcs)...,
    )
    boundary_op = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(bottom.func(args...)),
    )
    return broadcasted(boundary_op, tendency_bc)
end

function apply_vertical_boundary_conditions(
    tendency_bc,
    top::SetOperatorInputsAtBoundary,
    bottom::SetOperatorInputsAtBoundary,
    args...,
)
    if top.op_types !== bottom.op_types
        s = string(
            "top and bottom boundary conditions must have identical operator ",
            "types when they are both instances of SetOperatorInputsAtBoundary",
        ) # TODO: temporary fix for outdated formatter
        throw(ArgumentError(s))
    end
    return Base.afoldl(
        (tendency_bc, (op_type, top_f, bottom_f)) -> replace_op(
            tendency_bc,
            op_type(),
            op_type(
                top = Operators.SetValue(top_f(args...)),
                bottom = Operators.SetValue(bottom_f(args...)),
            ),
        ),
        tendency_bc,
        zip(top.op_types, top.funcs, bottom.funcs)...,
    )
end

function replace_op(tendency_bc, op, op′)
    if tendency_bc.f === op
        return broadcasted(op′, tendency_bc.args...)
    elseif tendency_bc.f isa Addition
        with_op, without_op =
            partition(x -> x isa Broadcasted && x.f === op, tendency_bc.args)
        if length(with_op) == 1
            op_bc = broadcasted(op′, with_op[1].args...)
            return broadcasted(+, op_bc, without_op...)
        elseif length(with_op) > 1
            s = "mistake detected in tendency broadcast factorization algorithm"
            throw(ErrorException(s))
        end
    end
    op_type = typeof(op).name.wrapper
    s = string(
        "unable to apply boundary condition for $op_type because there are no ",
        "tendency terms whose final non-arithmetic linear operation is $op",
    ) # TODO: temporary fix for outdated formatter
    @warn s maxlog = 1 # only show this warning on the first tendency evaluation
    return tendency_bc
end
