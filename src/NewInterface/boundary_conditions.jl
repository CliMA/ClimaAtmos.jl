using Base.Broadcast: broadcasted, Broadcasted
using ClimaCore: Operators

"""
    AbstractBoundaryCondition

Supertype for all boundary conditions.
"""
abstract type AbstractBoundaryCondition end

"""
    SetValueAtBoundary{F} <: AbstractBoundaryCondition

A boundary condition that sets the value of a tendency.

The field `func` contains a function (or a function-like object) that is used to
compute the value on every tendency evaluation. This function must have the type
signature `func(vars, Y, cache, consts, t)`.
"""
struct SetValueAtBoundary{F} <: AbstractBoundaryCondition
    func::F
end

"""
    SetValueAtBoundary(; value)

Alternative constructor for `SetValueAtBoundary` that automatically generates a
function which always returns the specified value.
"""
SetValueAtBoundary(; value) =
    SetValueAtBoundary((vars, Y, cache, consts, t) -> value)

"""
    SetOperatorInputsAtBoundary{N, O <: NTuple{N, Type}, F <: NTuple{N}} <:
        AbstractBoundaryCondition

A boundary condition that sets the inputs for ClimaCore operators that do not
already include boundary conditions.

The field `op_types` contains a tuple of operator types. The field `funcs`
contains a tuple of functions (or function-like objects) that are used to
compute the inputs for the corresponding operator types on every tendency
evaluation. Each of these functions must have the type signature
`func(vars, Y, cache, consts, t)`.

Before this boundary condition gets applied, all ClimaCore operators that have
been used in a linear manner (i.e., added, subtracted, or multiplied/divided by
scalars) are factored out of the sum of tendency terms. The resulting factored
tendency is either of the form `@. op₁(...) + C`, or it is of the form
`@. op₁(...) + op₂(...) + ... + opₙ(...) + ... + C`, where every distinct
operator `opₙ` only appears once. Then, for each `op_type` in `op_types`, the
tendency is scanned for an operator `opₙ` that is an instance of `op_type`
without any boundary conditions (i.e., `opₙ == op_type()`). If no such operator
is found, a warning is printed. Otherwise, `opₙ` is replaced with an instance of
`op_type` that includes a boundary condition. This automatically generated
boundary condition specifies that the input for the operator at the boundary is
the result of evaluating the `func` that corresponds to `op_type`.
"""
struct SetOperatorInputsAtBoundary{N, O <: NTuple{N, Type}, F <: NTuple{N}} <:
       AbstractBoundaryCondition
    op_types::O
    funcs::F
end

"""
    SetOperatorInputsAtBoundary([op_types], [funcs]; face_flux)

Alternative constructor for `SetOperatorInputsAtBoundary` that automatically
generates a function for the `ClimaCore.Operators.DivergenceF2C` operator type
based on the specified flux at the boundary face.

Since the tendency due to a flux is the negative of that flux's divergence, the
automatically generated function always returns the negative of the specified
flux. If `face_flux` is a mutable value, its negative is recomputed on every
tendency evaluation.
"""
function SetOperatorInputsAtBoundary(op_types = (), funcs = (); face_flux)
    if ismutable(face_flux)
        if face_flux isa Ref
            func = (vars, Y, cache, consts, t) -> -face_flux[]
        else
            input = similar(face_flux)
            func = (vars, Y, cache, consts, t) -> @. input = -face_flux
        end
    else
        input = -face_flux
        func = (vars, Y, cache, consts, t) -> input
    end
    return SetOperatorInputsAtBoundary(
        (op_types..., Operators.DivergenceF2C,),
        (funcs..., func,),
    )
end

################################################################################

# TODO: Add a dot to every use of SetValue once #325 is merged into ClimaCore.
# TODO: Add support for horizontal and full boundary conditions, if needed.

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
    top::SetValueAtBoundary,
    bottom::SetValueAtBoundary,
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
    top::SetValueAtBoundary,
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
    bottom::SetValueAtBoundary,
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
