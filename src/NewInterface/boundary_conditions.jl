using Base.Broadcast: broadcasted, Broadcasted
using ClimaCore: Operators

abstract type AbstractBoundaryCondition end

struct ValueBoundaryCondition{F} <: AbstractBoundaryCondition
    f::F
end

struct OperatorValuesBoundaryCondition{
    N,
    O <: NTuple{N, Type},
    F <: NTuple{N},
} <: AbstractBoundaryCondition
    op_types::O
    fs::F
end

##
## User-friendly boundary condition specifications
##

zero_value_boundary_condition(::Type{T}) where {T} =
    ValueBoundaryCondition((vars, Y, cache, consts, t) -> zero(T))

value_boundary_condition(value) =
    ValueBoundaryCondition((vars, Y, cache, consts, t) -> value)

function zero_flux_boundary_condition(::Type{T}) where {T}
    f = (vars, Y, cache, consts, t) -> zero(T)
    return OperatorValuesBoundaryCondition((Operators.DivergenceF2C,), (f,))
end

function constant_flux_boundary_condition(flux)
    value = -flux
    f = (vars, Y, cache, consts, t) -> value
    return OperatorValuesBoundaryCondition((Operators.DivergenceF2C,), (f,))
end

function mutable_flux_boundary_condition(flux)
    value = similar(flux)
    f = (vars, Y, cache, consts, t) -> @. value = -flux
    return OperatorValuesBoundaryCondition((Operators.DivergenceF2C,), (f,))
end

################################################################################

# TODO: Add a dot to every use of SetValue when #325 is merged into ClimaCore.
# TODO: Add support for horizontal boundary conditions, if needed.

abstract type AbstractBoundaryConditions end

struct NoBoundaryConditions <: AbstractBoundaryConditions end

struct VerticalBoundaryConditions{
    T <: AbstractBoundaryCondition,
    B <: AbstractBoundaryCondition,
} <: AbstractBoundaryConditions
    top::T
    bottom::B
end

(b::NoBoundaryConditions)(tendency_bc, args...) = tendency_bc

(b::VerticalBoundaryConditions)(tendency_bc, args...) =
    apply_vertical_boundary_conditions(tendency_bc, b.top, b.bottom, args...)

function apply_vertical_boundary_conditions(
    tendency_bc,
    top::ValueBoundaryCondition,
    bottom::ValueBoundaryCondition,
    args...,
)
    boundary_op = Operators.SetBoundaryOperator(
        top = Operators.SetValue(top.f(args...)),
        bottom = Operators.SetValue(bottom.f(args...)),
    )
    return broadcasted(boundary_op, tendency_bc)
end

function apply_vertical_boundary_conditions(
    tendency_bc,
    top::ValueBoundaryCondition,
    bottom::OperatorValuesBoundaryCondition,
    args...,
)
    tendency_bc = Base.afoldl(
        (tendency_bc, (op_type, f)) -> replace_op(
            tendency_bc,
            op_type(),
            op_type(bottom = Operators.SetValue(f(args...))),
        ),
        tendency_bc,
        zip(bottom.op_types, bottom.fs)...,
    )
    boundary_op =
        Operators.SetBoundaryOperator(top = Operators.SetValue(top.f(args...)))
    return broadcasted(boundary_op, tendency_bc)
end

function apply_vertical_boundary_conditions(
    tendency_bc,
    top::OperatorValuesBoundaryCondition,
    bottom::ValueBoundaryCondition,
    args...,
)
    tendency_bc = Base.afoldl(
        (tendency_bc, (op_type, f)) -> replace_op(
            tendency_bc,
            op_type(),
            op_type(top = Operators.SetValue(f(args...))),
        ),
        tendency_bc,
        zip(top.op_types, top.fs)...,
    )
    boundary_op = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(bottom.f(args...)),
    )
    return broadcasted(boundary_op, tendency_bc)
end

function apply_vertical_boundary_conditions(
    tendency_bc,
    top::OperatorValuesBoundaryCondition,
    bottom::OperatorValuesBoundaryCondition,
    args...,
)
    if top.op_types !== bottom.op_types
        s = string(
            "top and bottom boundary conditions must have the same operator ",
            "types when they are both \"OperatorValuesBoundaryCondition\"s",
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
        zip(top.op_types, top.fs, bottom.fs)...,
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
        "tendency terms whose final non-arithmetic operation is $op",
    ) # TODO: temporary fix for outdated formatter
    @warn s maxlog = 1 # only show this warning on the first tendency evaluation
    return tendency_bc
end
