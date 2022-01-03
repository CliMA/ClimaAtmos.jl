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

(b::NoBoundaryConditions)(tendency_bc, vars, Y, cache, consts, t) = tendency_bc

(b::VerticalBoundaryConditions)(tendency_bc, vars, Y, cache, consts, t) =
    apply_vert(tendency_bc, b.top, b.bottom, vars, Y, cache, consts, t)

function apply_vert(
    tendency_bc,
    top::ValueBoundaryCondition,
    bottom::ValueBoundaryCondition,
    args...,
)
    op = Operators.SetBoundaryOperator(
        top = Operators.SetValue(top.f(args...)),
        bottom = Operators.SetValue(bottom.f(args...)),
    )
    return broadcasted(op, tendency_bc)
end

function apply_vert(
    tendency_bc,
    top::ValueBoundaryCondition,
    bottom::OperatorValuesBoundaryCondition,
    args...,
)
    tendency_bc = Base.afoldl(
        (tendency_bc, (op_type, f)) -> replace_op(
            tendency_bc,
            op_type,
            op_type(bottom = Operators.SetValue(f(args...))),
        ),
        tendency_bc,
        zip(bottom.op_types, bottom.fs)...,
    )
    op = Operators.SetBoundaryOperator(top = Operators.SetValue(top.f(args...)))
    return broadcasted(op, tendency_bc)
end

function apply_vert(
    tendency_bc,
    top::OperatorValuesBoundaryCondition,
    bottom::ValueBoundaryCondition,
    args...,
)
    tendency_bc = Base.afoldl(
        (tendency_bc, (op_type, f)) -> replace_op(
            tendency_bc,
            op_type,
            op_type(top = Operators.SetValue(f(args...))),
        ),
        tendency_bc,
        zip(top.op_types, top.fs)...,
    )
    op = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(bottom.f(args...)),
    )
    return broadcasted(op, tendency_bc)
end

function apply_vert(
    tendency_bc,
    top::OperatorValuesBoundaryCondition,
    bottom::OperatorValuesBoundaryCondition,
    args...,
)
    if top.op_types !== bottom.op_types
        s = string(
            "top and bottom boundary conditions must have matching operator ",
            "types when each is an OperatorValuesBoundaryCondition",
        ) # TODO: temporary fix for outdated formatter
        throw(ArgumentError(s))
    end
    return Base.afoldl(
        (tendency_bc, (op_type, top_f, bottom_f)) -> replace_op(
            tendency_bc,
            op_type,
            op_type(
                top = Operators.SetValue(top_f(args...)),
                bottom = Operators.SetValue(bottom_f(args...)),
            ),
        ),
        tendency_bc,
        zip(top.op_types, top.fs, bottom.fs)...,
    )
end

function replace_op(tendency_bc::Broadcasted, op_type, new_op)
    if tendency_bc.f isa op_type
        return broadcasted(new_op, tendency_bc.args...)
    elseif tendency_bc.f isa Addition
        with_op_type, without_op_type = partition(
            x -> x isa Broadcasted && x.f isa op_type,
            tendency_bc.args,
        )
        if length(with_op_type) == 1
            new_op_bc = broadcasted(new_op, with_op_type[1].args...)
            return broadcasted(+, new_op_bc, without_op_type...)
        elseif length(with_op_type) > 1
            s = string(
                "boundary conditions specified in tendency terms (multiple ",
                "versions of $op_type detected)",
            ) # TODO: temporary fix for outdated formatter
            throw(ArgumentError(s))
        end
    end
    s = string(
        "unable to apply OperatorValuesBoundaryCondition with operator type ",
        "$op_type because no tendency terms were found with $op_type as their ",
        "final operation",
    ) # TODO: temporary fix for outdated formatter
    throw(ArgumentError(s))
end
