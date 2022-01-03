using Base.Broadcast: broadcasted, Broadcasted, BroadcastStyle
using ClimaCore: RecursiveApply, Operators

# TODO: Add support for multi-argument linear operators, if we come across any.

const LinearOneArgOperator = Union{
    Operators.FiniteDifferenceOperator,
    Operators.SpectralElementOperator,
    typeof(adjoint),
    typeof(transpose),
}

const ScalarValue = Union{Number, Ref, NTuple{1}}

const Addition = Union{typeof(+), typeof(RecursiveApply.radd)}
const Subtraction = Union{typeof(-), typeof(RecursiveApply.rsub)}
const Multiplication = Union{typeof(*), typeof(RecursiveApply.rmul)}
const Division = Union{typeof(/), typeof(RecursiveApply.rdiv)}

# Speed up the materialization of a Broadcasted object by recursively
#     - expanding out nested sums,
#     - factoring out negations from sums, and
#     - factoring out linear operations from sums, negations, differences,
#       products with scalar values, and quotients with scalar values.
# This also allows a single boundary condition to apply to a differential
# operator that is used by multiple tendency terms.
# TODO: Should muladd be factorized as well? Or would that be excessive?
factorize_bc(x) = x
function factorize_bc(bc::Broadcasted)
    args = map(factorize_bc, bc.args)
    bc.f isa Addition && return factorize_sum(expand_sums(args)...)
    bc.f isa Subtraction && return factorize_negation_or_difference(args...)
    bc.f isa Multiplication && return factorize_product(args...)
    bc.f isa Division && return factorize_quotient(args...)
    return broadcasted(bc.f, args...)
end

expand_sums(args) = Base.afoldl(_expand_sums, (), args...)
_expand_sums(expanded_args, arg) = arg isa Broadcasted && arg.f isa Addition ?
    (expanded_args..., arg.args...) : (expanded_args..., arg)

factorize_sum(args...) = _factorize_sum((), args...) # can't use foldl for this
_factorize_sum(factorized_args::NTuple{1}) = factorized_args[1] # sum of 1 term
_factorize_sum(factorized_args) = broadcasted(+, factorized_args...)
function _factorize_sum(factorized_args, arg, args...)
    if (
        is_linear_1arg_bc(arg) ||
        (arg isa Broadcasted && arg.f isa Subtraction && length(arg.args) == 1)
    )
        same_f, not_same_f = partition(
            x -> x isa Broadcasted && x.f === arg.f && length(x.args) == 1,
            args,
        )
        if length(same_f) > 0
            bc = broadcasted(+, arg.args[1], map(bc -> bc.args[1], same_f)...)
            arg = broadcasted(arg.f, bc)
        end
        return _factorize_sum((factorized_args..., arg), not_same_f...)
    else
        return _factorize_sum((factorized_args..., arg), args...)
    end
end

function factorize_negation_or_difference(arg)
    if is_linear_1arg_bc(arg)
        return broadcasted(arg.f, broadcasted(-, arg.args[1]))
    else
        return broadcasted(-, arg)
    end
end
function factorize_negation_or_difference(arg1, arg2)
    if is_linear_1arg_bc(arg1) && is_linear_1arg_bc(arg2) && arg1.f === arg2.f
        return broadcasted(arg1.f, broadcasted(-, arg1.args[1], arg2.args[1]))
    else
        return broadcasted(-, arg1, arg2)
    end
end

function factorize_product(args...)
    scalars, not_scalars = partition(x -> x isa ScalarValue, args)
    if length(not_scalars) == 1 && is_linear_1arg_bc(not_scalars[1])
        bc = broadcasted(*, not_scalars[1].args[1], scalars...)
        return broadcasted(not_scalars[1].f, bc)
    else
        return broadcasted(*, args...)
    end
end

function factorize_quotient(arg1, arg2)
    if is_linear_1arg_bc(arg1) && arg2 isa ScalarValue
        return broadcasted(arg1[1].f, broadcasted(/, arg1.args[1], arg2))
    else
        return broadcasted(/, arg1, arg2)
    end
end

is_linear_1arg_bc(x) =
    x isa Broadcasted && x.f isa LinearOneArgOperator && length(x.args) == 1

# Return a tuple of xs that satisfy condition f and a tuple of xs that don't.
# (This is like Base.filter_rec(f, xs::Tuple), except that it returns both the
# "do"s and the "don't"s, rather than just the "do"s.)
partition(f, xs) = Base.afoldl(
    ((dos, donts), x) -> f(x) ? ((dos..., x), donts) : (dos, (donts..., x)),
    ((), ()),
    xs...,
)
