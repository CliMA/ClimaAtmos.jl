using Base.Broadcast: broadcasted, Broadcasted, BroadcastStyle
using ClimaCore: RecursiveApply, Operators

# TODO: Think of a way to allow errors during the materialization of Broadcasted
# objects to be traced back to the user's code more easily. Can @lazydots attach
# source code information to Broadcasted objects (or to new objects that wrap
# Broadcasted objects) that can be printed in stacktraces? This might be made
# easier by putting a limit on the recursiveness of broadcast factorization; if
# broadcast factorization stops when a linear operation is factored out, then
# everything inside that linear operation will be exactly as it appears in the
# user-defined function. Such a modification of broadcast factorization will not
# affect the capabilities of boundary condition application, which only requires
# the outermost linear operators in a tendency to be factored out.

# TODO: Do broadcast factorization and boundary condition application using a
# lazier form of broadcasting (e.g., one that does not call `broadcasted`, and
# hence does not compute the BroadcastStyle or the axes, and does not perform
# other actions specified in `broadcasted` overloads, like those in ClimaCore).
# Convert to this lazy form before factorization and convert back to the regular
# form afterward. The broadcasts returned by user-defined functions must still
# be in the regular form, since otherwise any errors during the construction of
# Broadcasted objects will be very difficult to trace back to the user's code.

# TODO: Add support for multi-argument linear operations; e.g.,
# Operators.WeightedInterpolationOperator or Geometry.transform.

const LinearOneArgOperation = Union{
    Operators.FiniteDifferenceOperator,
    Operators.SpectralElementOperator,
    typeof(adjoint),
    typeof(transpose),
} # anything else?

const Addition = Union{typeof(+), typeof(RecursiveApply.radd)}
const Subtraction = Union{typeof(-), typeof(RecursiveApply.rsub)}
const Multiplication = Union{typeof(*), typeof(RecursiveApply.rmul)}
const Division = Union{typeof(/), typeof(RecursiveApply.rdiv)}
const MulAddOperation = Union{typeof(muladd), typeof(RecursiveApply.rmuladd)}

const Scalar = Union{Number, Ref, NTuple{1}} # anything else?

# Speed up the materialization of a Broadcasted object by recursively
#     - expanding out nested sums,
#     - factoring out negations from sums, and
#     - factoring out linear operations from sums, negations, differences,
#       products with scalars, quotients with a scalar in the denominator, and
#       "muladd"s with a scalar in one of the first two arguments.
# This also allows a single boundary condition to apply to a ClimaCore operator
# that is used in multiple tendency terms, since that operator will only be
# evaluated once per tendency evaluation.
factorize_bc(x) = x
function factorize_bc(bc::Broadcasted)
    args = map(factorize_bc, bc.args)
    bc.f isa Addition && return factorize_sum(args...)
    bc.f isa Subtraction && return factorize_negation_or_difference(args...)
    bc.f isa Multiplication && return factorize_product(args...)
    bc.f isa Division && return factorize_quotient(args...)
    bc.f isa MulAddOperation && return factorize_muladd(args...)
    return broadcasted(bc.f, args...)
end

factorize_sum(args...) = _factorize_sum((), expand_sums(args)...)
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
            bc = factorize_sum(arg.args[1], map(bc -> bc.args[1], same_f)...)
            arg = broadcasted(arg.f, bc)
        end
        return _factorize_sum((factorized_args..., arg), not_same_f...)
    end
    return _factorize_sum((factorized_args..., arg), args...)
end

function factorize_negation_or_difference(arg)
    if is_linear_1arg_bc(arg)
        bc = factorize_negation_or_difference(arg.args[1])
        return broadcasted(arg.f, bc)
    end
    return broadcasted(-, arg)
end
function factorize_negation_or_difference(arg1, arg2)
    if is_linear_1arg_bc(arg1) && is_linear_1arg_bc(arg2) && arg1.f === arg2.f
        bc = factorize_negation_or_difference(arg1.args[1], arg2.args[1])
        return broadcasted(arg1.f, bc)
    end
    return broadcasted(-, arg1, arg2)
end

function factorize_product(args...)
    scalars, not_scalars = partition(x -> x isa Scalar, args)
    if length(not_scalars) == 1 && is_linear_1arg_bc(not_scalars[1])
        bc = factorize_product(not_scalars[1].args[1], scalars...)
        return broadcasted(not_scalars[1].f, bc)
    end
    return broadcasted(*, args...)
end

function factorize_quotient(arg1, arg2)
    if is_linear_1arg_bc(arg1) && arg2 isa Scalar
        bc = factorize_quotient(arg1.args[1], arg2)
        return broadcasted(arg1[1].f, bc)
    end
    return broadcasted(/, arg1, arg2)
end

function factorize_muladd(arg1, arg2, arg3)
    if is_linear_1arg_bc(arg3)
        if is_linear_1arg_bc(arg1) && arg2 isa Scalar && arg1.f === arg3.f
            bc = factorize_muladd(arg1.args[1], arg2, arg3.args[1])
            return broadcasted(arg3[1].f, bc)
        elseif arg1 isa Scalar && is_linear_1arg_bc(arg2) && arg2.f === arg3.f
            bc = factorize_muladd(arg1, arg2.args[1], arg3.args[1])
            return broadcasted(arg3[1].f, bc)
        end
    end
    return broadcasted(muladd, arg1, arg2, arg3)
end

##
## Helper functions
##

expand_sums(args) = Base.afoldl(_expand_sums, (), args...)
_expand_sums(expanded_args, arg) = arg isa Broadcasted && arg.f isa Addition ?
    (expanded_args..., arg.args...) : (expanded_args..., arg)

is_linear_1arg_bc(x) =
    x isa Broadcasted && x.f isa LinearOneArgOperation && length(x.args) == 1

# Return a tuple of xs that satisfy condition f and a tuple of xs that don't.
# (This is like Base.filter_rec(f, xs::Tuple), except that it returns both the
# "do"s and the "don't"s, rather than just the "do"s.)
partition(f, xs) = Base.afoldl(
    ((dos, donts), x) -> f(x) ? ((dos..., x), donts) : (dos, (donts..., x)),
    ((), ()),
    xs...,
)
