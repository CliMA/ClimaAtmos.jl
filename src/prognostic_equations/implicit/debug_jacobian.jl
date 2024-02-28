"""
    DebugJacobian(
        approx_jacobian_algorithm,
        use_exact_jacobian,
        only_debug_first_column_jacobian,
    )

A `JacobianAlgorithm` that simultaneously computes an `ExactJacobian` and an
`ApproxJacobian`, so that the quality of the approximation can be assessed. The
`use_exact_jacobian` flag controls whether the exact Jacobian is used in the
implicit solver instead of the approximation, and, when `use_exact_jacobian` is
`false`, the `only_debug_first_column_jacobian` flag controls whether the exact
Jacobian is only evaluated in the first column.
"""
struct DebugJacobian{A <: ApproxJacobian} <: JacobianAlgorithm
    approx_jacobian_algorithm::A
    use_exact_jacobian::Bool
    only_debug_first_column_jacobian::Bool
end

contains_any_fields(::Union{Fields.Field, Fields.FieldVector}) = true
contains_any_fields(x::T) where {T} =
    fieldcount(T) == 0 ? false : unrolled_any(StaticOneTo(fieldcount(T))) do i
        contains_any_fields(getfield(x, i))
    end

first_column(x::Union{Fields.Field, Fields.FieldVector}) =
    Fields.column(x, 1, 1, 1)
first_column(x::Union{Tuple, NamedTuple}) = unrolled_map(first_column, x)
first_column(x::T) where {T} =
    fieldcount(T) == 0 || !contains_any_fields(x) ? x :
    T.name.wrapper(
        ntuple(i -> first_column(getfield(x, i)), Val(fieldcount(T)))...,
    )

required_columns_for_exact_jacobian(alg, x) =
    alg.use_exact_jacobian || !alg.only_debug_first_column_jacobian ? x :
    first_column(x)

function jacobian_cache(alg::DebugJacobian, Y, atmos)
    Y_or_column = required_columns_for_exact_jacobian(alg, Y)
    exact_cache = jacobian_cache(ExactJacobian(), Y_or_column, atmos)
    approx_cache = jacobian_cache(alg.approx_jacobian_algorithm, Y, atmos)
    return (; exact_cache..., approx_cache...)
end

update_jacobian!(alg::DebugJacobian, cache, Y, p, dtγ, t) =
    alg.use_exact_jacobian ?
    update_jacobian!(ExactJacobian(), cache, Y, p, dtγ, t) :
    update_jacobian!(alg.approx_jacobian_algorithm, cache, Y, p, dtγ, t)

function update_and_check_jacobian!(alg::DebugJacobian, cache, Y, p, dtγ, t)
    Y_or_column = required_columns_for_exact_jacobian(alg, Y)
    p_or_column = required_columns_for_exact_jacobian(alg, p)
    update_jacobian!(ExactJacobian(), cache, Y_or_column, p_or_column, dtγ, t)
    update_jacobian!(alg.approx_jacobian_algorithm, cache, Y, p, dtγ, t)
    # TODO: Add a quantitative check of the Jacobian approximation.
end

invert_jacobian!(alg::DebugJacobian, cache, ΔY, R) =
    alg.use_exact_jacobian ? invert_jacobian!(ExactJacobian(), cache, ΔY, R) :
    invert_jacobian!(alg.approx_jacobian_algorithm, cache, ΔY, R)

function save_jacobian!(alg::DebugJacobian, cache, Y, dtγ, t)
    Y_or_column = required_columns_for_exact_jacobian(alg, Y)
    save_jacobian!(ExactJacobian(), cache, Y_or_column, dtγ, t)
    save_jacobian!(alg.approx_jacobian_algorithm, cache, Y, dtγ, t)
    # TODO: Save the average/maximum difference between the approximate and
    # exact Jacobians, instead of computing the difference between their
    # averages/maxima when plotting.
end
