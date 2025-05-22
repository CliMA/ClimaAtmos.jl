"""
    DebugJacobian(
        sparse_jacobian_algorithm,
        use_auto_jacobian,
        only_debug_first_column_jacobian,
    )

A `JacobianAlgorithm` that simultaneously computes an `AutoDenseJacobian` and a
sparse `JacobianAlgorithm`, so that the quality of the sparse approximation can
be assessed. The `use_auto_jacobian` flag controls whether the exact Jacobian
is used in the implicit solver instead of the approximation, and, when
`use_auto_jacobian` is `false`, the `only_debug_first_column_jacobian` flag
controls whether the exact Jacobian is only evaluated in the first column.
"""
struct DebugJacobian{A <: JacobianAlgorithm, EJ, FC} <: JacobianAlgorithm
    sparse_jacobian_algorithm::A
end
DebugJacobian(
    sparse_jacobian_algorithm,
    use_auto_jacobian,
    only_debug_first_column_jacobian,
) = DebugJacobian{
    typeof(sparse_jacobian_algorithm),
    use_auto_jacobian,
    only_debug_first_column_jacobian,
}(
    sparse_jacobian_algorithm,
)

# TODO: Change use_auto_jacobian to use_dense_jacobian.

use_auto_jacobian(::DebugJacobian{A, EJ, FC}) where {A, EJ, FC} = EJ

only_debug_first_column_jacobian(::DebugJacobian{A, EJ, FC}) where {A, EJ, FC} =
    FC

required_columns_for_exact_jacobian(alg, x) =
    use_auto_jacobian(alg) || !only_debug_first_column_jacobian(alg) ? x :
    first_column(x)

function jacobian_cache(alg::DebugJacobian, Y, atmos)
    Y_for_exact = required_columns_for_exact_jacobian(alg, Y)
    exact_cache = jacobian_cache(AutoDenseJacobian(), Y_for_exact, atmos)
    approx_cache = jacobian_cache(alg.sparse_jacobian_algorithm, Y, atmos)
    return (; exact_cache..., approx_cache...)
end

update_jacobian!(alg::DebugJacobian, cache, Y, p, dtγ, t) =
    use_auto_jacobian(alg) ?
    update_jacobian!(AutoDenseJacobian(), cache, Y, p, dtγ, t) :
    update_jacobian!(alg.sparse_jacobian_algorithm, cache, Y, p, dtγ, t)

function update_and_check_jacobian!(alg::DebugJacobian, cache, Y, p, dtγ, t)
    update_for_exact! =
        use_auto_jacobian(alg) ? update_jacobian! :
        update_jacobian_skip_factorizing!
    Y_for_exact = required_columns_for_exact_jacobian(alg, Y)
    p_for_exact = required_columns_for_exact_jacobian(alg, p)
    update_for_exact!(
        AutoDenseJacobian(),
        cache,
        Y_for_exact,
        p_for_exact,
        dtγ,
        t,
    )
    update_jacobian!(alg.sparse_jacobian_algorithm, cache, Y, p, dtγ, t)
    # TODO: Add a quantitative check of the Jacobian approximation.
end

invert_jacobian!(alg::DebugJacobian, cache, ΔY, R) =
    use_auto_jacobian(alg) ?
    invert_jacobian!(AutoDenseJacobian(), cache, ΔY, R) :
    invert_jacobian!(alg.sparse_jacobian_algorithm, cache, ΔY, R)

function save_jacobian!(alg::DebugJacobian, cache, Y, dtγ, t)
    Y_for_exact = required_columns_for_exact_jacobian(alg, Y)
    save_jacobian!(AutoDenseJacobian(), cache, Y_for_exact, dtγ, t)
    save_jacobian!(alg.sparse_jacobian_algorithm, cache, Y, dtγ, t)
    # TODO: Save the average/maximum difference between the approximate and
    # exact Jacobians, instead of just computing the difference between their
    # averages/maxima when plotting.
end
