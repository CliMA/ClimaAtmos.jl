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
struct DebugJacobian{A <: ApproxJacobian, EJ, FC} <: JacobianAlgorithm
    approx_jacobian_algorithm::A
end
DebugJacobian(
    approx_jacobian_algorithm,
    use_exact_jacobian,
    only_debug_first_column_jacobian,
) = DebugJacobian{
    typeof(approx_jacobian_algorithm),
    use_exact_jacobian,
    only_debug_first_column_jacobian,
}(
    approx_jacobian_algorithm,
)

use_exact_jacobian(::DebugJacobian{A, EJ, FC}) where {A, EJ, FC} = EJ

only_debug_first_column_jacobian(::DebugJacobian{A, EJ, FC}) where {A, EJ, FC} =
    FC

required_columns_for_exact_jacobian(alg, x) =
    use_exact_jacobian(alg) || !only_debug_first_column_jacobian(alg) ? x :
    first_column(x)

function jacobian_cache(alg::DebugJacobian, Y, atmos)
    Y_for_exact = required_columns_for_exact_jacobian(alg, Y)
    exact_cache = jacobian_cache(ExactJacobian(), Y_for_exact, atmos)
    approx_cache = jacobian_cache(alg.approx_jacobian_algorithm, Y, atmos)
    return (; exact_cache..., approx_cache...)
end

update_jacobian!(alg::DebugJacobian, cache, Y, p, dtγ, t) =
    use_exact_jacobian(alg) ?
    update_jacobian!(ExactJacobian(), cache, Y, p, dtγ, t) :
    update_jacobian!(alg.approx_jacobian_algorithm, cache, Y, p, dtγ, t)

function update_and_check_jacobian!(alg::DebugJacobian, cache, Y, p, dtγ, t)
    update_for_exact! =
        use_exact_jacobian(alg) ? update_jacobian! :
        update_jacobian_skip_factorizing!
    Y_for_exact = required_columns_for_exact_jacobian(alg, Y)
    p_for_exact = required_columns_for_exact_jacobian(alg, p)
    update_for_exact!(ExactJacobian(), cache, Y_for_exact, p_for_exact, dtγ, t)
    update_jacobian!(alg.approx_jacobian_algorithm, cache, Y, p, dtγ, t)
    # TODO: Add a quantitative check of the Jacobian approximation.
end

invert_jacobian!(alg::DebugJacobian, cache, ΔY, R) =
    use_exact_jacobian(alg) ? invert_jacobian!(ExactJacobian(), cache, ΔY, R) :
    invert_jacobian!(alg.approx_jacobian_algorithm, cache, ΔY, R)

function save_jacobian!(alg::DebugJacobian, cache, Y, dtγ, t)
    Y_for_exact = required_columns_for_exact_jacobian(alg, Y)
    save_jacobian!(ExactJacobian(), cache, Y_for_exact, dtγ, t)
    save_jacobian!(alg.approx_jacobian_algorithm, cache, Y, dtγ, t)
    # TODO: Save the average/maximum difference between the approximate and
    # exact Jacobians, instead of just computing the difference between their
    # averages/maxima when plotting.
end
