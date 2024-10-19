struct DebugJacobian{E <: ExactJacobian, A <: ApproxJacobian} <:
       JacobianAlgorithm
    exact_jacobian_alg::E
    approx_jacobian_alg::A
    use_exact_jacobian::Bool
end
DebugJacobian(
    exact_jacobian_alg,
    approx_jacobian_alg;
    use_exact_jacobian = true,
) = DebugJacobian(exact_jacobian_alg, approx_jacobian_alg, use_exact_jacobian)

jacobian_cache(alg::DebugJacobian, Y, atmos) = (;
    jacobian_cache(alg.exact_jacobian_alg, Y, atmos)...,
    jacobian_cache(alg.approx_jacobian_alg, Y, atmos)...,
)

always_update_exact_jacobian(alg::DebugJacobian) =
    always_update_exact_jacobian(alg.exact_jacobian_alg)

factorize_exact_jacobian!(alg::DebugJacobian, cache, Y, p, dtγ, t) =
    factorize_exact_jacobian!(alg.exact_jacobian_alg, cache, Y, p, dtγ, t)

approximate_jacobian!(alg::DebugJacobian, cache, Y, p, dtγ, t) =
    approximate_jacobian!(alg.approx_jacobian_alg, cache, Y, p, dtγ, t)

invert_jacobian!(alg::DebugJacobian, cache, x, b) =
    alg.use_exact_jacobian ?
    invert_jacobian!(alg.exact_jacobian_alg, cache, x, b) :
    invert_jacobian!(alg.approx_jacobian_alg, cache, x, b)
