struct DebugJacobian{E <: ExactJacobian, A <: ApproxJacobian} <:
       JacobianAlgorithm
    exact_jacobian_mode::E
    approx_jacobian_mode::A
    use_exact_jacobian::Bool
end
function DebugJacobian(
    approx_jacobian_mode;
    use_exact_jacobian = true,
    exact_jacobian_mode_kwargs...,
)
    exact_jacobian_mode = ExactJacobian(;
        preserve_unfactorized_jacobian = true,
        exact_jacobian_mode_kwargs...,
    )
    return DebugJacobian(
        exact_jacobian_mode,
        approx_jacobian_mode,
        use_exact_jacobian,
    )
end

jacobian_cache(alg::DebugJacobian, Y, p) = (;
    exact_jacobian_cache = jacobian_cache(alg.exact_jacobian_mode, Y, p),
    approx_jacobian_cache = jacobian_cache(alg.approx_jacobian_mode, Y, p),
    similar_to_x = similar(Y),
    x_error = similar(Y),
    x_rel_error = similar(Y),
    compute_error_ref = Ref{Bool}(),
    t_ref = Ref{typeof(p.t_end)}(),
)

always_update_exact_jacobian(alg::DebugJacobian) =
    always_update_exact_jacobian(alg.exact_jacobian_mode)

function factorize_exact_jacobian!(alg::DebugJacobian, cache, Y, p, dtγ, t)
    factorize_exact_jacobian!(
        alg.exact_jacobian_mode,
        cache.exact_jacobian_cache,
        Y,
        p,
        dtγ,
        t,
    )
    cache.compute_error_ref[] = true
    cache.t_ref[] = t
end

approximate_jacobian!(alg::DebugJacobian, cache, Y, p, dtγ, t) =
    (!alg.use_exact_jacobian || cache.compute_error_ref[]) &&
    approximate_jacobian!(
        alg.approx_jacobian_mode,
        cache.approx_jacobian_cache,
        Y,
        p,
        dtγ,
        t,
    )

function invert_jacobian!(alg::DebugJacobian, cache, x, b)
    (; exact_jacobian_mode, approx_jacobian_mode, use_exact_jacobian) = alg
    (; exact_jacobian_cache, approx_jacobian_cache) = cache
    (; similar_to_x, x_error, x_rel_error, compute_error_ref, t_ref) = cache
    if compute_error_ref[]
        x_exact = use_exact_jacobian ? x : similar_to_x
        x_approx = use_exact_jacobian ? similar_to_x : x
        x_exact .= x_approx .= NaN # TODO: Remove this sanity check
        invert_jacobian!(exact_jacobian_mode, exact_jacobian_cache, x_exact, b)
        invert_jacobian!(
            approx_jacobian_mode,
            approx_jacobian_cache,
            x_approx,
            b,
        )
        compute_error_ref[] = false

        @. x_error = abs(x_approx - x_exact)
        @. x_rel_error = ifelse(
            x_approx == x_exact == 0,
            0,
            abs((x_approx - x_exact) / x_exact),
        )

        rms(x) = sqrt(mean(value -> value^2, x))
        field_vector_rms(x) =
            rms(map(rms ∘ Fields.backing_array, Fields._values(x)))
        field_vector_max(x) =
            maximum(map(maximum ∘ Fields.backing_array, Fields._values(x)))
        rel_rms_error = field_vector_rms(x_error) / field_vector_rms(x_exact)
        rel_max_error = field_vector_max(x_error) / field_vector_max(x_exact)
        rms_rel_error = field_vector_rms(x_rel_error)
        max_rel_error = field_vector_max(x_rel_error)
        @info "Error of approximate implicit solve at t = $(t_ref[]):\n  \
               relative RMS error = $(@sprintf("%.3e", rel_rms_error))\n  \
               relative max error = $(@sprintf("%.3e", rel_max_error))\n  \
               RMS relative error = $(@sprintf("%.3e", rms_rel_error))\n  \
               max relative error = $(@sprintf("%.3e", max_rel_error))"
        @assert rel_rms_error < 1 # TODO: Make this test more rigorous
    elseif use_exact_jacobian
        invert_jacobian!(exact_jacobian_mode, exact_jacobian_cache, x, b)
    else
        invert_jacobian!(approx_jacobian_mode, approx_jacobian_cache, x, b)
    end
end
