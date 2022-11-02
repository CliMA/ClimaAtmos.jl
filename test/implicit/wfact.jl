
function verify_wfact_matrix(W, Y, p, dtγ, t)
    (; ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄, ∂ᶜ𝕋ₜ∂ᶠ𝕄_field) = W
    (; ᶜts) = p

    if eltype(ᶜts) <: TD.PhaseEquil
        error("This function is incompatible with $(typeof(ᶜts))")
    end

    # Checking every column takes too long, so just check one.
    i, j, h = 1, 1, 1
    args = (implicit_tendency!, Y, p, t, i, j, h)
    ᶜ𝔼_name = filter(CA.is_energy_var, propertynames(Y.c))[1]

    @assert matrix_column(∂ᶜρₜ∂ᶠ𝕄, axes(Y.f), i, j, h) ≈
            exact_column_jacobian_block(args..., (:c, :ρ), (:f, :w))
    @assert matrix_column(∂ᶠ𝕄ₜ∂ᶜ𝔼, axes(Y.c), i, j, h) ≈
            exact_column_jacobian_block(args..., (:f, :w), (:c, ᶜ𝔼_name))
    @assert matrix_column(∂ᶠ𝕄ₜ∂ᶠ𝕄, axes(Y.f), i, j, h) ≈
            exact_column_jacobian_block(args..., (:f, :w), (:f, :w))
    for ᶜρc_name in filter(CA.is_tracer_var, propertynames(Y.c))
        ∂ᶜρcₜ∂ᶠ𝕄 = getproperty(∂ᶜ𝕋ₜ∂ᶠ𝕄_field, ᶜρc_name)
        ᶜρc_tuple = (:c, ᶜρc_name)
        @assert matrix_column(∂ᶜρcₜ∂ᶠ𝕄, axes(Y.f), i, j, h) ≈
                exact_column_jacobian_block(args..., ᶜρc_tuple, (:f, :w))
    end

    ∂ᶜ𝔼ₜ∂ᶠ𝕄_approx = matrix_column(∂ᶜ𝔼ₜ∂ᶠ𝕄, axes(Y.f), i, j, h)
    ∂ᶜ𝔼ₜ∂ᶠ𝕄_exact =
        exact_column_jacobian_block(args..., (:c, ᶜ𝔼_name), (:f, :w))
    if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact
        @assert ∂ᶜ𝔼ₜ∂ᶠ𝕄_approx ≈ ∂ᶜ𝔼ₜ∂ᶠ𝕄_exact
    else
        err = norm(∂ᶜ𝔼ₜ∂ᶠ𝕄_approx .- ∂ᶜ𝔼ₜ∂ᶠ𝕄_exact) / norm(∂ᶜ𝔼ₜ∂ᶠ𝕄_exact)
        @assert err < 1e-6
        # Note: the highest value seen so far is ~3e-7 (only applies to ρe_tot)
    end

    ∂ᶠ𝕄ₜ∂ᶜρ_approx = matrix_column(∂ᶠ𝕄ₜ∂ᶜρ, axes(Y.c), i, j, h)
    ∂ᶠ𝕄ₜ∂ᶜρ_exact = exact_column_jacobian_block(args..., (:f, :w), (:c, :ρ))
    if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
        @assert ∂ᶠ𝕄ₜ∂ᶜρ_approx ≈ ∂ᶠ𝕄ₜ∂ᶜρ_exact
    else
        err = norm(∂ᶠ𝕄ₜ∂ᶜρ_approx .- ∂ᶠ𝕄ₜ∂ᶜρ_exact) / norm(∂ᶠ𝕄ₜ∂ᶜρ_exact)
        @assert err < 0.03
        # Note: the highest value seen so far for ρe_tot is ~0.01, and the
        # highest value seen so far for ρθ is ~0.02
    end
end
