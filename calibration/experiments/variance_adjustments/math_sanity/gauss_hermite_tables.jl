# Physicists' Gauss–Hermite nodes/weights for ∫ exp(-x²) f(x) dx ≈ Σ wᵢ f(xᵢ).
# Same closed forms as ClimaAtmos `gauss_hermite` (sgs_quadrature.jl).
function mathsanity_gauss_hermite(::Type{FT}, N::Int) where {FT}
    if N == 1
        return (FT[0], FT[sqrt(π)])
    elseif N == 2
        a = sqrt(FT(0.5))
        return (FT[-a, a], FT[sqrt(π) / 2, sqrt(π) / 2])
    elseif N == 3
        a = sqrt(FT(1.5))
        w0 = FT(2) * sqrt(FT(π)) / 3
        w1 = sqrt(FT(π)) / 6
        return (FT[-a, 0, a], FT[w1, w0, w1])
    elseif N == 4
        a1 = sqrt(FT(3) - sqrt(FT(6))) / sqrt(FT(2))
        a2 = sqrt(FT(3) + sqrt(FT(6))) / sqrt(FT(2))
        w1 = sqrt(FT(π)) / (4 * (FT(3) - sqrt(FT(6))))
        w2 = sqrt(FT(π)) / (4 * (FT(3) + sqrt(FT(6))))
        return (FT[-a2, -a1, a1, a2], FT[w2, w1, w1, w2])
    elseif N == 5
        a1 = sqrt(FT(5) - sqrt(FT(10))) / sqrt(FT(2))
        a2 = sqrt(FT(5) + sqrt(FT(10))) / sqrt(FT(2))
        w0 = FT(8) * sqrt(FT(π)) / 15
        w1 = sqrt(FT(π)) * (FT(7) + FT(2) * sqrt(FT(10))) / 60
        w2 = sqrt(FT(π)) * (FT(7) - FT(2) * sqrt(FT(10))) / 60
        return (FT[-a2, -a1, 0, a1, a2], FT[w2, w1, w0, w1, w2])
    else
        error("Gauss–Hermite order $N not implemented; use N ∈ 1:5.")
    end
end

"""Weights ωᵢ such that E[g(Z)] ≈ Σᵢ ωᵢ g(√2 χᵢ) for Z ~ N(0,1), χᵢ GH nodes."""
function mathsanity_std_normal_expectation_weights(FT, N::Int)
    χ, w = mathsanity_gauss_hermite(FT, N)
    ω = w ./ sqrt(FT(π))
    return χ, ω
end

mathsanity_z_from_χ(χ) = sqrt(2) * χ
