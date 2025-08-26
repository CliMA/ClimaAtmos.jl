using Test

import Distributions

import ClimaAtmos

function percentile_bounds_mean_norm_distributions(
    low_percentile::FT,
    high_percentile::FT,
) where {FT <: Real}
    gauss_int(x) = -exp(-x * x / 2) / sqrt(2 * pi)
    xp_low = Distributions.quantile(Distributions.Normal(), low_percentile)
    xp_high = Distributions.quantile(Distributions.Normal(), high_percentile)
    return (gauss_int(xp_high) - gauss_int(xp_low)) / max(
        Distributions.cdf(Distributions.Normal(), xp_high) -
        Distributions.cdf(Distributions.Normal(), xp_low),
        eps(FT),
    )
end

@testset "Gauss quantile" begin
    for p in 0.0:0.1:1.0
        @test ClimaAtmos.gauss_quantile(p) ≈
              Distributions.quantile(Distributions.Normal(), p) rtol = 1e-3
    end

    for p in 0.0:0.1:1.0
        for q in 0.0:0.1:1.0
            @test ClimaAtmos.percentile_bounds_mean_norm(p, q) ≈
                  percentile_bounds_mean_norm_distributions(p, q) rtol = 1e-3
        end
    end
end
