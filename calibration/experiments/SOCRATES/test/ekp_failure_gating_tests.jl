using Test
import ClimaCalibrate as CAL
import EnsembleKalmanProcesses as EKP
using LinearAlgebra
using Random

@testset "SOCRATES EKP downstream failure gating (synthetic)" begin
    Random.seed!(11)

    mktempdir() do tmp
        obs = EKP.Observation(
            Dict(
                "samples" => [1.0, 2.0],
                "covariances" => Diagonal([1e-3, 1e-3]),
                "names" => "case",
            ),
        )
        obs_series = EKP.ObservationSeries([obs], EKP.FixedMinibatcher([1]), ["case"])
        prior = CAL.get_prior(joinpath(@__DIR__, "..", "prior.toml"))

        CAL.initialize(
            12,
            obs_series,
            nothing,
            prior,
            tmp;
            scheduler = EKP.DataMisfitController(on_terminate = "continue"),
        )

        # Use non-identical synthetic forward maps to avoid artificial clash warnings.
        G_good = randn(2, 12)
        G_good[1, :] .+= range(0.0, 0.55; length = 12)
        G_good[2, :] .+= range(0.25, 0.8; length = 12)

        CAL.save_G_ensemble(tmp, 0, G_good)
        eki1 = CAL.update_ensemble(tmp, 0, prior)
        @test all(isfinite, EKP.get_u_final(eki1))

        # Partial failure case (>50% NaN columns): should still return via failure handler.
        G_many_nan = copy(G_good)
        G_many_nan[:, 1:9] .= NaN
        CAL.save_G_ensemble(tmp, 1, G_many_nan)
        eki2 = CAL.update_ensemble(tmp, 1, prior)
        @test all(isfinite, EKP.get_u_final(eki2))

        # Full failure case (all NaN columns): EKP update should hard-fail.
        G_all_nan = fill(NaN, 2, 12)
        CAL.save_G_ensemble(tmp, 2, G_all_nan)
        @test_throws Exception CAL.update_ensemble(tmp, 2, prior)
    end
end
