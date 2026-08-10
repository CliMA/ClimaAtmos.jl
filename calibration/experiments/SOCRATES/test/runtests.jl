using Test: Test

include(joinpath(@__DIR__, "..", "src", "calibration", "SocratesCalibration.jl"))
using .SocratesCalibration: SocratesCalibration as SC

"""The cases with a reference to score against: RF11 has no Obs forcing."""
testcases() =
    [c for c in SC.SS.SM.socrates_cases() if SC.SS.case_name(c) != "RF11_Obs"]

Test.@testset "SOCRATES" begin

    Test.@testset "cases" begin
        Test.@test length(SC.SS.SM.socrates_cases()) == 11
        Test.@test SC.SS.case_name(SC.SS.SM.socrates_case("RF09_Obs")) == "RF09_Obs"
        # A flight with no artifact for that forcing is rejected at parse, not at the first run.
        Test.@test_throws Exception SC.SS.SM.socrates_case("RF99_Obs")
        Test.@test_throws Exception SC.SS.SM.socrates_case("RF11_Obs")
        Test.@test_throws Exception SC.SS.SM.socrates_case("nonsense")
        for case in testcases()
            t0, t1 = SC.SS.SM.score_window(case)
            Test.@test 0 <= t0 < t1 <= SC.SS.SM.t_end(case)
        end
    end

    Test.@testset "grid" begin
        for case in testcases()
            zc = SC.SS.SM.native_z(case)
            zf = SC.SS.SM.native_faces(case)
            Test.@test length(zf) == length(zc) + 1
            Test.@test issorted(zf)
            Test.@test first(zf) == 0.0
            Test.@test SC.SS.SM.centers_from_faces(zf) ≈ zc
            Test.@test SC.SS.SM.socrates_z(Float64, case) ≈ zc
            Test.@test SC.SS.SM.z_max_default(case) == last(zf)
        end

        # Coarsening drops faces, so the result is a sorted subset with every cell >= dz_min.
        case = SC.SS.SM.socrates_case("RF09_Obs")
        native = SC.SS.SM.native_faces(case)
        for dz in (50, 100, 200, 500)
            zf = SC.SS.SM.coarsen_faces_to_dz_min(native, dz)
            Test.@test issorted(zf)
            Test.@test issubset(Set(zf), Set(native))
            Test.@test minimum(diff(zf)) >= dz
            Test.@test first(zf) == first(native) && last(zf) == last(native)
            Test.@test length(SC.SS.SM.socrates_z(Float64, case; dz_min = dz)) ==
                       length(zf) - 1
        end
        Test.@test SC.SS.SM.coarsen_faces_to_dz_min(native, nothing) == native

        # Centres that are not midpoints of a grid from the surface have no faces.
        Test.@test_throws Exception SC.SS.SM.faces_from_centers([12.5, 112.0, 130.0])
        Test.@test SC.SS.SM.faces_from_centers([12.5, 112.0, 212.0, 300.0]) ==
                   [0.0, 25.0, 199.0, 225.0, 375.0]
    end

    Test.@testset "score transform" begin
        transform = SC.SS.ScoreTransform()
        Test.@test SC.SS.mean_nonzero_elements([0.0, 2.0, 4.0]) == 3.0
        Test.@test SC.SS.mean_nonzero_elements([0.0, 0.0]; all_zero = 7.0) == 7.0

        characteristic = SC.SS.DEFAULT_CHARACTERISTIC["clw"]
        scaling = SC.SS.DEFAULT_OBS_VAR_SCALING["clw"]
        Test.@test SC.SS.pool_var(transform, "clw", zeros(5)) ≈ scaling * characteristic^2
        Test.@test SC.SS.pool_var(transform, "clw", fill(1.0e-3, 5)) ≈ scaling * (1.0e-3)^2
        Test.@test SC.SS.pool_var(transform, "clw", fill(1.0e-9, 5)) ≈
                   scaling * characteristic^2
        Test.@test SC.SS.pool_var(transform, "clw", [0.0, 2.0e-3, 0.0, 2.0e-3]) ≈
                   scaling * (2.0e-3)^2
        Test.@test_throws Exception SC.SS.pool_var(transform, "not_a_variable", ones(3))

        for name in SC.SS.REFERENCE_VARS
            Test.@test SC.SS.normalized_characteristic(transform, name) ≈
                       1 / sqrt(SC.SS.DEFAULT_OBS_VAR_SCALING[name])
        end

        # A relative term plus a floor that does not depend on the case.
        diagonal = SC.SS.uncertainty_diagonal(transform, "clw", [1.0 1.0; 0.0 0.0])
        floor =
            SC.SS.DEFAULT_UNCERTAINTY_FLOOR["clw"] *
            SC.SS.normalized_characteristic(transform, "clw")
        Test.@test diagonal[2] ≈ floor^2
        Test.@test diagonal[1] ≈ SC.SS.DEFAULT_ADDITIONAL_UNCERTAINTY["clw"]^2 + floor^2
        Test.@test all(>(0), diagonal)

        Test.@test SC.SS.nanmean([1.0, NaN, 3.0]) == 2.0
        Test.@test SC.SS.nanmean([NaN, NaN]) == 0.0
    end

    Test.@testset "reference" begin
        case = SC.SS.SM.socrates_case("RF13_Obs")
        reference = SC.SS.les_outputvars(case; vars = ("clw", "lwp"))
        Test.@test Set(keys(reference)) == Set(["clw", "lwp"])
        Test.@test SC.ClimaAnalysis.has_altitude(reference["clw"])
        Test.@test !SC.ClimaAnalysis.has_altitude(reference["lwp"])
        Test.@test first(reference["clw"].dims["time"]) == 0.0

        # A cell-centred profile is valid across its end cells, and a level a hair past the face —
        # which the file's Float32 altitudes produce — is on the boundary, not outside it.
        zc = SC.SS.SM.native_z(case)
        face = last(SC.SS.SM.faces_from_centers(zc))
        mid = (last(zc) + face) / 2
        out = SC.SS.reference_on_levels(reference["clw"], [last(zc), mid, face])
        Test.@test out.dims["z"] == [last(zc), mid, face]
        Test.@test out.data[1, 1] == out.data[2, 1] == out.data[3, 1]
        Test.@test_throws Exception SC.SS.reference_on_levels(
            reference["clw"],
            [face + 30.0],
        )

        # Resampling onto the model's own levels is faithful to the file.
        levels = SC.SS.scored_levels(SC.SS.SM.socrates_z(Float64, case), (0.0, 2000.0))
        resampled = SC.SS.reference_on_levels(reference["clw"], levels)
        raw = Array{Float64}(reference["clw"].data)[1:length(levels), :]
        Test.@test maximum(abs.(Array{Float64}(resampled.data) .- raw)) < 1.0e-8
    end

    Test.@testset "z_bounds" begin
        for case in testcases()
            lo, hi = SC.SS.z_bounds(case)
            Test.@test lo == 0.0
            Test.@test 0 < hi <= SC.SS.SM.z_max_default(case)
            Test.@test !isempty(
                SC.SS.scored_levels(SC.SS.SM.socrates_z(Float64, case), (lo, hi)),
            )
        end
        Test.@test last(SC.SS.z_bounds(SC.SS.SM.socrates_case("RF09_Obs"))) ==
                   SC.SS.OBS_Z_TOP[9]
        Test.@test last(SC.SS.z_bounds(SC.SS.SM.socrates_case("RF13_Obs"))) ==
                   SC.SS.OBS_Z_TOP[13]
    end

    Test.@testset "observations" begin
        for case in testcases()
            observation = SC.case_observation(case)
            y = SC.EKP.get_obs(observation)
            covariance = SC.EKP.get_obs_noise_cov(observation; build = false)
            covariance = covariance isa Vector ? only(covariance) : covariance
            n_levels = length(
                SC.SS.scored_levels(
                    SC.SS.SM.socrates_z(Float64, case),
                    SC.SS.z_bounds(case),
                ),
            )

            # Four profiles over the scored levels, plus four scalars.
            Test.@test length(y) == 4 * n_levels + 4
            Test.@test all(isfinite, y)

            # A low-rank measured part plus a strictly positive diagonal.
            diagonal = SC.LinearAlgebra.diag(SC.EKP.get_diag_cov(covariance))
            Test.@test length(diagonal) == length(y)
            Test.@test all(>(0), diagonal)
            Test.@test length(SC.EKP.get_svd_cov(covariance).S) <=
                       size(SC.normalized_reference_series(case).series, 2)

            dense = SC.EKP.get_obs_noise_cov(observation; build = true)
            Test.@test size(dense) == (length(y), length(y))
            Test.@test SC.LinearAlgebra.isposdef(SC.LinearAlgebra.Symmetric(dense))
            # Cross-variable covariance survives: the block is not diagonal.
            Test.@test count(
                !=(0.0),
                dense - SC.LinearAlgebra.Diagonal(SC.LinearAlgebra.diag(dense)),
            ) > 0
        end

        # Observations follow the grid they are told to use.
        case = SC.SS.SM.socrates_case("RF09_Obs")
        coarse_grid = SC.SS.SM.socrates_grid(Float64, case; dz_min = 200)
        coarse = SC.case_observation(case; grid = coarse_grid)
        n_coarse = length(
            SC.SS.scored_levels(
                SC.SS.SM.socrates_z(coarse_grid),
                SC.SS.z_bounds(case),
            ),
        )
        Test.@test length(SC.EKP.get_obs(coarse)) == 4 * n_coarse + 4
        Test.@test length(SC.EKP.get_obs(coarse)) <
                   length(SC.EKP.get_obs(SC.case_observation(case)))
    end

    Test.@testset "prior" begin
        prior = SC.default_prior()
        names = SC.EKP.ParameterDistributions.get_name(prior)
        Test.@test names == SC.prior_names()
        Test.@test length(names) == 4

        # The physical median is the requested mean, and samples stay inside the bounds.
        samples = SC.EKP.ParameterDistributions.sample(
            SC.Random.MersenneTwister(7),
            prior,
            20_000,
        )
        physical = SC.EKP.ParameterDistributions.transform_unconstrained_to_constrained(
            prior,
            samples,
        )
        for (i, name) in enumerate(names)
            mean, lower, upper, _ = SC.DEFAULT_PRIOR_SPEC[Symbol(name)]
            values = vec(physical[i, :])
            Test.@test all(lower .<= values .<= upper)
            Test.@test isapprox(SC.Statistics.median(values), mean; rtol = 0.05)
        end

        Test.@test_throws Exception SC.default_prior((; bad = (1.0, 10.0, 5.0, 1.0)))
        Test.@test_throws Exception SC.default_prior((; bad = (5.0, 1.0, 10.0, 0.0)))
    end

    Test.@testset "interface and ekp" begin
        cases = testcases()
        interface = SC.SocratesInterface(; cases, output_dir = mktempdir())
        Test.@test length(interface.cases) == length(cases)

        # The grid belongs to the interface, since it also fixes the observation levels.
        Test.@test_throws Exception SC.SocratesInterface(;
            cases,
            output_dir = mktempdir(),
            run_kwargs = (; dz_min = 200),
        )

        # Constructing an interface never removes output: a previous attempt is left to be resumed.
        used = mktempdir()
        mkpath(joinpath(used, "iteration_001"))
        SC.SocratesInterface(; cases, output_dir = used)
        Test.@test isdir(joinpath(used, "iteration_001"))

        ekp = SC.build_ekp(interface, SC.default_prior(); ensemble_size = 6)
        Test.@test SC.EKP.get_N_ens(ekp) == 6
        Test.@test SC.EKP.get_process(ekp) isa SC.EKP.Inversion

        # Observation length equals the summed per-variable metadata length, which is what
        # GEnsembleBuilder fills against.
        obs_length = length(SC.EKP.get_obs(ekp))
        metadata = SC.ClimaCalibrate.get_metadata_for_nth_iteration(
            SC.EKP.get_observation_series(ekp),
            1,
        )
        Test.@test obs_length == sum(SC.ClimaAnalysis.flattened_length, metadata)

        # Block-diagonal across cases.
        covariance = SC.EKP.get_obs_noise_cov(ekp; build = true)
        lengths = [
            length(SC.EKP.get_obs(o)) for
            o in SC.EKP.get_observations(SC.EKP.get_observation_series(ekp))
        ]
        Test.@test sum(lengths) == obs_length
        offset = 0
        for len in lengths
            rows = (offset + 1):(offset + len)
            Test.@test all(==(0.0), covariance[rows, setdiff(1:obs_length, rows)])
            offset += len
        end
    end

    Test.@testset "T_stops ratchet" begin
        interface =
            SC.SocratesInterface(; cases = testcases()[1:1], output_dir = mktempdir())
        ekp = SC.build_ekp(
            interface,
            SC.default_prior();
            ensemble_size = 4,
            T_stops = [1.0, 10.0],
        )
        Test.@test ekp.scheduler.terminate_at == 1.0
        # No advance before any algorithmic time is consumed.
        Test.@test SC.ratchet_terminate_at(ekp, [1.0, 10.0]).scheduler.terminate_at == 1.0
        push!(SC.EKP.get_Δt(ekp), 5.0)
        advanced = SC.ratchet_terminate_at(ekp, [1.0, 10.0])
        Test.@test advanced.scheduler.terminate_at == 10.0
        Test.@test typeof(advanced) === typeof(ekp)
        # The scheduler's mutable iteration history survives the rebuild.
        Test.@test advanced.scheduler.iteration === ekp.scheduler.iteration
    end
end
