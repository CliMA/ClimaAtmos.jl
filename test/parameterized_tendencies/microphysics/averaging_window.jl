#=
Unit tests for the 1M microphysics averaging-window plumbing: the
`imex_gamma` extractor, the `MicrophysicsAveragingWindow` dispatch tags,
and the `AtmosCache` field resolution driven by `update_cache_every`.
=#

using Test
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS

include("../../test_helpers.jl")

@testset "Microphysics averaging window" begin

    @testset "imex_gamma extracts the DIRK diagonal" begin
        newton = CTS.NewtonsMethod(; max_iters = 1)
        γ_ars222_expected = 1 - 1 / sqrt(2.0)
        γ_ars343_expected = 0.4358665215084590
        γ_ars233_expected = (3 + sqrt(3.0)) / 6
        for (name, γ_expected) in (
            (CTS.ARS222(), γ_ars222_expected),
            (CTS.ARS343(), γ_ars343_expected),
            (CTS.ARS233(), γ_ars233_expected),
        )
            alg = CTS.IMEXAlgorithm(name, newton)
            γ = CA.imex_gamma(alg)
            @test γ !== nothing
            @test γ ≈ γ_expected rtol = 1e-6
        end

        # Fully explicit algorithm has no implicit stages → `nothing`.
        exp_alg = CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher())
        @test CA.imex_gamma(exp_alg) === nothing

        # `nothing` (no ODE scheme) → `nothing`.
        @test CA.imex_gamma(nothing) === nothing
    end

    @testset "mp_averaging_dt dispatches to the right scalar" begin
        p_mock = (; dt = 100.0, dtγ = 42.0)
        @test CA.mp_averaging_dt(p_mock, CA.StepAveraging()) == 100.0
        @test CA.mp_averaging_dt(p_mock, CA.StageAveraging()) == 42.0
    end

    @testset "AtmosCache resolution end-to-end" begin
        for ucae in ("step", "stage")
            config = CA.AtmosConfig(
                Dict(
                    "initial_condition" => "PrecipitatingColumn",
                    "microphysics_model" => "1M",
                    "config" => "column",
                    "ode_algo" => "ARS343",
                    "update_cache_every" => ucae,
                    "output_default_diagnostics" => false,
                ),
                job_id = "mp_avg_$(ucae)",
            )
            (; Y, p) = generate_test_simulation(config)
            FT = eltype(p.dt)
            γ_ars343 = CA.imex_gamma(
                CTS.IMEXAlgorithm(
                    CTS.ARS343(),
                    CTS.NewtonsMethod(; max_iters = 1),
                ),
            )
            @test p.dtγ ≈ FT(γ_ars343) * p.dt rtol = 1e-6
            if ucae == "step"
                @test p.mp_averaging isa CA.StepAveraging
                @test CA.mp_averaging_dt(p, p.mp_averaging) == p.dt
            else
                @test p.mp_averaging isa CA.StageAveraging
                @test CA.mp_averaging_dt(p, p.mp_averaging) == p.dtγ
            end
        end
    end

    @testset "Non-IMEX schemes fall back to step averaging" begin
        # Even if the user asks for `update_cache_every = "stage"`, an
        # explicit scheme has no γ and we fall back to `StepAveraging`
        # (behavior-preserving, since `dtγ` is undefined).
        config = CA.AtmosConfig(
            Dict(
                "initial_condition" => "PrecipitatingColumn",
                "microphysics_model" => "1M",
                "config" => "column",
                "ode_algo" => "SSP33ShuOsher",
                "update_cache_every" => "stage",
                "output_default_diagnostics" => false,
            ),
            job_id = "mp_avg_nonimex",
        )
        (; p) = generate_test_simulation(config)
        @test p.mp_averaging isa CA.StepAveraging
        @test p.dtγ == 0
        @test CA.mp_averaging_dt(p, p.mp_averaging) == p.dt
    end
end
