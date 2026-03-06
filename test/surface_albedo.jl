using Test
using ClimaComms
import ClimaAtmos
import Random

Random.seed!(1234)

@testset "set_surface_albedo!" begin
    FT = Float32
    base_config = Dict(
        "initial_condition" => "DYCOMS_RF02",
        "microphysics_model" => "0M",
        "config" => "column",
        "rad" => "clearsky",
        "FLOAT_TYPE" => string(FT),
        "output_default_diagnostics" => false,
    )

    @testset "ConstantAlbedo" begin
        config = ClimaAtmos.AtmosConfig(base_config; job_id = "surface_albedo_constant")
        simulation = ClimaAtmos.AtmosSimulation(config)
        (; u, p, t) = simulation.integrator

        ClimaAtmos.set_surface_albedo!(u, p, t, p.atmos.surface_albedo)

        @test all(p.radiation.rrtmgp_model.direct_sw_surface_albedo .== FT(0.38))
        @test all(p.radiation.rrtmgp_model.diffuse_sw_surface_albedo .== FT(0.38))
    end

    @testset "RegressionFunctionAlbedo" begin
        config_dict = merge(base_config, Dict("albedo_model" => "RegressionFunctionAlbedo"))
        config = ClimaAtmos.AtmosConfig(config_dict; job_id = "surface_albedo_regression")
        simulation = ClimaAtmos.AtmosSimulation(config)
        (; u, p, t) = simulation.integrator

        ClimaAtmos.set_surface_albedo!(u, p, t, p.atmos.surface_albedo)

        # Verify albedo is initialized (not NaN) - precise values checked in testset below
        @test !isnan(sum(p.radiation.rrtmgp_model.direct_sw_surface_albedo))
        @test !isnan(sum(p.radiation.rrtmgp_model.diffuse_sw_surface_albedo))
    end

    @testset "CouplerAlbedo" begin
        config_dict = merge(base_config, Dict("albedo_model" => "CouplerAlbedo"))
        config = ClimaAtmos.AtmosConfig(config_dict; job_id = "surface_albedo_coupler")
        simulation = ClimaAtmos.AtmosSimulation(config)
        (; u, p) = simulation.integrator

        # At t=0, should initialize to default (0.38)
        ClimaAtmos.set_surface_albedo!(u, p, 0.0, p.atmos.surface_albedo)
        @test all(p.radiation.rrtmgp_model.direct_sw_surface_albedo .== FT(0.38))
        @test all(p.radiation.rrtmgp_model.diffuse_sw_surface_albedo .== FT(0.38))

        # At t>0, should remain unchanged (coupler controls updates)
        ClimaAtmos.set_surface_albedo!(u, p, 1.0, p.atmos.surface_albedo)
        @test all(p.radiation.rrtmgp_model.direct_sw_surface_albedo .== FT(0.38))
        @test all(p.radiation.rrtmgp_model.diffuse_sw_surface_albedo .== FT(0.38))
    end
end

@testset "RegressionFunctionAlbedo values (Jin et al 2011)" begin
    FT = Float32

    @testset "Direct albedo (Figure 2)" begin
        # Reference values from Jin et al 2011 Figure 2 (n = 1.34 and 1.2)
        # Rows: (n=1.34, wind=0), (n=1.34, wind=3), (n=1.34, wind=12),
        #       (n=1.2, wind=0), (n=1.2, wind=3), (n=1.2, wind=12)
        reference = [
            [0.021, 0.021, 0.021, 0.022, 0.024, 0.030, 0.044, 0.078, 0.165, 0.379, 0],
            [0.021, 0.021, 0.021, 0.022, 0.024, 0.030, 0.045, 0.078, 0.154, 0.313, 0],
            [0.022, 0.022, 0.022, 0.023, 0.025, 0.032, 0.046, 0.077, 0.135, 0.231, 0],
            [0.008, 0.008, 0.008, 0.009, 0.010, 0.013, 0.022, 0.045, 0.114, 0.316, 0],
            [0.008, 0.008, 0.008, 0.009, 0.010, 0.013, 0.022, 0.045, 0.107, 0.261, 0],
            [0.008, 0.008, 0.009, 0.009, 0.010, 0.014, 0.023, 0.044, 0.093, 0.193, 0],
        ]

        θ_list = FT.(0:(π / 20):(π / 2))
        idx = 0
        for n in FT.([1.34, 1.2])
            for wind in FT.([0, 3, 12])
                idx += 1
                albedo_model = ClimaAtmos.RegressionFunctionAlbedo{FT}(; n)
                α_direct = ClimaAtmos.surface_albedo_direct(albedo_model)
                computed = [α_direct(FT(0), cos(θ), wind) for θ in θ_list]
                @test all(isapprox.(computed, reference[idx]; atol = 1e-3))
            end
        end
    end

    @testset "Diffuse albedo (Figure 4)" begin
        # Reference values from Jin et al 2011 Figure 4
        # Rows: n = 1.2, 1.34, 1.45; Columns: wind = 0:5:25
        reference = [
            [0.042499, 0.037779, 0.035150, 0.033099, 0.031358, 0.029819],
            [0.064824, 0.059713, 0.056866, 0.054646, 0.052761, 0.051095],
            [0.082365, 0.076947, 0.073930, 0.071576, 0.069578, 0.067811],
        ]

        for (i, n) in enumerate(FT.([1.2, 1.34, 1.45]))
            albedo_model = ClimaAtmos.RegressionFunctionAlbedo{FT}(; n)
            α_diffuse = ClimaAtmos.surface_albedo_diffuse(albedo_model)
            computed = [α_diffuse(FT(0), FT(cosd(30)), wind) for wind in FT.(0:5:25)]
            @test all(isapprox.(computed, reference[i]; atol = sqrt(eps(FT))))
        end
    end
end
