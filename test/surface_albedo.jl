using ClimaComms
using Logging
using ClimaAtmos
import Random
Random.seed!(1234)

redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))

@testset "set_surface_albedo!" begin
    config = ClimaAtmos.AtmosConfig(; job_id = "surface_albedo1")
    FT = Float32

    # test set_surface_albedo!(Y, p, t, α_type::ConstantAlbedo)
    config.parsed_args["rad"] = "clearsky"
    config.parsed_args["FLOAT_TYPE"] = string(FT)
    config.parsed_args["output_default_diagnostics"] = false
    simulation = ClimaAtmos.AtmosSimulation(config)
    (; u, p, t) = simulation.integrator
    ClimaAtmos.set_surface_albedo!(u, p, t, p.atmos.surface_albedo)

    @test all(p.radiation.rrtmgp_model.direct_sw_surface_albedo .== FT(0.38))
    @test all(p.radiation.rrtmgp_model.diffuse_sw_surface_albedo .== FT(0.38))

    # test set_surface_albedo!(Y, p, t, α_type::RegressionFunctionAlbedo)
    config.parsed_args["rad"] = "clearsky"
    config.parsed_args["albedo_model"] = "RegressionFunctionAlbedo"
    simulation = ClimaAtmos.AtmosSimulation(config)
    (; u, p, t) = simulation.integrator

    ClimaAtmos.set_surface_albedo!(u, p, t, p.atmos.surface_albedo)

    # test that the albedo is initiated (not NaN) - precise values are checked in the testset below
    @test !isnan(sum(p.radiation.rrtmgp_model.direct_sw_surface_albedo))
    @test !isnan(sum(p.radiation.rrtmgp_model.diffuse_sw_surface_albedo))

    # test set_surface_albedo!(Y, p, t, α_type::CouplerAlbedo)
    config.parsed_args["rad"] = "clearsky"
    config.parsed_args["albedo_model"] = "CouplerAlbedo"
    simulation = ClimaAtmos.AtmosSimulation(config)
    (; u, p) = simulation.integrator

    ClimaAtmos.set_surface_albedo!(u, p, Float64(0), p.atmos.surface_albedo)

    # test that the albedo initialized to 0.38 at t = 0 in the coupled case
    @test all(p.radiation.rrtmgp_model.direct_sw_surface_albedo .== FT(0.38))
    @test all(p.radiation.rrtmgp_model.diffuse_sw_surface_albedo .== FT(0.38))

    ClimaAtmos.set_surface_albedo!(u, p, Float64(1), p.atmos.surface_albedo)

    # test that the albedo is unchanged when updated at t > 0 in the coupled case
    @test all(p.radiation.rrtmgp_model.direct_sw_surface_albedo .== FT(0.38))
    @test all(p.radiation.rrtmgp_model.diffuse_sw_surface_albedo .== FT(0.38))
end


@testset "values of RegressionFunctionAlbedo against Jin et al 2011" begin
    FT = Float32

    # DIRECT
    # these values were visually confirmed by comparing `α_dir_list` generated using this script with Figure 2 from Jin et al 2011 for n = 1.34
    correct_vals = [
        [
            0.021,
            0.021,
            0.021,
            0.022,
            0.024,
            0.03,
            0.044,
            0.078,
            0.165,
            0.379,
            0,
        ],
        [
            0.021,
            0.021,
            0.021,
            0.022,
            0.024,
            0.03,
            0.045,
            0.078,
            0.154,
            0.313,
            0,
        ],
        [
            0.022,
            0.022,
            0.022,
            0.023,
            0.025,
            0.032,
            0.046,
            0.077,
            0.135,
            0.231,
            0,
        ],
        [
            0.008,
            0.008,
            0.008,
            0.009,
            0.01,
            0.013,
            0.022,
            0.045,
            0.114,
            0.316,
            0,
        ],
        [
            0.008,
            0.008,
            0.008,
            0.009,
            0.01,
            0.013,
            0.022,
            0.045,
            0.107,
            0.261,
            0,
        ],
        [
            0.008,
            0.008,
            0.009,
            0.009,
            0.01,
            0.014,
            0.023,
            0.044,
            0.093,
            0.193,
            0,
        ],
    ]

    n_list = FT.([1.34, 1.2])
    θ_list = FT.(collect(0:(π / 20):(π / 2)))
    u_list = FT.([0, 3, 12])
    i = 0
    for n in n_list
        for wind in u_list
            i += 1
            α_dir_list = []
            for θ_rad in θ_list
                albedo_model = ClimaAtmos.RegressionFunctionAlbedo{FT}(n = n)
                push!(
                    α_dir_list,
                    ClimaAtmos.surface_albedo_direct(albedo_model)(
                        FT(0),
                        cos(θ_rad),
                        FT(wind),
                    ),
                )
            end
            @test all(isapprox.(α_dir_list, correct_vals[i]; atol = 1e-3))
        end

    end

    # DIFFUSE
    # these values were visually confirmed by comparing `α_diff_list` generated using this script with Figure 4 from Jin et al 2011
    correct_vals = [
        [0.042499, 0.037779, 0.03515, 0.033099, 0.031358, 0.029819],
        [0.064824, 0.059713, 0.056866, 0.054646, 0.052761, 0.051095],
        [0.082365, 0.076947, 0.07393, 0.071576, 0.069578, 0.067811],
    ]

    n_list = FT.([1.2, 1.34, 1.45])
    u_list = FT.(collect(0:5:25))
    i = 0
    for n in n_list
        i += 1
        α_diff_list = []
        for wind in u_list
            albedo_model = ClimaAtmos.RegressionFunctionAlbedo{FT}(n = n)
            push!(
                α_diff_list,
                ClimaAtmos.surface_albedo_diffuse(albedo_model)(
                    FT(0),
                    FT(cos(30 * π / 180)),
                    FT(wind),
                ),
            )
        end
        @test all(isapprox.(α_diff_list, correct_vals[i]; atol = sqrt(eps(FT))))
    end
end
