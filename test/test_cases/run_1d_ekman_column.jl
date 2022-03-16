if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(dirname(@__DIR__))))
end
using Test

import ArgParse
using JLD2
using OrdinaryDiffEq: SSPRK33, CallbackSet, DiscreteCallback
using ClimaCorePlots, Plots
using UnPack

using CLIMAParameters
using ClimaCore: Geometry, Fields
using ClimaAtmos.Utils.InitialConditions: init_1d_ekman_column
using ClimaAtmos.Domains
using ClimaAtmos.BoundaryConditions
using ClimaAtmos.Models
using ClimaAtmos.Models.SingleColumnModels
using ClimaAtmos.Callbacks
using ClimaAtmos.Simulations

# Set up parameters
Base.@kwdef struct EkmanParameters{FT} <:
                   CLIMAParameters.AbstractEarthParameterSet
    Cd::FT = 0.01 / (2e2 / 30) # drag coefficients
    # Ch::FT = 0.01 / (2e2 / 30)
    # T_surf::FT = 300 # surface temperature
    f::FT = 5e-5 # Coriolis parameters
    ν::FT = 0.01 # diffusivity
    uh_g = Geometry.UVVector(FT(1.0), FT(0.0))
end

function run_1d_ekman_column(
    ::Type{FT};
    stepper = SSPRK33(),
    nelements = 30,
    dt = 0.01,
    callbacks = (),
    test_mode = :validation,
) where {FT}
    params = EkmanParameters{FT}()

    domain = Column(FT, zlim = (0.0, 2e2), nelements = nelements)

    # bulk_coefficients = (Cd = params.Cd, Ch = params.Ch)
    # boundary_conditions = (;
    #     base = (;
    #         ρ = (top = NoFluxCondition(), bottom = NoFluxCondition()),
    #         uh = (top = nothing, bottom = DragLawCondition(bulk_coefficients)),
    #         w = (top = NoFluxCondition(), bottom = NoFluxCondition()),
    #     ),
    #     thermodynamics = (;
    #         ρθ = (
    #             top = NoFluxCondition(),
    #             bottom = BulkFormulaCondition(bulk_coefficients, params.T_surf),
    #         ),
    #     ),
    # )

    model = SingleColumnModel(
        domain = domain,
        thermodynamics = PotentialTemperature(),
        vertical_diffusion = ConstantViscosity(ν = params.ν),
        boundary_conditions = nothing, #boundary_conditions,
        parameters = params,
    )

    anelastic_model = SingleColumnModel(
        domain = domain,
        base = AnelasticAdvectiveForm(),
        thermodynamics = PotentialTemperature(),
        vertical_diffusion = ConstantViscosity(ν = params.ν),
        boundary_conditions = nothing, #boundary_conditions,
        parameters = params,
    )

    # execute differently depending on testing mode
    if test_mode == :regression
        # test compressible model
        # Generate simple simulation data for test
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, dt))
        @unpack ρ, uv, w, ρθ = init_1d_ekman_column(FT, params)
        set!(simulation, :base, ρ = ρ, uh = uv, w = w)
        set!(simulation, :thermodynamics, ρθ = ρθ)
        step!(simulation)
        u = simulation.integrator.u.base

        # perform regression check
        current_min = -0.0
        current_max = 0.0
        @test minimum(parent(u.w)) ≈ current_min atol = 1e-3
        @test maximum(parent(u.w)) ≈ current_max atol = 1e-3

        # test anelastic model
        dt = FT(2.0)
        simulation =
            Simulation(anelastic_model, stepper, dt = dt, tspan = (0.0, dt))
        @unpack ρ, uv, ρθ = init_1d_ekman_column(FT, params)
        set!(simulation, :base, ρ = ρ, uh = uv)
        set!(simulation, :thermodynamics, ρθ = ρθ)
        step!(simulation)
        u = simulation.integrator.u.base

        # perform regression check 
        current_min = FT(0)
        current_max = FT(1)
        @test minimum(parent(u.uh)) ≈ current_min atol = 1e-3
        @test maximum(parent(u.uh)) ≈ current_max atol = 1e-3
    elseif test_mode == :validation
        simulation = Simulation(
            anelastic_model,
            stepper,
            dt = 2.0,
            tspan = (0.0, 30 * 3600.0),
        )
        @unpack ρ, uv, ρθ = init_1d_ekman_column(FT, params)
        set!(simulation, :base, ρ = ρ, uh = uv)
        set!(simulation, :thermodynamics, ρθ = ρθ)
        run!(simulation)
        anelastic_u_end = simulation.integrator.u.base

        # post-processing
        ENV["GKSwstype"] = "nul"
        Plots.GRBackend()

        # make output directory
        path = joinpath(@__DIR__, first(split(basename(@__FILE__), ".jl")))
        mkpath(path)

        # plot final state
        function ekman_plot(Y, params; title = "", size = (1024, 600))
            ug = parent(params.uh_g)[1]
            vg = parent(params.uh_g)[2]

            d = sqrt(2.0 * 0.01 / 5e-5)
            z_centers = parent(Fields.coordinate_field(axes(Y.ρ)))

            u_ref =
                ug .-
                exp.(-z_centers / d) .*
                (ug * cos.(z_centers / d) + vg * sin.(z_centers / d))
            sub_plt1 = Plots.plot(
                u_ref,
                z_centers,
                marker = :circle,
                xlabel = "u",
                label = "Ref",
            )
            sub_plt1 = Plots.plot!(
                sub_plt1,
                parent(Y.uh)[:, 1],
                z_centers,
                label = "Comp",
            )

            v_ref =
                vg .+
                exp.(-z_centers / d) .*
                (ug * sin.(z_centers / d) - vg * cos.(z_centers / d))
            sub_plt2 = Plots.plot(
                v_ref,
                z_centers,
                marker = :circle,
                xlabel = "v",
                label = "Ref",
            )
            sub_plt2 = Plots.plot!(
                sub_plt2,
                parent(Y.uh)[:, 2],
                z_centers,
                label = "Comp",
            )

            return Plots.plot(
                sub_plt1,
                sub_plt2,
                title = title,
                layout = (1, 2),
                size = size,
            )
        end

        foi = ekman_plot(anelastic_u_end, params)
        Plots.png(
            foi,
            joinpath(
                path,
                "ekman_column_1d_FT_$(FT)_model_$(anelastic_model.base)",
            ),
        )

        @test true # check is visual
    else
        throw(ArgumentError("$test_mode incompatible with test case."))
    end

    nothing
end

s = ArgParse.ArgParseSettings()
ArgParse.@add_arg_table s begin
    "--FT"
    help = "Float type"
    arg_type = String
    "--test_mode"
    help = "Testing mode (regression vs validation)"
    arg_type = String
    default = "regression"
end

parsed_args = ArgParse.parse_args(ARGS, s)
FT = eval(Symbol(parsed_args["FT"]))
test_mode = Symbol(parsed_args["test_mode"])

run_1d_ekman_column(FT; test_mode)
