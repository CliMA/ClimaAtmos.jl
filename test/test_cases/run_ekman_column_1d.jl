include("initial_conditions/ekman_column_1d.jl")

# Set up parameters
using CLIMAParameters
Base.@kwdef struct EkmanParameters{FT} <:
                   CLIMAParameters.AbstractEarthParameterSet
    Cd::FT = 0.01 / (2e2 / 30) # drag coefficients
    Ch::FT = 0.01 / (2e2 / 30)
    T_surf::FT = 300 # surface temperature
    f::FT = 5e-5 # Coriolis parameters
    ν::FT = 0.01 # diffusivity
    uvg = Geometry.UVVector(FT(1), FT(0)) # geostrophic velocity
end

function run_ekman_column_1d(
    ::Type{FT};
    stepper = SSPRK33(),
    nelements = 30,
    dt = 0.01,
    callbacks = (),
    mode = :regression,
) where {FT}
    params = EkmanParameters{FT}()

    domain = Column(FT, zlim = (0.0, 2e2), nelements = nelements)

    bulk_coefficients = (Cd = params.Cd, Ch = params.Ch)
    boundary_conditions = (
        ρ = (top = NoFluxCondition(), bottom = NoFluxCondition()),
        uv = (top = nothing, bottom = DragLawCondition(bulk_coefficients)),
        w = (top = NoFluxCondition(), bottom = NoFluxCondition()),
        ρθ = (
            top = NoFluxCondition(),
            bottom = BulkFormulaCondition(bulk_coefficients, params.T_surf),
        ),
    )

    model = SingleColumnModel(
        domain = domain,
        boundary_conditions = boundary_conditions,
        parameters = params,
    )

    # execute differently depending on testing mode
    if mode == :integration
        # Populate Callback Containers
        temp_filepath = joinpath(@__DIR__, "callback_tests")
        mkpath(temp_filepath)
        cb_1 = JLD2Output(model, temp_filepath, "TestFilename1", 0.01)
        cb_2 = JLD2Output(model, temp_filepath, "TestFilename2", 0.02)

        # Generate CallbackSet 
        cb_set = CallbackSet(generate_callback(cb_1), generate_callback(cb_2))

        # Type Checks
        @test generate_callback(cb_1) isa DiscreteCallback
        @test generate_callback(cb_2) isa DiscreteCallback

        # Generate simple simulation data for test
        simulation = Simulation(
            model,
            SSPRK33(),
            dt = 0.01,
            tspan = (0.0, 0.03),
            callbacks = cb_set,
        )
        @unpack ρ, uv, w, ρθ = init_ekman_column_1d(FT, params)
        set!(simulation, :base, ρ = ρ, uv = uv, w = w)
        set!(simulation, :thermodynamics, ρθ = ρθ)
        run!(simulation)

        # Test simulation restart
        simulation = Simulation(
            model,
            SSPRK33(),
            dt = 0.01,
            tspan = (0.0, 0.03),
            callbacks = cb_set,
            restart = Restart(
                restartfile = joinpath(
                    cb_1.filedir,
                    cb_1.filename * "_0.02.jld2",
                ),
                end_time = 0.05,
            ),
        )
        set!(simulation, :base, ρ = ρ, uv = uv, w = w)
        set!(simulation, :thermodynamics, ρθ = ρθ)
        run!(simulation)
        @test simulation.integrator.t == 0.05

        # Delete test output files
        @test isfile(joinpath(
            cb_1.filedir,
            cb_1.filename * "_0.01" * ".jld2",
        )) == true
        @test isfile(joinpath(
            cb_2.filedir,
            cb_2.filename * "_0.02" * ".jld2",
        )) == true

        rm(temp_filepath, recursive = true)
    elseif mode == :regression
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
        @unpack ρ, uv, w, ρθ = init_ekman_column_1d(FT, params)
        set!(simulation, :base, ρ = ρ, uv = uv, w = w)
        set!(simulation, :thermodynamics, ρθ = ρθ)
        step!(simulation)
        u = simulation.integrator.u.base

        # perform regression check
        current_min = -0.0
        current_max = 0.0
        @test minimum(parent(u.w)) ≈ current_min atol = 1e-3
        @test maximum(parent(u.w)) ≈ current_max atol = 1e-3
    elseif mode == :validation
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 3600.0))
        @unpack ρ, uv, w, ρθ = init_ekman_column_1d(FT, params)
        set!(simulation, :base, ρ = ρ, uv = uv, w = w)
        set!(simulation, :thermodynamics, ρθ = ρθ)
        run!(simulation)
        u_end = simulation.integrator.u.base

        # post-processing
        ENV["GKSwstype"] = "nul"
        Plots.GRBackend()

        # make output directory
        path = joinpath(@__DIR__, "output_validation")
        mkpath(path)

        # plot final state
        function ekman_plot(Y, params; title = "", size = (1024, 600))
            uvg = parent(params.uvg)
            ug = uvg[1]
            vg = uvg[2]

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
                parent(Y.uv)[:, 1],
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
                parent(Y.uv)[:, 2],
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
        foi = ekman_plot(u_end, params)
        Plots.png(foi, joinpath(path, "ekman_column_1d_FT_$FT"))

        @test true # check is visual
    else
        throw(ArgumentError("$mode incompatible with test case."))
    end

    nothing
end
