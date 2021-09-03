# includes init_bickley_jet_2d_plane(
include("initial_conditions/ekman_column_1d.jl")

function run_ekman_column_1d(
    FT;
    stepper = SSPRK33(),
    nelements = 30,
    dt = 0.01,
    callbacks = (),
    mode = :regression,
)
    if FT <: Float32
        @info "Ekman column 1D test does not run for $FT."
        return nothing
    end

    params = (
        MSLP = FT(1e5), # mean sea level pressure
        grav = FT(9.8), # gravitational constant
        R_d = FT(287.058), # R dry (gas constant / mol mass dry air)
        C_p = FT(287.058 * 7 / 2), # heat capacity at constant pressure
        C_v = FT(287.058 * 5 / 2), # heat capacity at constant volume
        R_m = FT(287.058), # moist R, assumed to be dry
        f = FT(5e-5), # Coriolis parameters
        ν = FT(0.01),
        Cd = FT(0.01 / (2e2 / 30.0)),
        uvg = Geometry.Cartesian12Vector(FT(1.0), FT(0.0)),
        T_surf = FT(300.0),
        T_min_ref = FT(230.0),
        u0 = FT(1.0),
        v0 = FT(0.0),
        w0 = FT(0.0),
    )

    # set up domain
    domain = Column(FT, zlim = (0.0, 2e2), nelements = nelements)

    # set up boundary conditions
    # boundary_conditions = (
    #     ρ = (top = NoFluxCondition(), bottom = NoFluxCondition()),
    #     u = (top = nothing, bottom = DragLawCondition()),
    #     v = (top = nothing, bottom = DragLawCondition()),
    #     w = (top = NoFluxCondition(), bottom = NoFluxCondition()),
    #     ρθ = (top = NoFluxCondition(), bottom = NoFluxCondition()),
    # )

    # set up model
    model = SingleColumnModel(
        domain = domain,
        boundary_conditions = nothing,
        parameters = params,
    )

    # execute differently depending on testing mode
    if mode == :unit
        # TODO!: run with input callbacks = ...
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
        @unpack ρ, uv, w, ρθ = init_ekman_column_1d(params)
        set!(simulation, ρ = ρ, uv = uv, w = w, ρθ = ρθ)
        step!(simulation)

        @test true # either error or integration runs
    elseif mode == :regression
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
        @unpack ρ, uv, w, ρθ = init_ekman_column_1d(params)
        set!(simulation, :scm, ρ = ρ, uv = uv, w = w, ρθ = ρθ)
        step!(simulation)
        u = simulation.integrator.u.scm

        # perform regression check
        current_min = -0.0
        current_max = 0.0
        @test minimum(parent(u.w)) ≈ current_min atol = 1e-3
        @test maximum(parent(u.w)) ≈ current_max atol = 1e-3
    elseif mode == :validation
        # TODO!: run with callbacks = ...
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 3600.0))
        @unpack ρ, uv, w, ρθ = init_ekman_column_1d(params)
        set!(simulation, ρ = ρ, uv = uv, w = w, ρθ = ρθ)
        run!(simulation)
        u_end = simulation.integrator.u.scm

        # post-processing
        ENV["GKSwstype"] = "nul"
        Plots.GRBackend()

        # make output directory
        path = joinpath(@__DIR__, "output_validation")
        mkpath(path)

        # plot final state
        function ekman_plot(Y, params; title = "", size = (1024, 600))
            @unpack uvg = params
            uvg = parent(uvg)
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
