#=
Water and energy budget of a precipitating column over a slab ocean with the
stage-weighted surface precipitation accumulator. The atmosphere's implicit
stage tendency is the residual at the last Newton iterate plus the last Newton
increment, so its bottom cell loses the sedimentation flux linearized at that
iterate. The accumulator sums the same flux, and the stage's `cache!` sample
adds to the slab the part it does not receive inside the Newton solve
(`add_stage_slab_precipitation_increment!`). The tests check that the
column-plus-slab water and energy close over full IMEX steps, and that the slab
increment uses the level-1 condensate from before the post-Newton clip of
`enforce_grid_mean_microphysics_constraints!`, which leaves `ρq_tot` unchanged.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Geometry, Operators
import ClimaTimeSteppers as CTS

include("../test_helpers.jl")

first_value(f) = Array(parent(f))[1]

function _measure_allocs(f!, args...)
    f!(args...)
    return @allocated f!(args...)
end

slab_precipitation_config(extra; job_id) = CA.AtmosConfig(
    Dict{String, Any}(
        "config" => "column",
        "FLOAT_TYPE" => "Float64",
        "dt" => "10secs",
        "z_stretch" => false,
        "initial_condition" => "PrecipitatingColumn",
        "microphysics_model" => "1M",
        "prognostic_surface" => "SlabOceanSST",
        "use_itime" => false,
        "disable_surface_flux_tendency" => true,
        "output_default_diagnostics" => false,
        extra...,
    );
    job_id,
)

"""
    column_plus_slab_water_and_energy(Y)

Column integrals of `ρq_tot` and `ρe_tot` plus the slab's water and heat
content, in the first column.
"""
function column_plus_slab_water_and_energy(Y, p)
    slab = p.atmos.surface.temperature
    C_slab = slab.ρ_ocean * slab.cp_ocean * slab.depth_ocean
    col_q = similar(Y.sfc.water)
    col_e = similar(Y.sfc.water)
    Operators.column_integral_definite!(col_q, Y.c.ρq_tot)
    Operators.column_integral_definite!(col_e, Y.c.ρe_tot)
    water = first_value(col_q) + first_value(Y.sfc.water)
    energy = first_value(col_e) + C_slab * first_value(Y.sfc.T)
    return (water, energy)
end

function add_rain_and_snow!(Y)
    FT = eltype(Y)
    z_c = Fields.coordinate_field(Y.c).z
    @. Y.c.ρq_rai = Y.c.ρ * FT(1.5e-4) * exp(-z_c / FT(2000))
    @. Y.c.ρq_sno = Y.c.ρ * FT(6e-5) * exp(-z_c / FT(3000))
    @. Y.c.ρq_tot =
        max(Y.c.ρq_tot, Y.c.ρq_lcl + Y.c.ρq_icl + Y.c.ρq_rai + Y.c.ρq_sno)
    return nothing
end

@testset "Column plus slab budget over full IMEX steps" begin
    # `SSP333` has an explicit first stage with `b_1 ≠ 0`, so its implicit
    # stages do not start at the first stage of the step.
    for algo_name in ("ARS343", "SSP333")
        config = slab_precipitation_config(
            Dict("ode_algo" => algo_name);
            job_id = "slab_precip_budget_$(algo_name)",
        )
        (; Y, p, simulation) = generate_test_simulation(config)
        FT = eltype(Y)
        @test CA.has_precip_stage_accumulator(p)
        add_rain_and_snow!(Y)
        CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)

        integrator = if algo_name == "SSP333"
            # `ClimaTimeSteppers` IMEX-SSPRK steps with plain floating-point `t`
            # rather than `ITime`.
            CTS.init(
                CTS.ODEProblem(
                    simulation.integrator.sol.prob.f,
                    Y,
                    (FT(0), FT(100)),
                    p,
                ),
                simulation.integrator.alg;
                dt = FT(p.dt),
                saveat = [FT(0), FT(100)],
            )
        else
            simulation.integrator
        end
        (water_0, energy_0) = column_plus_slab_water_and_energy(Y, p)
        for _ in 1:3
            CTS.step!(integrator)
            (water, energy) = column_plus_slab_water_and_energy(Y, p)
            @test abs(water - water_0) <= FT(1e-12) * abs(water_0)
            @test abs(energy - energy_0) <= FT(1e-8) * abs(energy_0)
        end
        @test first_value(Y.sfc.water) > FT(0)

        # A cache built at the step boundary matches the stepped one in every
        # flag, as a restart that checks the cache for consistency (as
        # ClimaCoupler does) requires.
        fresh = CA.build_cache(
            Y,
            p.atmos,
            p.params,
            p.dt,
            simulation.start_date,
            p.steady_state_velocity;
            ode_algo = simulation.integrator.alg,
            update_cache_every = "stage",
        )
        for name in propertynames(p.scratch)
            flag = getproperty(p.scratch, name)
            flag isa Base.RefValue || continue
            @test getproperty(fresh.scratch, name)[] == flag[]
        end
    end
end

@testset "Stage sample uses the condensate before the post-Newton clip" begin
    config = slab_precipitation_config(
        Dict("ode_algo" => "ARS343");
        job_id = "slab_precip_clip",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    FT = eltype(Y)
    t = simulation.integrator.t
    mp = p.atmos.microphysics_model
    add_rain_and_snow!(Y)
    CA.set_precomputed_quantities!(Y, p, t)

    # Open the first implicit stage and evaluate the residual at the last
    # Newton iterate `Y`, which records the linearization there.
    p.scratch.precip_stage_idx[] = p.scratch.n_imp_stages
    CA.initialize_implicit_stage_problem!(Y, p, p.dtγ)
    @test p.scratch.precip_stage_idx[] == 0
    @test p.scratch.precip_stage_pending[]
    Yₜ = similar(Y)
    CA.implicit_tendency!(Yₜ, Y, p, t)
    @test !p.scratch.precip_solved_condensate_recorded[]
    lin = p.scratch.sfc_precip_lin_state

    # The Newton step overshoots the level-1 cloud liquid below zero and drains
    # half of the level-1 rain.
    Y_solved = copy(Y)
    ρ₁ = first_value(Y.c.ρ)
    parent(Y_solved.c.ρq_lcl)[1] = -FT(1e-6) * ρ₁
    parent(Y_solved.c.ρq_rai)[1] = first_value(Y.c.ρq_rai) / 2
    names = (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
    solved = map(name -> first_value(getproperty(Y_solved.c, name)), names)

    CA.enforce_grid_mean_microphysics_constraints!(Y_solved, p, t)
    @test first_value(Y_solved.c.ρq_lcl) >= FT(0)
    @test p.scratch.precip_solved_condensate_recorded[]

    w_jac = map(
        name -> first_value(getproperty(lin, name)),
        (:w_lcl_jac, :w_icl_jac, :w_rai_jac, :w_sno_jac),
    )
    guess = map(
        name -> first_value(getproperty(lin, name)),
        (:ρq_lcl_guess, :ρq_icl_guess, :ρq_rai_guess, :ρq_sno_guess),
    )
    f_guess = map(
        name -> first_value(getproperty(lin, name)),
        (:f_guess_lcl, :f_guess_icl, :f_guess_rai, :f_guess_sno),
    )
    Δf = w_jac .* (solved .- guess)
    @test Δf[1] != FT(0) && Δf[3] != FT(0)

    # The `cache!` call at the solved state samples the stage and adds the
    # slab's share; later calls in the stage do neither.
    water_0 = first_value(Y_solved.sfc.water)
    CA.set_precomputed_quantities!(Y_solved, p, t)
    @test !p.scratch.precip_stage_pending[]
    @test p.scratch.precip_stage_idx[] == 1
    @test first_value(Y_solved.sfc.water) - water_0 ≈ -p.dtγ * sum(Δf) rtol =
        FT(1e-12)
    b_1 = p.scratch.b_imp_stages[1]
    f_rain = f_guess[1] + Δf[1] + f_guess[3] + Δf[3]
    f_snow = f_guess[2] + Δf[2] + f_guess[4] + Δf[4]
    @test first_value(p.scratch.surface_rain_flux_accum) ≈ b_1 * f_rain rtol =
        FT(1e-12)
    @test first_value(p.scratch.surface_snow_flux_accum) ≈ b_1 * f_snow rtol =
        FT(1e-12)
    water_1 = first_value(Y_solved.sfc.water)
    CA.set_precomputed_quantities!(Y_solved, p, t)
    @test first_value(Y_solved.sfc.water) == water_1
    @test p.scratch.precip_stage_idx[] == 1

    # Without a constraint since the last residual, the sample reads the
    # condensate of the current state.
    CA.initialize_implicit_stage_problem!(Y, p, p.dtγ)
    CA.implicit_tendency!(Yₜ, Y, p, t)
    Y_unclipped = copy(Y)
    parent(Y_unclipped.c.ρq_rai)[1] = first_value(Y.c.ρq_rai) / 2
    water_0 = first_value(Y_unclipped.sfc.water)
    CA.set_precomputed_quantities!(Y_unclipped, p, t)
    Δf_rai =
        first_value(lin.w_rai_jac) *
        (first_value(Y_unclipped.c.ρq_rai) - first_value(lin.ρq_rai_guess))
    @test first_value(Y_unclipped.sfc.water) - water_0 ≈ -p.dtγ * Δf_rai rtol =
        FT(1e-12)
end

@testset "PrognosticEDMFX surface precipitation energy and budget" begin
    config = slab_precipitation_config(
        Dict("turbconv" => "prognostic_edmfx");
        job_id = "edmf_sfc_sgs_precip_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    FT = eltype(Y)
    t = simulation.integrator.t
    @. Y.c.sgsʲs.:(1).ρa = Y.c.ρ * FT(0.1)
    @. Y.c.sgsʲs.:(1).q_lcl = FT(5e-4)
    @. Y.c.sgsʲs.:(1).q_rai = FT(1e-3)
    @. Y.c.sgsʲs.:(1).q_tot = FT(1.2e-2)
    @. Y.c.sgsʲs.:(1).mse = p.core.ᶜΦ + FT(3.05e5)
    @. Y.c.ρq_lcl = Y.c.ρ * FT(2e-4)
    @. Y.c.ρq_rai = Y.c.ρ * FT(5e-4)
    CA.set_precomputed_quantities!(Y, p, t)

    Yₜ = similar(Y)
    fill!(Yₜ, 0)
    CA.vertical_advection_of_water_tendency!(Yₜ, Y, p, t)
    f_e_sgs = first_value(p.scratch.sfc_precip_lin_state.f_e_sgs)
    @test isfinite(f_e_sgs) && f_e_sgs != FT(0)
    @test _measure_allocs(CA.record_stage_precipitation_linearization!, Y, p) ==
          0

    # `f_e_sgs` is `J_c1 / J_f0 * (ρʲ fʲ + ρ⁰ f⁰)` at level 1.
    thp = CA.CAP.thermodynamics_params(p.params)
    J_c1 = first_value(Fields.local_geometry_field(Y.c).J)
    J_f0 = first_value(Fields.local_geometry_field(Y.f).J)
    J_ratio = J_c1 / J_f0
    lg1 = Fields.local_geometry_field(Fields.level(Y.c, 1))[]
    uvw_gm = Geometry.UVWVector(Fields.level(p.precomputed.ᶜu, 1)[], lg1)
    uvw_up = Geometry.UVWVector(Fields.level(p.precomputed.ᶜuʲs.:(1), 1)[], lg1)
    uvw_en = Geometry.UVWVector(Fields.level(p.precomputed.ᶜu⁰, 1)[], lg1)
    T_gm = first_value(p.precomputed.ᶜT)
    T_up = first_value(p.precomputed.ᶜTʲs.:(1))
    T_en = first_value(p.precomputed.ᶜT⁰)
    ρ_up = first_value(p.precomputed.ᶜρʲs.:(1))
    ρ_en = first_value(p.scratch.sfc_precip_env_density)
    ρa_up = first_value(Y.c.sgsʲs.:(1).ρa)
    f_e_sgs_exact = FT(0)
    for (q_name, ρq_name, wʲ_name, w_name, e_fn) in (
        (:q_lcl, :ρq_lcl, :ᶜwₗʲs, :ᶜwₗ, CA.TD.internal_energy_liquid),
        (:q_icl, :ρq_icl, :ᶜwᵢʲs, :ᶜwᵢ, CA.TD.internal_energy_ice),
        (:q_rai, :ρq_rai, :ᶜwᵣʲs, :ᶜwᵣ, CA.TD.internal_energy_liquid),
        (:q_sno, :ρq_sno, :ᶜwₛʲs, :ᶜwₛ, CA.TD.internal_energy_ice),
    )
        qʲ = first_value(getproperty(Y.c.sgsʲs.:(1), q_name))
        ρq = first_value(getproperty(Y.c, ρq_name))
        wʲ = first_value(getproperty(p.precomputed, wʲ_name).:(1))
        w = first_value(getproperty(p.precomputed, w_name))
        f_e_sgs_exact +=
            J_ratio * (
                ρ_up * CA.updraft_precip_energy_flux(
                    qʲ, wʲ, T_up, uvw_up, w, T_gm, uvw_gm, ρa_up, ρ_up, e_fn, thp,
                ) +
                ρ_en * CA.environment_precip_energy_flux(
                    qʲ, wʲ, ρq, w, T_gm, uvw_gm, T_en, uvw_en, ρ_en, ρa_up, e_fn,
                    thp,
                )
            )
    end
    @test f_e_sgs ≈ f_e_sgs_exact rtol = FT(1e-13)

    (water_0, energy_0) = column_plus_slab_water_and_energy(Y, p)
    CTS.step!(simulation.integrator)
    (water, energy) = column_plus_slab_water_and_energy(Y, p)
    @test abs(water - water_0) <= FT(1e-12) * abs(water_0)
    @test abs(energy - energy_0) <= FT(1e-8) * abs(energy_0)
end
