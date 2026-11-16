#=
julia --project
using Revise; include("test/prognostic_equations/post_step_hook_tests.jl")
=#
using Test
using Dates
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Geometry, Spaces
import ClimaDiagnostics as CD
import ClimaTimeSteppers as CTS
import ClimaUtilities.TimeManager: ITime

@testset "StageOnlyUpdateHandler and NeverUpdateHandler signals" begin
    stage_only = CA.StageOnlyUpdateHandler()
    @test CTS.needs_update!(stage_only, CTS.EndOfStageSignal())
    @test !CTS.needs_update!(stage_only, CTS.EndOfStepSignal())
    @test !CTS.needs_update!(stage_only, CTS.WithDSSSignal())
    @test !CTS.needs_update!(stage_only, CTS.NewTimeStep(0.0))
    @test !CTS.needs_update!(stage_only, CTS.NewNewtonSolve())
    @test !CTS.needs_update!(stage_only, CTS.NewNewtonIteration())

    with_dss_only = CA.WithDSSOnlyUpdateHandler()
    @test CTS.needs_update!(with_dss_only, CTS.EndOfStageSignal())
    @test !CTS.needs_update!(with_dss_only, CTS.EndOfStepSignal())
    @test CTS.needs_update!(with_dss_only, CTS.WithDSSSignal())
    @test !CTS.needs_update!(with_dss_only, CTS.NewTimeStep(0.0))
    @test !CTS.needs_update!(with_dss_only, CTS.NewNewtonSolve())
    @test !CTS.needs_update!(with_dss_only, CTS.NewNewtonIteration())

    never = CA.NeverUpdateHandler()
    @test !CTS.needs_update!(never, CTS.EndOfStageSignal())
    @test !CTS.needs_update!(never, CTS.EndOfStepSignal())
    @test !CTS.needs_update!(never, CTS.WithDSSSignal())
    @test !CTS.needs_update!(never, CTS.NewTimeStep(0.0))
    @test !CTS.needs_update!(never, CTS.NewNewtonSolve())
    @test !CTS.needs_update!(never, CTS.NewNewtonIteration())

    @test CA.update_cache_signal_handler("stage") ===
          CA.StageOnlyUpdateHandler()
    @test CA.update_cache_signal_handler("step") === CA.NeverUpdateHandler()

    @test CA.update_constrain_state_signal_handler("stage") ===
          CA.StageOnlyUpdateHandler()
    @test CA.update_constrain_state_signal_handler("step") ===
          CA.NeverUpdateHandler()
    @test CA.update_constrain_state_signal_handler("dss") ===
          CA.WithDSSOnlyUpdateHandler()
end

@testset "End-of-step suppression in CTS on ARS343 and SSP333" begin
    for (alg_name, expected_stage_calls) in (
        (CTS.ARS343(), 3),
        (CTS.SSP333(), 2),
    )
        nm = CTS.NewtonsMethod(;
            max_iters = 1,
            update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
        )
        alg = CTS.IMEXAlgorithm(alg_name, nm)

        for (handler, expected_step_u_calls) in (
            (CA.StageOnlyUpdateHandler(), expected_stage_calls),
            (CA.NeverUpdateHandler(), 0),
        )
            events = Symbol[]
            W = Matrix{Float64}(-I, 1, 1)
            T_imp = CTS.ODEFunction(
                (du, u, p, t) -> (du .= -0.1 .* u);
                jac_prototype = W,
                Wfact = (J, u, p, dtγ, t) -> (J[1, 1] = -1.0 - 0.1 * dtγ),
            )
            f = CTS.ClimaODEFunction(;
                T_exp! = (du, u, p, t) -> (du .= 0.0),
                T_imp! = T_imp,
                cache! = (u, p, t) -> (push!(events, :cache); nothing),
                cache_imp! = (u, p, t) -> nothing,
                constrain_state! = (u, p, t) ->
                    (push!(events, :constrain); nothing),
                update_cache = handler,
                update_constrain_state = handler,
            )
            post_cb = CTS.DiscreteCallback(
                (u, t, integrator) -> true,
                integrator -> begin
                    push!(events, :post_hook)
                    f.constrain_state!(integrator.u, integrator.p, integrator.t)
                    f.cache!(integrator.u, integrator.p, integrator.t)
                    return nothing
                end,
            )
            user_cb = CTS.DiscreteCallback(
                (u, t, integrator) -> true,
                integrator -> (push!(events, :user_cb); nothing),
            )
            prob = CTS.ODEProblem(f, [1.0], (0.0, 10.0), nothing)
            integrator = CTS.init(
                prob,
                alg;
                dt = 10.0,
                callback = CTS.CallbackSet(post_cb, user_cb),
            )
            empty!(events)
            CTS.step!(integrator)

            hook_pos = findfirst(==(:post_hook), events)
            @test !isnothing(hook_pos)
            pre_hook = events[1:(hook_pos - 1)]
            post_hook = events[(hook_pos + 1):end]
            @test count(==(:cache), pre_hook) == expected_step_u_calls
            @test count(==(:constrain), pre_hook) == expected_step_u_calls
            @test post_hook == [:constrain, :cache, :user_cb]
        end
    end
end

@noinline _measure_post_step_allocs(integrator) =
    @allocated CA.post_step_callback!(integrator)

@testset "AtmosSimulation post-step plumbing on ARS343 and SSP333" begin
    FT = Float64
    ctx = ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    grid = CA.ColumnGrid(
        FT;
        context = ctx,
        z_elem = 16,
        z_max = 6000.0,
        z_stretch = false,
    )
    params = CA.ClimaAtmosParameters(FT)
    model = CA.AtmosModel(
        grid;
        params,
        disable_surface_flux_tendency = true,
    )

    mktempdir() do tmpdir
        # Default simulation activates post-ARK split physics and installs the
        # post-step callback first; Rosenbrock schemes (`SSPKnoth`) are rejected.
        sim_default = CA.AtmosSimulation(
            model;
            dt = 50,
            t_end = 100,
            output_dir = joinpath(tmpdir, "default_on"),
            default_callbacks = false,
        )
        @test hasproperty(sim_default.integrator.p.scratch, :post_step)
        @test first(
            sim_default.integrator.callback.discrete_callbacks,
        ).affect! === CA.post_step_callback!
        @test CA.post_step_integrator_cache(sim_default.integrator.p) ===
              sim_default.integrator.cache
        @test_throws ArgumentError CA.ode_configuration(
            FT,
            "SSPKnoth",
            "stage",
            1,
            false,
            false,
            1.5,
            0.1,
            false,
            1e-5,
            1.0,
        )
        @test_throws ArgumentError CA.AtmosSimulation(
            model;
            dt = 50,
            t_end = 100,
            ode_config = CTS.RosenbrockAlgorithm(CTS.tableau(CTS.SSPKnoth())),
            output_dir = joinpath(tmpdir, "rejected_rosenbrock"),
            default_callbacks = false,
        )

        for (alg_name, constraint) in (
            (CTS.ARS343(), CTS.Unconstrained()),
            (CTS.SSP333(), CTS.Unconstrained()),
        )
            ode_config = CTS.IMEXAlgorithm(
                alg_name,
                CTS.NewtonsMethod(;
                    max_iters = 1,
                    update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
                ),
                constraint,
            )
            alg_tag = string(nameof(typeof(alg_name)))

            # 1. `default_callbacks = false` with user callback, `tstops` step
            # reduction, stage-index reset, and exact `b_imp`/`T_imp` density
            # increment on both `ARS343` (`IMEXARKCache`) and `SSP333`
            # (`IMEXSSPRKCache`, including explicit-diagonal stage 1).
            call_order = Symbol[]
            recorded_dts = FT[]
            stage_indices_at_entry = Int[]
            ρ_prev = Vector{FT}()
            scale_last_step = FT(1 + 1e-4)
            user_saw_scaled_ρ = Ref(false)
            user_saw_refreshed_p = Ref(false)

            hook_fn! = function (Y, p, t, Δt, integrator_cache)
                push!(call_order, :post_step_hook)
                push!(recorded_dts, Δt)
                push!(
                    stage_indices_at_entry,
                    p.scratch.post_step.stage_imp_idx,
                )
                @test integrator_cache === CA.post_step_integrator_cache(p)
                @test CA.has_implicit_stage_tendencies(integrator_cache)

                # Reconstruct `Δρ = Δt * Σᵢ b_imp[i] * T_imp[i].c.ρ` and check
                # exact agreement with the step density change in 1D.
                Δρ_imp = zeros(FT, length(ρ_prev))
                weight_sum = Ref(zero(FT))
                visited_stages = Int[]
                CA.foreach_b_imp_stage(integrator_cache) do i, w_i, T_imp_i
                    push!(visited_stages, i)
                    weight_sum[] += FT(w_i)
                    Δρ_imp .+= Δt .* FT(w_i) .* vec(Array(parent(T_imp_i.c.ρ)))
                end
                @test weight_sum[] ≈ one(FT) atol = 1e-14
                if alg_name isa CTS.ARS343
                    @test visited_stages == [2, 3, 4]
                else
                    @test visited_stages == [1, 2, 3]
                end
                ρ_now = vec(Array(parent(Y.c.ρ)))
                @test maximum(abs, Δρ_imp) > FT(1e-6)
                @test (ρ_now .- ρ_prev) ≈ Δρ_imp rtol = 1e-12 atol = 1e-14

                # Simulate stage-pairing counter advances during a step so the
                # next step verifies `reset_post_step_stage_index!`.
                CA.advance_post_step_stage_index!(p)
                CA.advance_post_step_stage_index!(p)

                # On the final step (`t == 200 s`), scale density and total
                # energy so downstream user and saving callbacks verify
                # ordering and cache refresh.
                if FT(t) ≈ FT(200)
                    @. Y.c.ρ *= scale_last_step
                    @. Y.c.ρe_tot *= scale_last_step
                end
                ρ_prev .= vec(Array(parent(Y.c.ρ)))
                return nothing
            end

            p_before_scale = Ref(zero(FT))
            user_cb = CTS.DiscreteCallback(
                (u, t, integrator) -> true,
                integrator -> begin
                    push!(call_order, :user_cb)
                    if FT(integrator.t) ≈ FT(135)
                        p_before_scale[] =
                            vec(Array(parent(integrator.p.precomputed.ᶜp)))[1]
                    elseif FT(integrator.t) ≈ FT(200)
                        ρ_cb = vec(Array(parent(integrator.u.c.ρ)))
                        user_saw_scaled_ρ[] = ρ_cb ≈ ρ_prev
                        # `post_step_callback!` refreshes `ᶜp` after `hook_fn!`
                        # scales `(ρ, ρe_tot)` at fixed `(e_tot, u)`, scaling
                        # ideal-gas pressure in level 1 by `scale_last_step`.
                        p_after =
                            vec(Array(parent(integrator.p.precomputed.ᶜp)))[1]
                        thp = CA.CAP.thermodynamics_params(integrator.p.params)
                        T_1 = vec(Array(parent(integrator.p.precomputed.ᶜT)))[1]
                        p_expected =
                            CA.TD.air_pressure(thp, T_1, ρ_cb[1])
                        user_saw_refreshed_p[] =
                            isapprox(p_after, p_expected; rtol = 1e-12)
                    end
                    return nothing
                end,
            )

            start_date = DateTime(2010, 1, 1)
            sim = CA.AtmosSimulation(
                model;
                dt = 100,
                t_start = 0,
                t_end = 200,
                start_date,
                ode_config,
                output_dir = joinpath(tmpdir, "custom_cbs_$alg_tag"),
                default_callbacks = false,
                callbacks = (user_cb,),
                post_step_hook! = hook_fn!,
            )

            @test hasproperty(sim.integrator.p.scratch, :post_step)
            @test sim.integrator.sol.prob.f.update_cache ===
                  CA.StageOnlyUpdateHandler()
            @test sim.integrator.sol.prob.f.update_constrain_state ===
                  CA.StageOnlyUpdateHandler()

            # Impose a smooth interior vertical velocity wave so continuity
            # tendencies `T_imp[i].c.ρ` are non-zero on every stage.
            z_f = Fields.coordinate_field(sim.integrator.u.f).z
            z_max = Spaces.z_max(axes(z_f))
            @. sim.integrator.u.f.u₃ = Geometry.Covariant3Vector(
                Geometry.WVector(FT(2.0) * sin(FT(π) * z_f / z_max)),
            )
            CA.set_precomputed_quantities!(
                sim.integrator.u,
                sim.integrator.p,
                sim.integrator.t,
            )
            ρ_prev = copy(vec(Array(parent(sim.integrator.u.c.ρ))))

            tstop_135 =
                first(promote(ITime(135; epoch = start_date), sim.integrator.t))
            CTS.add_tstop!(sim.integrator, tstop_135)

            CA.solve_atmos!(sim)

            @test recorded_dts ≈ FT[100, 35, 65]
            @test stage_indices_at_entry == [0, 0, 0]
            @test call_order == [
                :post_step_hook,
                :user_cb,
                :post_step_hook,
                :user_cb,
                :post_step_hook,
                :user_cb,
            ]
            @test user_saw_scaled_ρ[]
            @test user_saw_refreshed_p[]
            @test vec(Array(parent(sim.integrator.sol.u[end].c.ρ))) ≈ ρ_prev

            # 2. `default_callbacks = true` + `IntegratorWithDiagnostics`:
            # verify `post_step_callback!` runs before scheduled diagnostics.
            diag_events = Symbol[]
            ρ_expected_diag = Vector{FT}()
            diag_hook! = function (Y, p, t, Δt, integrator_cache)
                push!(diag_events, :post_step_hook)
                @. Y.c.ρ *= FT(1.01)
                @. Y.c.ρe_tot *= FT(1.01)
                ρ_expected_diag = copy(vec(Array(parent(Y.c.ρ))))
                return nothing
            end
            diag_spec = Dict{String, Any}(
                "short_name" => "rhoa",
                "period" => "1steps",
                "writer" => "dict",
            )
            sim_diag = CA.AtmosSimulation(
                model;
                dt = 50,
                t_start = 0,
                t_end = 50,
                start_date,
                ode_config,
                output_dir = joinpath(tmpdir, "diag_cbs_$alg_tag"),
                default_callbacks = true,
                callback_kwargs = (; log_progress = false, check_nan_every = 1),
                diagnostics = CA.DiagnosticsConfig(;
                    default = false,
                    additional = [diag_spec],
                ),
                post_step_hook! = diag_hook!,
            )
            @test first(
                sim_diag.integrator.callback.discrete_callbacks,
            ).affect! === CA.post_step_callback!
            CA.solve_atmos!(sim_diag)
            @test diag_events == [:post_step_hook]

            dict_writer = sim_diag.output_writers[1]
            rhoa_key = only(collect(keys(dict_writer.dict)))
            diag_series = collect(values(dict_writer.dict[rhoa_key]))
            @test length(diag_series) >= 2
            diag_rhoa_step1 = vec(Array(parent(diag_series[end])))
            @test diag_rhoa_step1 ≈ ρ_expected_diag rtol = 1e-12
        end

        # Direct `IMEXSSPRKCache` (`SSP` constraint) integration check with
        # float time span, verifying `post_step_discrete_callback()` and
        # `foreach_b_imp_stage` on `IMEXSSPRKCache` as well.
        ssp_alg = CTS.IMEXAlgorithm(
            CTS.SSP333(),
            CTS.NewtonsMethod(;
                max_iters = 1,
                update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
            ),
        )
        Y_ssp = CA.initial_state(model)
        z_f_ssp = Fields.coordinate_field(Y_ssp.f).z
        z_max_ssp = Spaces.z_max(axes(z_f_ssp))
        @. Y_ssp.f.u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(FT(2.0) * sin(FT(π) * z_f_ssp / z_max_ssp)),
        )
        ρ_prev_ssp = copy(vec(Array(parent(Y_ssp.c.ρ))))
        ssp_dts = FT[]
        ssp_hook! = function (Y, p, t, Δt, integrator_cache)
            push!(ssp_dts, Δt)
            @test integrator_cache isa CTS.IMEXSSPRKCache
            @test integrator_cache === CA.post_step_integrator_cache(p)
            Δρ_imp = zeros(FT, length(ρ_prev_ssp))
            visited = Int[]
            CA.foreach_b_imp_stage(integrator_cache) do i, w_i, T_imp_i
                push!(visited, i)
                Δρ_imp .+= Δt .* FT(w_i) .* vec(Array(parent(T_imp_i.c.ρ)))
            end
            @test visited == [1, 2, 3]
            @test (vec(Array(parent(Y.c.ρ))) .- ρ_prev_ssp) ≈ Δρ_imp rtol =
                1e-12 atol = 1e-14
            ρ_prev_ssp .= vec(Array(parent(Y.c.ρ)))
            return nothing
        end
        p_ssp = CA.build_cache(
            Y_ssp,
            model,
            params,
            FT(100),
            DateTime(2010, 1, 1),
            nothing;
            ode_algo = ssp_alg,
            update_cache_every = "stage",
            post_step_hook! = ssp_hook!,
        )
        cb_ssp = CTS.CallbackSet(CA.post_step_discrete_callback())
        int_args, int_kwargs = CA.args_integrator(
            Y_ssp,
            p_ssp,
            (FT(0), FT(200)),
            ssp_alg,
            cb_ssp,
            CA.ManualSparseJacobian(),
            false,
            model.prescribed_flow,
            FT(100),
            "stage",
            "stage",
        )
        int_ssp = CTS.init(int_args...; int_kwargs...)
        CTS.add_tstop!(int_ssp, FT(135))
        CTS.solve!(int_ssp)
        @test ssp_dts ≈ FT[100, 35, 65]
    end
end
