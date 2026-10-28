#=
Advection operator tests for ClimaAtmos.jl

Impenetrable boundary condition: the state filter must ensure ᶠu³ = 0 at
both the surface and model top. Over sloped terrain-following coordinates
this is non-trivial because the horizontal wind projects onto the vertical
contravariant component (ᶠuₕ³ ≠ 0).

See src/cache/precomputed_quantities.jl for set_velocity_at_surface! and
set_velocity_at_top!.
=#

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Spaces, Geometry
const TD = CA.TD

include("../test_helpers.jl")

@testset "Updraft sedimentation mass and energy consistency" begin
    config = CA.AtmosConfig(
        Dict{String, Any}(
            "initial_condition" => "Bomex",
            "FLOAT_TYPE" => "Float64",
            "config" => "column",
            "turbconv" => "prognostic_edmfx",
            "edmfx_entr_model" => "Generalized",
            "edmfx_detr_model" => "Generalized",
            "edmfx_sgs_mass_flux" => true,
            "edmfx_sgs_diffusive_flux" => false,
            "edmfx_vertical_diffusion" => false,
            "edmfx_nh_pressure" => true,
            "implicit_diffusion" => false,
            "prognostic_tke" => false,
            "microphysics_model" => "1M",
            "z_max" => 3000.0,
            "z_elem" => 15,
            "z_stretch" => false,
            "dt" => "10secs",
            "t_end" => "10secs",
            "ode_algo" => "SSP333",
            "toml" => [joinpath(pkgdir(CA), "toml", "prognostic_edmfx.toml")],
            "output_default_diagnostics" => false,
        );
        job_id = "advection_updraft_sed_mse_test",
    )
    (; Y, p) = generate_test_simulation(config)
    FT = eltype(Y)
    thp = CAP.thermodynamics_params(p.params)
    ᶜz = Fields.coordinate_field(Y.c).z
    sgsʲ = Y.c.sgsʲs.:(1)
    arr(field) = Array(parent(field))

    # With the updraft velocity, buoyancy anomaly, and the entrainment,
    # detrainment, and surface sources zeroed, sedimentation is the only
    # process in edmfx_sgs_vertical_advection_tendency! and in the analytic
    # ρa solve.
    fill!(Y.f.sgsʲs.:(1).u₃, Geometry.Covariant3Vector(FT(0)))
    fill!(p.precomputed.ᶠu³ʲs.:(1), Geometry.Contravariant3Vector(FT(0)))
    for name in (
        :ᶜρ_diffʲs,
        :ᶜentr_vel_scaleʲs,
        :ᶜentr_nonvel_rateʲs,
        :ᶜarea_bounding_entr_detrʲs,
        :sfc_mass_flux_sourceʲs,
    )
        fill!(getproperty(p.precomputed, name).:(1), FT(0))
    end
    @. sgsʲ.ρa = FT(0.1) * p.precomputed.ᶜρʲs.:(1)

    # Updraft and grid-mean terminal-velocity fields of each species.
    velocity_fields = (;
        q_lcl = (:ᶜwₗʲs, :ᶜwₗ),
        q_icl = (:ᶜwᵢʲs, :ᶜwᵢ),
        q_rai = (:ᶜwᵣʲs, :ᶜwᵣ),
        q_sno = (:ᶜwₛʲs, :ᶜwₛ),
    )
    liquid = (
        (;
            name = :q_lcl, state = CA.@name(c.sgsʲs.:(1).q_lcl),
            scale = FT(5e-4), depth = FT(800), w = FT(0.5),
        ),
        (;
            name = :q_rai, state = CA.@name(c.sgsʲs.:(1).q_rai),
            scale = FT(1e-3), depth = FT(1000), w = FT(4),
        ),
    )
    ice = (
        (;
            name = :q_icl, state = CA.@name(c.sgsʲs.:(1).q_icl),
            scale = FT(3e-4), depth = FT(900), w = FT(0.3),
        ),
        (;
            name = :q_sno, state = CA.@name(c.sgsʲs.:(1).q_sno),
            scale = FT(8e-4), depth = FT(1100), w = FT(1.5),
        ),
    )

    jac_cache = CA.jacobian_cache(CA.ManualSparseJacobian(), Y, p.atmos)
    matrix = jac_cache.matrix.matrix
    Yₜ = similar(Y)
    S = similar(sgsʲ.ρa)
    r = similar(sgsʲ.ρa)
    dtγ = FT(10)
    turbconv_model = p.atmos.turbconv_model

    # Sedimentation mass source S = Σ_μ vtt_μ of the updraft.
    function sedimentation_mass_source!(S)
        fill!(S, FT(0))
        CA.foreach_updraft_sedimentation(Y, p, 1) do vtt, _, _, _, _, _, _
            @. S += vtt
        end
        return S
    end
    # Tʲ from mse and the water partition, as
    # `set_prognostic_edmf_precomputed_quantities_draft_thermo!` diagnoses it,
    # with ρʲ held fixed.
    function diagnose_draft_temperature!()
        (; ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = p.precomputed
        @. ᶜq_liqʲs.:(1) = max(0, sgsʲ.q_lcl + sgsʲ.q_rai)
        @. ᶜq_iceʲs.:(1) = max(0, sgsʲ.q_icl + sgsʲ.q_sno)
        @. ᶜq_tot_nonnegʲs.:(1) =
            max(ᶜq_liqʲs.:(1) + ᶜq_iceʲs.:(1), sgsʲ.q_tot)
        @. ᶜTʲs.:(1) = TD.air_temperature(
            thp,
            TD.ph(),
            sgsʲ.mse - p.core.ᶜΦ,
            ᶜq_tot_nonnegʲs.:(1),
            ᶜq_liqʲs.:(1),
            ᶜq_iceʲs.:(1),
        )
        return nothing
    end
    function mse_and_q_tot_tendencies()
        fill!(Yₜ, FT(0))
        CA.edmfx_sgs_vertical_advection_tendency!(Yₜ, Y, p, FT(0), turbconv_model)
        return (
            copy(arr(Yₜ.c.sgsʲs.:(1).mse)),
            copy(arr(Yₜ.c.sgsʲs.:(1).q_tot)),
        )
    end

    # The two `active` species of one phase have non-uniform updraft profiles
    # and positive terminal velocities, and the grid mean has the same
    # specific humidity, so the environment residual flux is positive; the
    # other two species are zero. The budgets use the phase enthalpy `h_cond`.
    S_h = similar(sgsʲ.ρa)
    function check_sedimentation_budgets(active, h_cond)
        for (name, (wʲ, w_gs)) in pairs(velocity_fields)
            fill!(getproperty(sgsʲ, name), FT(0))
            fill!(getproperty(Y.c, Symbol(:ρ, name)), FT(0))
            fill!(getproperty(p.precomputed, wʲ).:(1), FT(0))
            fill!(getproperty(p.precomputed, w_gs), FT(0))
        end
        for s in active
            q = getproperty(sgsʲ, s.name)
            @. q = s.scale * exp(-(ᶜz) / s.depth)
            ρq = getproperty(Y.c, Symbol(:ρ, s.name))
            @. ρq = Y.c.ρ * q
            (wʲ, w_gs) = velocity_fields[s.name]
            fill!(getproperty(p.precomputed, wʲ).:(1), s.w)
            fill!(getproperty(p.precomputed, w_gs), s.w)
        end

        fill!(Yₜ, FT(0))
        CA.edmfx_sgs_vertical_advection_tendency!(
            Yₜ,
            Y,
            p,
            FT(0),
            p.atmos.turbconv_model,
        )
        sedimentation_mass_source!(S)

        fill!(S_h, FT(0))
        ᶜh_condʲ = @. h_cond(thp, p.precomputed.ᶜTʲs.:(1)) + p.core.ᶜΦ
        ᶜh_cond⁰ = @. h_cond(thp, p.precomputed.ᶜT⁰) + p.core.ᶜΦ
        CA.foreach_updraft_sedimentation(
            Y,
            p,
            1,
        ) do vtt, _, ᶜqʲ, ᶜwʲ, ᶜa, ᶠJ, ᶜρ⁰w⁰χ⁰
            CA.updraft_sedimentation!(
                vtt,
                p,
                p.precomputed.ᶜρʲs.:(1),
                ᶜwʲ,
                ᶜa,
                (@. ᶜqʲ * ᶜh_condʲ),
                ᶠJ,
                (@. ᶜρ⁰w⁰χ⁰ * ᶜh_cond⁰),
            )
            @. S_h += vtt
        end

        ρa, q_tot, mse, S_arr = arr(sgsʲ.ρa), arr(sgsʲ.q_tot), arr(sgsʲ.mse), arr(S)
        q_cond = sum(arr(getproperty(sgsʲ, s.name)) for s in active)
        q_cond_tend =
            sum(arr(getproperty(Yₜ.c.sgsʲs.:(1), s.name)) for s in active)

        # Flux-form budgets with ∂ₜ(ρa) = S: ∂ₜ(ρa χ) = ρa ∂ₜχ + χ S equals
        # S for water and condensate and S_h for static energy.
        @test maximum(abs, S_arr) > 0
        @test ρa .* arr(Yₜ.c.sgsʲs.:(1).q_tot) .+ q_tot .* S_arr ≈ S_arr rtol =
            1e-10
        @test ρa .* q_cond_tend .+ q_cond .* S_arr ≈ S_arr rtol = 1e-10
        @test ρa .* arr(Yₜ.c.sgsʲs.:(1).mse) .+ mse .* S_arr ≈ arr(S_h) rtol =
            1e-10

        # The rate r = Σ_μ Δq_μ / dtγ takes one linearized backward-Euler step
        # of each species' sedimentation, so r = ρ̂⁻¹ S(q + r dtγ) species by
        # species, here at Courant numbers w dtγ / Δz of about 0.2 and 10.
        ᶜinv_ρ̂ = @. CA.specific(
            FT(1),
            sgsʲ.ρa,
            FT(0),
            p.precomputed.ᶜρʲs.:(1),
            turbconv_model,
        )
        # Zero terminal velocities isolate one species.
        function velocity(s)
            (wʲ, w_gs) = velocity_fields[s.name]
            return (
                getproperty(p.precomputed, wʲ).:(1),
                getproperty(p.precomputed, w_gs),
            )
        end
        for dtγ_test in (dtγ, FT(500))
            r_sum = zeros(FT, size(arr(r)))
            for s in active
                others = filter(o -> o.name != s.name, active)
                for o in others, w in velocity(o)
                    fill!(w, FT(0))
                end
                q = getproperty(sgsʲ, s.name)
                q_old = copy(q)
                CA.updraft_sedimentation_rate!(r, Y, p, 1, dtγ_test)
                @test maximum(abs, arr(r)) > 0
                r_sum .+= arr(r)
                @. q = q_old + r * dtγ_test
                @test minimum(arr(q)) >= 0
                sedimentation_mass_source!(S)
                @test arr(r) ≈ arr(S) .* arr(ᶜinv_ρ̂) rtol = 1e-10
                @. q = q_old
                for o in others, w in velocity(o)
                    fill!(w, o.w)
                end
            end
            CA.updraft_sedimentation_rate!(r, Y, p, 1, dtγ_test)
            @test arr(r) ≈ r_sum rtol = 1e-10
        end
        sedimentation_mass_source!(S)

        # The analytic ρa solve applies the rate at the stage-start area, so
        # the stage source is r ρa_new.
        ρa_old = copy(sgsʲ.ρa)
        CA.updraft_sedimentation_rate!(r, Y, p, 1, dtγ)
        r_old = copy(arr(r))
        CA.solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ)
        @test (arr(sgsʲ.ρa) .- arr(ρa_old)) ./ dtγ ≈ r_old .* arr(sgsʲ.ρa) rtol =
            1e-8
        @. sgsʲ.ρa = ρa_old

        # The (mse, mse), (mse, q_μ), and (q_tot, q_μ) blocks match central
        # differences of the tendencies in a smooth direction, with Tʲ
        # rediagnosed and ρʲ held fixed. The blocks contain dtγ ∂T/∂Y - I for
        # dtγ = 1 and omit the dilution diagonal -ρ̂⁻¹ S of the (mse, mse)
        # block.
        diagnose_draft_temperature!()
        CA.update_sgs_advection_jacobian!(matrix, Y, p, FT(1))
        ᶜv = @. sin(ᶜz / FT(700)) + FT(0.5)
        mse_state = CA.@name(c.sgsʲs.:(1).mse)
        for (state, χ) in (
            (mse_state, sgsʲ.mse),
            map(s -> (s.state, getproperty(sgsʲ, s.name)), active)...,
        )
            χ_old = copy(χ)
            δ = FT(1e-6) * maximum(abs, arr(χ_old))
            @. χ = χ_old + δ * ᶜv
            diagnose_draft_temperature!()
            (mse_plus, q_tot_plus) = mse_and_q_tot_tendencies()
            @. χ = χ_old - δ * ᶜv
            diagnose_draft_temperature!()
            (mse_minus, q_tot_minus) = mse_and_q_tot_tendencies()
            @. χ = χ_old
            diagnose_draft_temperature!()
            ∂mse = matrix[mse_state, state]
            Jv = arr(@. ∂mse * ᶜv)
            if state == mse_state
                Jv .+= arr(ᶜv) .- arr(ᶜinv_ρ̂) .* arr(S) .* arr(ᶜv)
            end
            fd = (mse_plus .- mse_minus) ./ (2δ)
            @test maximum(abs, fd) > 0
            @test Jv ≈ fd rtol = 1e-6
            state == mse_state && continue
            ∂q_tot = matrix[CA.@name(c.sgsʲs.:(1).q_tot), state]
            fd_q_tot = (q_tot_plus .- q_tot_minus) ./ (2δ)
            @test maximum(abs, fd_q_tot) > 0
            @test arr(@. ∂q_tot * ᶜv) ≈ fd_q_tot rtol = 1e-6
        end
        return ᶜinv_ρ̂
    end

    check_sedimentation_budgets(liquid, TD.enthalpy_liquid)
    check_sedimentation_budgets(ice, TD.enthalpy_ice)

    # Lateral entrainment branch (∂_z a < 0, downward-widening updraft):
    # 1. When updraft and environment have identical uniform ρ, w, q_μ, and T,
    #    the vertical flux convergence and the lateral environment influx cancel
    #    at interior levels.
    ρ_orig = copy(Y.c.ρ)
    ρʲ_orig = copy(p.precomputed.ᶜρʲs.:(1))
    T⁰_orig = copy(p.precomputed.ᶜT⁰)
    fill!(Y.c.ρ, FT(1))
    fill!(p.precomputed.ᶜρʲs.:(1), FT(1))
    @. sgsʲ.ρa = (FT(0.25) - FT(0.15) * (ᶜz / FT(3000))) * p.precomputed.ᶜρʲs.:(1)
    for (name, (wʲ, w_gs)) in pairs(velocity_fields)
        fill!(getproperty(sgsʲ, name), FT(0))
        fill!(getproperty(Y.c, Symbol(:ρ, name)), FT(0))
        fill!(getproperty(p.precomputed, wʲ).:(1), FT(0))
        fill!(getproperty(p.precomputed, w_gs), FT(0))
    end
    fill!(sgsʲ.q_rai, FT(1e-3))
    @. Y.c.ρq_rai = Y.c.ρ * sgsʲ.q_rai
    fill!(p.precomputed.ᶜwᵣʲs.:(1), FT(4))
    fill!(p.precomputed.ᶜwᵣ, FT(4))
    CA.foreach_updraft_sedimentation(Y, p, 1) do vtt, name, _, _, _, _, _
        name == CA.@name(q_rai) &&
            @test maximum(abs, arr(vtt)[2:(end - 1)]) <= 100 * eps(FT)
    end
    sedimentation_mass_source!(S)
    @test maximum(abs, arr(S)[2:(end - 1)]) <= 100 * eps(FT)
    # Clamping of negative residual environment flux density: when ρq_gs = 0
    # while qʲ > 0 and ∂_z a < 0, the residual is negative everywhere and
    # ᶜρ⁰w⁰χ⁰ is clamped to 0.
    fill!(Y.c.ρq_rai, FT(0))
    CA.foreach_updraft_sedimentation(Y, p, 1) do _, name, _, _, _, _, ᶜρ⁰w⁰χ⁰
        name == CA.@name(q_rai) &&
            @test all(iszero, arr(Base.materialize(ᶜρ⁰w⁰χ⁰)))
    end

    # 2. With non-uniform profiles, ∂_z a < 0, and T⁰ ≠ Tʲ, the flux-form mass,
    #    condensate, and mse budgets hold and the mse tendency depends on T⁰.
    @. Y.c.ρ = ρ_orig
    @. p.precomputed.ᶜρʲs.:(1) = ρʲ_orig
    @. sgsʲ.ρa = (FT(0.25) - FT(0.15) * (ᶜz / FT(3000))) * p.precomputed.ᶜρʲs.:(1)
    @. p.precomputed.ᶜT⁰ = p.precomputed.ᶜTʲs.:(1) - FT(5)
    check_sedimentation_budgets(liquid, TD.enthalpy_liquid)
    mse_tend_cold_env = copy(arr(Yₜ.c.sgsʲs.:(1).mse))
    @. p.precomputed.ᶜT⁰ = p.precomputed.ᶜTʲs.:(1) + FT(5)
    check_sedimentation_budgets(liquid, TD.enthalpy_liquid)
    @test !(arr(Yₜ.c.sgsʲs.:(1).mse) ≈ mse_tend_cold_env)
    ᶜinv_ρ̂ = check_sedimentation_budgets(ice, TD.enthalpy_ice)

    # 3. The lateral Jacobian diagonal vanishes where the residual environment
    #    flux is clamped: zeroing the grid-mean snow removes exactly the
    #    unclamped lateral coefficient from the (q_tot, q_sno) block and leaves
    #    the off-diagonals unchanged.
    q_sno_state = CA.@name(c.sgsʲs.:(1).q_sno)
    CA.update_sgs_advection_jacobian!(matrix, Y, p, FT(1))
    ∂q_tot_open = copy(matrix[CA.@name(c.sgsʲs.:(1).q_tot), q_sno_state])
    ᶜ∂a∂z = copy(p.scratch.ᶜtemp_scalar_7)
    ᶜρʲ = p.precomputed.ᶜρʲs.:(1)
    ᶜa = @. sgsʲ.ρa / ᶜρʲ
    ᶜwʲ = p.precomputed.ᶜwₛʲs.:(1)
    ᶜlateral_open = @. ᶜinv_ρ̂ * (1 - sgsʲ.q_tot) *
       ifelse(
           ᶜ∂a∂z < 0,
           ᶜ∂a∂z * ᶜa * ᶜρʲ * ᶜwʲ / max(1 - ᶜa, eps(FT)),
           FT(0),
       )
    @test maximum(abs, arr(ᶜlateral_open)) > 0
    fill!(Y.c.ρq_sno, FT(0))
    CA.update_sgs_advection_jacobian!(matrix, Y, p, FT(1))
    ∂q_tot_clamped = matrix[CA.@name(c.sgsʲs.:(1).q_tot), q_sno_state]
    expected_clamped =
        @. ∂q_tot_open - CA.DiagonalMatrixRow(ᶜlateral_open)
    @test arr(∂q_tot_clamped) ≈ arr(expected_clamped) rtol = 1e-10
    @. p.precomputed.ᶜT⁰ = T⁰_orig
end

@testset "Impenetrable boundary condition over topography" begin
    # Uniform 10 m/s wind over a Schar mountain with linear mesh warping, so
    # coordinate surfaces are sloped up to the model top and ᶠuₕ³ ≠ 0 there.
    config = CA.AtmosConfig(
        Dict(
            "config" => "plane",
            "initial_condition" => "ConstantBuoyancyFrequencyProfile",
            "topography" => "Schar",
            "mesh_warp_type" => "Linear",
            "FLOAT_TYPE" => "Float64",
            "x_max" => 100e3,
            "x_elem" => 8,
            "z_max" => 21e3,
            "z_elem" => 20,
            "dz_bottom" => 500.0,
            "dt" => "1secs",
            "t_end" => "10secs",
            "output_default_diagnostics" => false,
        );
        job_id = "advection_boundary_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    FT = eltype(Y)

    # The state filter must cancel the vertical contravariant projection of
    # the horizontal wind at both boundaries, so that the total contravariant
    # velocity ᶠu³ vanishes there even over sloped coordinate surfaces.
    (; ᶠu³) = p.precomputed
    ᶠuₕ³ = Base.materialize(CA.compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ))
    top_level = Spaces.nlevels(axes(Y.c)) + CA.half
    for level in (CA.half, top_level)
        u³_boundary = Fields.level(ᶠu³.components.data.:1, level)
        uₕ³_boundary = Fields.level(ᶠuₕ³.components.data.:1, level)
        max_uₕ³ = maximum(abs, parent(uₕ³_boundary))
        # Nonzero uₕ³ at the boundary is what makes this test non-vacuous.
        @test max_uₕ³ > 0
        @test maximum(abs, parent(u³_boundary)) <= 100 * eps(FT) * max_uₕ³
    end
end
