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

    # Zero the updraft velocity, buoyancy anomaly, and the entrainment,
    # detrainment, and surface sources, so that sedimentation is the only
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
    dtγ = FT(10)

    # Prescribe the two `active` species of one phase (non-uniform updraft
    # profiles with positive terminal velocities, and the grid mean carrying
    # the same specific humidity so the environment residual flux is positive),
    # zero the other two, and check the budgets with phase enthalpy `h_cond`.
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
        CA.updraft_sedimentation_mass_source!(S, Y, p, 1)

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

        # The analytic ρa solve adds the same mass source to ρa.
        ρa_old = copy(sgsʲ.ρa)
        CA.solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ)
        @test (arr(sgsʲ.ρa) .- arr(ρa_old)) ./ dtγ ≈ S_arr rtol = 1e-8
        @. sgsʲ.ρa = ρa_old

        # The Jacobian couples each sedimenting species to q_tot with weight
        # (1 - q_tot)/ρa and to mse with h_μ(Tʲ) + Φ at the donor cell and
        # -mse at the receiver cell.
        CA.update_sgs_advection_jacobian!(matrix, Y, p, FT(1))
        for s in active
            ∂q_tot = matrix[CA.@name(c.sgsʲs.:(1).q_tot), s.state]
            ∂mse = matrix[CA.@name(c.sgsʲs.:(1).mse), s.state]
            @test maximum(abs, arr(∂q_tot)) > 0
            sed_op = @. CA.DiagonalMatrixRow(1 / (1 - sgsʲ.q_tot)) * ∂q_tot
            expected_∂mse =
                @. sed_op * CA.DiagonalMatrixRow(ᶜh_condʲ) -
                   CA.DiagonalMatrixRow(sgsʲ.mse) * sed_op
            @test arr(∂mse) ≈ arr(expected_∂mse) rtol = 1e-10
        end
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
    CA.updraft_sedimentation_mass_source!(S, Y, p, 1)
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
    check_sedimentation_budgets(ice, TD.enthalpy_ice)
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
