#=
Consistency of the implicit vertical transport of energy, total water, and
condensates. Upwinding of `e_tot - Φ`, `q_tot`, and sedimenting condensates via
`energy_humidity_upwinding` is applied directly in the Newton residual by
`add_energy_transport!` and `add_vertical_transport!`. Checks:
- Affine consistency of `implicit_vertical_advection_tendency!` with
  `energy_humidity_upwinding` in the residual.
- Jacobian blocks for the scalar and velocity transport.
- ρtke first-order upwind residual and Jacobian.
=#
using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Geometry, Operators, Spaces
import ClimaCore: MatrixFields
import ClimaTimeSteppers as CTS
import ClimaAtmos: TD

include("../test_helpers.jl")

function _measure_allocs(f!, args...)
    f!(args...)
    return @allocated f!(args...)
end

column_config(; dt = "10secs", kwargs...) = CA.AtmosConfig(
    Dict{String, Any}(
        "config" => "column",
        "FLOAT_TYPE" => "Float64",
        "dt" => dt,
        "z_max" => 6000.0,
        "z_elem" => 60,
        "z_stretch" => false,
        "initial_condition" => "PrecipitatingColumn",
        "microphysics_model" => "1M",
        "disable_surface_flux_tendency" => true,
        "output_default_diagnostics" => false,
        (string(key) => value for (key, value) in kwargs)...,
    );
    job_id = "consistent_transport_test",
)

@testset "Affine consistency of energy and water transport" begin
    (; Y, p, simulation) = generate_test_simulation(column_config())
    t = simulation.integrator.t
    FT = eltype(Y)
    (; dtγ) = p
    (; ᶠu³, ᶜp) = p.precomputed
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶠρ = @. CA.ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
    z_c = Fields.coordinate_field(Y.c).z
    z_f = Fields.coordinate_field(Y.f).z
    n_levels = Spaces.nlevels(axes(Y.c))
    z_max = Spaces.z_max(axes(Y.f))
    @test p.atmos.numerics.energy_humidity_upwinding == Val(:vanleer_limiter)

    # Zero sedimentation velocities and pressure so the material transport
    # alone is under test.
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end
    for ρχ_name in (:ρq_icl, :ρq_rai, :ρq_sno)
        fill!(parent(getproperty(Y.c, ρχ_name)), FT(0))
    end
    fill!(parent(ᶜp), FT(0))
    A, B, C = FT(2.5e5), FT(2.4e6), FT(-3.3e5)
    u3_arr = parent(ᶠu³.components.data.:1)
    Yₜ = similar(Y)
    Yₜ_corr = similar(Y)
    Yₜ_wired = similar(Y)

    function test_affine_fields(ρ_f, ρe_f, ρq_f, ρl_f, label)
        ρe = Array(parent(ρe_f))
        affine =
            A .* Array(parent(ρ_f)) .+ B .* Array(parent(ρq_f)) .+
            C .* Array(parent(ρl_f))
        scale = maximum(abs, affine)
        @test scale > 0 || label == "correction"
        # Subtracting two `O(A)` energy fluxes (`A = 2.5e5`) in `upwind - base`
        # leaves `O(A * eps(FT)) ≈ 1e-11` roundoff when `ρ`'s correction is 0.
        @test maximum(abs, ρe .- affine) <= FT(1e-10) * max(scale, FT(1))
    end
    test_affine(Yₜ_k, label) = test_affine_fields(
        Yₜ_k.c.ρ,
        Yₜ_k.c.ρe_tot,
        Yₜ_k.c.ρq_tot,
        Yₜ_k.c.ρq_lcl,
        label,
    )

    # `implicit_vertical_advection_tendency!` always uses the upwinding mode
    # from `p.atmos.numerics.energy_humidity_upwinding`, which is `Val(:vanleer_limiter)`
    # for the column config above. Van Leer commutes with affine combinations only when
    # all transported profiles share one shape `ψ(z)` with coefficients of one sign
    # (the limiter then sees the same relative reconstruction for every component).
    # Mixed-profile combinations (ψ_step + ψ_smooth) break this and are not tested here.
    # The `Val(:first_order)` / `Val(:third_order)` labels document the scheme that WOULD
    # be used if the config were overridden; with van Leer, same-shape profiles are required.
    ψ_step = @. ifelse(z_c < z_max / 2, FT(1), FT(0))
    ψ_smooth = @. FT(0.5) * (1 + tanh((z_max / 2 - z_c) / FT(300)))
    cases = (
        (Val(:first_order), ψ_step, ψ_step),
        (Val(:third_order), ψ_smooth, ψ_smooth),
        (Val(:vanleer_limiter), ψ_step, ψ_step),
        (Val(:vanleer_limiter), ψ_smooth, ψ_smooth),
    )
    for (upwinding, ψ_q, ψ_l) in cases,
        C_val in (FT(0.1), FT(0.2), FT(0.3), FT(0.4), FT(0.5), FT(2.5))

        @. Y.c.ρq_tot = Y.c.ρ * (FT(2e-3) + FT(8e-3) * ψ_q)
        @. Y.c.ρq_lcl = Y.c.ρ * (FT(1e-4) + FT(9e-4) * ψ_l)
        # Keep physical `p.core.ᶜΦ = g z` nonzero so that `e_tot - Φ` is affine
        # in `(1, q_tot, q_lcl)` even in a gravitational field.
        @. Y.c.ρe_tot =
            Y.c.ρ * (A + p.core.ᶜΦ) + B * Y.c.ρq_tot + C * Y.c.ρq_lcl
        u3_arr .= C_val / FT(dtγ)
        ᶜe_mat = @. CA.specific(Y.c.ρe_tot, Y.c.ρ) - p.core.ᶜΦ
        ᶜq_tot = @. CA.specific(Y.c.ρq_tot, Y.c.ρ)
        ᶜq_lcl = @. CA.specific(Y.c.ρq_lcl, Y.c.ρ)

        Yₜ .= zero(eltype(Yₜ))
        CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
        ᶜρe_mat_tend =
            Base.materialize(@. Yₜ.c.ρe_tot + CA.ᶜadvdivᵥ(ᶠρ * ᶠu³ * CA.ᶠinterp(p.core.ᶜΦ)))
        test_affine_fields(
            Yₜ.c.ρ,
            ᶜρe_mat_tend,
            Yₜ.c.ρq_tot,
            Yₜ.c.ρq_lcl,
            "residual",
        )
    end

    # Terrain-following `u₃` and `uₕ` Jacobian actions preserve the same affine
    # relation at `ᶜp = 0`.
    topo_cfg = CA.AtmosConfig(
        Dict{String, Any}(
            "config" => "plane",
            "z_stretch" => false,
            "initial_condition" => "ConstantBuoyancyFrequencyProfile",
            "microphysics_model" => "1M",
            "topography" => "Schar",
            "mesh_warp_type" => "Linear",
            "FLOAT_TYPE" => "Float64",
            "x_max" => 100e3,
            "x_elem" => 4,
            "z_max" => 10e3,
            "z_elem" => 10,
            "dt" => "2secs",
            "implicit_microphysics" => false,
            "disable_surface_flux_tendency" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "consistent_transport_topo_test",
    )
    nt_t = generate_test_simulation(topo_cfg)
    Y_t, p_t, sim_t = nt_t.Y, nt_t.p, nt_t.simulation
    mat_t = sim_t.integrator.cache.newtons_method_cache.j.cache.matrix.matrix
    z_ct = Fields.coordinate_field(Y_t.c).z
    z_ft = Fields.coordinate_field(Y_t.f).z
    x_ct = Fields.coordinate_field(Y_t.c).x
    @. Y_t.c.ρq_tot = Y_t.c.ρ * (FT(3e-3) + FT(5e-3) * exp(-z_ct / FT(2500)))
    @. Y_t.c.ρq_lcl = Y_t.c.ρ * FT(8e-4) * exp(-(z_ct - FT(2000))^2 / FT(1e6))
    for ρχ_name in (:ρq_icl, :ρq_rai, :ρq_sno)
        fill!(parent(getproperty(Y_t.c, ρχ_name)), FT(0))
    end
    @. Y_t.c.ρe_tot = Y_t.c.ρ * A + B * Y_t.c.ρq_tot + C * Y_t.c.ρq_lcl
    fill!(parent(p_t.precomputed.ᶜp), FT(0))
    dtγ_t = FT(p_t.dtγ)
    CA.update_advection_jacobian!(mat_t, Y_t, p_t, dtγ_t, CA.UseDerivative())
    @test _measure_allocs(
        CA.update_advection_jacobian!,
        mat_t,
        Y_t,
        p_t,
        dtγ_t,
        CA.UseDerivative(),
    ) == 0

    ᶠδu₃ = @. Geometry.Covariant3Vector(
        Geometry.WVector(sin(FT(π) * z_ft / FT(10e3))),
    )
    ᶜδuₕ = @. Geometry.Covariant12Vector(
        Geometry.UVWVector(
            FT(3) * cos(FT(2π) * x_ct / FT(100e3)) * sin(z_ct / FT(2000)),
            FT(0),
            FT(0),
        ),
    )
    jact(row, col, δv) =
        Base.materialize(@. $(mat_t[row, col]) * δv)
    test_affine_fields(
        jact(MatrixFields.@name(c.ρ), MatrixFields.@name(f.u₃), ᶠδu₃),
        jact(MatrixFields.@name(c.ρe_tot), MatrixFields.@name(f.u₃), ᶠδu₃),
        jact(MatrixFields.@name(c.ρq_tot), MatrixFields.@name(f.u₃), ᶠδu₃),
        jact(MatrixFields.@name(c.ρq_lcl), MatrixFields.@name(f.u₃), ᶠδu₃),
        "J_u3",
    )
    test_affine_fields(
        jact(MatrixFields.@name(c.ρ), MatrixFields.@name(c.uₕ), ᶜδuₕ),
        jact(MatrixFields.@name(c.ρe_tot), MatrixFields.@name(c.uₕ), ᶜδuₕ),
        jact(MatrixFields.@name(c.ρq_tot), MatrixFields.@name(c.uₕ), ᶜδuₕ),
        jact(MatrixFields.@name(c.ρq_lcl), MatrixFields.@name(c.uₕ), ᶜδuₕ),
        "J_uh",
    )
end

@testset "Jacobian of the scalar and velocity transport blocks" begin
    (; Y, p, simulation) = generate_test_simulation(column_config())
    t = simulation.integrator.t
    FT = eltype(Y)
    dtγ = FT(60)
    z_c = Fields.coordinate_field(Y.c).z
    z_f = Fields.coordinate_field(Y.f).z
    z_max = Spaces.z_max(axes(Y.f))
    matrix = simulation.integrator.cache.newtons_method_cache.j.cache.matrix.matrix

    # Condensates stay above the finite-difference perturbation everywhere, so
    # the `max(0, q_liq)` of the thermodynamic state is inactive.
    @. Y.c.ρq_lcl =
        Y.c.ρ * (FT(1e-5) + FT(1e-3) * exp(-(z_c - FT(2000))^2 / FT(1e6)))
    @. Y.c.ρq_rai =
        Y.c.ρ * (FT(1e-5) + FT(2e-3) * exp(-(z_c - FT(1500))^2 / FT(1e6)))
    @. Y.c.ρq_tot =
        Y.c.ρ * FT(8e-3) * (1 + FT(0.3) * sin(FT(2π) * z_c / z_max)) +
        Y.c.ρq_lcl + Y.c.ρq_rai
    @. Y.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(FT(2.5) * sin(FT(π) * z_f / z_max)),
    )
    CA.set_precomputed_quantities!(Y, p, t)
    # Zero sedimentation velocities so the blocks hold grid-mean advection.
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end
    CA.update_sedimentation_jacobian!(matrix, Y, p, dtγ)
    @test _measure_allocs(CA.update_sedimentation_jacobian!, matrix, Y, p, dtγ) ==
          0

    arr(f) = Array(parent(f))
    ᶜv = @. sin(z_c / FT(700)) + FT(0.5)
    # The grid-mean advection of every scalar is central and advanced
    # explicitly by the Newton step, so with zero sedimentation velocities the
    # scalar diagonals are `-I` and the `(ρe_tot, ρq_tot)` coupling is zero.
    for (row_sym, col_sym) in (
        (:ρq_tot, :ρq_tot),
        (:ρe_tot, :ρe_tot),
        (:ρq_lcl, :ρq_lcl),
        (:ρq_rai, :ρq_rai),
    )
        block = matrix[
            MatrixFields.FieldName(:c, row_sym),
            MatrixFields.FieldName(:c, col_sym),
        ]
        @test arr(@. block * ᶜv) ≈ -arr(ᶜv)
    end
    @test maximum(
        abs,
        arr(
            @. $(matrix[MatrixFields.@name(c.ρe_tot), MatrixFields.@name(c.ρq_tot)]) *
               ᶜv
        ),
    ) == 0

    # Velocity blocks: each `(·, u₃)` block is the exact derivative of the
    # central-face flux `-ᶜadvdivᵥ((ᶠρ ᶠinterp(e_tot) + ᶠinterp(p)) ᶠu³)` or
    # `-ᶜadvdivᵥ(ᶠρ ᶠinterp(χ) ᶠu³)` with `e_tot`, `p`, and `χ` held fixed
    # (omitting `K(u₃)` dependence).
    CA.update_advection_jacobian!(matrix, Y, p, dtγ, CA.IgnoreDerivative())
    @test _measure_allocs(
        CA.update_advection_jacobian!,
        matrix,
        Y,
        p,
        dtγ,
        CA.IgnoreDerivative(),
    ) == 0
    # Zero boundary perturbations match the impenetrable boundary condition.
    ᶠv = @. Geometry.Covariant3Vector(
        Geometry.WVector(sin(FT(π) * z_f / z_max)),
    )
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶠρ_fixed = Base.materialize(@. CA.ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ)
    ᶜp_fixed = copy(p.precomputed.ᶜp)
    ᶜe_fixed = Base.materialize(@. CA.specific(Y.c.ρe_tot, Y.c.ρ))
    u₃_orig = copy(Y.f.u₃)
    δ_u₃ = FT(1e-3)
    ᶜflux_div = similar(Y.c.ρ)
    function fd_central_u₃_derivative(row_sym)
        tendencies = map((+1, -1)) do sgn
            ᶠu³_p = @. CA.CT3(u₃_orig + sgn * δ_u₃ * ᶠv)
            if row_sym == :ρ
                @. ᶜflux_div = -(CA.ᶜadvdivᵥ(ᶠρ_fixed * ᶠu³_p))
            elseif row_sym == :ρe_tot
                @. ᶜflux_div = -(CA.ᶜadvdivᵥ(
                    (ᶠρ_fixed * CA.ᶠinterp(ᶜe_fixed) + CA.ᶠinterp(ᶜp_fixed)) *
                    ᶠu³_p,
                ))
            else
                ᶜχ_f = @. CA.specific($(getproperty(Y.c, row_sym)), Y.c.ρ)
                @. ᶜflux_div =
                    -(CA.ᶜadvdivᵥ(ᶠρ_fixed * CA.ᶠinterp(ᶜχ_f) * ᶠu³_p))
            end
            copy(arr(ᶜflux_div))
        end
        return dtγ .* (tendencies[1] .- tendencies[2]) ./ (2 * δ_u₃)
    end
    for row_sym in (:ρ, :ρe_tot, :ρq_tot, :ρq_lcl, :ρq_rai)
        block = matrix[
            MatrixFields.FieldName(:c, row_sym),
            MatrixFields.FieldName(:f, :u₃),
        ]
        Jv = arr(@. block * ᶠv)
        fd = fd_central_u₃_derivative(row_sym)
        @test norm(Jv .- fd) / norm(fd) < FT(1e-10)
    end
end

@testset "Prognostic ρtke implicit upwind1 residual and Jacobian blocks" begin
    # 1. Dry EDOnly with explicit diffusion exercises the `DryModel` + `c.ρtke`
    # `sedimentation_jacobian_blocks` and iterative arrowhead paths without
    # diffusion blocks.
    dry_tke_cfg = column_config(;
        initial_condition = "DecayingProfile",
        microphysics_model = "dry",
        turbconv = "edonly_edmfx",
        prognostic_tke = true,
        implicit_diffusion = false,
    )
    nt_dry = generate_test_simulation(dry_tke_cfg)
    Y_d, p_d, sim_d = nt_dry.Y, nt_dry.p, nt_dry.simulation
    t_d = sim_d.integrator.t
    FT = eltype(Y_d)
    dtγ_d = FT(p_d.dtγ)
    z_cd = Fields.coordinate_field(Y_d.c).z
    z_fd = Fields.coordinate_field(Y_d.f).z
    z_max_d = Spaces.z_max(axes(Y_d.f))
    @test MatrixFields.has_field(Y_d, MatrixFields.@name(c.ρtke))
    # Sign-changing vertical velocity with interior zero crossing away from grid
    # faces exercises both donor branches of `ᶠupwind1` smoothly.
    @. Y_d.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(
            FT(1.8) *
            sin(FT(2π) * (z_fd / z_max_d - FT(0.23))) *
            sin(FT(π) * z_fd / z_max_d),
        ),
    )
    @. Y_d.c.ρtke =
        Y_d.c.ρ * (FT(0.4) + FT(0.3) * cos(FT(3π) * z_cd / z_max_d))
    CA.set_precomputed_quantities!(Y_d, p_d, t_d)

    Yₜ_d = similar(Y_d)
    fill!(Yₜ_d, 0)
    CA.implicit_vertical_advection_tendency!(Yₜ_d, Y_d, p_d, t_d)
    ᶜJ_d = Fields.local_geometry_field(Y_d.c).J
    ᶠJ_d = Fields.local_geometry_field(Y_d.f).J
    ᶠρ_d = @. CA.ᶠinterp(Y_d.c.ρ * ᶜJ_d) / ᶠJ_d
    ᶜtke_d = @. CA.specific(Y_d.c.ρtke, Y_d.c.ρ)
    ᶜρtke_exp = Base.materialize(
        @. -(CA.ᶜadvdivᵥ(ᶠρ_d * CA.ᶠupwind1(p_d.precomputed.ᶠu³, ᶜtke_d)))
    )
    @test Array(parent(Yₜ_d.c.ρtke)) ≈ Array(parent(ᶜρtke_exp)) rtol = 1e-13

    mat_d = sim_d.integrator.cache.newtons_method_cache.j.cache.matrix.matrix
    CA.update_sedimentation_jacobian!(mat_d, Y_d, p_d, dtγ_d)
    CA.update_advection_jacobian!(mat_d, Y_d, p_d, dtγ_d, CA.IgnoreDerivative())
    @test _measure_allocs(
        CA.update_sedimentation_jacobian!, mat_d, Y_d, p_d, dtγ_d,
    ) == 0
    @test _measure_allocs(
        CA.update_advection_jacobian!,
        mat_d,
        Y_d,
        p_d,
        dtγ_d,
        CA.IgnoreDerivative(),
    ) == 0

    # Finite-difference check of `(c.ρtke, c.ρtke)` at fixed `ρ` and `u³`.
    ᶜδtke = @. FT(0.2) + FT(0.1) * sin(FT(5π) * z_cd / z_max_d)
    ρtke_orig = copy(Y_d.c.ρtke)
    δ_s = FT(1e-3)
    err_tke = map((+1, -1)) do sgn
        @. Y_d.c.ρtke = ρtke_orig + sgn * δ_s * ᶜδtke
        fill!(Yₜ_d, 0)
        CA.implicit_vertical_advection_tendency!(Yₜ_d, Y_d, p_d, t_d)
        copy(Array(parent(@. dtγ_d * Yₜ_d.c.ρtke - Y_d.c.ρtke)))
    end
    @. Y_d.c.ρtke = ρtke_orig
    fd_diag = (err_tke[1] .- err_tke[2]) ./ (2 * δ_s)
    Jv_diag = Array(
        parent(
            @. $(mat_d[MatrixFields.@name(c.ρtke), MatrixFields.@name(c.ρtke)]) *
               ᶜδtke
        ),
    )
    @test norm(Jv_diag .- fd_diag) / norm(fd_diag) < FT(1e-10)

    # Velocity blocks `(c.ρtke, f.u₃)` are omitted so `u³` sign reversals cannot
    # turn a frozen-donor upwind flux downwind.
    @test !haskey(mat_d, (MatrixFields.@name(c.ρtke), MatrixFields.@name(f.u₃)))
    CTS.step!(sim_d.integrator)
    @test all(isfinite, parent(sim_d.integrator.u.c.ρtke))

    # 2. Implicit-diffusion accumulation (`+=`, single `-I`) in dry and moist
    # EDMF (`update_diffusion_jacobian!`, which handles every closure).
    for (tc, mp) in (("edonly_edmfx", "dry"), ("prognostic_edmfx", "1M"))
        nt_imp = generate_test_simulation(
            column_config(;
                initial_condition = mp == "dry" ? "DecayingProfile" : "PrecipitatingColumn",
                microphysics_model = mp,
                turbconv = tc,
                prognostic_tke = true,
                implicit_diffusion = true,
            ),
        )
        Y_m, p_m, sim_m = nt_imp.Y, nt_imp.p, nt_imp.simulation
        t_m = sim_m.integrator.t
        dtγ_m = FT(p_m.dtγ)
        z_fm = Fields.coordinate_field(Y_m.f).z
        z_max_m = Spaces.z_max(axes(Y_m.f))
        @. Y_m.f.u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(FT(1.2) * sin(FT(π) * z_fm / z_max_m)),
        )
        fill!(parent(Y_m.c.ρtke), FT(0))
        CA.set_precomputed_quantities!(Y_m, p_m, t_m)
        fill!(parent(p_m.precomputed.ᶠK_u), FT(0))
        j_cache = sim_m.integrator.cache.newtons_method_cache.j.cache
        mat_m = j_cache.matrix.matrix
        CA.update_sedimentation_jacobian!(mat_m, Y_m, p_m, dtγ_m)
        adv_tke_v = copy(
            Array(
                parent(
                    @. $(mat_m[MatrixFields.@name(c.ρtke), MatrixFields.@name(c.ρtke)]) *
                       ᶜδtke
                ),
            ),
        )
        eddy = CA.eddy_diffusivity_coefficients!(Y_m, p_m)
        CA.update_diffusion_jacobian!(
            mat_m, Y_m, p_m, dtγ_m, CA.UseDerivative(), eddy,
        )
        # With nonzero `u³`, `K_u = 0`, and `tke = 0`, `+=` preserves the
        # tridiagonal upwind advection block initialized by
        # `update_sedimentation_jacobian!` without subtracting `-I` twice.
        comb_tke_v = Array(
            parent(
                @. $(mat_m[MatrixFields.@name(c.ρtke), MatrixFields.@name(c.ρtke)]) *
                   ᶜδtke
            ),
        )
        @test comb_tke_v ≈ adv_tke_v rtol = 1e-13
    end

    # 3. Over Schar topography with a buoyancy perturbation reversing `u₃`
    # across a sharp TKE step, `(c.ρtke, c.uₕ)` and `(c.ρtke, f.u₃)` are
    # omitted and the implicit Newton solve preserves `ρtke >= 0`.
    tke_topo_cfg = CA.AtmosConfig(
        Dict{String, Any}(
            "config" => "plane",
            "z_stretch" => false,
            "initial_condition" => "ConstantBuoyancyFrequencyProfile",
            "microphysics_model" => "dry",
            "turbconv" => "edonly_edmfx",
            "prognostic_tke" => true,
            "implicit_diffusion" => true,
            "topography" => "Schar",
            "mesh_warp_type" => "Linear",
            "FLOAT_TYPE" => "Float64",
            "x_max" => 100e3,
            "x_elem" => 4,
            "z_max" => 10e3,
            "z_elem" => 10,
            "dt" => "2secs",
            "disable_surface_flux_tendency" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "consistent_transport_tke_topo_test",
    )
    nt_tt = generate_test_simulation(tke_topo_cfg)
    Y_tt, p_tt, sim_tt = nt_tt.Y, nt_tt.p, nt_tt.simulation
    t_tt = sim_tt.integrator.t
    dtγ_tt = FT(p_tt.dtγ)
    z_ctt = Fields.coordinate_field(Y_tt.c).z
    z_ftt = Fields.coordinate_field(Y_tt.f).z
    @. Y_tt.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(
            FT(-6.0) *
            sin(FT(2π) * (z_ftt / FT(10e3) - FT(0.20))) *
            sin(FT(π) * z_ftt / FT(10e3)),
        ),
    )
    @. Y_tt.c.ρtke =
        ifelse((z_ctt > FT(1500)) & (z_ctt < FT(3500)), Y_tt.c.ρ * FT(4.0), FT(0))
    ρe_tt_orig = copy(Y_tt.c.ρe_tot)
    @. Y_tt.c.ρe_tot *=
        FT(1.0) + FT(0.03) * exp(-((z_ctt - FT(1500)) / FT(400))^2)
    CA.set_implicit_precomputed_quantities!(Y_tt, p_tt, t_tt)
    nm_j = sim_tt.integrator.cache.newtons_method_cache.j
    mat_tt = nm_j.cache.matrix.matrix
    @test !haskey(mat_tt, (MatrixFields.@name(c.ρtke), MatrixFields.@name(c.uₕ)))
    @test !haskey(mat_tt, (MatrixFields.@name(c.ρtke), MatrixFields.@name(f.u₃)))
    CA.update_advection_jacobian!(mat_tt, Y_tt, p_tt, dtγ_tt, CA.UseDerivative())
    @test _measure_allocs(
        CA.update_advection_jacobian!,
        mat_tt,
        Y_tt,
        p_tt,
        dtγ_tt,
        CA.UseDerivative(),
    ) == 0
    CA.update_jacobian!(nm_j.alg, nm_j.cache, Y_tt, p_tt, dtγ_tt, t_tt)
    Yₜ_tt = similar(Y_tt)
    CA.implicit_tendency!(Yₜ_tt, Y_tt, p_tt, t_tt)
    R_tt = similar(Y_tt)
    @. R_tt = dtγ_tt * Yₜ_tt
    ΔY_tt = similar(Y_tt)
    CA.invert_jacobian!(nm_j.alg, nm_j.cache, ΔY_tt, R_tt)
    ρtke_newton = Array(parent(@. Y_tt.c.ρtke - ΔY_tt.c.ρtke))
    @test all(isfinite, ρtke_newton)
    @test minimum(ρtke_newton) >= -FT(1e-12)
    @. Y_tt.c.ρe_tot = ρe_tt_orig
    CA.set_precomputed_quantities!(Y_tt, p_tt, t_tt)
    CTS.step!(sim_tt.integrator)
    @test all(isfinite, parent(sim_tt.integrator.u.c.ρtke))
end
