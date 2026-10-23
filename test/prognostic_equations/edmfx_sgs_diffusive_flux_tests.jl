#=
Tests the EDMFX vertical SGS diffusive flux (`edmfx_sgs_diffusive_flux_tendency!`)
and its implicit Jacobian (`update_diffusion_jacobian!`, `update_sgs_diffusion_jacobian!`):
- Suspended cloud species diffuse down their own subdomain gradients while rain
  and snow receive no `K_h` transport.
- Updraft homogenization: uniform `mseʲ` gives zero updraft tendency; perturbed
  `mseʲ` diffuses down its own vertical gradient `∇mseʲ`.
- Area-gradient signature (`U-c`): uniform environment and updraft scalars with
  vertically varying draft area `aʲ(z)` produce zero grid-mean moisture and
  cloud-species diffusive tendencies.
- Kirchhoff `T_0` reference-state invariance holds for both grid-mean `ρe_tot`
  and updraft `mseʲ`.
- Implicit Jacobian blocks for `ρq_lcl`, `mseʲ`, `q_totʲ`, and `q_lclʲ` match
  finite differences.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaParams as CP
import ClimaCore: Fields, MatrixFields
import ClimaCore.MatrixFields: @name
const TD = CA.TD

include("../test_helpers.jl")

@testset "EDMFX SGS diffusive flux and Jacobian (Bomex column)" begin
    config = CA.AtmosConfig(
        Dict{String, Any}(
            "initial_condition" => "Bomex",
            "FLOAT_TYPE" => "Float64",
            "config" => "column",
            "turbconv" => "prognostic_edmfx",
            "edmfx_entr_model" => "Generalized",
            "edmfx_detr_model" => "Generalized",
            "edmfx_sgs_mass_flux" => true,
            "edmfx_sgs_diffusive_flux" => true,
            "edmfx_vertical_diffusion" => true,
            "edmfx_nh_pressure" => true,
            "implicit_diffusion" => true,
            "prognostic_tke" => true,
            "microphysics_model" => "1M",
            "z_max" => 3000.0,
            "z_elem" => 30,
            "z_stretch" => false,
            "dt" => "1secs",
            "t_end" => "10secs",
            "ode_algo" => "ARS222",
            "toml" => [joinpath(pkgdir(CA), "toml", "prognostic_edmfx.toml")],
            "output_default_diagnostics" => false,
        );
        job_id = "edmfx_sgs_diffusive_flux_water_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    FT = eltype(Y)
    ϵ_FT = eps(FT)
    turbconv_params = CAP.turbconv_params(p.params)
    updraft_kh_factor = CAP.edmfx_updraft_kh_diffusion_factor(turbconv_params)

    ᶜz = Fields.coordinate_field(Y.c).z
    @. Y.c.ρtke = FT(0.5) * Y.c.ρ
    @. Y.c.sgsʲs.:(1).ρa =
        Y.c.ρ * (FT(0.08) + FT(0.04) * cos(FT(2π) * ᶜz / FT(3000)))
    @. Y.c.ρq_lcl = Y.c.ρ * FT(2e-4) * (1 + sin(FT(2π) * ᶜz / FT(3000)))
    @. Y.c.ρq_icl = Y.c.ρ * FT(1e-4) * (1 + sin(FT(2π) * ᶜz / FT(3000) + FT(1)))
    @. Y.c.ρq_rai = Y.c.ρ * FT(1e-5) * (1 + sin(FT(2π) * ᶜz / FT(3000) + FT(2)))
    @. Y.c.ρq_sno = zero(Y.c.ρ)
    @. Y.c.ρq_tot =
        Y.c.ρ *
        FT(8e-3) *
        (1 + FT(0.2) * sin(FT(2π) * ᶜz / FT(3000) + FT(0.5))) +
        Y.c.ρq_rai +
        Y.c.ρq_sno
    @. Y.c.sgsʲs.:(1).q_lcl =
        FT(3e-4) * (1 + FT(0.5) * cos(FT(2π) * ᶜz / FT(3000)))
    @. Y.c.sgsʲs.:(1).q_icl =
        FT(1.5e-4) * (1 + FT(0.5) * cos(FT(2π) * ᶜz / FT(3000) + FT(0.7)))
    @. Y.c.sgsʲs.:(1).q_rai =
        FT(2e-5) * (1 + sin(FT(2π) * ᶜz / FT(3000) + FT(1.2)))
    @. Y.c.sgsʲs.:(1).q_sno = zero(ᶜz)
    @. Y.c.sgsʲs.:(1).q_tot =
        FT(1e-2) * (1 + FT(0.15) * cos(FT(2π) * ᶜz / FT(3000) + FT(0.3))) +
        Y.c.sgsʲs.:(1).q_rai +
        Y.c.sgsʲs.:(1).q_sno
    @. Y.c.sgsʲs.:(1).mse =
        FT(3.05e5) + FT(2e3) * sin(FT(2π) * ᶜz / FT(3000))
    CA.set_precomputed_quantities!(Y, p, t)

    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_diffusive_flux_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.turbconv_model,
    )

    # Aggregate water tendency is nonzero and mirrored to ρ.
    @test maximum(abs, parent(Yₜ.c.ρq_tot)) > 0
    @test parent(Yₜ.c.ρ) == parent(Yₜ.c.ρq_tot)

    # Suspended cloud species diffuse down their own subdomain gradients:
    # F_qμ = -(ᶠw⁰ ∇q_μ⁰ + ᶠw¹ ∇q_μ¹), while updraft species use
    # -(1/ρa¹) ∇·(-ᶠw¹ ∇q_μ¹).
    (; ᶠK_h) = p.precomputed
    ᶜρa⁰ = similar(Y.c.ρ)
    @. ᶜρa⁰ = CA.ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model)
    ᶠw⁰ = similar(ᶠK_h)
    ᶠw¹ = similar(ᶠK_h)
    @. ᶠw⁰ = CA.ᶠinterp(ᶜρa⁰) * ᶠK_h
    @. ᶠw¹ =
        updraft_kh_factor * ᶠK_h /
        CA.ᶠinterp(1 / max(Y.c.sgsʲs.:(1).ρa, ϵ_FT))
    ᶜexpected = similar(Y.c.ρ)
    ᶜexpected_up = similar(Y.c.ρ)
    ᶜq_tot_eff¹ = similar(Y.c.ρ)
    @. ᶜq_tot_eff¹ =
        Y.c.sgsʲs.:(1).q_tot - Y.c.sgsʲs.:(1).q_rai - Y.c.sgsʲs.:(1).q_sno
    @. ᶜexpected_up =
        -CA.ᶜdiffdivᵥ(-(ᶠw¹ * CA.ᶠgradᵥ(ᶜq_tot_eff¹))) /
        max(Y.c.sgsʲs.:(1).ρa, ϵ_FT)
    @test parent(Yₜ.c.sgsʲs.:(1).q_tot) ≈ parent(ᶜexpected_up) rtol = FT(1e-11)
    for (χ_name, ρq_sym, q_sym) in (
        (@name(q_lcl), :ρq_lcl, :q_lcl),
        (@name(q_icl), :ρq_icl, :q_icl),
    )
        ᶜχ⁰ = CA.ᶜspecific_env_value(χ_name, Y, p)
        ᶜχ¹ = getproperty(Y.c.sgsʲs.:(1), q_sym)
        @. ᶜexpected_up =
            -CA.ᶜdiffdivᵥ(-(ᶠw¹ * CA.ᶠgradᵥ(ᶜχ¹))) /
            max(Y.c.sgsʲs.:(1).ρa, ϵ_FT)
        @. ᶜexpected =
            -CA.ᶜdiffdivᵥ(-(ᶠw⁰ * CA.ᶠgradᵥ(ᶜχ⁰))) -
            CA.ᶜdiffdivᵥ(-(ᶠw¹ * CA.ᶠgradᵥ(ᶜχ¹)))
        @test parent(getproperty(Yₜ.c.sgsʲs.:(1), q_sym)) ≈
              parent(ᶜexpected_up) rtol = FT(1e-11)
        @test parent(getproperty(Yₜ.c, ρq_sym)) ≈ parent(ᶜexpected) rtol =
            FT(1e-11)
    end

    # Rain and snow receive no K_h transport.
    @test maximum(abs, parent(Y.c.ρq_rai)) > 0
    @test all(iszero, parent(Yₜ.c.ρq_rai))
    @test all(iszero, parent(Yₜ.c.ρq_sno))

    # Test 4: Updraft homogenization — own-gradient mse¹ diffusion matches
    # -(1/ρa¹) ∇·(-ᶠw¹ ∇mse¹) in the absence of updraft precipitation, and
    # vanishes when mse¹ is vertically uniform even though grid-mean ρe_tot has
    # strong vertical gradients.
    @. Y.c.sgsʲs.:(1).q_tot -= Y.c.sgsʲs.:(1).q_rai
    @. Y.c.sgsʲs.:(1).q_rai = zero(ᶜz)
    CA.set_precomputed_quantities!(Y, p, t)
    @. ᶠw¹ =
        updraft_kh_factor * p.precomputed.ᶠK_h /
        CA.ᶠinterp(1 / max(Y.c.sgsʲs.:(1).ρa, ϵ_FT))
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_diffusive_flux_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.turbconv_model,
    )
    @. ᶜexpected_up =
        -CA.ᶜdiffdivᵥ(-(ᶠw¹ * CA.ᶠgradᵥ(Y.c.sgsʲs.:(1).mse))) /
        max(Y.c.sgsʲs.:(1).ρa, ϵ_FT)
    @test parent(Yₜ.c.sgsʲs.:(1).mse) ≈ parent(ᶜexpected_up) rtol = FT(1e-11)

    @. Y.c.sgsʲs.:(1).mse = FT(3.05e5)
    CA.set_precomputed_quantities!(Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_diffusive_flux_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.turbconv_model,
    )
    @test maximum(abs, parent(Yₜ.c.ρe_tot)) > 0
    @test maximum(abs, parent(Yₜ.c.sgsʲs.:(1).mse)) <= FT(1e-12)

    # Test 3: Kirchhoff T_0 reference-state invariance for both grid-mean ρe_tot
    # and updraft mse¹ (with nonzero rain in both subdomains).
    @. Y.c.sgsʲs.:(1).q_rai =
        FT(2e-5) * (1 + sin(FT(2π) * ᶜz / FT(3000) + FT(1.2)))
    @. Y.c.sgsʲs.:(1).q_tot += Y.c.sgsʲs.:(1).q_rai
    @. Y.c.sgsʲs.:(1).mse =
        FT(3.05e5) + FT(2e3) * sin(FT(2π) * ᶜz / FT(3000))
    CA.set_precomputed_quantities!(Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_diffusive_flux_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.turbconv_model,
    )
    δT_0 = FT(25.0)
    toml_shifted = deepcopy(config.toml_dict)
    toml_shifted.data["thermodynamics_temperature_reference"]["value"] += δT_0
    toml_shifted.data["latent_heat_vaporization_at_reference"]["value"] +=
        (CAP.cp_v(p.params) - CAP.cp_l(p.params)) * δT_0
    toml_shifted.data["latent_heat_sublimation_at_reference"]["value"] +=
        (CAP.cp_v(p.params) - CAP.cp_i(p.params)) * δT_0
    params_shifted = CA.ClimaAtmosParameters(
        toml_shifted;
        microphysics_model = p.atmos.microphysics_model,
    )
    tp_0 = CAP.thermodynamics_params(p.params)
    tp_s = CAP.thermodynamics_params(params_shifted)
    δh_d =
        TD.dry_static_energy(tp_s, FT(280), FT(0)) -
        TD.dry_static_energy(tp_0, FT(280), FT(0))
    δh_water =
        TD.enthalpy_vapor(tp_s, FT(280)) - TD.enthalpy_vapor(tp_0, FT(280))
    Y_s = deepcopy(Y)
    @. Y_s.c.sgsʲs.:(1).mse =
        Y.c.sgsʲs.:(1).mse +
        (1 - Y.c.sgsʲs.:(1).q_tot) * δh_d +
        Y.c.sgsʲs.:(1).q_tot * δh_water
    p_s = CA.AtmosCache(
        (
            k == :params ? params_shifted : getfield(p, k) for
            k in fieldnames(typeof(p))
        )...,
    )
    Yₜ_s = similar(Y)
    Yₜ_s .= zero(eltype(Yₜ_s))
    CA.edmfx_sgs_diffusive_flux_tendency!(
        Yₜ_s,
        Y_s,
        p_s,
        t,
        p_s.atmos.turbconv_model,
    )
    @. ᶜexpected = (Yₜ_s.c.ρe_tot - Yₜ.c.ρe_tot) - δh_water * Yₜ.c.ρq_tot
    @test maximum(abs, parent(ᶜexpected)) <=
          FT(1e-10) * maximum(abs, parent(Yₜ.c.ρe_tot))
    @. ᶜexpected_up =
        (Yₜ_s.c.sgsʲs.:(1).mse - Yₜ.c.sgsʲs.:(1).mse) -
        (δh_water - δh_d) * Yₜ.c.sgsʲs.:(1).q_tot
    @test maximum(abs, parent(ᶜexpected_up)) <=
          FT(1e-10) * maximum(abs, parent(Yₜ.c.sgsʲs.:(1).mse))

    # Test 5: Area-gradient signature (U-c) — when environment χ⁰ and updraft χ¹
    # are each vertically uniform (with χ¹ ≠ χ⁰) while draft area a¹(z) varies
    # vertically so that grid-mean χ̄ = a⁰ χ⁰ + a¹ χ¹ has a nonzero vertical
    # gradient, the subdomain-decomposed diffusive tendencies of ρq_tot, ρq_lcl,
    # ρq_icl, q_tot¹, and q_lcl¹ are zero.
    q_tot⁰_const = FT(8e-3)
    q_tot¹_const = FT(1.4e-2)
    q_lcl⁰_const = FT(1e-4)
    q_lcl¹_const = FT(6e-4)
    @. Y.c.ρq_rai = FT(0)
    @. Y.c.ρq_sno = FT(0)
    @. Y.c.sgsʲs.:(1).q_rai = FT(0)
    @. Y.c.sgsʲs.:(1).q_sno = FT(0)
    @. Y.c.sgsʲs.:(1).q_tot = q_tot¹_const
    @. Y.c.sgsʲs.:(1).q_lcl = q_lcl¹_const
    @. Y.c.sgsʲs.:(1).q_icl = FT(0)
    @. Y.c.ρq_tot =
        (Y.c.ρ - Y.c.sgsʲs.:(1).ρa) * q_tot⁰_const +
        Y.c.sgsʲs.:(1).ρa * q_tot¹_const
    @. Y.c.ρq_lcl =
        (Y.c.ρ - Y.c.sgsʲs.:(1).ρa) * q_lcl⁰_const +
        Y.c.sgsʲs.:(1).ρa * q_lcl¹_const
    @. Y.c.ρq_icl = FT(0)
    CA.set_precomputed_quantities!(Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_diffusive_flux_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.turbconv_model,
    )
    @test maximum(abs, parent(Yₜ.c.ρq_tot)) <= FT(1e-15)
    @test maximum(abs, parent(Yₜ.c.ρq_lcl)) <= FT(1e-15)
    @test maximum(abs, parent(Yₜ.c.sgsʲs.:(1).q_tot)) <= FT(1e-15)
    @test maximum(abs, parent(Yₜ.c.sgsʲs.:(1).q_lcl)) <= FT(1e-15)

    # When `edmfx_vertical_diffusion = false`, `PrognosticEDMFX` reduces to the
    # textbook environment-only diffusive flux
    # (`-ᶜdiffdivᵥ(-(ᶠw⁰ * ᶠgradᵥ(ᶜχ⁰)))`) with zero updraft diffusive
    # tendencies.
    @. Y.c.ρq_lcl = Y.c.ρ * FT(2e-4) * (1 + sin(FT(2π) * ᶜz / FT(3000)))
    @. Y.c.sgsʲs.:(1).q_lcl =
        FT(3e-4) * (1 + FT(0.5) * cos(FT(2π) * ᶜz / FT(3000)))
    @. Y.c.sgsʲs.:(1).q_tot =
        FT(1e-2) * (1 + FT(0.15) * cos(FT(2π) * ᶜz / FT(3000) + FT(0.3)))
    @. Y.c.sgsʲs.:(1).mse =
        FT(3.05e5) + FT(2e3) * sin(FT(2π) * ᶜz / FT(3000))
    CA.set_precomputed_quantities!(Y, p, t)
    @. ᶜρa⁰ = CA.ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model)
    @. ᶠw⁰ = CA.ᶠinterp(ᶜρa⁰) * p.precomputed.ᶠK_h
    edmfx_no_up = CA.EDMFXModel(
        (
            k == :vertical_diffusion ? false : getfield(p.atmos.edmfx_model, k)
            for k in fieldnames(typeof(p.atmos.edmfx_model))
        )...,
    )
    turbconv_no_up = CA.AtmosTurbconv(
        (
            k == :edmfx_model ? edmfx_no_up : getfield(p.atmos.turbconv, k) for
            k in fieldnames(typeof(p.atmos.turbconv))
        )...,
    )
    atmos_no_up = CA.AtmosModel(
        (
            k == :turbconv ? turbconv_no_up : getfield(p.atmos, k) for
            k in fieldnames(typeof(p.atmos))
        )...,
    )
    p_no_up = CA.AtmosCache(
        (
            k == :atmos ? atmos_no_up : getfield(p, k) for
            k in fieldnames(typeof(p))
        )...,
    )
    @test !p_no_up.atmos.edmfx_model.vertical_diffusion
    Yₜ_no_up = similar(Y)
    Yₜ_no_up .= zero(eltype(Yₜ_no_up))
    CA.edmfx_sgs_diffusive_flux_tendency!(
        Yₜ_no_up,
        Y,
        p_no_up,
        t,
        p_no_up.atmos.turbconv_model,
    )
    ᶜq_lcl⁰ = CA.ᶜspecific_env_value(@name(q_lcl), Y, p_no_up)
    ᶜq_scratch = similar(Y.c.ρ)
    @. ᶜq_scratch = ᶜq_lcl⁰
    @. ᶜexpected = -CA.ᶜdiffdivᵥ(-(ᶠw⁰ * CA.ᶠgradᵥ(ᶜq_scratch)))
    @test parent(Yₜ_no_up.c.ρq_lcl) ≈ parent(ᶜexpected) rtol = FT(1e-11)
    @test all(iszero, parent(Yₜ_no_up.c.sgsʲs.:(1).mse))
    @test all(iszero, parent(Yₜ_no_up.c.sgsʲs.:(1).q_tot))
    @test all(iszero, parent(Yₜ_no_up.c.sgsʲs.:(1).q_lcl))

    # Test 6: Finite-difference verification of the new Jacobian diagonals for
    # suspended cloud species (`c.ρq_lcl`) and updraft own-gradient diffusion
    # (`c.sgsʲs.:(1).mse`, `c.sgsʲs.:(1).q_tot`, `c.sgsʲs.:(1).q_lcl`).
    # Use a vertically uniform draft area fraction so that the `(1 - a¹)`
    # factors in `ᶠinterp(ρa⁰)` and `∂q_lcl⁰/∂ρq_lcl = 1/ρa⁰` cancel.
    @. Y.c.sgsʲs.:(1).ρa = FT(0.08) * Y.c.ρ
    CA.set_precomputed_quantities!(Y, p, t)

    jac = CA.Jacobian(CA.ManualSparseJacobian(), Y, p.atmos)
    matrix = jac.cache.matrix
    for state_name in (
        @name(c.ρq_lcl),
        @name(c.sgsʲs.:(1).mse),
        @name(c.sgsʲs.:(1).q_tot),
        @name(c.sgsʲs.:(1).q_lcl),
    )
        fill!(parent(matrix[state_name, state_name]), FT(0))
    end
    dtγ = FT(1.0)
    eddy_diffusivities = CA.eddy_diffusivity_coefficients!(Y, p)
    CA.update_diffusion_jacobian!(
        matrix,
        Y,
        p,
        dtγ,
        CA.UseDerivative(),
        eddy_diffusivities,
    )
    CA.update_sgs_diffusion_jacobian!(
        matrix,
        Y,
        p,
        dtγ,
        CA.UseDerivative(),
    )

    # Verify that the Jacobian diagonal step on an interior cell k matches a
    # finite-difference perturbation of the diffusive tendency alone.
    k_test = 15
    for (state_name, scale_eps) in (
        (@name(c.ρq_lcl), FT(1e-7)),
        (@name(c.sgsʲs.:(1).mse), FT(1.0)),
        (@name(c.sgsʲs.:(1).q_tot), FT(1e-6)),
        (@name(c.sgsʲs.:(1).q_lcl), FT(1e-7)),
    )
        Y_plus = deepcopy(Y)
        field_plus = MatrixFields.get_field(Y_plus, state_name)
        parent(field_plus)[k_test] += scale_eps
        Yₜ_0 = similar(Y)
        Yₜ_0 .= zero(eltype(Yₜ_0))
        Yₜ_p = similar(Y)
        Yₜ_p .= zero(eltype(Yₜ_p))
        CA.edmfx_sgs_diffusive_flux_tendency!(
            Yₜ_0,
            Y,
            p,
            t,
            p.atmos.turbconv_model,
        )
        CA.edmfx_sgs_diffusive_flux_tendency!(
            Yₜ_p,
            Y_plus,
            p,
            t,
            p.atmos.turbconv_model,
        )
        fd_col =
            (
                parent(MatrixFields.get_field(Yₜ_p, state_name)) .-
                parent(MatrixFields.get_field(Yₜ_0, state_name))
            ) ./ scale_eps

        δχ = zero(Y.c.ρ)
        parent(δχ)[k_test] = FT(1)
        jac_col = similar(Y.c.ρ)
        block = matrix[state_name, state_name]
        @. jac_col = block * δχ
        @test parent(jac_col) ≈ fd_col rtol = FT(1e-6)
    end
end
