#=
Tests for the horizontal component of the EDMFX SGS diffusive flux
(`edmfx_sgs_horizontal_diffusive_flux_tendency!`): tendencies vanish for
horizontally uniform states and on single-column geometry, the moist air mass
tendency matches the `ρq_tot` tendency bitwise, the total-enthalpy tendency
equals the sum of its dry-static-energy, vapor, liquid, and ice constituents,
the tracer flux carries the `α_vert_diff_tracer` scaling, the TKE and
horizontal-wind tendencies match directly assembled references, the
mixing-length grid-scale limit is controlled by the `grid_scale` keyword and
attained, the flux is wired into `additional_tendency!`, and incompatible
option combinations are rejected at model construction.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Spaces

include("../test_helpers.jl")

function bomex_edmfx_config_dict(; extra...)
    dict = Dict{String, Any}(
        "initial_condition" => "Bomex",
        "FLOAT_TYPE" => "Float64",
        "turbconv" => "prognostic_edmfx",
        "edmfx_entr_model" => "Generalized",
        "edmfx_detr_model" => "Generalized",
        "edmfx_sgs_mass_flux" => true,
        "edmfx_sgs_diffusive_flux" => true,
        "edmfx_sgs_horizontal_diffusive_flux" => true,
        "edmfx_nh_pressure" => true,
        "prognostic_tke" => true,
        "microphysics_model" => "1M",
        "z_max" => 3000.0,
        "z_elem" => 10,
        "z_stretch" => false,
        "dt" => "1secs",
        "t_end" => "10secs",
        "ode_algo" => "ARS222",
        "toml" => [joinpath(pkgdir(CA), "toml", "prognostic_edmfx.toml")],
        "output_default_diagnostics" => false,
    )
    for (key, value) in extra
        dict[String(key)] = value
    end
    return dict
end

box_config_dict(; extra...) = bomex_edmfx_config_dict(;
    config = "box", x_max = 6400.0, x_elem = 2, y_max = 6400.0, y_elem = 2, extra...,
)

@testset "EDMFX horizontal diffusive flux consistency (Bomex box)" begin
    config = CA.AtmosConfig(
        box_config_dict();
        job_id = "edmfx_horizontal_diffusion_box_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    FT = eltype(Y)

    @test p.atmos.edmfx_model.sgs_diffusive_flux_horizontal isa Val{true}

    # Nonzero TKE gives a nonzero eddy diffusivity.
    @. Y.c.ρtke = FT(0.5) * Y.c.ρ
    CA.set_horizontal_diffusivities!(Y, p)

    rows = (:ρ, :ρe_tot, :ρq_tot, :ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno, :ρtke)
    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    # DSS is applied to the weak-divergence tendencies, as in the solver.
    Spaces.weighted_dss!(Yₜ.c.uₕ)
    Spaces.weighted_dss!(Yₜ.f.u₃)
    uniform_max = Dict(
        name => maximum(abs, parent(getproperty(Yₜ.c, name))) for name in rows
    )
    uniform_max_uₕ = maximum(abs, parent(Yₜ.c.uₕ))
    uniform_max_u₃ = maximum(abs, parent(Yₜ.f.u₃))

    # Perturb the state horizontally to activate every flux pathway. The
    # enthalpy flux reads the precomputed `ᶜT` and the momentum stress
    # reads the precomputed `ᶜu`/`ᶠu`, which are perturbed directly.
    ᶜx = Fields.coordinate_field(Y.c).x
    ᶠx = Fields.coordinate_field(Y.f).x
    ᶜpert = @. 1 + FT(0.1) * sin(FT(2π) * ᶜx / FT(6400))
    @. Y.c.ρq_tot *= ᶜpert
    @. Y.c.ρq_rai = FT(1e-5) * Y.c.ρ * ᶜpert
    @. Y.c.ρtke *= ᶜpert
    @. p.precomputed.ᶜT *= 1 + FT(1e-3) * sin(FT(2π) * ᶜx / FT(6400))
    @. p.precomputed.ᶜu *= ᶜpert
    @. p.precomputed.ᶠu *= 1 + FT(0.1) * sin(FT(2π) * ᶠx / FT(6400))
    CA.set_horizontal_diffusivities!(Y, p)

    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    Spaces.weighted_dss!(Yₜ.c.uₕ)
    Spaces.weighted_dss!(Yₜ.f.u₃)
    perturbed_max = Dict(
        name => maximum(abs, parent(getproperty(Yₜ.c, name))) for name in rows
    )

    perturbed_max_uₕ = maximum(abs, parent(Yₜ.c.uₕ))
    perturbed_max_u₃ = maximum(abs, parent(Yₜ.f.u₃))

    # Nonzero fluxes for every perturbed pathway.
    @test perturbed_max[:ρq_tot] > 0
    @test perturbed_max[:ρe_tot] > 0
    @test perturbed_max[:ρq_rai] > 0
    @test perturbed_max[:ρtke] > 0
    @test perturbed_max_uₕ > 0
    @test perturbed_max_u₃ > 0

    # The moist air mass tendency matches the ρq_tot tendency bitwise.
    @test parent(Yₜ.c.ρ) == parent(Yₜ.c.ρq_tot)

    # The sedimenting-tracer flux carries the α_vert_diff_tracer scaling.
    α = CA.CAP.α_vert_diff_tracer(p.params)
    ᶜq_rai = similar(Y.c.ρ)
    @. ᶜq_rai = Y.c.ρq_rai / Y.c.ρ
    ᶜρq_rai_ref = similar(Y.c.ρ)
    @. ᶜρq_rai_ref =
        CA.wdivₕ(Y.c.ρ * p.precomputed.ᶜK_h_h * α * CA.gradₕ(ᶜq_rai))
    @test parent(Yₜ.c.ρq_rai) ≈ parent(ᶜρq_rai_ref) rtol = FT(1e-10)

    # The horizontal-wind tendency matches an independently assembled stress
    # divergence.
    ᶜτ = similar(p.scratch.ᶜtemp_UVWxUVW)
    CA.compute_strain_rate_center_full!(ᶜτ, p.precomputed.ᶜu, p.precomputed.ᶠu)
    @. ᶜτ = -2 * p.precomputed.ᶜK_u_h * ᶜτ
    ᶜuₕ_ref = similar(Yₜ.c.uₕ)
    @. ᶜuₕ_ref = -CA.C12(CA.wdivₕ(Y.c.ρ * ᶜτ) / Y.c.ρ)
    Spaces.weighted_dss!(ᶜuₕ_ref)
    @test parent(Yₜ.c.uₕ) ≈ parent(ᶜuₕ_ref) rtol = FT(1e-10)

    # Horizontally uniform fields produce no tendency (up to spectral
    # roundoff, measured against the perturbed flux scale).
    for name in (:ρ, :ρe_tot, :ρq_tot, :ρtke)
        reference = max(perturbed_max[name], perturbed_max[:ρq_tot])
        @test uniform_max[name] <= FT(1e-10) * reference
    end
    @test uniform_max_uₕ <= FT(1e-10) * perturbed_max_uₕ
    @test uniform_max_u₃ <= FT(1e-10) * perturbed_max_u₃

    # TKE shear production from horizontal gradients is positive definite:
    # with uniform TKE the transport term vanishes and the production from
    # the horizontally sheared velocity above remains.
    @. Y.c.ρtke = FT(0.5) * Y.c.ρ
    CA.set_horizontal_diffusivities!(Y, p)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    production_max = maximum(parent(Yₜ.c.ρtke))
    @test production_max > 0
    @test minimum(parent(Yₜ.c.ρtke)) >= -FT(1e-10) * production_max

    # The TKE tendency matches the transport + production reference.
    @. Y.c.ρtke = FT(0.5) * Y.c.ρ * ᶜpert
    CA.set_horizontal_diffusivities!(Y, p)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    ᶜtke = similar(Y.c.ρ)
    @. ᶜtke = Y.c.ρtke / Y.c.ρ
    ᶜS_h = CA.compute_strain_rate_center_horizontal(p.precomputed.ᶜu)
    ᶜρtke_ref = similar(Y.c.ρ)
    @. ᶜρtke_ref = CA.wdivₕ(Y.c.ρ * p.precomputed.ᶜK_u_h * CA.gradₕ(ᶜtke))
    @. ᶜρtke_ref += 2 * Y.c.ρ * p.precomputed.ᶜK_u_h * CA.norm_sqr(ᶜS_h)
    @test parent(Yₜ.c.ρtke) ≈ parent(ᶜρtke_ref) rtol = FT(1e-10)
    # Identically zero tracers stay identically zero.
    for name in (:ρq_lcl, :ρq_icl, :ρq_sno)
        @test uniform_max[name] == 0
        @test perturbed_max[name] == 0
    end

    # Passing the default grid scale explicitly is an identity.
    ᶜl_default = similar(Y.c.ρ)
    ᶜl_default .= CA.ᶜmixing_length(Y, p)
    ᶜl_explicit = similar(Y.c.ρ)
    ᶜl_explicit .= CA.ᶜmixing_length(
        Y, p; grid_scale = CA.resolvability_filter_scale(axes(Y.c)),
    )
    @test parent(ᶜl_explicit) == parent(ᶜl_default)

    # A tighter grid scale is monotone, a small grid scale is attained
    # exactly, and the 1 m lower bound holds.
    ᶜl_scalar = similar(Y.c.ρ)
    ᶜl_scalar .= CA.ᶜmixing_length(Y, p; grid_scale = FT(300))
    @test all(parent(ᶜl_scalar) .<= parent(ᶜl_default))
    ᶜl_limited = similar(Y.c.ρ)
    ᶜl_limited .= CA.ᶜmixing_length(Y, p; grid_scale = FT(5))
    @test maximum(parent(ᶜl_limited)) == FT(5)
    @test minimum(parent(ᶜl_limited)) >= FT(1)

    # The flux enters the solver through `additional_tendency!`: zeroing the
    # cached diffusivities removes exactly the direct tendency.
    CA.set_horizontal_diffusivities!(Y, p)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    Yₜ_on = similar(Y)
    Yₜ_on .= zero(eltype(Yₜ_on))
    CA.additional_tendency!(Yₜ_on, Y, p, t)
    p.precomputed.ᶜK_h_h .= 0
    p.precomputed.ᶜK_u_h .= 0
    Yₜ_off = similar(Y)
    Yₜ_off .= zero(eltype(Yₜ_off))
    CA.additional_tendency!(Yₜ_off, Y, p, t)
    ᶜwired = similar(Y.c.ρ)
    @. ᶜwired = Yₜ_on.c.ρe_tot - Yₜ_off.c.ρe_tot
    @test parent(ᶜwired) ≈ parent(Yₜ.c.ρe_tot) rtol = FT(1e-6)
end

@testset "EDMFX total-enthalpy horizontal diffusion constituents (Bomex box)" begin
    config = CA.AtmosConfig(
        box_config_dict();
        job_id = "edmfx_horizontal_diffusion_enthalpy_box_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    FT = eltype(Y)

    @. Y.c.ρtke = FT(0.5) * Y.c.ρ

    # Independent horizontal variation in each enthalpy constituent.
    ᶜx = Fields.coordinate_field(Y.c).x
    @. p.precomputed.ᶜT += FT(2) * sin(FT(2π) * ᶜx / FT(6400))
    @. p.precomputed.ᶜq_tot_nonneg +=
        FT(1e-3) * sin(FT(2π) * ᶜx / FT(6400) + FT(0.5))
    @. p.precomputed.ᶜq_liq = FT(2e-4) * (1 + sin(FT(2π) * ᶜx / FT(6400) + FT(1)))
    @. p.precomputed.ᶜq_ice = FT(2e-4) * (1 + sin(FT(2π) * ᶜx / FT(6400) + FT(2)))
    CA.set_horizontal_diffusivities!(Y, p)

    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )

    # The computed tendency and the reference constituents read the same
    # cached diffusivity.
    ᶜK_h = p.precomputed.ᶜK_h_h

    thermo_params = CA.CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p.core
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    ᶜq_vap = similar(Y.c.ρ)
    @. ᶜq_vap = CA.TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)

    ᶜflux_dse = similar(Y.c.ρ)
    ᶜflux_vap = similar(Y.c.ρ)
    ᶜflux_liq = similar(Y.c.ρ)
    ᶜflux_ice = similar(Y.c.ρ)
    @. ᶜflux_dse = CA.wdivₕ(
        Y.c.ρ * ᶜK_h * CA.gradₕ(CA.TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)),
    )
    @. ᶜflux_vap = CA.wdivₕ(
        Y.c.ρ * ᶜK_h * (CA.TD.enthalpy_vapor(thermo_params, ᶜT) + ᶜΦ) *
        CA.gradₕ(ᶜq_vap),
    )
    @. ᶜflux_liq = CA.wdivₕ(
        Y.c.ρ * ᶜK_h * (CA.TD.enthalpy_liquid(thermo_params, ᶜT) + ᶜΦ) *
        CA.gradₕ(ᶜq_liq),
    )
    @. ᶜflux_ice = CA.wdivₕ(
        Y.c.ρ * ᶜK_h * (CA.TD.enthalpy_ice(thermo_params, ᶜT) + ᶜΦ) *
        CA.gradₕ(ᶜq_ice),
    )

    # Each constituent carries a nonzero flux.
    @test maximum(abs, parent(ᶜflux_dse)) > 0
    @test maximum(abs, parent(ᶜflux_vap)) > 0
    @test maximum(abs, parent(ᶜflux_liq)) > 0
    @test maximum(abs, parent(ᶜflux_ice)) > 0

    # The ρe_tot tendency is the sum of the four constituents.
    ᶜflux_total = similar(Y.c.ρ)
    @. ᶜflux_total = ᶜflux_dse + ᶜflux_vap + ᶜflux_liq + ᶜflux_ice
    @test parent(Yₜ.c.ρe_tot) ≈ parent(ᶜflux_total) rtol = FT(1e-10)
end

@testset "EDMFX updraft horizontal diffusion (Bomex box)" begin
    config = CA.AtmosConfig(
        box_config_dict(; edmfx_horizontal_diffusion = true);
        job_id = "edmfx_updraft_horizontal_diffusion_box_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    FT = eltype(Y)

    @. Y.c.ρtke = FT(0.5) * Y.c.ρ
    CA.set_horizontal_diffusivities!(Y, p)
    # A nonzero updraft area makes the ρa counterpart nonzero.
    @. Y.c.sgsʲs.:(1).ρa = FT(0.1) * Y.c.ρ

    up_row_max(Yₜ) = (
        maximum(abs, parent(Yₜ.c.sgsʲs.:(1).mse)),
        maximum(abs, parent(Yₜ.c.sgsʲs.:(1).q_tot)),
        maximum(abs, parent(Yₜ.c.sgsʲs.:(1).ρa)),
    )

    # The updraft scalars receive the grid-mean specific tendencies, so
    # updraft-internal variation alone produces no updraft tendency.
    ᶜx = Fields.coordinate_field(Y.c).x
    @. Y.c.sgsʲs.:(1).q_tot *= 1 + FT(0.1) * sin(FT(2π) * ᶜx / FT(6400))
    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    uniform_mse, uniform_q_tot, uniform_ρa = up_row_max(Yₜ)

    # Horizontal variation in the grid-mean state drives the updraft
    # tendencies.
    @. p.precomputed.ᶜT += FT(2) * sin(FT(2π) * ᶜx / FT(6400))
    @. Y.c.ρq_tot *= 1 + FT(1e-3) * sin(FT(2π) * ᶜx / FT(6400) + FT(0.5))
    @. Y.c.ρq_rai =
        FT(1e-5) * Y.c.ρ * (1 + FT(0.1) * sin(FT(2π) * ᶜx / FT(6400)))
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    perturbed_mse, perturbed_q_tot, perturbed_ρa = up_row_max(Yₜ)
    @test perturbed_mse > 0
    @test perturbed_q_tot > 0
    @test perturbed_ρa > 0
    @test uniform_mse <= FT(1e-10) * perturbed_mse
    @test uniform_q_tot <= FT(1e-10) * perturbed_q_tot
    @test uniform_ρa <= FT(1e-10) * perturbed_ρa

    # The updraft tendencies are the grid-mean specific tendencies: in this
    # isolated call, Yₜ.c.ρe_tot and Yₜ.c.ρq_tot hold exactly the grid-mean
    # diffusion terms.
    ᶜexpected = similar(Y.c.ρ)
    @. ᶜexpected = Yₜ.c.ρe_tot / Y.c.ρ
    @test parent(Yₜ.c.sgsʲs.:(1).mse) ≈ parent(ᶜexpected) rtol = FT(1e-12)
    @. ᶜexpected = Yₜ.c.ρq_tot / Y.c.ρ
    @test parent(Yₜ.c.sgsʲs.:(1).q_tot) ≈ parent(ᶜexpected) rtol = FT(1e-12)

    # The ρa tendency is the q_tot tendency scaled by ρa / (1 - q_totʲ).
    @. ᶜexpected =
        Y.c.sgsʲs.:(1).ρa / (1 - Y.c.sgsʲs.:(1).q_tot) *
        Yₜ.c.sgsʲs.:(1).q_tot
    @test parent(Yₜ.c.sgsʲs.:(1).ρa) ≈ parent(ᶜexpected) rtol = FT(1e-12)

    # The updraft SGS tracers receive the grid-mean specific tendencies.
    @test maximum(abs, parent(Yₜ.c.ρq_rai)) > 0
    @. ᶜexpected = Yₜ.c.ρq_rai / Y.c.ρ
    @test parent(Yₜ.c.sgsʲs.:(1).q_rai) ≈ parent(ᶜexpected) rtol = FT(1e-12)
end

@testset "EDMFX horizontal diffusive flux vanishes on a column" begin
    config = CA.AtmosConfig(
        bomex_edmfx_config_dict(;
            config = "column",
            implicit_diffusion = true,
            approximate_linear_solve_iters = 2,
            dt = "120secs",
            t_end = "30mins",
            z_elem = 30,
            edmfx_horizontal_diffusion = true,
        );
        job_id = "edmfx_horizontal_diffusion_column_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    FT = eltype(Y)

    @. Y.c.ρtke = FT(0.5) * Y.c.ρ

    # Spectral operators return exact zeros on a column, so all existing
    # single-column results are unaffected by the options.
    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    for name in (:ρ, :ρe_tot, :ρq_tot, :ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno, :ρtke)
        @test maximum(abs, parent(getproperty(Yₜ.c, name))) == 0
    end
    @test maximum(abs, parent(Yₜ.c.uₕ)) == 0
    @test maximum(abs, parent(Yₜ.f.u₃)) == 0

    @test maximum(abs, parent(Yₜ.c.sgsʲs.:(1).mse)) == 0
    @test maximum(abs, parent(Yₜ.c.sgsʲs.:(1).q_tot)) == 0
    @test maximum(abs, parent(Yₜ.c.sgsʲs.:(1).ρa)) == 0
end

@testset "Incompatible horizontal diffusion options are rejected" begin
    smag_config = CA.AtmosConfig(
        box_config_dict(; smagorinsky_lilly = "UVW");
        job_id = "edmfx_horizontal_diffusion_smag_test",
    )
    @test_throws "cannot be combined" generate_test_simulation(smag_config)

    amd_config = CA.AtmosConfig(
        box_config_dict(; amd_les = true);
        job_id = "edmfx_horizontal_diffusion_amd_test",
    )
    @test_throws "cannot be combined" generate_test_simulation(amd_config)

    pairing_config = CA.AtmosConfig(
        box_config_dict(;
            edmfx_sgs_horizontal_diffusive_flux = false,
            edmfx_horizontal_diffusion = true,
        );
        job_id = "edmfx_horizontal_diffusion_pairing_test",
    )
    @test_throws "requires" generate_test_simulation(pairing_config)
end
