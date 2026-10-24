#=
Tests for the horizontal component of the EDMFX SGS diffusive flux
(`edmfx_sgs_horizontal_diffusive_flux_tendency!`): tendencies vanish for
horizontally uniform states and on single-column geometry, the moist air mass
tendency matches the `ρq_tot` tendency bitwise, the total-enthalpy tendency
equals the dry-static-energy flux plus the Yatunin et al. (2026, Eq. 33)
water-enthalpy fluxes and differs from diffusing `h_tot` directly, suspended
cloud species (`ρq_lcl`, `ρq_icl`, `ρn_lcl`, `ρn_ice`) diffuse down their own
horizontal gradients while rain and snow receive no horizontal transport,
updraft scalars (`mse`, `q_tot`, suspended cloud species, and passive tracers)
diffuse down their own horizontal gradients under `edmfx_horizontal_diffusion`
with the element-wise power-mean area weight, add the same flux divergence to the
grid mean (no environment residual), and stay bounded where the draft area
collapses, the TKE and horizontal-wind tendencies match
directly assembled references, the mixing-length grid-scale limit is controlled
by the `grid_scale` keyword and attained, the flux is wired into
`additional_tendency!`, and incompatible option combinations are rejected at
model construction.
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
    @. Y.c.ρq_lcl = FT(2e-5) * Y.c.ρ * ᶜpert
    @. Y.c.ρtke *= ᶜpert
    @. p.precomputed.ᶜT *= 1 + FT(1e-3) * sin(FT(2π) * ᶜx / FT(6400))
    @. p.precomputed.ᶜT⁰ *= 1 + FT(1e-3) * sin(FT(2π) * ᶜx / FT(6400))
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
    @test perturbed_max[:ρq_lcl] > 0
    @test perturbed_max[:ρtke] > 0
    @test perturbed_max_uₕ > 0
    @test perturbed_max_u₃ > 0

    # The moist air mass tendency matches the ρq_tot tendency bitwise.
    @test parent(Yₜ.c.ρ) == parent(Yₜ.c.ρq_tot)

    # Rain and snow get no horizontal transport, even when perturbed, as they
    # get none in the vertical.
    @test maximum(abs, parent(Y.c.ρq_rai)) > 0
    @test perturbed_max[:ρq_rai] == 0

    # The aggregate water flux is built from q_tot_eff⁰, excluding rain and
    # snow.
    ᶜρa⁰ = similar(Y.c.ρ)
    @. ᶜρa⁰ = CA.ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model)
    ᶜq_tot⁰ = CA.ᶜspecific_env_value(CA.MatrixFields.@name(q_tot), Y, p)
    ᶜq_rai⁰ = CA.ᶜspecific_env_value(CA.MatrixFields.@name(q_rai), Y, p)
    ᶜq_sno⁰ = CA.ᶜspecific_env_value(CA.MatrixFields.@name(q_sno), Y, p)
    ᶜq_tot_eff = similar(Y.c.ρ)
    @. ᶜq_tot_eff = ᶜq_tot⁰ - ᶜq_rai⁰ - ᶜq_sno⁰
    ᶜρq_tot_ref = similar(Y.c.ρ)
    @. ᶜρq_tot_ref =
        CA.wdivₕ(ᶜρa⁰ * p.precomputed.ᶜK_h_h * CA.gradₕ(ᶜq_tot_eff))
    @test parent(Yₜ.c.ρq_tot) ≈ parent(ᶜρq_tot_ref) rtol = FT(1e-10)

    # Suspended cloud liquid diffuses down its own horizontal subdomain
    # gradient.
    ᶜq_lcl⁰ = CA.ᶜspecific_env_value(CA.MatrixFields.@name(q_lcl), Y, p)
    ᶜq_lcl⁰_buf = similar(Y.c.ρ)
    @. ᶜq_lcl⁰_buf = ᶜq_lcl⁰
    ᶜρq_lcl_ref = similar(Y.c.ρ)
    @. ᶜρq_lcl_ref =
        CA.wdivₕ(ᶜρa⁰ * p.precomputed.ᶜK_h_h * CA.gradₕ(ᶜq_lcl⁰_buf))
    @test parent(Yₜ.c.ρq_lcl) ≈ parent(ᶜρq_lcl_ref) rtol = FT(1e-10)

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
    # The shear production uses the lumped restriction of the strain-rate invariant,
    # as in the model (see `edmfx_sgs_horizontal_diffusive_flux_tendency!`).
    @. ᶜρtke_ref +=
        2 * Y.c.ρ * p.precomputed.ᶜK_u_h * CA.lumpedₕ(CA.norm_sqr(ᶜS_h))
    @test parent(Yₜ.c.ρtke) ≈ parent(ᶜρtke_ref) rtol = FT(1e-10)
    # Identically zero tracers stay identically zero.
    for name in (:ρq_icl, :ρq_sno)
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

@testset "EDMFX total-enthalpy horizontal diffusion form (Bomex box)" begin
    config = CA.AtmosConfig(
        box_config_dict();
        job_id = "edmfx_horizontal_diffusion_enthalpy_box_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    FT = eltype(Y)

    @. Y.c.ρtke = FT(0.5) * Y.c.ρ

    # Independent horizontal variation in temperature, in the aggregate water,
    # and in the phase partition, so the dry and water parts of the flux are
    # separately identifiable and h_eff is not spatially constant.
    ᶜx = Fields.coordinate_field(Y.c).x
    @. p.precomputed.ᶜT⁰ += FT(2) * sin(FT(2π) * ᶜx / FT(6400))
    @. Y.c.ρq_tot *= 1 + FT(1e-3) * sin(FT(2π) * ᶜx / FT(6400) + FT(0.5))
    @. Y.c.ρq_lcl = FT(2e-4) * Y.c.ρ * (1 + sin(FT(2π) * ᶜx / FT(6400) + FT(1)))
    @. Y.c.ρq_icl = FT(2e-4) * Y.c.ρ * (1 + sin(FT(2π) * ᶜx / FT(6400) + FT(2)))
    @. Y.c.ρq_rai = FT(1e-5) * Y.c.ρ * (1 + sin(FT(2π) * ᶜx / FT(6400) + FT(3)))
    CA.set_horizontal_diffusivities!(Y, p)

    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )

    # The computed tendency and the reference read the same cached diffusivity.
    ᶜK_h = p.precomputed.ᶜK_h_h

    thermo_params = CA.CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p.core
    (; ᶜT⁰) = p.precomputed
    ᶜρa⁰ = similar(Y.c.ρ)
    @. ᶜρa⁰ = CA.ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model)
    ᶜq_lcl⁰ = CA.ᶜspecific_env_value(CA.MatrixFields.@name(q_lcl), Y, p)
    ᶜq_icl⁰ = CA.ᶜspecific_env_value(CA.MatrixFields.@name(q_icl), Y, p)
    ᶜq_tot⁰ = CA.ᶜspecific_env_value(CA.MatrixFields.@name(q_tot), Y, p)
    ᶜq_rai⁰ = CA.ᶜspecific_env_value(CA.MatrixFields.@name(q_rai), Y, p)
    ᶜq_sno⁰ = CA.ᶜspecific_env_value(CA.MatrixFields.@name(q_sno), Y, p)
    ᶜq_lcl = similar(Y.c.ρ)
    ᶜq_icl = similar(Y.c.ρ)
    @. ᶜq_lcl = ᶜq_lcl⁰
    @. ᶜq_icl = ᶜq_icl⁰

    # Reference enthalpy-flux coefficients and q_tot_eff, assembled
    # independently of the source.
    ᶜq_tot_eff = similar(Y.c.ρ)
    @. ᶜq_tot_eff = ᶜq_tot⁰ - ᶜq_rai⁰ - ᶜq_sno⁰
    (; ᶜcp, ᶜh_vap_plus_Φ, ᶜh_lv, ᶜh_iv) = enthalpy_flux_coefficients(
        thermo_params,
        ᶜT⁰,
        ᶜΦ,
        (ᶜq_tot_eff, ᶜq_lcl, ᶜq_icl),
    )

    ᶜflux_dse = similar(Y.c.ρ)
    ᶜflux_cond = similar(Y.c.ρ)
    ᶜflux_water = similar(Y.c.ρ)
    @. ᶜflux_dse =
        CA.wdivₕ(ᶜρa⁰ * ᶜK_h * ᶜcp * CA.gradₕ(ᶜT⁰)) +
        CA.wdivₕ(ᶜρa⁰ * ᶜK_h * CA.gradₕ(ᶜΦ))
    @. ᶜflux_cond =
        CA.wdivₕ(ᶜρa⁰ * ᶜK_h * ᶜh_lv * CA.gradₕ(ᶜq_lcl)) +
        CA.wdivₕ(ᶜρa⁰ * ᶜK_h * ᶜh_iv * CA.gradₕ(ᶜq_icl))
    @. ᶜflux_water =
        CA.wdivₕ(ᶜρa⁰ * ᶜK_h * ᶜh_vap_plus_Φ * CA.gradₕ(ᶜq_tot_eff))

    # All three parts are nonzero, so no assertion below passes trivially.
    @test maximum(abs, parent(ᶜflux_dse)) > 0
    @test maximum(abs, parent(ᶜflux_cond)) > 0
    @test maximum(abs, parent(ᶜflux_water)) > 0

    # The ρe_tot tendency is the sensible-heat and geopotential flux plus the
    # condensate latent-enthalpy flux plus the vapor water flux.
    ᶜflux_total = similar(Y.c.ρ)
    @. ᶜflux_total = ᶜflux_dse + ᶜflux_cond + ᶜflux_water
    @test parent(Yₜ.c.ρe_tot) ≈ parent(ᶜflux_total) rtol = FT(1e-10)

    # Diffusing h_tot directly is a different tendency: the difference is the
    # spurious transport the decomposition removes.
    ᶜh_tot = p.precomputed.ᶜh_tot
    ᶜflux_h_tot = similar(Y.c.ρ)
    @. ᶜflux_h_tot = CA.wdivₕ(Y.c.ρ * ᶜK_h * CA.gradₕ(ᶜh_tot))
    ᶜdiff = similar(Y.c.ρ)
    @. ᶜdiff = ᶜflux_h_tot - ᶜflux_total
    @test maximum(abs, parent(ᶜdiff)) >
          FT(1e-6) * maximum(abs, parent(ᶜflux_total))
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
    # A nonzero updraft area would make a ρa dry-mass tendency nonzero, so the
    # assertion below that it vanishes is not satisfied trivially.
    @. Y.c.sgsʲs.:(1).ρa = FT(0.1) * Y.c.ρ

    up_row_max(Yₜ) = (
        maximum(abs, parent(Yₜ.c.sgsʲs.:(1).mse)),
        maximum(abs, parent(Yₜ.c.sgsʲs.:(1).q_tot)),
    )

    turbconv_params = CA.CAP.turbconv_params(p.params)
    updraft_kh_factor =
        CA.CAP.edmfx_updraft_kh_diffusion_factor(turbconv_params)

    # Horizontally uniform updraft scalars produce zero updraft tendency even
    # when the grid-mean state varies horizontally.
    ᶜx = Fields.coordinate_field(Y.c).x
    @. p.precomputed.ᶜT⁰ += FT(2) * sin(FT(2π) * ᶜx / FT(6400))
    @. Y.c.ρq_tot *= 1 + FT(1e-3) * sin(FT(2π) * ᶜx / FT(6400) + FT(0.5))
    @. Y.c.ρq_lcl =
        FT(2e-5) * Y.c.ρ * (1 + FT(0.1) * sin(FT(2π) * ᶜx / FT(6400)))
    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    uniform_mse, uniform_q_tot = up_row_max(Yₜ)

    # Horizontal variation in the updraft scalars drives own-gradient updraft
    # diffusion.
    @. Y.c.sgsʲs.:(1).mse += FT(1.5e3) * sin(FT(2π) * ᶜx / FT(6400))
    @. Y.c.sgsʲs.:(1).q_tot *= 1 + FT(0.1) * sin(FT(2π) * ᶜx / FT(6400))
    @. Y.c.sgsʲs.:(1).q_lcl =
        FT(3e-5) * (1 + FT(0.2) * sin(FT(2π) * ᶜx / FT(6400) + FT(0.4)))
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    perturbed_mse, perturbed_q_tot = up_row_max(Yₜ)
    @test perturbed_mse > 0
    @test perturbed_q_tot > 0
    @test uniform_mse <= FT(1e-10) * perturbed_mse
    @test uniform_q_tot <= FT(1e-10) * perturbed_q_tot

    # With a uniform draft area, the element-wise power mean is the area
    # itself, so the updraft weight is `ρ f K_h_h a¹` and the intensive
    # tendency is the flux divergence divided by `ρa¹`.
    ᶜρa¹ = Y.c.sgsʲs.:(1).ρa
    ᶜwʲ = similar(Y.c.ρ)
    CA.ᶜupdraft_horizontal_diffusion_weight!(
        ᶜwʲ, Y.c.ρ, ᶜρa¹, p.precomputed.ᶜK_h_h, updraft_kh_factor,
    )
    ᶜw_ref = similar(Y.c.ρ)
    @. ᶜw_ref = ᶜρa¹ * updraft_kh_factor * p.precomputed.ᶜK_h_h
    @test parent(ᶜwʲ) ≈ parent(ᶜw_ref) rtol = FT(1e-12)
    ᶜexpected = similar(Y.c.ρ)
    @. ᶜexpected = CA.wdivₕ(ᶜw_ref * CA.gradₕ(Y.c.sgsʲs.:(1).mse)) / ᶜρa¹
    @test parent(Yₜ.c.sgsʲs.:(1).mse) ≈ parent(ᶜexpected) rtol = FT(1e-10)
    @. ᶜexpected = CA.wdivₕ(ᶜw_ref * CA.gradₕ(Y.c.sgsʲs.:(1).q_tot)) / ᶜρa¹
    @test parent(Yₜ.c.sgsʲs.:(1).q_tot) ≈ parent(ᶜexpected) rtol = FT(1e-10)

    # The ρa dry-mass counterpart is omitted, matching the vertical path.
    @test maximum(abs, parent(Yₜ.c.sgsʲs.:(1).ρa)) == 0

    # Updraft cloud species also diffuse down their own horizontal gradients.
    @test maximum(abs, parent(Yₜ.c.ρq_lcl)) > 0
    @. ᶜexpected = CA.wdivₕ(ᶜw_ref * CA.gradₕ(Y.c.sgsʲs.:(1).q_lcl)) / ᶜρa¹
    @test parent(Yₜ.c.sgsʲs.:(1).q_lcl) ≈ parent(ᶜexpected) rtol = FT(1e-10)

    # The grid mean receives the environment flux plus exactly the extensive
    # updraft tendency, so the diagnosed environment receives no residual.
    ᶜρa⁰ = similar(Y.c.ρ)
    @. ᶜρa⁰ = CA.ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model)
    ᶜq_lcl⁰ = similar(Y.c.ρ)
    ᶜq_lcl⁰ .= CA.ᶜspecific_env_value(CA.MatrixFields.@name(q_lcl), Y, p)
    @. ᶜexpected =
        CA.wdivₕ(ᶜρa⁰ * p.precomputed.ᶜK_h_h * CA.gradₕ(ᶜq_lcl⁰)) +
        ᶜρa¹ * Yₜ.c.sgsʲs.:(1).q_lcl
    @test parent(Yₜ.c.ρq_lcl) ≈ parent(ᶜexpected) rtol = FT(1e-10)

    # Rain has no horizontal transport, in the grid mean or the updraft.
    @test maximum(abs, parent(Yₜ.c.ρq_rai)) == 0
    @test maximum(abs, parent(Yₜ.c.sgsʲs.:(1).q_rai)) == 0

    # Where the updraft area collapses to zero next to nodes with finite area,
    # an element containing a draft-free node carries almost no updraft flux,
    # and the intensive updraft tendency stays bounded (the nodal area in the
    # weight would amplify it by the area ratio between nodes, about
    # 0.1 / 1e-8 here).
    Yₜ_ref = deepcopy(Yₜ)
    ᶜy = Fields.coordinate_field(Y.c).y
    @. Y.c.sgsʲs.:(1).ρa = ifelse(ᶜy < FT(3200), FT(0.1) * Y.c.ρ, FT(0))
    CA.ᶜupdraft_horizontal_diffusion_weight!(
        ᶜwʲ, Y.c.ρ, Y.c.sgsʲs.:(1).ρa, p.precomputed.ᶜK_h_h, updraft_kh_factor,
    )
    @test maximum(parent(ᶜwʲ)) <=
          FT(4e-8) * maximum(parent(Y.c.ρ .* p.precomputed.ᶜK_h_h))
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    @test maximum(abs, parent(Yₜ.c.sgsʲs.:(1).mse)) <=
          4 * maximum(abs, parent(Yₜ_ref.c.sgsʲs.:(1).mse))
    @test maximum(abs, parent(Yₜ.c.ρq_tot)) > 0
    @test !(parent(Yₜ.c.ρq_tot) ≈ parent(Yₜ_ref.c.ρq_tot))
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
