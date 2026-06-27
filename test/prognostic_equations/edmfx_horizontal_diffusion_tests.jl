#=
Tests for the horizontal component of the EDMFX SGS diffusive flux
(`edmfx_sgs_horizontal_diffusive_flux_tendency!`): tendencies vanish for
horizontally uniform states and on single-column geometry, the moist air mass
tendency matches the `ρq_tot` tendency bitwise, the mixing-length grid-scale
limit is directional, and incompatible closure combinations are rejected at
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

    # A nonzero TKE makes the eddy diffusivity nonvacuous.
    @. Y.c.ρtke = FT(0.5) * Y.c.ρ

    rows = (:ρ, :ρe_tot, :ρq_tot, :ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno, :ρtke)
    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    # The weak divergence is assembled by DSS, which the solver applies each
    # stage. The momentum stress has nonzero components for sheared states,
    # so its element-boundary residues cancel only after DSS.
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

    # Non-vacuous fluxes for every perturbed pathway.
    @test perturbed_max[:ρq_tot] > 0
    @test perturbed_max[:ρe_tot] > 0
    @test perturbed_max[:ρq_rai] > 0
    @test perturbed_max[:ρtke] > 0
    @test perturbed_max_uₕ > 0
    @test perturbed_max_u₃ > 0

    # The moist air mass tendency matches the ρq_tot tendency bitwise.
    @test parent(Yₜ.c.ρ) == parent(Yₜ.c.ρq_tot)

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
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_horizontal_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )
    production_max = maximum(parent(Yₜ.c.ρtke))
    @test production_max > 0
    @test minimum(parent(Yₜ.c.ρtke)) >= -FT(1e-10) * production_max
    # Identically zero tracers stay identically zero.
    for name in (:ρq_lcl, :ρq_icl, :ρq_sno)
        @test uniform_max[name] == 0
        @test perturbed_max[name] == 0
    end

    # Directional grid-scale limit: a scalar horizontal grid scale equal to
    # the uniform cell thickness reproduces the vertical mixing length.
    ᶜl_default = similar(Y.c.ρ)
    ᶜl_default .= CA.ᶜmixing_length(Y, p)
    ᶜl_scalar = similar(Y.c.ρ)
    ᶜl_scalar .= CA.ᶜmixing_length(Y, p; grid_scale = FT(300))
    @test parent(ᶜl_scalar) ≈ parent(ᶜl_default) rtol = FT(1e-10)

    # The grid-scale limit binds: a small cap is attained exactly, and the
    # 1 m lower bound holds.
    ᶜl_capped = similar(Y.c.ρ)
    ᶜl_capped .= CA.ᶜmixing_length(Y, p; grid_scale = FT(5))
    @test maximum(parent(ᶜl_capped)) == FT(5)
    @test minimum(parent(ᶜl_capped)) >= FT(1)

    # Passing the default stability input explicitly is an identity.
    ᶜl_buoygrad = similar(Y.c.ρ)
    ᶜl_buoygrad .= CA.ᶜmixing_length(
        Y, p; buoyancy_gradient = p.precomputed.ᶜlinear_buoygrad,
    )
    @test parent(ᶜl_buoygrad) == parent(ᶜl_default)
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
        );
        job_id = "edmfx_horizontal_diffusion_column_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    FT = eltype(Y)

    @. Y.c.ρtke = FT(0.5) * Y.c.ρ

    # Spectral operators return exact zeros on a column, so all existing
    # single-column results are unaffected by the option.
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
end

@testset "Incompatible horizontal SGS closures are rejected" begin
    smag_config = CA.AtmosConfig(
        box_config_dict(; smagorinsky_lilly = "UVW");
        job_id = "edmfx_horizontal_diffusion_smag_test",
    )
    @test_throws ErrorException generate_test_simulation(smag_config)

    amd_config = CA.AtmosConfig(
        box_config_dict(; amd_les = true);
        job_id = "edmfx_horizontal_diffusion_amd_test",
    )
    @test_throws ErrorException generate_test_simulation(amd_config)
end
