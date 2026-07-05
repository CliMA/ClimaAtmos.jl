#=
Regression tests for discrete consistency between tracer transport and the
continuity equation (the "q ≡ 1 test").

If a specific tracer concentration χ is uniform (χ ≡ 1, i.e. ρχ = ρ), then
every transport pathway must produce ∂(ρχ)/∂t = ∂ρ/∂t discretely, and the
EDMFX difference-form SGS fluxes must vanish identically. These tests guard
against three violations that were fixed on branch `ts/top-bc-fixes`:

  - T1: spurious mass flux through the model top over sloped terrain-following
    coordinates (`set_velocity_at_top!` must cancel `ᶠuₕ³` so that `ᶠu³ = 0`
    at the lid, and the continuity equation must use the same zero-boundary-
    flux divergence operator as the tracers).
  - T2: EDMFX SGS transport of auto-discovered tracers must use difference
    form ρᵏaᵏ(u³ᵏ - u³)(χᵏ - χ), which vanishes for uniform χ, instead of
    absolute subdomain fluxes.
  - T3: the EDMFX SGS mass flux of q_tot must have an identical ρ counterpart
    (vertical redistribution of water mass changes moist air mass).
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Spaces
import ClimaTimeSteppers as CTS

include("../test_helpers.jl")

@testset "Grid-mean transport consistency over topography (T1)" begin
    # Uniform 10 m/s wind over a Schar mountain with linear mesh warping, so
    # coordinate surfaces are sloped up to the model top and ᶠuₕ³ ≠ 0 there.
    config = CA.AtmosConfig(
        Dict(
            "config" => "plane",
            "initial_condition" => "ConstantBuoyancyFrequencyProfile",
            "microphysics_model" => "1M",
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
            "implicit_microphysics" => false,
            "disable_surface_flux_tendency" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "tracer_consistency_topo_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
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

    # With every specific concentration χ ≡ 1 (ρχ = ρ), the transport of each
    # tracer must reproduce the corresponding mass transport term-by-term.
    # The full implicit tendency preserves ∂(ρq_tot)/∂t = ∂ρ/∂t for q_tot ≡ 1:
    # the advective fluxes match because both use ᶜadvdivᵥ with ᶠu³ = 0 at the
    # boundaries, and the water sedimentation flux is added identically to
    # both ρ and ρq_tot.
    for ρχ_name in (:ρq_tot, :ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
        ᶜρχ = getproperty(Y.c, ρχ_name)
        ᶜρχ .= Y.c.ρ
    end
    Yₜ_imp = similar(Y)
    CA.implicit_tendency!(Yₜ_imp, Y, p, t)
    ρ_tendency = parent(Yₜ_imp.c.ρ)
    @test maximum(abs, ρ_tendency) > 0
    tol = 100 * eps(FT) * maximum(abs, ρ_tendency)
    @test maximum(abs, parent(Yₜ_imp.c.ρq_tot) .- ρ_tendency) <= tol

    # Passive-tracer advection with the grid-mean flow (explicit pathway) must
    # reproduce the advective part of the continuity tendency for uniform χ.
    # (The ρ row of Yₜ_imp cannot serve as the reference here because it also
    # includes water sedimentation, which is not part of this pathway.)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜρ_advection = similar(Y.c.ρ)
    @. ᶜρ_advection = -CA.ᶜadvdivᵥ(CA.ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠu³)
    ρ_advection = parent(ᶜρ_advection)
    @test maximum(abs, ρ_advection) > 0
    adv_tol = 100 * eps(FT) * maximum(abs, ρ_advection)
    Yₜ_exp = similar(Y)
    Yₜ_exp .= zero(eltype(Yₜ_exp))
    CA.explicit_vertical_advection_tendency!(Yₜ_exp, Y, p, t)
    for ρχ_name in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
        ρχ_tendency = parent(getproperty(Yₜ_exp.c, ρχ_name))
        @test maximum(abs, ρχ_tendency .- ρ_advection) <= adv_tol
    end

    # Subsidence (advective form). With descending prescribed
    # flow (w < 0, inflow through the lid), the top-cell tendency must vanish:
    # the advective form with zero boundary fluxes on both divergences is
    # equivalent to a zero-gradient inflow condition χ(above lid) = χ_top.
    # The pre-fix `Extrapolate` top BC copied the cell below into the top
    # cell, which is nonzero for any non-constant profile.
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶠsubsidence³ = Base.materialize(
        @. -FT(0.01) * CA.CT3(CA.unit_basis_vector_data(CA.CT3, ᶠlg))
    )
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜχ_linear = ᶜz ./ Spaces.z_max(axes(Y.f))
    ᶜρχₜ = similar(Y.c.ρ)
    ᶜρχₜ .= 0
    CA.subsidence!(ᶜρχₜ, Y.c.ρ, ᶠsubsidence³, ᶜχ_linear, Val(:first_order))
    subsidence_scale = maximum(abs, parent(ᶜρχₜ))
    @test subsidence_scale > 0  # non-vacuous: interior tendency is active
    ᶜρχₜ_top = Fields.level(ᶜρχₜ, Spaces.nlevels(axes(Y.c)))
    @test maximum(abs, parent(ᶜρχₜ_top)) <= 100 * eps(FT) * subsidence_scale

    # With uniform χ, the subsidence tendency must vanish identically at
    # every level for every reconstruction scheme (q ≡ 1 consistency).
    ᶜχ_uniform = similar(Y.c.ρ)
    ᶜχ_uniform .= 1
    for scheme in (Val(:none), Val(:first_order), Val(:third_order))
        ᶜρχₜ .= 0
        CA.subsidence!(ᶜρχₜ, Y.c.ρ, ᶠsubsidence³, ᶜχ_uniform, scheme)
        @test maximum(abs, parent(ᶜρχₜ)) <= 100 * eps(FT) * subsidence_scale
    end
end

@testset "EDMFX SGS mass-flux consistency (T2, T3)" begin
    config = CA.AtmosConfig(
        Dict(
            "config" => "column",
            "initial_condition" => "DYCOMS_RF01",
            "FLOAT_TYPE" => "Float64",
            "turbconv" => "prognostic_edmfx",
            "implicit_diffusion" => true,
            "approximate_linear_solve_iters" => 2,
            "edmfx_entr_model" => "Generalized",
            "edmfx_detr_model" => "Generalized",
            "edmfx_sgs_mass_flux" => true,
            "edmfx_sgs_diffusive_flux" => true,
            "edmfx_nh_pressure" => true,
            "edmfx_vertical_diffusion" => true,
            "edmfx_filter" => true,
            "prognostic_tke" => true,
            "microphysics_model" => "1M",
            "z_max" => 1500.0,
            "z_elem" => 30,
            "z_stretch" => false,
            "dt" => "120secs",
            "t_end" => "30mins",
            "ode_algo" => "ARS222",
            "toml" => [joinpath(pkgdir(CA), "toml", "prognostic_edmfx.toml")],
            "output_default_diagnostics" => false,
        );
        job_id = "tracer_consistency_edmfx_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    (; integrator) = simulation
    FT = eltype(Y)

    # Step to develop nontrivial updraft area and velocity structure.
    for _ in 1:10
        CTS.step!(integrator)
    end
    Y = integrator.u
    p = integrator.p
    t = integrator.t

    # T3: the SGS mass flux of q_tot must be accompanied by an identical
    # moist air mass tendency, mirroring the diffusive-flux treatment.
    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    flux_scale = maximum(abs, parent(Yₜ.c.ρq_tot))
    @test flux_scale > 0  # non-vacuous: the SGS q_tot flux is active
    @test parent(Yₜ.c.ρ) == parent(Yₜ.c.ρq_tot)

    # T2: with χ uniform across the grid mean and all subdomains (χ ≡ 1), the
    # difference-form SGS fluxes must vanish identically. Only Y is modified;
    # the precomputed subdomain velocities and densities keep their spun-up
    # values, so the fluxes vanish because (χᵏ - χ) = 0, not because the SGS
    # circulation is trivial.
    for χ_name in (:q_tot, :q_lcl, :q_icl, :q_rai, :q_sno)
        ᶜρχ = getproperty(Y.c, Symbol(:ρ, χ_name))
        ᶜρχ .= Y.c.ρ
        ᶜχʲ = getproperty(Y.c.sgsʲs.:(1), χ_name)
        ᶜχʲ .= FT(1)
    end
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    tol = FT(1e-10) * flux_scale
    for ρχ_name in (:ρ, :ρq_tot, :ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
        @test maximum(abs, parent(getproperty(Yₜ.c, ρχ_name))) <= tol
    end
end
