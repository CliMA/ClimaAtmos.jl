#=
Regression tests for discrete consistency between tracer transport and the
continuity equation (the "χ ≡ 1 test").

If a specific tracer concentration χ is uniform (χ ≡ 1, i.e. ρχ = ρ), then
every transport pathway must produce ∂(ρχ)/∂t = ∂ρ/∂t discretely. These
tests check this tracer-mass consistency relation.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Spaces

include("../test_helpers.jl")

@testset "Grid-mean transport consistency over topography" begin
    # Uniform 10 m/s wind over a Schar mountain with linear mesh warping, so
    # coordinate surfaces are sloped up to the model top and ᶠuₕ³ ≠ 0 there.
    config = CA.AtmosConfig(
        Dict(
            "config" => "plane",
            "z_stretch" => false,
            "initial_condition" => "ConstantBuoyancyFrequencyProfile",
            "microphysics_model" => "1M",
            "topography" => "Schar",
            "mesh_warp_type" => "Linear",
            "FLOAT_TYPE" => "Float64",
            "x_max" => 100e3,
            "x_elem" => 8,
            "z_max" => 21e3,
            "z_elem" => 20,
            "dt" => "1secs",
            "t_end" => "10secs",
            "disable_surface_flux_tendency" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "tracer_consistency_topo_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    FT = eltype(Y)

    (; ᶠu³) = p.precomputed

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

    # Microphysics tracer advection with the grid-mean flow (in
    # `implicit_vertical_advection_tendency!`) must reproduce the advective part
    # of the continuity tendency for uniform χ when sedimentation velocities are
    # zeroed.
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜρ_advection = similar(Y.c.ρ)
    @. ᶜρ_advection = -CA.ᶜadvdivᵥ(CA.ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠu³)
    ρ_advection = parent(ᶜρ_advection)
    @test maximum(abs, ρ_advection) > 0
    adv_tol = 100 * eps(FT) * maximum(abs, ρ_advection)
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end
    Yₜ_imp_adv = similar(Y)
    Yₜ_imp_adv .= zero(eltype(Yₜ_imp_adv))
    CA.implicit_vertical_advection_tendency!(Yₜ_imp_adv, Y, p, t)
    for ρχ_name in (:ρq_tot, :ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
        ρχ_tendency = parent(getproperty(Yₜ_imp_adv.c, ρχ_name))
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
