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

@testset "Updraft sedimentation enthalpy consistency" begin
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

    # Zero out updraft velocity and buoyancy anomaly so only sedimentation
    # contributes to edmfx_sgs_vertical_advection_tendency!.
    fill!(Y.f.sgsʲs.:(1).u₃, Geometry.Covariant3Vector(FT(0)))
    fill!(p.precomputed.ᶠu³ʲs.:(1), Geometry.Contravariant3Vector(FT(0)))
    fill!(p.precomputed.ᶜρ_diffʲs.:(1), FT(0))

    # Prescribe non-uniform updraft liquid/rain and positive terminal velocities.
    ᶜz = Fields.coordinate_field(Y.c).z
    @. Y.c.sgsʲs.:(1).ρa = FT(0.1) * p.precomputed.ᶜρʲs.:(1)
    @. Y.c.sgsʲs.:(1).q_lcl = FT(5e-4) * exp(-(ᶜz) / FT(800))
    @. Y.c.sgsʲs.:(1).q_rai = FT(1e-3) * exp(-(ᶜz) / FT(1000))
    fill!(Y.c.sgsʲs.:(1).q_icl, FT(0))
    fill!(Y.c.sgsʲs.:(1).q_sno, FT(0))
    @. Y.c.ρq_lcl = Y.c.sgsʲs.:(1).ρa * Y.c.sgsʲs.:(1).q_lcl
    @. Y.c.ρq_rai = Y.c.sgsʲs.:(1).ρa * Y.c.sgsʲs.:(1).q_rai
    fill!(p.precomputed.ᶜwₗʲs.:(1), FT(0.5))
    fill!(p.precomputed.ᶜwₗ, FT(0.5))
    fill!(p.precomputed.ᶜwᵣʲs.:(1), FT(4.0))
    fill!(p.precomputed.ᶜwᵣ, FT(4.0))
    fill!(p.precomputed.ᶜwᵢʲs.:(1), FT(0))
    fill!(p.precomputed.ᶜwᵢ, FT(0))
    fill!(p.precomputed.ᶜwₛʲs.:(1), FT(0))
    fill!(p.precomputed.ᶜwₛ, FT(0))

    Yₜ = similar(Y)
    fill!(Yₜ, FT(0))
    CA.edmfx_sgs_vertical_advection_tendency!(
        Yₜ,
        Y,
        p,
        FT(0),
        p.atmos.turbconv_model,
    )

    q_tot_tend = Array(parent(Yₜ.c.sgsʲs.:(1).q_tot))
    q_lcl_tend = Array(parent(Yₜ.c.sgsʲs.:(1).q_lcl))
    q_rai_tend = Array(parent(Yₜ.c.sgsʲs.:(1).q_rai))
    mse_tend = Array(parent(Yₜ.c.sgsʲs.:(1).mse))
    T_up = Array(parent(p.precomputed.ᶜTʲs.:(1)))
    e_int_liq = TD.internal_energy_liquid.(Ref(thp), T_up)

    @test maximum(abs, q_lcl_tend) > 0
    @test maximum(abs, q_rai_tend) > 0
    @test q_tot_tend ≈ q_lcl_tend .+ q_rai_tend rtol = 100 * eps(FT)
    @test mse_tend ≈ q_tot_tend .* e_int_liq rtol = 100 * eps(FT)

    # Also verify frozen species (q_icl, q_sno) carry internal_energy_ice.
    fill!(Y.c.sgsʲs.:(1).q_lcl, FT(0))
    fill!(Y.c.sgsʲs.:(1).q_rai, FT(0))
    @. Y.c.sgsʲs.:(1).q_icl = FT(3e-4) * exp(-(ᶜz) / FT(900))
    @. Y.c.sgsʲs.:(1).q_sno = FT(8e-4) * exp(-(ᶜz) / FT(1100))
    @. Y.c.ρq_icl = Y.c.sgsʲs.:(1).ρa * Y.c.sgsʲs.:(1).q_icl
    @. Y.c.ρq_sno = Y.c.sgsʲs.:(1).ρa * Y.c.sgsʲs.:(1).q_sno
    fill!(p.precomputed.ᶜwₗʲs.:(1), FT(0))
    fill!(p.precomputed.ᶜwₗ, FT(0))
    fill!(p.precomputed.ᶜwᵣʲs.:(1), FT(0))
    fill!(p.precomputed.ᶜwᵣ, FT(0))
    fill!(p.precomputed.ᶜwᵢʲs.:(1), FT(0.3))
    fill!(p.precomputed.ᶜwᵢ, FT(0.3))
    fill!(p.precomputed.ᶜwₛʲs.:(1), FT(1.5))
    fill!(p.precomputed.ᶜwₛ, FT(1.5))

    fill!(Yₜ, FT(0))
    CA.edmfx_sgs_vertical_advection_tendency!(
        Yₜ,
        Y,
        p,
        FT(0),
        p.atmos.turbconv_model,
    )

    q_tot_ice_tend = Array(parent(Yₜ.c.sgsʲs.:(1).q_tot))
    q_icl_tend = Array(parent(Yₜ.c.sgsʲs.:(1).q_icl))
    q_sno_tend = Array(parent(Yₜ.c.sgsʲs.:(1).q_sno))
    mse_ice_tend = Array(parent(Yₜ.c.sgsʲs.:(1).mse))
    e_int_ice = TD.internal_energy_ice.(Ref(thp), T_up)

    @test maximum(abs, q_icl_tend) > 0
    @test maximum(abs, q_sno_tend) > 0
    @test q_tot_ice_tend ≈ q_icl_tend .+ q_sno_tend rtol = 100 * eps(FT)
    @test mse_ice_tend ≈ q_tot_ice_tend .* e_int_ice rtol = 100 * eps(FT)
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
