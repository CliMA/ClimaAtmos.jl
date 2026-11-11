#=
Regression tests for discrete consistency between tracer transport and the
continuity equation (the "χ ≡ 1 test").

If a specific tracer concentration χ is uniform (χ ≡ 1, i.e. ρχ = ρ), then
every transport pathway must produce ∂(ρχ)/∂t = ∂ρ/∂t discretely, and the
EDMFX difference-form SGS fluxes must vanish identically. These tests check
this tracer-mass consistency relation.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Spaces
import ClimaTimeSteppers as CTS

include("../test_helpers.jl")

# Measured behind a function barrier: an inline `@allocated` at testset scope
# counts the allocations of the dynamic dispatch to `f!`.
function _measure_allocs(f!, args...)
    f!(args...)
    return @allocated f!(args...)
end

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
            "implicit_microphysics" => false,
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

@testset "EDMFX SGS mass-flux consistency" begin
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

    # The SGS mass flux of q_tot must be accompanied by an identical
    # moist air mass tendency, mirroring the diffusive-flux treatment.
    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    flux_scale = maximum(abs, parent(Yₜ.c.ρq_tot))
    @test flux_scale > 0  # non-vacuous: the SGS q_tot flux is active
    @test parent(Yₜ.c.ρ) == parent(Yₜ.c.ρq_tot)

    # Solving the tridiagonal `(c.ρtke, c.ρtke)` advection block of
    # `update_sedimentation_jacobian!` (with `(c.ρtke, f.u₃)` omitted) keeps a
    # localized `ρtke` spike nonnegative and conservative at `C_γ = 2.5` while
    # `correct_implicit_advection_tendency!` leaves `ρtke` untouched.
    u₃_orig = copy(Y.f.u₃)
    tke_orig = copy(Y.c.ρtke)
    ρ_orig = copy(Y.c.ρ)
    z_f = Fields.coordinate_field(Y.f).z
    matrix = integrator.cache.newtons_method_cache.j.cache.matrix.matrix
    dtγ = FT(p.dtγ)
    @test !haskey(matrix, (CA.@name(c.ρtke), CA.@name(f.u₃)))

    # With condensates zeroed so `vertical_advection_of_water_tendency!` adds no
    # sedimentation mass flux to `Yₜ.c.ρ`, solving the tridiagonal
    # `(c.ρtke, c.ρtke)` advection block on a 1-cell `ρtke` spike at `C_γ ≈ 2.5`
    # keeps `ρtke` nonnegative and conserves column-integrated `ρtke`, while
    # `correct_implicit_advection_tendency!` leaves `Yₜ.c.ρtke == 0`.
    cond_orig = (;
        ρq_lcl = copy(Y.c.ρq_lcl),
        ρq_icl = copy(Y.c.ρq_icl),
        ρq_rai = copy(Y.c.ρq_rai),
        ρq_sno = copy(Y.c.ρq_sno),
        q_lclʲ = copy(Y.c.sgsʲs.:(1).q_lcl),
        q_iclʲ = copy(Y.c.sgsʲs.:(1).q_icl),
        q_raiʲ = copy(Y.c.sgsʲs.:(1).q_rai),
        q_snoʲ = copy(Y.c.sgsʲs.:(1).q_sno),
    )
    for ρχ_name in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
        fill!(parent(getproperty(Y.c, ρχ_name)), FT(0))
    end
    for χ_name in (:q_lcl, :q_icl, :q_rai, :q_sno)
        fill!(parent(getproperty(Y.c.sgsʲs.:(1), χ_name)), FT(0))
    end
    n_lev = Spaces.nlevels(axes(Y.c))
    Δz_col = FT(1500) / n_lev
    @. Y.c.ρ = ρ_orig
    fill!(parent(Y.c.ρtke), FT(0))
    parent(Y.c.ρtke)[div(n_lev, 2)] = parent(Y.c.ρ)[div(n_lev, 2)]
    @. Y.f.u₃ = CA.Geometry.Covariant3Vector(
        CA.Geometry.WVector(
            ifelse((z_f > FT(0)) & (z_f < FT(1500)), FT(2.5) * Δz_col / dtγ, FT(0)),
        ),
    )
    CA.set_implicit_precomputed_quantities_core!(Y, p, t)
    CA.update_sedimentation_jacobian!(matrix, Y, p, dtγ)
    Yₜ .= zero(eltype(Yₜ))
    CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    ᶜA_tke = similar(p.scratch.ᶜupwind_dissipation_system)
    @. ᶜA_tke = -($(matrix[CA.@name(c.ρtke), CA.@name(c.ρtke)]))
    ᶜΔρtke = similar(Y.c.ρtke)
    ᶜrhs_tke = @. dtγ * Yₜ.c.ρtke
    CA.MatrixFields.single_field_solve!(
        p.scratch.ᶜupwind_dissipation_solver_cache,
        ᶜΔρtke,
        ᶜA_tke,
        ᶜrhs_tke,
    )
    Y_solved = copy(Y)
    @. Y_solved.c.ρtke = Y.c.ρtke + ᶜΔρtke
    Yₜ_post = similar(Y)
    CA.correct_implicit_advection_tendency!(Yₜ_post, Y_solved, p, t)
    @test all(iszero, parent(Yₜ_post.c.ρtke))
    tke_spike_end = Array(parent(Y_solved.c.ρtke))
    @test minimum(tke_spike_end) >= -FT(1e-14)
    @test sum(tke_spike_end) ≈ sum(Array(parent(Y.c.ρtke))) rtol = FT(1e-13)

    @. Y.c.ρ = ρ_orig
    @. Y.c.ρtke = tke_orig
    @. Y.f.u₃ = u₃_orig
    @. Y.c.ρq_lcl = cond_orig.ρq_lcl
    @. Y.c.ρq_icl = cond_orig.ρq_icl
    @. Y.c.ρq_rai = cond_orig.ρq_rai
    @. Y.c.ρq_sno = cond_orig.ρq_sno
    @. Y.c.sgsʲs.:(1).q_lcl = cond_orig.q_lclʲ
    @. Y.c.sgsʲs.:(1).q_icl = cond_orig.q_iclʲ
    @. Y.c.sgsʲs.:(1).q_rai = cond_orig.q_raiʲ
    @. Y.c.sgsʲs.:(1).q_sno = cond_orig.q_snoʲ
    CA.set_implicit_precomputed_quantities!(Y, p, t)

    # With χ uniform across the grid mean and all subdomains (χ ≡ 1), the
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
