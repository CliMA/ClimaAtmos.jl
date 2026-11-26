#=
Wiring of the displacement length `l_d` of the SGS variance closure with prognostic
TKE: the buoyancy bound on `l_mix` and the Lagrangian time that limits the floor
both use `κ_b = c_b tke / (2 c_σ)`. `l_d` is a temporary of the fused column loop of
`set_covariance_cache!`, so it is checked through the moisture variance that loop
sets, `q′q′ = min(ℓ_h² |∇ₕq|² + 2 c_σ l_d² (∂q/∂z)², (r_max q_tot⁰)²)`, with
`q_tot⁰` the environment total water under `PrognosticEDMFX`. The pure helpers
are tested in `cloud_fraction.jl`; this checks the arguments the model passes
them, and that the bound is taken against the environment mean.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Spaces

include("../../test_helpers.jl")

bomex_column_config(job_id, tomls) = CA.AtmosConfig(
    Dict{String, Any}(
        "initial_condition" => "Bomex",
        "FLOAT_TYPE" => "Float64",
        "config" => "column",
        "turbconv" => "prognostic_edmfx",
        "prognostic_tke" => true,
        "microphysics_model" => "1M",
        "z_max" => 3000.0,
        "z_elem" => 30,
        "z_stretch" => false,
        "dt" => "1secs",
        "t_end" => "10secs",
        "toml" => [tomls...],
        "output_default_diagnostics" => false,
    );
    job_id,
)

@testset "Displacement length wiring through `set_covariance_cache!` (Bomex column)" begin
    config = bomex_column_config("displacement_length_wiring_test", ())
    (; Y, p, simulation) = generate_test_simulation(config)
    FT = eltype(Y)
    @. Y.c.ρtke = FT(0.5) * Y.c.ρ
    CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)

    (; ᶜq′q′, ᶜl_mix, ᶜbg_coeffs, ᶠ∂θli∂z, ᶠ∂qt∂z, ᶜgradᵥ_q_tot, ᶜq_tot_nonneg) =
        p.precomputed
    (; ᶜq_tot_nonneg⁰) = p.precomputed
    q′q′ = copy(ᶜq′q′)
    l_mix = copy(ᶜl_mix)
    N² = similar(l_mix)
    CA.set_displacement_buoyancy_gradient!(N², ᶜbg_coeffs, ᶠ∂θli∂z, ᶠ∂qt∂z)

    params = p.params
    c_b = FT(CAP.static_stab_coeff(CAP.turbconv_params(params)))
    c_σ = FT(CAP.diagnostic_covariance_coeff(params))
    w_min = CA.floor_velocity(
        CA.horizontal_filter_scale(axes(Y.c)),
        FT(CAP.sgs_variance_vertical_velocity_floor(params)),
        FT(CAP.sgs_variance_floor_reference_resolution(params)),
        FT(CAP.sgs_variance_floor_resolution_exponent(params)),
    )
    τ_0 = FT(CAP.sgs_variance_floor_time_scale(params))
    r_max = FT(CAP.sgs_variance_max_rel_std(params))
    c_g = FT(CAP.sgs_variance_geometric_coeff(params))
    c_Δx = FT(CAP.sgs_variance_horizontal_scale_factor(params))
    Δx_h = FT(Spaces.node_horizontal_length_scale(Spaces.horizontal_space(axes(Y.c))))
    ℓ_h_q² = c_g * (c_Δx * Δx_h)^2
    inv_q = zeros(axes(Y.c))
    iszero(c_Δx) || CA.hgrad_invariant!(inv_q, ᶜq_tot_nonneg)
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜΔz = Fields.Δz_field(axes(Y.c))
    ᶜlg = Fields.local_geometry_field(Y.c)
    ∂q∂z = similar(l_mix)
    @. ∂q∂z = CA.projected_vector_data(CA.C3, ᶜgradᵥ_q_tot, ᶜlg)

    # Expected `q′q′` for a given `κ_b / tke`, through `l_mix_disp` and `l_d`.
    function expected_q′q′(κ_b_per_tke)
        κ_b = @. κ_b_per_tke * max(Y.c.ρtke / Y.c.ρ, 0)
        bounded = @. CA.buoyancy_bounded_mixing_length(l_mix, κ_b, N²)
        τ_turb = @. CA.turbulent_lagrangian_time(bounded, κ_b)
        l_d = @. CA.displacement_length(
            bounded,
            CA.floor_displacement_length(
                w_min^2 / (2 * c_σ), N², τ_turb, τ_0, ᶜz - z_sfc, ᶜΔz,
            ),
            ᶜz - z_sfc,
        )
        return @. min(
            CA.displacement_covariance(ℓ_h_q², 2 * c_σ * l_d^2, inv_q, ∂q∂z, ∂q∂z),
            (r_max * ᶜq_tot_nonneg⁰)^2,
        )
    end
    @test parent(q′q′) ≈ parent(expected_q′q′(c_b / (2 * c_σ)))

    # The test is not vacuous: the unnormalized `κ_b = c_b tke` gives a
    # different `q′q′` in this column.
    @test !(parent(q′q′) ≈ parent(expected_q′q′(c_b)))
end

@testset "Moisture-variance bound against the environment mean (Bomex column)" begin
    # A small `sgs_variance_max_rel_std` makes the bound bind wherever the
    # displacement variance is nonzero, so `q′q′` there equals
    # `(r_max q_tot⁰)²` of the environment. Bomex starts without updraft area,
    # so the updraft is given a tenth of the area and more total water than
    # the grid mean, which puts the environment total water below it.
    r_max = 1e-4
    toml = joinpath(mktempdir(), "variance_bound.toml")
    write(
        toml,
        "[sgs_variance_max_rel_std]\nvalue = $r_max\ntype = \"float\"\n",
    )
    config = bomex_column_config("environment_variance_bound_test", (toml,))
    (; Y, p, simulation) = generate_test_simulation(config)
    FT = eltype(Y)
    @test FT(CAP.sgs_variance_max_rel_std(p.params)) == FT(r_max)
    @. Y.c.ρtke = FT(0.5) * Y.c.ρ
    @. Y.c.sgsʲs.:(1).ρa = FT(0.1) * Y.c.ρ
    @. Y.c.sgsʲs.:(1).q_tot *= FT(1.2)
    CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)

    (; ᶜq′q′, ᶜq_tot_nonneg, ᶜq_tot_nonneg⁰) = p.precomputed
    q′q′ = parent(ᶜq′q′)
    bound⁰ = parent(@. (FT(r_max) * ᶜq_tot_nonneg⁰)^2)
    bound_gm = parent(@. (FT(r_max) * ᶜq_tot_nonneg)^2)
    @test all(q′q′ .<= bound⁰ .* (1 + 16 * eps(FT)))
    @test all(parent(ᶜq_tot_nonneg⁰) .< parent(ᶜq_tot_nonneg))
    binds = q′q′ .≈ bound⁰
    @test count(binds) > 0
    @test !any(q′q′[binds] .≈ bound_gm[binds])
end

@noinline _alloc_hgrad_invariants_dss!(ᶜinvariants, buffer) =
    @allocated Spaces.weighted_dss!(ᶜinvariants, buffer)

@testset "Packed 3D horizontal-gradient-invariant DSS parity and 0 B allocations" begin
    for FT in (Float32, Float64)
        toml = joinpath(mktempdir(), "hgrad_scale_$FT.toml")
        write(
            toml,
            "[sgs_variance_horizontal_scale_factor]\nvalue = 1.0\ntype = \"float\"\n",
        )
        config = CA.AtmosConfig(
            Dict{String, Any}(
                "initial_condition" => "Bomex",
                "FLOAT_TYPE" => string(FT),
                "config" => "box",
                "microphysics_model" => "0M",
                "x_max" => 6400.0,
                "x_elem" => 2,
                "y_max" => 6400.0,
                "y_elem" => 2,
                "z_max" => 3000.0,
                "z_elem" => 6,
                "z_stretch" => false,
                "dt" => "1secs",
                "t_end" => "10secs",
                "toml" => [toml],
                "output_default_diagnostics" => false,
            );
            job_id = "batched_hgrad_dss_$FT",
        )
        (; Y, p, simulation) = generate_test_simulation(config)
        @test CA.do_dss(axes(Y.c))
        @test !isnothing(p.scratch.ᶜhgrad_invariants_dss_buffer)

        ᶜcoord = Fields.coordinate_field(Y.c)
        ᶜpert_q = @. FT(1) +
           FT(0.2) *
           sin(FT(2π) * ᶜcoord.x / FT(6400)) *
           cos(FT(2π) * ᶜcoord.y / FT(6400)) *
           (FT(1) + ᶜcoord.z / FT(3000))
        ᶜpert_e = @. FT(1) +
           FT(0.03) *
           cos(FT(4π) * ᶜcoord.x / FT(6400)) *
           sin(FT(2π) * ᶜcoord.y / FT(6400)) *
           cos(FT(π) * ᶜcoord.z / FT(3000))
        @. Y.c.ρq_tot *= ᶜpert_q
        @. Y.c.ρe_tot *= ᶜpert_e
        CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)

        thermo_params = CAP.thermodynamics_params(p.params)
        CA.set_buoyancy_gradient_inputs!(Y, p, thermo_params, true)
        ᶜθ_li = copy(p.scratch.ᶜtemp_scalar)
        ref_q = similar(ᶜθ_li)
        ref_θ = similar(ᶜθ_li)
        ref_θq = similar(ᶜθ_li)
        CA.hgrad_invariant!(ref_q, p.precomputed.ᶜq_tot_nonneg)
        CA.hgrad_invariant!(ref_θ, ᶜθ_li)
        CA.hgrad_cross_invariant!(ref_θq, ᶜθ_li, p.precomputed.ᶜq_tot_nonneg)

        # `set_covariance_cache_and_cloud_fraction!` writes the three DSSed
        # horizontal invariants into the packed field `ᶜhgrad_invariants`, whose
        # components are the scratch fields `ᶜtemp_scalar_5`, `temp_data`, and
        # `ᶜtemp_scalar_6`.
        CA.set_covariance_cache_and_cloud_fraction!(Y, p)
        (; ᶜhgrad_invariants) = p.scratch
        ᶜinv_q = ᶜhgrad_invariants.q
        ᶜinv_θ = ᶜhgrad_invariants.θ
        ᶜinv_θq = ᶜhgrad_invariants.θq
        @test all(parent(p.scratch.ᶜtemp_scalar_5) .=== parent(ᶜinv_q))
        @test all(
            parent(Fields.Field(p.scratch.temp_data, axes(Y.c))) .===
            parent(ᶜinv_θ),
        )
        @test all(parent(p.scratch.ᶜtemp_scalar_6) .=== parent(ᶜinv_θq))
        @test maximum(abs, parent(ref_q)) > zero(FT)
        @test maximum(abs, parent(ref_θ)) > zero(FT)
        @test maximum(abs, parent(ref_θq)) > zero(FT)
        @test all(parent(ᶜinv_q) .=== parent(ref_q))
        @test all(parent(ᶜinv_θ) .=== parent(ref_θ))
        @test all(parent(ᶜinv_θq) .=== parent(ref_θq))

        if Base.JLOptions().check_bounds != 1
            buffer = p.scratch.ᶜhgrad_invariants_dss_buffer
            _alloc_hgrad_invariants_dss!(ᶜhgrad_invariants, buffer)
            @test _alloc_hgrad_invariants_dss!(ᶜhgrad_invariants, buffer) == 0
        end
    end
end
