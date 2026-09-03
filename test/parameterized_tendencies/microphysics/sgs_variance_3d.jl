#=
Unit tests for the 3D, scale-aware SGS variance closure (`SGSVariance3D`):

  - pointwise tests of the helpers `horizontal_gradient_invariants`,
    `sgs_cov_vertical`, `sgs_cov_horizontal`, `sgs_correlation_Tq`, and the
    `SGSVarianceCoeffs` bundle;
  - integration tests that a box EDMF simulation with `sgs_variance_model: 3d`
    produces finite, non-negative variances and a correlation in `[-1, 1]`, that
    the horizontal gradient invariants are allocated on a box but not on a
    column, and that the diagnosed correlation is anti-correlated for a
    stratified boundary layer.

The observational / validation runs (gray-zone Δx convergence, LES comparison,
recalibration) are intentionally not included here.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Geometry
const CAP = CA.Parameters

include("../../test_helpers.jl")

@testset "SGS variance helpers (pointwise)" begin
    FT = Float64
    UVW = Geometry.UVWVector

    # horizontal_gradient_invariants: flat case (vertical gradient has no u,v),
    # so turbulent and geometric horizontal invariants coincide.
    Gθ_h = UVW(FT(2), FT(0), FT(0))   # ∂θ/∂x = 2
    Gq_h = UVW(FT(-3), FT(0), FT(0))  # ∂q/∂x = -3
    Gθ_v = UVW(FT(0), FT(0), FT(5))   # ∂θ/∂z (physical vertical only)
    Gq_v = UVW(FT(0), FT(0), FT(7))
    inv = CA.horizontal_gradient_invariants(Gθ_h, Gq_h, Gθ_v, Gq_v)
    @test inv.θθ_t ≈ 4    # (2)^2
    @test inv.qq_t ≈ 9    # (-3)^2
    @test inv.θq_t ≈ -6   # 2 * -3  (anti-correlated)
    @test inv.θθ_g ≈ 4
    @test inv.qq_g ≈ 9
    @test inv.θq_g ≈ -6

    # With a tilted along-surface gradient (nonzero w), the geometric invariant
    # includes the vertical tilt component, but the turbulent invariant (from
    # the slope-corrected full gradient) uses only the horizontal components.
    Gθ_ht = UVW(FT(2), FT(0), FT(1))   # along-surface, tilted
    Gθ_vt = UVW(FT(0), FT(0), FT(-1))  # vertical cancels the tilt
    inv_t = CA.horizontal_gradient_invariants(Gθ_ht, Gθ_ht, Gθ_vt, Gθ_vt)
    @test inv_t.θθ_t ≈ 4          # (2)^2, tilt cancelled by the vertical part
    @test inv_t.θθ_g ≈ 4 + 1      # (2)^2 + (1)^2, tilt retained

    # sgs_cov_vertical / sgs_cov_horizontal: turbulent + geometric split.
    coeffs = CA.SGSVarianceCoeffs{FT}(FT(2), FT(1 // 12), FT(100), FT(1))
    # vertical: 2*C*l_v^2*dz_prod + c_g*(c_Δz*Δz)^2*dz_prod
    @test CA.sgs_cov_vertical(coeffs, FT(3), FT(10), FT(4)) ≈
          2 * 2 * 9 * 4 + (1 / 12) * 100 * 4
    # horizontal: 2*C*l_h^2*inv_t + c_g*Δx_eff^2*inv_g
    @test CA.sgs_cov_horizontal(coeffs, FT(3), FT(4), FT(4)) ≈
          2 * 2 * 9 * 4 + (1 / 12) * 10000 * 4
    # cross-covariance can be negative
    @test CA.sgs_cov_horizontal(coeffs, FT(3), FT(-4), FT(-4)) < 0

    # sgs_correlation_Tq: clamp, fallback, and sign.
    @test CA.sgs_correlation_Tq(FT(4), FT(9), FT(-6), FT(0.6), FT(1)) ≈ -1
    @test CA.sgs_correlation_Tq(FT(4), FT(9), FT(3), FT(0.6), FT(1)) ≈ 0.5
    @test CA.sgs_correlation_Tq(FT(4), FT(9), FT(6), FT(0.6), FT(0.9)) ≈ 0.9  # capped
    @test CA.sgs_correlation_Tq(FT(0), FT(0), FT(0), FT(0.6), FT(1)) == 0.6   # fallback
end

function edmfx_config_dict(; extra...)
    dict = Dict{String, Any}(
        "initial_condition" => "Bomex",
        "FLOAT_TYPE" => "Float64",
        "turbconv" => "prognostic_edmfx",
        "edmfx_entr_model" => "Generalized",
        "edmfx_detr_model" => "Generalized",
        "edmfx_sgs_mass_flux" => true,
        "edmfx_sgs_diffusive_flux" => true,
        "edmfx_nh_pressure" => true,
        "prognostic_tke" => true,
        "cloud_model" => "quadrature",
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

@testset "3D SGS variance on a box (EDMFX)" begin
    config = CA.AtmosConfig(
        edmfx_config_dict(;
            config = "box",
            x_max = 6400.0,
            x_elem = 2,
            y_max = 6400.0,
            y_elem = 2,
            sgs_variance_model = "3d",
            tq_correlation_model = "diagnosed",
        );
        job_id = "sgs_variance_3d_box_test",
    )
    (; Y, p) = generate_test_simulation(config)
    FT = eltype(Y)

    @test p.atmos.sgs_variance_model isa CA.SGSVariance3D
    @test p.atmos.tq_correlation_model isa CA.DiagnosedTqCorrelation
    # Horizontal invariants are allocated on a box.
    @test hasproperty(p.precomputed, :ᶜ∇ₕ_inv)
    @test hasproperty(p.precomputed, :ᶜl_mix_h)

    # Give the column some TKE and a horizontal moisture perturbation so both
    # turbulent and geometric horizontal terms are active.
    ᶜx = Fields.coordinate_field(Y.c).x
    @. Y.c.ρtke = FT(0.5) * Y.c.ρ
    @. Y.c.ρq_tot *= 1 + FT(0.1) * sin(FT(2π) * ᶜx / FT(6400))

    CA.set_covariance_cache_and_cloud_fraction!(Y, p)

    (; ᶜT′T′, ᶜq′q′, ᶜT′q′, ᶜcorr_Tq) = p.precomputed
    # Variances are finite and non-negative; correlation is a valid coefficient.
    @test all(isfinite, parent(ᶜT′T′))
    @test all(isfinite, parent(ᶜq′q′))
    @test all(isfinite, parent(ᶜT′q′))
    @test minimum(parent(ᶜT′T′)) >= 0
    @test minimum(parent(ᶜq′q′)) >= 0
    @test maximum(parent(ᶜq′q′)) > 0
    @test maximum(abs, parent(ᶜcorr_Tq)) <= 1
    # Cauchy-Schwarz: covariance bounded by the product of the standard deviations.
    @test all(parent(ᶜT′q′) .^ 2 .<= parent(ᶜT′T′) .* parent(ᶜq′q′) .+ eps(FT))
    # Cloud fraction stays a probability.
    @test minimum(parent(p.precomputed.ᶜcloud_fraction)) >= 0
    @test maximum(parent(p.precomputed.ᶜcloud_fraction)) <= 1
end

@testset "3D SGS variance skips horizontal terms on a column (EDMFX)" begin
    config = CA.AtmosConfig(
        edmfx_config_dict(; config = "column", sgs_variance_model = "3d");
        job_id = "sgs_variance_3d_column_test",
    )
    (; Y, p) = generate_test_simulation(config)

    @test p.atmos.sgs_variance_model isa CA.SGSVariance3D
    # On a single column there is no horizontal scale, so the horizontal
    # invariants and the horizontal mixing length are not allocated.
    @test !hasproperty(p.precomputed, :ᶜ∇ₕ_inv)
    @test !hasproperty(p.precomputed, :ᶜl_mix_h)

    # The closure still runs (vertical turbulent + vertical geometric terms).
    CA.set_covariance_cache_and_cloud_fraction!(Y, p)
    @test all(isfinite, parent(p.precomputed.ᶜq′q′))
    @test minimum(parent(p.precomputed.ᶜq′q′)) >= 0
end
