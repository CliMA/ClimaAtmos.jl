#=
Wiring of the displacement length `l_d` of the SGS variance closure with prognostic
TKE: the buoyancy bound on `l_mix` and the Lagrangian time that limits the floor
both use `κ_b = c_b tke / (2 c_σ)`. `l_d` is a temporary of the fused column loop of
`set_covariance_cache!`, so it is checked through the moisture variance that loop
sets, `q′q′ = min(ℓ_h² |∇ₕq|² + 2 c_σ l_d² (∂q/∂z)², (r_max q_tot)²)`. The pure
helpers are tested in `cloud_fraction.jl`; this checks the arguments the model
passes them.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Spaces

include("../../test_helpers.jl")

@testset "Displacement length wiring through `set_covariance_cache!` (Bomex column)" begin
    config = CA.AtmosConfig(
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
            "toml" => [joinpath(pkgdir(CA), "toml", "prognostic_edmfx.toml")],
            "output_default_diagnostics" => false,
        );
        job_id = "displacement_length_wiring_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    FT = eltype(Y)
    @. Y.c.ρtke = FT(0.5) * Y.c.ρ
    CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)

    (; ᶜq′q′, ᶜl_mix, ᶜbg_coeffs, ᶠ∂θli∂z, ᶠ∂qt∂z, ᶜgradᵥ_q_tot, ᶜq_tot_nonneg) =
        p.precomputed
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
    iszero(c_Δx) || CA.hgrad_invariant!(inv_q, ᶜq_tot_nonneg, p)
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
            (r_max * ᶜq_tot_nonneg)^2,
        )
    end
    @test parent(q′q′) ≈ parent(expected_q′q′(c_b / (2 * c_σ)))

    # The test is not vacuous: the unnormalized `κ_b = c_b tke` gives a
    # different `q′q′` in this column.
    @test !(parent(q′q′) ≈ parent(expected_q′q′(c_b)))
end
