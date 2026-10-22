#=
Wiring of `materialized_displacement_length!` with prognostic TKE: the buoyancy
bound on `l_mix` and the Lagrangian time that limits the floor both use
`κ_b = c_b tke / (2 c_σ)`. The pure helpers are tested in `cloud_fraction.jl`;
this checks the arguments the model passes them.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields

include("../../test_helpers.jl")

@testset "`materialized_displacement_length!` wiring (Bomex column)" begin
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

    ᶜl_d_model, ᶜl_mix_disp_model = CA.materialized_displacement_length!(Y, p)
    l_d = copy(ᶜl_d_model)
    l_mix_disp = similar(l_d)
    @. l_mix_disp = ᶜl_mix_disp_model
    N² = copy(p.scratch.ᶜtemp_scalar_2)
    l_mix = copy(p.precomputed.ᶜl_mix)

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
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜΔz = Fields.Δz_field(axes(Y.c))

    # Expected `l_mix_disp` and `l_d` for a given `κ_b / tke`.
    function expected(κ_b_per_tke)
        κ_b = @. κ_b_per_tke * max(Y.c.ρtke / Y.c.ρ, 0)
        bounded = @. CA.buoyancy_bounded_mixing_length(l_mix, κ_b, N²)
        τ_turb = @. CA.turbulent_lagrangian_time(bounded, κ_b)
        displacement = @. CA.displacement_length(
            bounded,
            CA.floor_displacement_length(
                w_min^2 / (2 * c_σ), N², τ_turb, τ_0, ᶜz - z_sfc, ᶜΔz,
            ),
            ᶜz - z_sfc,
        )
        return (bounded, displacement)
    end
    l_mix_disp_expected, l_d_expected = expected(c_b / (2 * c_σ))
    @test parent(l_mix_disp) ≈ parent(l_mix_disp_expected)
    @test parent(l_d) ≈ parent(l_d_expected)

    # The test is not vacuous: the unnormalized `κ_b = c_b tke` gives a
    # different `l_mix_disp` and `l_d` in this column.
    l_mix_disp_unnormalized, l_d_unnormalized = expected(c_b)
    @test !(parent(l_mix_disp) ≈ parent(l_mix_disp_unnormalized))
    @test !(parent(l_d) ≈ parent(l_d_unnormalized))
end
