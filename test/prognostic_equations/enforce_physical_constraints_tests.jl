using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

include("../test_helpers.jl")

function create_physical_constraints_config(job_id = "physical_constraints_test")
    config_dict = Dict(
        "config" => "column",
        "initial_condition" => "DecayingProfile",
        "turbconv" => "prognostic_edmfx",
        "edmfx_filter" => true,
        "microphysics_model" => "1M",
        "output_default_diagnostics" => false,
    )
    return CA.AtmosConfig(config_dict; job_id)
end

function create_diagnostic_physical_constraints_config(
    job_id = "diag_physical_constraints_test",
)
    config_dict = Dict(
        "config" => "column",
        "initial_condition" => "DecayingProfile",
        "turbconv" => "diagnostic_edmfx",
        "edmfx_filter" => true,
        "microphysics_model" => "1M",
        "output_default_diagnostics" => false,
    )
    return CA.AtmosConfig(config_dict; job_id)
end

function introduce_constant!(field, value)
    FT = eltype(field)
    parent(field) .= FT(value)
    return nothing
end

function all_geq_zero(field)
    return minimum(parent(field)) >= zero(eltype(field))
end

@testset "Enforce Physical Constraints" begin

    @testset "Non-equilibrium microphysics condensate non-negativity" begin
        config = create_physical_constraints_config("physical_constraints_nonnegativity")
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)

        introduce_constant!(Y.c.ρq_lcl, -1e-7)
        introduce_constant!(Y.c.ρq_icl, 2e-7)
        introduce_constant!(Y.c.ρq_rai, -3e-7)
        introduce_constant!(Y.c.ρq_sno, 4e-7)

        @test minimum(parent(Y.c.ρq_lcl)) < FT(0)
        @test minimum(parent(Y.c.ρq_rai)) < FT(0)

        CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

        @test all_geq_zero(Y.c.ρq_lcl)
        @test all_geq_zero(Y.c.ρq_icl)
        @test all_geq_zero(Y.c.ρq_rai)
        @test all_geq_zero(Y.c.ρq_sno)
    end

    @testset "Condensate mass does not exceed total moisture" begin
        config = create_physical_constraints_config("physical_constraints_condensate_bound")
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)

        introduce_constant!(Y.c.ρq_tot, 1e-3)

        introduce_constant!(Y.c.ρq_lcl, 8e-4)
        introduce_constant!(Y.c.ρq_icl, 8e-4)
        introduce_constant!(Y.c.ρq_rai, 8e-4)
        introduce_constant!(Y.c.ρq_sno, 8e-4)

        ρq_cond_before =
            Y.c.ρq_lcl .+
            Y.c.ρq_icl .+
            Y.c.ρq_rai .+
            Y.c.ρq_sno

        @test maximum(parent(ρq_cond_before .- Y.c.ρq_tot)) > FT(0)

        CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

        ρq_cond_after =
            Y.c.ρq_lcl .+
            Y.c.ρq_icl .+
            Y.c.ρq_rai .+
            Y.c.ρq_sno

        @test maximum(parent(ρq_cond_after .- Y.c.ρq_tot)) <= 10 * eps(FT)
        @test all_geq_zero(Y.c.ρq_lcl)
        @test all_geq_zero(Y.c.ρq_icl)
        @test all_geq_zero(Y.c.ρq_rai)
        @test all_geq_zero(Y.c.ρq_sno)
    end

    @testset "Condensate is removed when total moisture is non-positive" begin
        config = create_physical_constraints_config("physical_constraints_negative_qtot")
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)

        introduce_constant!(Y.c.ρq_tot, -1e-6)

        introduce_constant!(Y.c.ρq_lcl, 1e-4)
        introduce_constant!(Y.c.ρq_icl, 1e-4)
        introduce_constant!(Y.c.ρq_rai, 1e-4)
        introduce_constant!(Y.c.ρq_sno, 1e-4)

        CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

        @test maximum(abs.(parent(Y.c.ρq_lcl))) <= 10 * eps(FT)
        @test maximum(abs.(parent(Y.c.ρq_icl))) <= 10 * eps(FT)
        @test maximum(abs.(parent(Y.c.ρq_rai))) <= 10 * eps(FT)
        @test maximum(abs.(parent(Y.c.ρq_sno))) <= 10 * eps(FT)
    end

    @testset "EDMF area fraction is clipped to physical bounds" begin
        config = create_physical_constraints_config("physical_constraints_area_bounds")
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)
        n = CA.n_mass_flux_subdomains(p.atmos.turbconv_model)

        CA.set_precomputed_quantities!(Y, p, FT(0))

        for j in 1:n
            introduce_constant!(Y.c.sgsʲs.:($j).ρa, -1e-12)

            CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

            @test minimum(parent(Y.c.sgsʲs.:($j).ρa)) >= FT(0)
        end
    end

    @testset "EDMF filter mixes small-area updraft state with environment" begin
        config = create_physical_constraints_config("physical_constraints_edmf_filter")
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)
        j = 1

        CA.set_precomputed_quantities!(Y, p, FT(0))
        (; ᶜh_tot, ᶜK) = p.precomputed

        introduce_constant!(Y.c.sgsʲs.:($j).ρa, 0)
        introduce_constant!(Y.c.sgsʲs.:($j).mse, 999)
        introduce_constant!(Y.c.sgsʲs.:($j).q_tot, 999)

        CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

        @test maximum(abs.(parent(Y.c.sgsʲs.:($j).mse .- (ᶜh_tot .- ᶜK)))) <=
              100 * eps(FT)

        @test maximum(
            abs.(
                parent(
                    Y.c.sgsʲs.:($j).q_tot .-
                    CA.specific.(Y.c.ρq_tot, Y.c.ρ),
                ),
            ),
        ) <= 100 * eps(FT)
    end

    @testset "EDMF updraft velocity is clipped to non-negative values" begin
        config = create_physical_constraints_config("physical_constraints_velocity")
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)
        j = 1

        parent(Y.f.sgsʲs.:($j).u₃.components.data.:1) .= -FT(1)

        @test minimum(parent(Y.f.sgsʲs.:($j).u₃.components.data.:1)) < FT(0)

        CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

        @test minimum(parent(Y.f.sgsʲs.:($j).u₃.components.data.:1)) >= FT(0)
    end

    @testset "EDMF updraft velocity is zero when area fraction is negligible" begin
        config =
            create_physical_constraints_config("physical_constraints_velocity_small_area")
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)
        j = 1

        introduce_constant!(Y.c.sgsʲs.:($j).ρa, 0)
        parent(Y.f.sgsʲs.:($j).u₃.components.data.:1) .= FT(1)

        CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

        @test maximum(abs.(parent(Y.f.sgsʲs.:($j).u₃.components.data.:1))) <=
              10 * eps(FT)
    end

    @testset "EDMF q_tot satisfies subdomain mass bound" begin
        config = create_physical_constraints_config("physical_constraints_subdomain_qtot")
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)
        j = 1

        introduce_constant!(Y.c.sgsʲs.:($j).ρa, 0.1)
        introduce_constant!(Y.c.ρq_tot, 1e-3)
        introduce_constant!(Y.c.sgsʲs.:($j).q_tot, 1)

        CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

        bound_violation =
            Y.c.sgsʲs.:($j).ρa .* Y.c.sgsʲs.:($j).q_tot .- Y.c.ρq_tot

        @test maximum(parent(bound_violation)) <= 10 * eps(FT)
        @test minimum(parent(Y.c.sgsʲs.:($j).q_tot)) >= FT(0)
    end

    @testset "No EDMF + 0M: enforce_physical_constraints! is a no-op" begin
        # Verifies that when neither 1M/2M microphysics nor EDMF is active,
        # calling enforce_physical_constraints! does not modify Y.
        config_dict = Dict(
            "config" => "column",
            "initial_condition" => "DecayingProfile",
            "microphysics_model" => "0M",
            "output_default_diagnostics" => false,
        )
        config = CA.AtmosConfig(
            config_dict;
            job_id = "physical_constraints_noop",
        )
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)
        ref_Y = deepcopy(Y)

        CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

        @test Y == ref_Y
    end
end

@testset "Enforce Physical Constraints — DiagnosticEDMFX" begin

    @testset "Diagnostic EDMF: condensate non-negativity" begin
        config = create_diagnostic_physical_constraints_config(
            "diag_physical_constraints_nonnegativity",
        )
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)

        introduce_constant!(Y.c.ρq_lcl, -1e-7)
        introduce_constant!(Y.c.ρq_icl, 2e-7)
        introduce_constant!(Y.c.ρq_rai, -3e-7)
        introduce_constant!(Y.c.ρq_sno, 4e-7)

        @test minimum(parent(Y.c.ρq_lcl)) < FT(0)
        @test minimum(parent(Y.c.ρq_rai)) < FT(0)

        CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

        @test all_geq_zero(Y.c.ρq_lcl)
        @test all_geq_zero(Y.c.ρq_icl)
        @test all_geq_zero(Y.c.ρq_rai)
        @test all_geq_zero(Y.c.ρq_sno)
    end

    @testset "Diagnostic EDMF: condensate mass does not exceed total moisture" begin
        config = create_diagnostic_physical_constraints_config(
            "diag_physical_constraints_condensate_bound",
        )
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)

        introduce_constant!(Y.c.ρq_tot, 1e-3)

        introduce_constant!(Y.c.ρq_lcl, 8e-4)
        introduce_constant!(Y.c.ρq_icl, 8e-4)
        introduce_constant!(Y.c.ρq_rai, 8e-4)
        introduce_constant!(Y.c.ρq_sno, 8e-4)

        ρq_cond_before =
            Y.c.ρq_lcl .+
            Y.c.ρq_icl .+
            Y.c.ρq_rai .+
            Y.c.ρq_sno

        @test maximum(parent(ρq_cond_before .- Y.c.ρq_tot)) > FT(0)

        CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

        ρq_cond_after =
            Y.c.ρq_lcl .+
            Y.c.ρq_icl .+
            Y.c.ρq_rai .+
            Y.c.ρq_sno

        @test maximum(parent(ρq_cond_after .- Y.c.ρq_tot)) <= 10 * eps(FT)
        @test all_geq_zero(Y.c.ρq_lcl)
        @test all_geq_zero(Y.c.ρq_icl)
        @test all_geq_zero(Y.c.ρq_rai)
        @test all_geq_zero(Y.c.ρq_sno)
    end

    @testset "Diagnostic EDMF: condensate is removed when total moisture is non-positive" begin
        config = create_diagnostic_physical_constraints_config(
            "diag_physical_constraints_negative_qtot",
        )
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)

        introduce_constant!(Y.c.ρq_tot, -1e-6)

        introduce_constant!(Y.c.ρq_lcl, 1e-4)
        introduce_constant!(Y.c.ρq_icl, 1e-4)
        introduce_constant!(Y.c.ρq_rai, 1e-4)
        introduce_constant!(Y.c.ρq_sno, 1e-4)

        CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

        @test maximum(abs.(parent(Y.c.ρq_lcl))) <= 10 * eps(FT)
        @test maximum(abs.(parent(Y.c.ρq_icl))) <= 10 * eps(FT)
        @test maximum(abs.(parent(Y.c.ρq_rai))) <= 10 * eps(FT)
        @test maximum(abs.(parent(Y.c.ρq_sno))) <= 10 * eps(FT)
    end

    @testset "Diagnostic EDMF: grid-mean fixers always run (edmfx_filter does not gate microphysics)" begin
        config_dict = Dict(
            "config" => "column",
            "initial_condition" => "DecayingProfile",
            "turbconv" => "diagnostic_edmfx",
            "edmfx_filter" => false,          # filter is off
            "microphysics_model" => "1M",
            "output_default_diagnostics" => false,
        )
        config = CA.AtmosConfig(
            config_dict;
            job_id = "diag_physical_constraints_filter_off",
        )
        (; Y, p) = generate_test_simulation(config)

        FT = eltype(Y)

        introduce_constant!(Y.c.ρq_lcl, -1e-7)
        introduce_constant!(Y.c.ρq_rai, -3e-7)

        CA.enforce_physical_constraints!(Y, p, FT(0), p.atmos)

        # Grid-mean microphysics fixers always run — negative values must be clipped
        @test all_geq_zero(Y.c.ρq_lcl)
        @test all_geq_zero(Y.c.ρq_rai)
    end
end
