using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

include("../test_helpers.jl")

function create_vwb_config(species = ["ρq_tot"], job_id = "vwb_test")
    config_dict = Dict(
        "config" => "column",
        "initial_condition" => "DecayingProfile",
        "microphysics_model" => "1M",
        "tracer_nonnegativity_method" => "vertical_water_borrowing",
        "output_default_diagnostics" => false,
    )
    if !isnothing(species)
        config_dict["vertical_water_borrowing_species"] = species
    end
    return CA.AtmosConfig(config_dict; job_id)
end

function introduce_negatives!(field_array, fraction = 0.2, magnitude = 1e-7)
    FT = eltype(field_array)
    n_points = length(field_array)
    n_negative = max(1, round(Int, n_points * fraction))
    for idx in (n_points - n_negative + 1):n_points
        field_array[idx] = -FT(magnitude)
    end
    return sum(field_array)
end

@testset "Vertical Water Borrowing Limiter" begin
    @testset "Non-negativity and mass conservation" begin
        config = create_vwb_config()
        (; Y, p) = generate_test_simulation(config)
        FT = eltype(Y)
        @test p.atmos.water.tracer_nonnegativity_method isa
              CA.TracerNonnegativityVerticalWaterBorrowing
        @test !isnothing(p.numerics.vertical_water_borrowing_limiter)

        ref_Y = deepcopy(Y)
        ρq_tot_array = parent(Y.c.ρq_tot)
        total_mass_before = introduce_negatives!(ρq_tot_array)
        @test minimum(ρq_tot_array) < FT(0)

        CA.limiters_func!(Y, p, FT(0), ref_Y)
        @test minimum(ρq_tot_array) >= FT(0)
        n_points = length(ρq_tot_array)
        mass_tolerance = n_points * eps(FT)
        # Mass conservation: limiter conserves sum(ρq) per column. Only check when total_mass_before > 0
        # (when total_mass_before ≤ 0, nonnegativity forces the sum to increase).
        if total_mass_before > eps(FT)
            @test abs(sum(ρq_tot_array) - total_mass_before) /
                  (abs(total_mass_before) + eps(FT)) < mass_tolerance
        end
    end

    @testset "Species filtering" begin
        config = create_vwb_config(["ρq_tot"], "vwb_species_test")
        (; Y, p) = generate_test_simulation(config)
        FT = eltype(Y)
        ref_Y = deepcopy(Y)
        introduce_negatives!(parent(Y.c.ρq_tot))
        if hasproperty(Y.c, :ρq_liq)
            introduce_negatives!(parent(Y.c.ρq_liq))
        end

        CA.limiters_func!(Y, p, FT(0), ref_Y)
        @test minimum(parent(Y.c.ρq_tot)) >= FT(0)
    end

    @testset "All tracers mode" begin
        config = create_vwb_config(nothing, "vwb_all_tracers_test")
        (; Y, p) = generate_test_simulation(config)
        FT = eltype(Y)
        ref_Y = deepcopy(Y)
        introduce_negatives!(parent(Y.c.ρq_tot))
        if hasproperty(Y.c, :ρq_liq)
            introduce_negatives!(parent(Y.c.ρq_liq))
        end

        CA.limiters_func!(Y, p, FT(0), ref_Y)
        @test minimum(parent(Y.c.ρq_tot)) >= FT(0)
        hasproperty(Y.c, :ρq_liq) && @test minimum(parent(Y.c.ρq_liq)) >= FT(0)
    end
end
