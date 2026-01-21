#=
Variable manipulations unit tests for ClimaAtmos.jl

Tests for functions in src/utils/variable_manipulations.jl that handle
density-weighted variables, SGS/EDMFX calculations, and tracer operations.
=#

using Test
import ClimaAtmos as CA
import ClimaCore
import ClimaCore: Fields, Spaces, Geometry
import ClimaCore.MatrixFields: @name
using ClimaComms

include("test_helpers.jl")

# Mock TurbconvModel for testing regularization parameters
struct MockTurbconvModel{T}
    a_half::T
end

@testset "sgs_weight_function" begin
    # Test smooth weight function properties
    # w(0) should be 0, w(1) should be 1
    @test CA.sgs_weight_function(0.0, 0.5) ≈ 0.0 atol = 1e-10
    @test CA.sgs_weight_function(1.0, 0.5) ≈ 1.0 atol = 1e-10

    # w(a_half) should be 0.5
    @test CA.sgs_weight_function(0.5, 0.5) ≈ 0.5 atol = 1e-6
    @test CA.sgs_weight_function(0.3, 0.3) ≈ 0.5 atol = 1e-6

    # Should be monotonically increasing
    a_values = 0.0:0.1:1.0
    w_values = [CA.sgs_weight_function(a, 0.5) for a in a_values]
    @test all(diff(w_values) .>= 0)

    # Test with different a_half values
    @test 0 < CA.sgs_weight_function(0.2, 0.1) < 1
    @test 0 < CA.sgs_weight_function(0.8, 0.9) < 1
end

@testset "specific (regularized division)" begin
    # 2-argument version (Grid Mean)
    @test CA.specific(10.0, 2.0) ≈ 5.0
    @test CA.specific(0.0, 1.0) ≈ 0.0
    @test isfinite(CA.specific(1e-20, 1e-20))

    # 5-argument version (Regularized SGS)
    # specific(ρaχ, ρa, ρχ_fallback, ρ_fallback, turbconv_model)

    tc_model = MockTurbconvModel(0.5)

    # Case 1: Large area fraction a = ρa/ρ = 0.8/1.0 = 0.8 > a_half (0.5)
    # Weight should be > 0.5, result dominated by SGS value (8/0.8 = 10)
    ρaχ = 8.0
    ρa = 0.8
    ρχ_fallback = 5.0
    ρ_fallback = 1.0
    val = CA.specific(ρaχ, ρa, ρχ_fallback, ρ_fallback, tc_model)
    @test val ≈ ρaχ / ρa rtol = 0.1 # Should be close to SGS value

    # Case 2: Small area fraction a = 0.1 < a_half (0.5)
    # Weight should be < 0.5, result blended with fallback (5/1 = 5)
    ρaχ = 1.0
    ρa = 0.1
    val_small = CA.specific(ρaχ, ρa, ρχ_fallback, ρ_fallback, tc_model)
    # Should be blended between 10 (SGS) and 5 (GridMean)
    # 5.0 comes from specific(ρχ_fallback, ρ_fallback)
    @test CA.specific(ρχ_fallback, ρ_fallback) < val_small < ρaχ / ρa

    # Case 3: Zero area - should return fallback exactly due to check
    @test CA.specific(1.0, 0.0, 5.0, 1.0, tc_model) == 5.0
end

@testset "Name utilities" begin
    # Test is_ρ_weighted_name
    @test CA.is_ρ_weighted_name(@name(ρq_tot))
    @test CA.is_ρ_weighted_name(@name(ρe_tot))
    @test !CA.is_ρ_weighted_name(@name(q_tot))
    @test !CA.is_ρ_weighted_name(@name(T))
    @test CA.is_ρ_weighted_name(@name(ρ)) # Returns true by implementation (starts with ρ)

    # Test specific_tracer_name: ρX -> X
    @test CA.specific_tracer_name(@name(ρq_tot)) == @name(q_tot)
    @test CA.specific_tracer_name(@name(ρe_tot)) == @name(e_tot)

    # Test get_ρχ_name: X -> ρX
    @test CA.get_ρχ_name(@name(q_tot)) == @name(ρq_tot)
    @test CA.get_ρχ_name(@name(h_tot)) == @name(ρh_tot)

    # Test get_χʲ_name_from_ρχ_name: ρX -> sgsʲs.:(1).X
    # Note: Requires MatrixFields internal knowledge
    # ρq_tot -> sgsʲs.:(1).q_tot
    χʲ_name = CA.get_χʲ_name_from_ρχ_name(@name(ρq_tot))
    @test string(χʲ_name) == "@name(sgsʲs.:(1).q_tot)"
end

@testset "Field operations" begin
    # Setup test spaces and fields
    (; cent_space) = get_cartesian_spaces()
    FT = Float32

    # Create mock state Y with tracers
    # Create the field using map over coordinates to ensure correct structure
    coords = Fields.coordinate_field(cent_space)
    Y_c = similar(coords, NamedTuple{(:ρ, :ρq_tot, :ρe_tot), Tuple{FT, FT, FT}})
    Y_c.ρ .= 1.0
    Y_c.ρq_tot .= 2.0
    Y_c.ρe_tot .= 5.0

    Y = (; c = Y_c)

    # Test gs_tracer_names 
    # Should find ρq_tot but exclude ρ and ρe_tot (as per implementation)
    names = CA.gs_tracer_names(Y)
    @test @name(ρq_tot) in names
    @test !(@name(ρ) in names)
    @test !(@name(ρe_tot) in names)

    # Test specific_gs_tracer_names
    spec_names = CA.specific_gs_tracer_names(Y)
    @test @name(q_tot) in spec_names

    # Test basic access works (sanity check)
    @test Y.c.ρ isa Fields.Field
    @test parent(Y.c.ρ) isa AbstractArray

    # Test ᶜgs_tracers
    tracers_lazy = CA.ᶜgs_tracers(Y)
    tracers = Base.materialize(tracers_lazy)

    # Check that we extracted the correct field (ρq_tot should have value 2.0)
    @test all(parent(tracers.ρq_tot) .≈ 2.0)

    # Test ᶜspecific_gs_tracers (lazy evaluation)

    spec_tracers_lazy = CA.ᶜspecific_gs_tracers(Y)
    spec_tracers = Base.materialize(spec_tracers_lazy)

    # Check values
    # q_tot = ρq_tot / ρ = 2 / 1 = 2
    @test all(parent(spec_tracers.q_tot) .≈ 2.0)
end

@testset "ᶜenv_value & helpers" begin
    # Make promote_type_mul work with Numbers for testing
    # variable_manipulations.jl only defines it for AxisTensor
    CA.promote_type_mul(::Number, ::Number) = Float64

    # Test draft_sum and ᶜenv_value (decomposition)
    FT = Float64

    # Simple struct for draft state
    struct MockDraft
        ρa::FT
        val::FT
    end

    # Create drafts: 2 drafts with ρa=0.1
    drafts = (MockDraft(0.1, 5.0), MockDraft(0.1, 3.0))
    gs = (; sgsʲs = drafts)

    # draft_sum
    sum_val = CA.draft_sum(d -> d.val, drafts)
    @test sum_val == 8.0 # 5 + 3

    # env_value: GridMean(10) - Sum(8) = 2
    grid_scale_val = 10.0
    env_val = CA.env_value(grid_scale_val, d -> d.val, drafts)
    @test env_val == 2.0

    # Test u₃⁰ (Vertical velocity decomposition)
    # GridMean ρw = 10, ρ = 1
    # Drafts: ρa1=0.1, w1=10 -> mom1 = 1; ρa2=0.1, w2=20 -> mom2 = 2
    # Env momentum = 10 - (1+2) = 7
    # Env area = 1 - (0.1+0.1) = 0.8
    # Env velocity = 7 / 0.8 = 8.75

    ρ = 1.0
    w_mean = 10.0
    ρaʲs = (0.1, 0.1)
    wʲs = (10.0, 20.0)
    tc_model = MockTurbconvModel(0.5)

    u3_env = CA.u₃⁰(ρaʲs, wʲs, ρ, w_mean, tc_model)
    # Verify composition:
    # Env Momentum = Grid(10) - Drafts(3) = 7.0
    # Env Area Density = Grid(1) - Drafts(0.2) = 0.8
    # Fallback = Grid(10) / Grid(1)
    expected_u3 = CA.specific(7.0, 0.8, 10.0, 1.0, tc_model)
    @test u3_env ≈ expected_u3
end

@testset "Mathematical helpers" begin
    # Test mapreduce_with_init
    vals = [1, 2, 3]
    @test CA.mapreduce_with_init(x -> x, +, vals) == 6

    # Test unrolled_dotproduct
    t1 = (1.0, 2.0, 3.0)
    t2 = (2.0, 1.0, 4.0)
    # 2 + 2 + 12 = 16
    @test CA.unrolled_dotproduct(t1, t2) == 16.0

end
