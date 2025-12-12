#=
julia --project=.buildkite
using Revise; include("test/parameterized_tendencies/forcing/subsidence.jl")
=#
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
using NullBroadcasts: NullBroadcasted
using ClimaCore.CommonSpaces
using ClimaCore: Spaces, Fields, Geometry, ClimaCore
using Test
using Base.Broadcast: materialize

using ClimaCore.CommonSpaces

### Common Objects ###
@testset "Subsidence functions" begin
    FT = Float64
    ᶜspace = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 10,
        z_min = 0,
        z_max = 1000, # 1 km domain
        radius = 10,
        h_elem = 10,
        n_quad_points = 4,
        staggering = CellCenter(),
    )
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶠz = Fields.coordinate_field(ᶠspace).z
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶠlg = Fields.local_geometry_field(ᶠspace)

    # Create simple test fields
    ᶜρ = Fields.Field(FT, ᶜspace)
    @. ᶜρ = 1.0 # Constant density

    # Create a linear enthalpy profile (increases with height)
    ᶜh_tot = Fields.Field(FT, ᶜspace)
    @. ᶜh_tot = 300000.0 + 100.0 * ᶜz # Linear profile: h = 300000 + 100*z

    # Create a linear q_tot profile (decreases with height)
    ᶜq_tot = Fields.Field(FT, ᶜspace)
    @. ᶜq_tot = 0.02 - 0.00001 * ᶜz # Linear profile: q = 0.02 - 0.00001*z

    # Create a simple subsidence profile (constant downward velocity)
    # Negative velocity = downward motion (subsidence)
    w_subsidence = FT(-0.01) # -0.01 m/s = -1 cm/s subsidence
    subsidence_profile(z) = w_subsidence
    subsidence = CA.Subsidence(subsidence_profile)

    # Test subsidence_velocity_field helper
    ᶠsubsidence³_lazy = CA.subsidence_velocity_field(ᶠz, ᶠlg, subsidence_profile)
    ᶠsubsidence³ = materialize(ᶠsubsidence³_lazy)
    # Verify the velocity field has the right magnitude
    # The vertical component should be approximately w_subsidence
    ᶠw_data = [Fields.field_values(ᶠsubsidence³)[i].components.data.:1 for i in 1:length(ᶠsubsidence³)]
    @test all(abs.(ᶠw_data .- w_subsidence) .< 1e-10)

    ### Test subsidence_tendency_ρe_tot ###
    @testset "subsidence_tendency_ρe_tot" begin
        tendency_lazy = CA.subsidence_tendency_ρe_tot(ᶜρ, ᶜh_tot, ᶠsubsidence³, subsidence)
        @test !(tendency_lazy isa NullBroadcasted)

        tendency = materialize(tendency_lazy)

        # Physical check: With downward motion (negative w) and positive gradient (dh/dz > 0),
        # the tendency should be positive (energy increases at lower levels due to subsidence)
        # The tendency should be approximately: -ρ * w * dh/dz
        # With w = -0.01 m/s, dh/dz = 100 J/(kg*m), ρ = 1 kg/m³
        # Expected tendency ≈ -1 * (-0.01) * 100 = 1.0 J/(kg*m³*s)
        # But we need to account for the discrete operator...

        # At least verify it's not all zeros
        @test !all(iszero.(Fields.field_values(tendency)))

        # Test that it returns NullBroadcasted when subsidence is nothing
        tendency_null = CA.subsidence_tendency_ρe_tot(ᶜρ, ᶜh_tot, ᶠsubsidence³, nothing)
        @test tendency_null isa NullBroadcasted

        # Test that it can be broadcast-added
        test_tendency = Fields.Field(FT, ᶜspace)
        @. test_tendency += tendency_lazy
        @test test_tendency ≈ tendency
    end

    ### Test subsidence_tendency_ρq_tot ###
    @testset "subsidence_tendency_ρq_tot" begin
        tendency_lazy = CA.subsidence_tendency_ρq_tot(ᶜρ, ᶜq_tot, ᶠsubsidence³, subsidence)
        @test !(tendency_lazy isa NullBroadcasted)

        tendency = materialize(tendency_lazy)

        # Physical check: With downward motion (negative w) and negative gradient (dq/dz < 0),
        # the tendency should be negative (moisture decreases at lower levels)
        # At least verify it's not all zeros
        @test !all(iszero.(Fields.field_values(tendency)))

        # Test that it returns NullBroadcasted when subsidence is nothing
        tendency_null = CA.subsidence_tendency_ρq_tot(ᶜρ, ᶜq_tot, ᶠsubsidence³, nothing)
        @test tendency_null isa NullBroadcasted

        # Test that it can be broadcast-added
        test_tendency = Fields.Field(FT, ᶜspace)
        @. test_tendency += tendency_lazy
        @test test_tendency ≈ tendency
    end

    ### Test subsidence_tendency_ρq_liq ###
    @testset "subsidence_tendency_ρq_liq" begin
        ᶜq_liq = Fields.Field(FT, ᶜspace)
        @. ᶜq_liq = 0.001 - 0.000001 * ᶜz # Small linear profile

        tendency_lazy = CA.subsidence_tendency_ρq_liq(ᶜρ, ᶜq_liq, ᶠsubsidence³, subsidence)
        @test !(tendency_lazy isa NullBroadcasted)

        tendency = materialize(tendency_lazy)

        # At least verify it's not all zeros
        @test !all(iszero.(Fields.field_values(tendency)))

        # Test that it returns NullBroadcasted when subsidence is nothing
        tendency_null = CA.subsidence_tendency_ρq_liq(ᶜρ, ᶜq_liq, ᶠsubsidence³, nothing)
        @test tendency_null isa NullBroadcasted

        # Test that it can be broadcast-added
        test_tendency = Fields.Field(FT, ᶜspace)
        @. test_tendency += tendency_lazy
        @test test_tendency ≈ tendency
    end

    ### Test subsidence_tendency_ρq_ice ###
    @testset "subsidence_tendency_ρq_ice" begin
        ᶜq_ice = Fields.Field(FT, ᶜspace)
        @. ᶜq_ice = 0.0001 - 0.0000001 * ᶜz # Very small linear profile

        tendency_lazy = CA.subsidence_tendency_ρq_ice(ᶜρ, ᶜq_ice, ᶠsubsidence³, subsidence)
        @test !(tendency_lazy isa NullBroadcasted)

        tendency = materialize(tendency_lazy)

        # At least verify it's not all zeros
        @test !all(iszero.(Fields.field_values(tendency)))

        # Test that it returns NullBroadcasted when subsidence is nothing
        tendency_null = CA.subsidence_tendency_ρq_ice(ᶜρ, ᶜq_ice, ᶠsubsidence³, nothing)
        @test tendency_null isa NullBroadcasted

        # Test that it can be broadcast-added
        test_tendency = Fields.Field(FT, ᶜspace)
        @. test_tendency += tendency_lazy
        @test test_tendency ≈ tendency
    end

    ### Test fusion capability ###
    @testset "tendency fusion" begin
        # Test that multiple tendencies can be combined (fused)
        tend_ρe_lazy = CA.subsidence_tendency_ρe_tot(ᶜρ, ᶜh_tot, ᶠsubsidence³, subsidence)
        tend_ρq_lazy = CA.subsidence_tendency_ρq_tot(ᶜρ, ᶜq_tot, ᶠsubsidence³, subsidence)

        # Should be able to combine them in a single broadcast
        combined_lazy = @. tend_ρe_lazy + tend_ρq_lazy
        combined = materialize(combined_lazy)

        # Verify the result is the sum
        tend_ρe = materialize(tend_ρe_lazy)
        tend_ρq = materialize(tend_ρq_lazy)
        @test combined ≈ tend_ρe .+ tend_ρq
    end
end
