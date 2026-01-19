using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
using NullBroadcasts: NullBroadcasted
using ClimaCore: Spaces, Fields, Geometry
using ClimaCore.CommonSpaces
using Base.Broadcast: materialize

@testset "Viscous Sponge" begin
    FT = Float64
    z_max = FT(1)
    z_damping = FT(0.5)  # Damping starts at z = 0.5
    κ₂ = FT(2)  # Viscosity coefficient

    # Create extruded cubed sphere space
    ᶜspace = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 10,
        z_min = 0,
        z_max = z_max,
        radius = 10,
        h_elem = 4,
        n_quad_points = 4,
        staggering = CellCenter(),
    )
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶠz = Fields.coordinate_field(ᶠspace).z

    # Create test fields
    ᶜuₕ = Fields.Field(Geometry.Covariant12Vector{FT}, ᶜspace)
    @. ᶜuₕ = Geometry.Covariant12Vector(FT(1), FT(1))
    ᶠu₃ = Fields.Field(Geometry.Covariant3Vector{FT}, ᶠspace)
    @. ᶠu₃ = Geometry.Covariant3Vector(FT(1))
    ᶜρ = ones(ᶜspace)
    ᶜh_tot = ones(ᶜspace)
    ᶜχ = ones(ᶜspace)

    @testset "Damping coefficient β_viscous" begin
        s = CA.ViscousSponge(; zd = z_damping, κ₂ = κ₂)

        # β = κ₂ * sin²(π/2 * (z - zd) / (zmax - zd)) for z > zd, else 0
        expected = @. ifelse(
            ᶜz > z_damping,
            κ₂ * sin(FT(π) / 2 * (ᶜz - z_damping) / (z_max - z_damping))^2,
            FT(0),
        )

        @test CA.β_viscous.(s, ᶜz, z_max) ≈ expected

        # Test that damping is maximum at z_max (= κ₂)
        @test CA.β_viscous(s, z_max, z_max) ≈ κ₂
    end

    @testset "Tendency functions return finite values" begin
        s = CA.ViscousSponge(; zd = z_damping, κ₂ = κ₂)

        # Test viscous_sponge_tendency_uₕ returns finite values
        tendency_uₕ = CA.viscous_sponge_tendency_uₕ(ᶜuₕ, s)
        @test all(isfinite, parent(materialize(tendency_uₕ)))

        # Test viscous_sponge_tendency_u₃ returns finite values
        tendency_u₃ = CA.viscous_sponge_tendency_u₃(ᶠu₃, s)
        @test all(isfinite, parent(materialize(tendency_u₃)))

        # Test viscous_sponge_tendency_ρe_tot returns finite values
        tendency_ρe_tot = CA.viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, s)
        @test all(isfinite, parent(materialize(tendency_ρe_tot)))

        # Test viscous_sponge_tendency_tracer returns finite values
        tendency_tracer = CA.viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, s)
        @test all(isfinite, parent(materialize(tendency_tracer)))
    end

    @testset "No sponge (nothing) returns NullBroadcasted" begin
        @test CA.viscous_sponge_tendency_uₕ(ᶜuₕ, nothing) isa NullBroadcasted
        @test CA.viscous_sponge_tendency_u₃(ᶠu₃, nothing) isa NullBroadcasted
        @test CA.viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, nothing) isa NullBroadcasted
        @test CA.viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, nothing) isa NullBroadcasted
    end
end
