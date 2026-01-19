using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
using NullBroadcasts: NullBroadcasted
using ClimaCore: Spaces, Fields, Geometry
using ClimaCore.CommonSpaces
using Base.Broadcast: materialize

@testset "Rayleigh Sponge" begin
    FT = Float64
    z_max = FT(1)
    z_damping = FT(0.5)  # Damping starts at z = 0.5

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

    # Create test velocity fields
    ᶜuₕ = Fields.Field(Geometry.Covariant12Vector{FT}, ᶜspace)
    fill!(parent(ᶜuₕ), FT(1))  # Both components = 1
    ᶠw = Fields.Field(Geometry.Covariant3Vector{FT}, ᶠspace)
    fill!(parent(ᶠw), FT(1))
    ᶜχ = ones(ᶜspace)  # Tracer field
    ᶜχʲ = 2 .* ones(ᶜspace)  # Updraft tracer field

    @testset "Damping coefficient β" begin
        rs = CA.RayleighSponge(;
            zd = z_damping,
            α_uₕ = FT(1),
            α_w = FT(2),
            α_sgs_tracer = FT(3),
        )

        # β = α * sin²(π/2 * (z - zd) / (zmax - zd)) for z > zd, else 0
        expected_uₕ = @. ifelse(
            ᶜz > z_damping,
            FT(1) * sin(FT(π) / 2 * (ᶜz - z_damping) / (z_max - z_damping))^2,
            FT(0),
        )
        expected_w = @. ifelse(
            ᶠz > z_damping,
            FT(2) * sin(FT(π) / 2 * (ᶠz - z_damping) / (z_max - z_damping))^2,
            FT(0),
        )

        @test CA.β_rayleigh_uₕ.(rs, ᶜz, z_max) ≈ expected_uₕ
        @test CA.β_rayleigh_w.(rs, ᶠz, z_max) ≈ expected_w
    end

    @testset "Tendency for horizontal velocity" begin
        rs = CA.RayleighSponge(;
            zd = z_damping,
            α_uₕ = FT(1),
            α_w = FT(1),
            α_sgs_tracer = FT(1),
        )

        # Tendency = -β * uₕ
        tendency = CA.rayleigh_sponge_tendency_uₕ(ᶜuₕ, rs)
        β = CA.β_rayleigh_uₕ.(rs, ᶜz, z_max)
        expected = @. -β * ᶜuₕ

        @test materialize(tendency) ≈ expected
    end

    @testset "Tendency for SGS tracer (single argument)" begin
        rs = CA.RayleighSponge(;
            zd = z_damping,
            α_uₕ = FT(1),
            α_w = FT(1),
            α_sgs_tracer = FT(2),
        )

        # Tendency = -β * χ
        tendency = CA.rayleigh_sponge_tendency_sgs_tracer(ᶜχ, rs)
        β = CA.β_rayleigh_sgs_tracer.(rs, ᶜz, z_max)
        expected = @. -β * ᶜχ

        @test materialize(tendency) ≈ expected
    end

    @testset "Tendency for SGS tracer (updraft-environment difference)" begin
        rs = CA.RayleighSponge(;
            zd = z_damping,
            α_uₕ = FT(1),
            α_w = FT(1),
            α_sgs_tracer = FT(2),
        )

        # Tendency = -β * (χʲ - χ)
        tendency = CA.rayleigh_sponge_tendency_sgs_tracer(ᶜχʲ, ᶜχ, rs)
        β = CA.β_rayleigh_sgs_tracer.(rs, ᶜz, z_max)
        expected = @. -β * (ᶜχʲ - ᶜχ)

        @test materialize(tendency) ≈ expected
    end

    @testset "No sponge (nothing) returns NullBroadcasted" begin
        @test CA.rayleigh_sponge_tendency_uₕ(ᶜuₕ, nothing) isa NullBroadcasted
        @test CA.rayleigh_sponge_tendency_sgs_tracer(ᶜχ, nothing) isa NullBroadcasted
        @test CA.rayleigh_sponge_tendency_sgs_tracer(ᶜχʲ, ᶜχ, nothing) isa NullBroadcasted
    end
end
