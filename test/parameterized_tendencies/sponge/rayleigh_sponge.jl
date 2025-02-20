#=
julia --project=examples
using Revise; include("test/parameterized_tendencies/sponge/rayleigh_sponge.jl")
=#
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
using NullBroadcasts: NullBroadcasted
using ClimaCore.CommonSpaces
using ClimaCore: Spaces, Fields, Geometry, ClimaCore
using Test
using Base.Broadcast: materialize

pkgversion(ClimaCore) < v"0.14.20" && exit() # CommonSpaces
using ClimaCore.CommonSpaces

### Common Objects ###
@testset "Rayleigh-sponge functions" begin
    FT = Float64
    ᶜspace = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 10,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 10,
        n_quad_points = 4,
        staggering = CellCenter(),
    )
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶠz = Fields.coordinate_field(ᶠspace).z
    ᶜz = Fields.coordinate_field(ᶜspace).z
    zmax = maximum(ᶠz)
    ᶜuₕ = map(z -> zero(Geometry.Covariant12Vector{eltype(z)}), ᶜz)
    @. ᶜuₕ.components.data.:1 = 1
    @. ᶜuₕ.components.data.:2 = 1
    ### Component test begins here
    rs = CA.RayleighSponge(; zd = FT(0), α_uₕ = FT(1), α_w = FT(1))
    expected = @. sin(FT(π) / 2 * ᶜz / zmax)^2
    computed = CA.rayleigh_sponge_tendency_uₕ(ᶜuₕ, rs)
    @test CA.β_rayleigh_uₕ.(rs, ᶜz, zmax) == expected
    @test materialize(computed) == .-expected .* ᶜuₕ

    # Test when not using a Rayleigh sponge.
    computed = CA.rayleigh_sponge_tendency_uₕ(ᶜuₕ, nothing)
    @test computed isa NullBroadcasted
    @. ᶜuₕ += computed # test that it can broadcast
end
