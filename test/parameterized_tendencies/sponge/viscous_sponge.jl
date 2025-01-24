#=
julia --project=examples
using Revise; include("test/parameterized_tendencies/sponge/viscous_sponge.jl")
=#
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
using ClimaCore.CommonSpaces
using ClimaCore: Spaces, Fields, Geometry, ClimaCore
using Test
using Base.Broadcast: materialize

pkgversion(ClimaCore) < v"0.14.20" && exit() # CommonSpaces
using ClimaCore.CommonSpaces

### Common Objects ###
@testset "Viscous-sponge functions" begin
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
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶠz = Fields.coordinate_field(ᶠspace).z
    zmax = maximum(ᶠz)
    ### Component test begins here
    s = CA.ViscousSponge{FT}(; zd = 0, κ₂ = 1)
    @test CA.β_viscous.(s, ᶜz, zmax) == @. ifelse(ᶜz > s.zd, s.κ₂, FT(0)) *
             sin(FT(π) / 2 * (ᶜz - s.zd) / (zmax - s.zd))^2
end
