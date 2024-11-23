#=
julia --project=examples
using Revise; include("test/parameterized_tendencies/sponge/viscous_sponge.jl")
=#
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore
using ClimaCore: Spaces, Grids, Fields
if pkgversion(ClimaCore) ≥ v"0.14.18"
    using ClimaCore.CommonGrids
    using Test

    ### Common Objects ###
    @testset "Viscous-sponge functions" begin
        grid = ExtrudedCubedSphereGrid(;
            z_elem = 10,
            z_min = 0,
            z_max = 1,
            radius = 10,
            h_elem = 10,
            n_quad_points = 4,
        )
        cspace = Spaces.ExtrudedFiniteDifferenceSpace(grid, Grids.CellCenter())
        fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
        z = Fields.coordinate_field(cspace).z
        zmax = maximum(Fields.coordinate_field(fspace).z)
        FT = typeof(zmax)
        ### Component test begins here
        s = CA.ViscousSponge{FT}(; zd = 0, κ₂ = 1)
        @test CA.β_viscous.(s, z, zmax) == @. ifelse(z > s.zd, s.κ₂, FT(0)) *
                 sin(FT(π) / 2 * (z - s.zd) / (zmax - s.zd))^2
    end
end
