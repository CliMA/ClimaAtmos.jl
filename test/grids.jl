using Test
import ClimaAtmos as CA
import ClimaCore as CC

@testset "Grid generation" begin
    @testset "SphereGrid" begin
        # Test sphere grid space-filling curve
        grid = CA.SphereGrid(Float64, h_elem = 3)
        mesh = grid.horizontal_grid.topology.mesh
        sfc_elemorder = CC.Topologies.spacefillingcurve(mesh)

        # ClimaCore.jl/test/Topologies/cubedsphere_sfc.jl indices test for spacefillingcurve
        sfc_orderindex = CC.Meshes.linearindices(sfc_elemorder)
        for (order, cartindex) in enumerate(sfc_elemorder)
            @test sfc_orderindex[cartindex] == order
        end

        # The spacefillingcurve order is not the same as the linear order
        linearorder = vec(collect(CC.Meshes.elements(mesh)))
        @test linearorder != sfc_elemorder

    end

    @testset "BoxGrid" begin
        # use nh_poly=1 for clearer checks of element ordering (1 node per element)
        # Set x_max and y_max to x_elem and y_elem so coordinates align with indices
        x_elem = x_max = 3
        y_elem = y_max = 6
        grid = CA.BoxGrid(Float64; x_elem, y_elem, x_max, y_max, nh_poly = 1)

        # Extract the space-filling curve from the topology's mesh
        mesh = grid.horizontal_grid.topology.mesh
        spacefilling = CC.Topologies.spacefillingcurve(mesh)

        # We can now verify the coordinates directly (add 1 to account for 1-based indexing)
        coords_x = parent(grid.horizontal_grid.local_geometry.coordinates.x)[1, 1, 1, :]
        coords_y = parent(grid.horizontal_grid.local_geometry.coordinates.y)[1, 1, 1, :]
        coords = @. CartesianIndex(Int(coords_x + 1), Int(coords_y + 1))

        @test coords == spacefilling

        # ClimaCore.jl/test/Topologies/rectangle_sfc.jl indices test for spacefillingcurve
        sfc_orderindex = CC.Meshes.linearindices(spacefilling)
        for (order, cartindex) in enumerate(spacefilling)
            @test sfc_orderindex[cartindex] == order
        end

        # The spacefillingcurve order is not the same as the linear order
        linearorder = vec(collect(CC.Meshes.elements(mesh)))
        @test linearorder != spacefilling
    end
end
