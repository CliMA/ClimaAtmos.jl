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

        # Full atmospheric vertical grid uses BoundaryLayerHyperbolicTangentStretching
        if isdefined(CC.Meshes, :BoundaryLayerHyperbolicTangentStretching)
            atmos_grid = CA.SphereGrid(
                Float64;
                h_elem = 3,
                z_elem = 63,
                z_max = 60000.0,
                dz_bottom = 30.0,
            )
            z_stretch_rule = atmos_grid.vertical_grid.topology.mesh.stretch
            @test z_stretch_rule isa
                  CC.Meshes.BoundaryLayerHyperbolicTangentStretching{Float64}
            @test z_stretch_rule.dz_surface == 30.0
            @test z_stretch_rule.dz_top == 3000.0
            @test z_stretch_rule.z_pbl == 2200.0

            atmos_grid_z79 = CA.SphereGrid(
                Float64;
                h_elem = 3,
                z_elem = 79,
                z_max = 60000.0,
                dz_bottom = 28.0,
            )
            z_stretch_z79 = atmos_grid_z79.vertical_grid.topology.mesh.stretch
            @test z_stretch_z79 isa
                  CC.Meshes.BoundaryLayerHyperbolicTangentStretching{Float64}
            @test z_stretch_z79.dz_surface == 28.0
        end
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
        # `Nh` is the last dimension of the parent array; the number of leading
        # node/field dimensions depends on the ClimaCore data layout, so take
        # the first index of each instead of hard-coding their count.
        first_node(p) = p[ntuple(_ -> 1, ndims(p) - 1)..., :]
        coords_x = first_node(parent(grid.horizontal_grid.local_geometry.coordinates.x))
        coords_y = first_node(parent(grid.horizontal_grid.local_geometry.coordinates.y))
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
