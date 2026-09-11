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

    @testset "MultiColumnGrid" begin
        points = [
            CC.Geometry.LatLongPoint(0.0, 0.0),
            CC.Geometry.LatLongPoint(30.0, 90.0),
            CC.Geometry.LatLongPoint(-45.0, -120.0),
        ]
        radius = 6.371229e6
        grid = CA.MultiColumnGrid(Float64; points, radius, z_elem = 5)
        (; center_space) = CA.get_spaces(grid)
        @test CC.Spaces.ncolumns(center_space) == length(points)

        coords = CC.Fields.coordinate_field(center_space)
        for (h, point) in enumerate(points)
            column_coords = CC.Fields.column(coords, 1, 1, h)
            @test all(==(point.lat), parent(column_coords.lat))
            @test all(==(point.long), parent(column_coords.long))
        end

        for (deep_atmosphere, GlobalGeometry) in (
            (false, CC.Geometry.ShallowSphericalGlobalGeometry),
            (true, CC.Geometry.DeepSphericalGlobalGeometry),
        )
            geometry_grid =
                CA.MultiColumnGrid(Float64; points, radius, z_elem = 5, deep_atmosphere)
            @test CC.Grids.global_geometry(geometry_grid) isa GlobalGeometry
        end
    end
end
