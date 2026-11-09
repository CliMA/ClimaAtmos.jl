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

    @testset "Vertical stretching" begin
        stretching(grid) = CC.Grids.vertical_topology(grid).mesh.stretch

        uniform_grid = CA.ColumnGrid(Float64)
        @test stretching(uniform_grid) isa CC.Meshes.Uniform

        fine_grid = CA.ColumnGrid(
            Float64;
            z_elem = 63,
            z_max = 60000.0,
            z_stretch = true,
            dz_bottom = 30.0,
        )
        @test stretching(fine_grid) isa
              CC.Meshes.BoundaryLayerHyperbolicTangentStretching{Float64}
        @test stretching(fine_grid).dz_surface == 30.0
        @test stretching(fine_grid).dz_top == 3000.0
        @test stretching(fine_grid).z_pbl == 2200.0

        fine_sphere = CA.SphereGrid(
            Float64;
            h_elem = 3,
            z_elem = 79,
            z_max = 60000.0,
            z_stretch = true,
            dz_bottom = 28.0,
            dz_top = 2500.0,
            z_pbl = 2000.0,
        )
        @test stretching(fine_sphere).dz_top == 2500.0
        @test stretching(fine_sphere).z_pbl == 2000.0

        # The two-scale rule rejects a `z_elem` outside its admissible range:
        # the 10-level default grid lies below it, and 200 levels with
        # `dz_bottom = 30` m up to 45 km lie above it.
        @test_throws ArgumentError CA.ColumnGrid(Float64; z_stretch = true)
        @test_throws ArgumentError CA.ColumnGrid(
            Float64;
            z_elem = 200,
            z_max = 45000.0,
            z_stretch = true,
            dz_bottom = 30.0,
        )
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
        n_columns = 3
        grid = CA.MultiColumnGrid(Float64; n_columns, z_elem = 5)
        (; center_space) = CA.get_spaces(grid)
        @test center_space isa CC.Spaces.MultiColumnFiniteDifferenceSpace
        @test CC.Spaces.ncolumns(center_space) == n_columns

        coords = CC.Fields.coordinate_field(center_space)
        # We assign each point the coordinates of zero for both latitude and
        # longitude
        @test all(iszero, parent(coords.lat))
        @test all(iszero, parent(coords.long))
        column_space = CA.get_spaces(CA.ColumnGrid(Float64; z_elem = 5)).center_space
        @test vec(parent(CC.Fields.column(coords, 1, 1, n_columns).z)) ==
              vec(parent(CC.Fields.coordinate_field(column_space).z))

        @test_throws ErrorException CA.MultiColumnGrid(Float64; n_columns = 0)
    end
end
