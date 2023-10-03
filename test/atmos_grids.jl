using Test
import ClimaComms
import ClimaCore: Meshes
import ClimaAtmos as CA

# Here, we copy and paste the ClimaAtmos.get_spaces functions from commit fab2f3c to ensure
# that we produce the same spaces. (The function is modified to take planet_radius directly,
# otherwise it would be a massive pain)
# include("get_spaces.jl")

@testset "UniformColumnGrid" begin

    # Check that we can build some grids
    for FT in (Float64, Float32)
        z_elem = 10
        z_max = 30000

        column = CA.Grid.UniformColumnGrid(; z_elem, z_max, float_type = FT)
        @test column.z_max == convert(FT, z_max)
        @test column.z_stretch == Meshes.Uniform()
        @test CA.Grid.float_type(column) == FT
        @test CA.Grid.context(column) == ClimaComms.context()

        # The traditional interface takes arguments even if they are not needed
        parsed_args = Dict(
            "z_elem" => z_elem,
            "z_max" => z_max,
            "config" => "column",
            # Not used
            "h_elem" => NaN,
            "z_stretch" => false,
            "dz_bottom" => NaN,
            "dz_top" => NaN,
            "topography" => "NoWarp",
            "bubble" => false,
        )
        planet_radius = convert(FT, 1.0)

        traditional_spaces =
            get_spaces(parsed_args, planet_radius, ClimaComms.context())

        # We cannot compare directly the spaces because they are complex objects made with
        # mutable components, so we compare the relevant bits

        @test typeof(column.center_space) ==
              typeof(traditional_spaces.center_space)
        @test typeof(column.face_space) == typeof(traditional_spaces.face_space)

        @test parent(column.center_space.center_local_geometry) ==
              parent(traditional_spaces.center_space.center_local_geometry)
        @test parent(column.face_space.center_local_geometry) ==
              parent(traditional_spaces.face_space.center_local_geometry)

    end
end

@testset "StretchedColumnGrid" begin

    # Check that we can build some grids
    for FT in (Float64, Float32)
        dz_bottom = 500
        dz_top = 5000
        z_elem = 10
        z_max = 30000

        column = CA.Grid.StretchedColumnGrid(;
            z_elem,
            z_max,
            dz_bottom,
            dz_top,
            float_type = FT,
        )
        @test column.z_max == convert(FT, z_max)
        @test column.z_stretch ==
              Meshes.GeneralizedExponentialStretching(FT(dz_bottom), FT(dz_top))
        @test CA.Grid.float_type(column) == FT
        @test CA.Grid.context(column) == ClimaComms.context()

        # The traditional interface takes arguments even if they are not needed
        parsed_args = Dict(
            "z_elem" => z_elem,
            "z_max" => z_max,
            "dz_bottom" => dz_bottom,
            "dz_top" => dz_top,
            "z_stretch" => true,
            "config" => "column",
            # Not used
            "h_elem" => NaN,
            "topography" => "NoWarp",
            "bubble" => false,
        )
        planet_radius = convert(FT, 1.0)

        traditional_spaces =
            get_spaces(parsed_args, planet_radius, ClimaComms.context())

        # We cannot compare directly the spaces because they are complex objects made with
        # mutable components, so we compare the relevant bits

        @test typeof(column.center_space) ==
              typeof(traditional_spaces.center_space)
        @test typeof(column.face_space) == typeof(traditional_spaces.face_space)

        @test parent(column.center_space.center_local_geometry) ==
              parent(traditional_spaces.center_space.center_local_geometry)
        @test parent(column.face_space.center_local_geometry) ==
              parent(traditional_spaces.face_space.center_local_geometry)
    end
end

@testset "VerticallyStretchedBoxGrid" begin

    # Check that we can build some grids
    for FT in (Float64, Float32), enable_bubble in (true, false)
        nh_poly = 3
        dz_bottom = 500
        dz_top = 5000
        x_elem = 6
        y_elem = 6
        x_max = 300000
        y_max = 300000
        z_elem = 10
        z_max = 30000

        box = CA.Grid.Box(;
            nh_poly,
            x_elem,
            y_elem,
            z_elem,
            x_max,
            y_max,
            z_max,
            dz_bottom,
            dz_top,
            float_type = FT,
            enable_bubble,
        )
        @test box.x_max == convert(FT, x_max)
        @test box.y_max == convert(FT, y_max)
        @test box.z_max == convert(FT, z_max)
        @test CA.Grid.float_type(box) == FT
        @test CA.Grid.context(box) == ClimaComms.context()
        @test box.z_stretch ==
              Meshes.GeneralizedExponentialStretching(FT(dz_bottom), FT(dz_top))

        # The traditional interface takes arguments even if they are not needed
        parsed_args = Dict(
            "nh_poly" => nh_poly,
            "x_elem" => x_elem,
            "x_max" => x_max,
            "y_elem" => y_elem,
            "y_max" => y_max,
            "z_elem" => z_elem,
            "z_max" => z_max,
            "dz_bottom" => dz_bottom,
            "dz_top" => dz_top,
            "z_stretch" => true,
            "config" => "box",
            "bubble" => enable_bubble,
            # Not used
            "h_elem" => NaN,
            "topography" => "NoWarp",
        )
        planet_radius = convert(FT, 1.0)

        traditional_spaces =
            get_spaces(parsed_args, planet_radius, ClimaComms.context())

        # We cannot compare directly the spaces because they are complex objects made with
        # mutable components, so we compare the relevant bits

        @test typeof(box.center_space) ==
              typeof(traditional_spaces.center_space)
        @test typeof(box.face_space) == typeof(traditional_spaces.face_space)

        @test parent(box.center_space.center_local_geometry) ==
              parent(traditional_spaces.center_space.center_local_geometry)
        @test parent(box.face_space.center_local_geometry) ==
              parent(traditional_spaces.face_space.center_local_geometry)
    end
end

@testset "VerticallyUniformBoxGrid" begin

    # Check that we can build some grids
    for FT in (Float64, Float32), enable_bubble in (true, false)
        nh_poly = 3
        x_elem = 6
        y_elem = 6
        x_max = 300000
        y_max = 300000
        z_elem = 10
        z_max = 30000

        box = CA.Grid.VerticallyUniformBoxGrid(;
            nh_poly,
            x_elem,
            y_elem,
            z_elem,
            x_max,
            y_max,
            z_max,
            float_type = FT,
            enable_bubble,
        )
        @test box.x_max == convert(FT, x_max)
        @test box.y_max == convert(FT, y_max)
        @test box.z_max == convert(FT, z_max)
        @test CA.Grid.float_type(box) == FT
        @test CA.Grid.context(box) == ClimaComms.context()
        @test box.z_stretch == Meshes.Uniform()

        # The traditional interface takes arguments even if they are not needed
        parsed_args = Dict(
            "nh_poly" => nh_poly,
            "x_elem" => x_elem,
            "x_max" => x_max,
            "y_elem" => y_elem,
            "y_max" => y_max,
            "z_elem" => z_elem,
            "z_max" => z_max,
            "z_stretch" => false,
            "config" => "box",
            "bubble" => enable_bubble,
            # Not used
            "dz_bottom" => NaN,
            "dz_top" => NaN,
            "h_elem" => NaN,
            "topography" => "NoWarp",
        )
        planet_radius = convert(FT, 1.0)

        traditional_spaces =
            get_spaces(parsed_args, planet_radius, ClimaComms.context())

        # We cannot compare directly the spaces because they are complex objects made with
        # mutable components, so we compare the relevant bits

        @test typeof(box.center_space) ==
              typeof(traditional_spaces.center_space)
        @test typeof(box.face_space) == typeof(traditional_spaces.face_space)

        @test parent(box.center_space.center_local_geometry) ==
              parent(traditional_spaces.center_space.center_local_geometry)
        @test parent(box.face_space.center_local_geometry) ==
              parent(traditional_spaces.face_space.center_local_geometry)
    end
end

@testset "VerticallyStretchedSphereGrid" begin

    # Check that we can build some grids
    for FT in (Float64, Float32), enable_bubble in (true, false)
        nh_poly = 3
        dz_bottom = 500
        dz_top = 5000
        h_elem = 6
        radius = 100
        z_elem = 10
        z_max = 30000

        sphere = CA.Grid.Sphere(;
            nh_poly,
            h_elem,
            z_elem,
            radius,
            z_max,
            dz_bottom,
            dz_top,
            float_type = FT,
            enable_bubble,
        )
        @test sphere.radius == convert(FT, radius)
        @test CA.Grid.float_type(sphere) == FT
        @test CA.Grid.context(sphere) == ClimaComms.context()
        @test sphere.z_stretch ==
              Meshes.GeneralizedExponentialStretching(FT(dz_bottom), FT(dz_top))

        # The traditional interface takes arguments even if they are not needed
        parsed_args = Dict(
            "nh_poly" => nh_poly,
            "radius" => radius,
            "h_elem" => h_elem,
            "z_elem" => z_elem,
            "z_max" => z_max,
            "dz_bottom" => dz_bottom,
            "dz_top" => dz_top,
            "z_stretch" => true,
            "config" => "sphere",
            "bubble" => enable_bubble,
            # Not used
            "topography" => "NoWarp",
        )
        planet_radius = convert(FT, radius)

        traditional_spaces =
            get_spaces(parsed_args, planet_radius, ClimaComms.context())

        # We cannot compare directly the spaces because they are complex objects made with
        # mutable components, so we compare the relevant bits

        @test typeof(sphere.center_space) ==
              typeof(traditional_spaces.center_space)
        @test typeof(sphere.face_space) == typeof(traditional_spaces.face_space)

        @test parent(sphere.center_space.center_local_geometry) ==
              parent(traditional_spaces.center_space.center_local_geometry)
        @test parent(sphere.face_space.center_local_geometry) ==
              parent(traditional_spaces.face_space.center_local_geometry)
    end
end

@testset "VerticallyUniformSphereGrid" begin

    # Check that we can build some grids
    for FT in (Float64, Float32), enable_bubble in (true, false)
        nh_poly = 3
        h_elem = 6
        y_elem = 6
        x_max = 300000
        y_max = 300000
        z_elem = 10
        z_max = 30000

        sphere = CA.Grid.VerticallyUniformSphereGrid(;
            nh_poly,
            h_elem,
            radius,
            z_elem,
            z_max,
            float_type = FT,
            enable_bubble,
        )
        @test sphere.radius == convert(FT, radius)
        @test CA.Grid.float_type(sphere) == FT
        @test CA.Grid.context(sphere) == ClimaComms.context()
        @test sphere.z_stretch == Meshes.Uniform()

        # The traditional interface takes arguments even if they are not needed
        parsed_args = Dict(
            "nh_poly" => nh_poly,
            "radius" => radius,
            "h_elem" => h_elem,
            "z_elem" => z_elem,
            "z_max" => z_max,
            "z_stretch" => false,
            "config" => "sphere",
            "bubble" => enable_bubble,
            # Not used
            "dz_bottom" => NaN,
            "dz_top" => NaN,
            "topography" => "NoWarp",
        )
        planet_radius = convert(FT, radius)

        traditional_spaces =
            get_spaces(parsed_args, planet_radius, ClimaComms.context())

        # We cannot compare directly the spaces because they are complex objects made with
        # mutable components, so we compare the relevant bits

        @test typeof(sphere.center_space) ==
              typeof(traditional_spaces.center_space)
        @test typeof(sphere.face_space) == typeof(traditional_spaces.face_space)

        @test parent(sphere.center_space.center_local_geometry) ==
              parent(traditional_spaces.center_space.center_local_geometry)
        @test parent(sphere.face_space.center_local_geometry) ==
              parent(traditional_spaces.face_space.center_local_geometry)
    end
end
