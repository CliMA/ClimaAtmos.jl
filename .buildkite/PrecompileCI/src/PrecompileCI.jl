module PrecompileCI

using PrecompileTools, Logging
import ClimaAtmos as CA
import ClimaComms
import ClimaCore: InputOutput, Meshes, Spaces, Quadratures
import ClimaParams

@compile_workload begin
    with_logger(NullLogger()) do
        FT = Float32 # Float64?
        h_elem = 6 # 16, 30?
        z_elem = 10 # 30, 31, 63?
        x_elem = y_elem = 2
        x_max = y_max = 1e8
        z_max = FT(30000.0)
        dz_bottom = FT(500) # other values?
        z_stretch = true # Meshes.Uniform()
        bubble = true # false
        nh_poly = 3 # GLL{4} = nh_poly + 1
        context = ClimaComms.context(ClimaComms.CPUSingleThreaded())
        deep_atmosphere = false
        topography = CA.NoTopography()
        params = CA.ClimaAtmosParameters(FT)
        radius = CA.Parameters.planet_radius(params)

        sphere_grid = CA.SphereGrid(
            FT;
            context, radius, h_elem, nh_poly, z_elem, z_max, z_stretch, 
            dz_bottom, bubble, deep_atmosphere, 
            topography,
        )
        box_grid = CA.BoxGrid(
            FT; context, x_elem, x_max, y_elem, y_max, z_elem, z_max, nh_poly, 
            z_stretch, dz_bottom, bubble, deep_atmosphere, 
            periodic_x = true, periodic_y = true, topography,
        )
        plane_grid = CA.PlaneGrid(
            FT; context, x_elem, x_max, z_elem, z_max, nh_poly, z_stretch, 
            dz_bottom, bubble, deep_atmosphere, periodic_x = true, 
            topography,
        )
        column_grid = CA.ColGrid(
            FT; context, z_elem, z_max, z_stretch, dz_bottom,
        )

        for grid in [sphere_grid, box_grid, plane_grid, column_grid]
            CA.get_spaces(grid, context)
        end
    end
end

end # module PrecompileCI
