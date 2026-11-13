module PrecompileCI

using PrecompileTools, Logging
import ClimaAtmos as CA
import ClimaComms
import ClimaParams

@compile_workload begin
    with_logger(NullLogger()) do
        FT = Float32 # Float64?
        h_elem = 6 # 16, 30?
        # The 63-level grid up to 60 km with 30 m bottom spacing lies in the
        # range the two-scale stretching admits for the default `dz_top` and
        # `z_pbl`.
        z_elem = 63
        x_elem = y_elem = 2
        x_max = y_max = 1e8
        z_max = FT(60000.0)
        dz_bottom = FT(30)
        z_stretch = true
        bubble = true
        nh_poly = 3 # GLL{4} = nh_poly + 1
        # TODO: compile CUDA methods as well
        # The singleton context is constructed directly: `ClimaComms.context`
        # reads `CLIMACOMMS_CONTEXT`, which the MPI jobs set, and the
        # precompile process does not load MPI.jl.
        context =
            ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
        topography = CA.NoTopography()
        params = CA.ClimaAtmosParameters(FT)
        radius = CA.Parameters.planet_radius(params)

        sphere_grid = CA.SphereGrid(
            FT;
            context,
            radius, h_elem, nh_poly,
            z_elem, z_max, z_stretch, dz_bottom,
            bubble, topography,
        )
        box_grid = CA.BoxGrid(
            FT;
            context,
            x_elem, x_max, y_elem, y_max, nh_poly, periodic_x = true, periodic_y = true,
            z_elem, z_max, z_stretch, dz_bottom,
            bubble, topography,
        )
        plane_grid = CA.PlaneGrid(
            FT;
            context,
            x_elem, x_max, nh_poly, periodic_x = true,
            z_elem, z_max, z_stretch, dz_bottom,
            topography,
        )
        column_grid = CA.ColumnGrid(
            FT; context, z_elem, z_max, z_stretch, dz_bottom,
        )
        all_grids = (sphere_grid, box_grid, plane_grid, column_grid)
        foreach(CA.get_spaces, all_grids)
    end
end

end # module PrecompileCI
