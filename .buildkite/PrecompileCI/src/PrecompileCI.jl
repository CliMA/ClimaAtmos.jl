module PrecompileCI

using PrecompileTools
import ClimaAtmos as CA
import ClimaComms
import ClimaCore: InputOutput, Meshes, Spaces, Quadratures
import ClimaParams

@compile_workload begin
    FT = Float32 # Float64?
    h_elem = 6 # 16, 30?
    z_elem = 10 # 30, 31, 63?
    x_elem = y_elem = 2
    x_max = y_max = 1e8
    z_max = FT(30000.0)
    dz_bottom = FT(500) # other values?
    z_stretch = Meshes.HyperbolicTangentStretching(dz_bottom) # Meshes.Uniform()
    bubble = true # false
    parsed_args =
        Dict{String, Any}("topography" => "NoWarp", "topo_smoothing" => false)
    comms_ctx = ClimaComms.context()
    deep = false

    # constants
    quad = Quadratures.GLL{4}()
    params = CA.ClimaAtmosParameters(FT)
    radius = CA.Parameters.planet_radius(params)

    # Sphere
    horz_mesh = CA.cubed_sphere_mesh(; radius, h_elem)
    h_space = CA.make_horizontal_space(horz_mesh, quad, comms_ctx, bubble)
    CA.make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; parsed_args)

    # box
    horizontal_mesh = CA.periodic_rectangle_mesh(; x_max, y_max, x_elem, y_elem)
    h_space = CA.make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
    # This is broken
    # CA.make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; parsed_args)

    # plane
    horizontal_mesh = CA.periodic_line_mesh(; x_max, x_elem)
    h_space = CA.make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
end


end # module Precompile
