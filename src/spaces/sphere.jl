export CubedSphereSpace

function CubedSphereSpace(;
    context = ClimaComms.context(),
    float_type = Float64,
    radius = CAP.planet_radius(current_params()),
    horz_panel_elem = 8,
    horz_poly_order = 3,
    bubble_correction = true,
    vert_elem = 63,
    z_max = 30e3,
    dz_bottom = 0.5e3,
    dz_top = 5e3,
    topography = nothing,
)
    FT = float_type

    horz_domain = ClimaComms.Domains.SphereDomain(FT(radius))
    horz_mesh = ClimaComms.Meshes.EquiangularCubedSphere(horz_domain, horz_panel_elem)
    horz_topology = ClimaComms.Topology2D(context, horz_mesh)
    horz_quad = Spaces.Quadratures.GLL{horz_poly_order + 1}()
    horz_space = ClimaComms.Spaces.SpectralElementSpace2D(horz_topology, horz_quad;
    enable_bubble = bubble_correction)

    vert_space = ColumnSpace(;
        device = ClimaComms.device(context),
        float_type,
        vert_elem,
        z_max,
        dz_bottom,
        dz_top,
    )

    space = ClimaComms.Spaces.ExtrudedFiniteDifferenceSpace(horz_space, vert_space)
    return space
end
