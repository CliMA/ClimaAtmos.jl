"""
    make_vertical_space(;
                         z_elem,
                         z_max,
                         z_stretch::Meshes.StretchingRule,
                         comms_ctx::ClimaComms.AbstractCommsContext,
                         float_type
                         )

Return a vertical `Spaces.CenterFiniteDifferenceSpace` with `z_elem` and height `z_max` (in
meters) with resolution defined by `z_stretch`.
"""
function make_vertical_space(;
    z_elem,
    z_max,
    z_stretch::Meshes.StretchingRule,
    comms_ctx::ClimaComms.AbstractCommsContext,
    float_type,
)

    # Promote types
    z_max = float_type(z_max)

    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(float_type)),
        Geometry.ZPoint(z_max);
        boundary_tags = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, z_stretch; nelems = z_elem)
    z_topology = Topologies.IntervalTopology(comms_ctx, z_mesh)
    return Spaces.CenterFiniteDifferenceSpace(z_topology)
end


"""
    make_trivial_horizontal_space(;
                                   comms_ctx::ClimaComms.AbstractCommsContext,
                                   float_type
                                   )

Return a one-point wide 2D `Spaces.SpectralElementSpace2D`. This is useful for building columns.
"""
function make_trivial_horizontal_space(; comms_ctx, float_type)
    FT = float_type

    # 1-point wide horizontal domain
    x_elem = y_elem = 1
    x_domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(FT)),
        Geometry.XPoint(one(FT));
        periodic = true,
    )
    y_domain = Domains.IntervalDomain(
        Geometry.YPoint(zero(FT)),
        Geometry.YPoint(one(FT));
        periodic = true,
    )
    h_domain = Domains.RectangleDomain(x_domain, y_domain)
    h_mesh = Meshes.RectilinearMesh(h_domain, x_elem, y_elem)
    h_quadrature = Spaces.Quadratures.GL{1}()
    h_topology = Topologies.Topology2D(
        comms_ctx,
        h_mesh,
        Topologies.spacefillingcurve(h_mesh),
    )
    return Spaces.SpectralElementSpace2D(h_topology, h_quadrature;)
end


"""
    make_horizontal_space(; nh_poly,
                            h_mesh,
                            comms_ctx::ClimaComms.AbstractCommsContext,
                            float_type,
                            enable_bubble,
                            )

Return a 2D `Spaces.SpectralElementSpace2D` built from the given horizontal `h_mesh`.
"""
function make_horizontal_space(;
    nh_poly,
    h_mesh,
    comms_ctx,
    enable_bubble = false,
)
    h_quadrature = Spaces.Quadratures.GLL{nh_poly + 1}()

    # We have to pick different topologies depending if we are running on a single process or not.
    make_topology =
        comms_ctx isa ClimaComms.SingletonCommsContext ? Topologies.Topology2D :
        Topologies.DistributedTopology2D

    h_topology =
        make_topology(comms_ctx, h_mesh, Topologies.spacefillingcurve(h_mesh))
    return Spaces.SpectralElementSpace2D(
        h_topology,
        h_quadrature;
        enable_bubble = enable_bubble,
    )
end
