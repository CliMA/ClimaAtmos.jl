import ClimaCore: Fields, Hypsography

##########
# Column #
##########
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
    make_column(;
                z_elem,
                z_max,
                z_stretch::Meshes.StretchingRule,
                comms_ctx::ClimaComms.AbstractCommsContext,
                float_type
                )

Return center and face space for a column.
"""
function make_column(
    z_elem,
    z_max,
    z_stretch::Meshes.StretchingRule,
    comms_ctx,
    float_type,
)
    isa(comms_ctx, ClimaComms.SingletonCommsContext) ||
        error("ColumnGrids are incompatible with MPI")

    # Vertical space
    z_space =
        make_vertical_space(; z_elem, z_max, z_stretch, comms_ctx, float_type)

    # Horizontal space
    h_space = make_trivial_horizontal_space(; comms_ctx, float_type)

    # 3D space
    center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return ColumnGrid(; center_space, face_space, z_elem, z_max, z_stretch)
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
    # Method for column/box/sphere

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

"""
    make_hybrid_space(; h_space,
                        z_max,
                        z_elem,
                        z_stretch::Meshes.StretchingRule,
                        comms_ctx::ClimaComms.AbstractCommsContext,
                        float_type;
                        topography,
                        topo_smoothing)

Return the center and face spaces as constructed from the given horizontal space.
"""
function make_hybrid_spaces(
    h_space,
    z_max,
    z_elem,
    z_stretch::Meshes.StretchingRule,
    comms_ctx,
    float_type;
    topography = nothing,
    topo_smoothing = false,
)
    z_max = float_type(z_max)

    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_tags = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, z_stretch; nelems = z_elem)
    if isnothing(topography)
        z_topology = Topologies.IntervalTopology(comms_ctx, z_mesh)
        z_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
        center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
        face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    else
        z_surface = topography(Fields.coordinate_field(h_space))
        if topo_smoothing
            Hypsography.diffuse_surface_elevation!(z_surface)
        end
        z_face_space = Spaces.FaceFiniteDifferenceSpace(z_mesh)
        face_space = Spaces.ExtrudedFiniteDifferenceSpace(
            h_space,
            z_face_space,
            Hypsography.LinearAdaption(z_surface),
        )
        center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(face_space)
    end
    return center_space, face_space
end

#######
# Box #
#######

"""
function make_box_horizontal_space(;
                                    nh_poly,
                                    x_elem,
                                    x_max,
                                    y_elem,
                                    y_max,
                                    enable_bubble,
                                    comms_ctx,
                                    float_type,
                                   )

Return the horizontal space for a box configuration.
"""
function make_box_horizontal_space(;
    nh_poly,
    x_elem,
    x_max,
    y_elem,
    y_max,
    enable_bubble,
    comms_ctx,
    float_type,
)
    x_max, y_max = map(float_type, (x_max, y_max))

    x_domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(x_max)),
        Geometry.XPoint(x_max);
        periodic = true,
    )
    y_domain = Domains.IntervalDomain(
        Geometry.YPoint(zero(y_max)),
        Geometry.YPoint(y_max);
        periodic = true,
    )
    h_domain = Domains.RectangleDomain(x_domain, y_domain)
    h_mesh = Meshes.RectilinearMesh(h_domain, x_elem, y_elem)
    h_space = make_horizontal_space(; nh_poly, h_mesh, comms_ctx, enable_bubble)

    return h_space
end


function make_box(;
    nh_poly,
    x_elem,
    x_max,
    y_elem,
    y_max,
    z_elem,
    z_max,
    z_stretch,
    enable_bubble,
    topography,
    topo_smoothing,
    comms_ctx,
    float_type,
)

    h_space = make_box_horizontal_space(;
        nh_poly,
        x_elem,
        x_max,
        y_elem,
        y_max,
        enable_bubble,
        comms_ctx,
        float_type,
    )

    center_space, face_space = make_hybrid_spaces(
        h_space,
        z_max,
        z_elem,
        z_stretch,
        comms_ctx,
        float_type;
        topography,
        topo_smoothing,
    )

    return BoxGrid(;
        center_space,
        face_space,
        nh_poly,
        x_elem,
        x_max,
        y_elem,
        y_max,
        z_elem,
        z_max,
        z_stretch,
        enable_bubble,
        topography,
    )
end

##########
# Sphere #
##########

function make_sphere_horizontal_space(;
    nh_poly,
    h_elem,
    radius,
    enable_bubble,
    comms_ctx,
    float_type,
)
    radius = float_type(radius)

    h_domain = Domains.SphereDomain(radius)
    h_mesh = Meshes.EquiangularCubedSphere(h_domain, h_elem)
    h_space = make_horizontal_space(; nh_poly, h_mesh, comms_ctx, enable_bubble)
    return h_space
end

function make_sphere(;
    h_elem,
    radius,
    nh_poly,
    z_elem,
    z_max,
    z_stretch,
    topography,
    topo_smoothing,
    enable_bubble,
    comms_ctx,
    float_type,
)

    h_space = make_sphere_horizontal_space(;
        h_elem,
        radius,
        nh_poly,
        enable_bubble,
        comms_ctx,
        float_type,
    )

    center_space, face_space = make_hybrid_spaces(
        h_space,
        z_max,
        z_elem,
        z_stretch,
        comms_ctx,
        float_type;
        topography,
        topo_smoothing,
    )
    return SphereGrid(;
        center_space,
        face_space,
        nh_poly,
        h_elem,
        radius,
        z_elem,
        z_max,
        z_stretch,
        enable_bubble,
        topography,
    )
end

#########
# Plane #
#########

function make_plane_horizontal_space(; nh_poly, x_max, x_elem, comms_ctx)
    h_domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(x_max)),
        Geometry.XPoint(x_max);
        periodic = true,
    )
    h_mesh = Meshes.IntervalMesh(h_domain; nelems = x_elem)

    h_quadrature = Spaces.Quadratures.GLL{nh_poly + 1}()
    h_topology = Topologies.IntervalTopology(comms_ctx, h_mesh)

    return Spaces.SpectralElementSpace1D(h_topology, h_quadrature;)
end
