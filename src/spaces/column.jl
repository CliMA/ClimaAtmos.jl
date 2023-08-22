export ColumnSpace

function ColumnSpace(;
    device = ClimaComms.device(),
    float_type = Float64,
    vert_elem = 63,
    z_max = 30e3,
    dz_bottom = 0.5e3,
    dz_top = 5e3,
    )
    FT = float_type
    vert_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(FT)),
        Geometry.ZPoint(FT(z_max));
        boundary_tags = (:bottom, :top),
    )
    stretch = Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
    vert_mesh = Meshes.IntervalMesh(vert_domain, stretch; nelems = z_elem)
    vert_topology = Topologies.IntervalTopology(ClimaComms.SingletonCommsContext(device), vert_mesh)
    vert_space = Spaces.CenterFiniteDifferenceSpace(vert_topology)
    return vert_space
end
