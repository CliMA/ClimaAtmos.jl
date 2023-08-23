import ClimaComms
import ClimaCore: Domains, Geometry, InputOutput, Meshes, Topologies, Spaces

"""
    AtmosDomain

Computational domain over which the simulation is run. Thin wrapper around a
`ClimaCore.Space` for the cell centers. Convenience functions are provided to access data
such `face_space` (`Space` for the cell faces), or `comms_ctx` (for the context of the
computing device).

"""
struct AtmosDomain{T1, T2}
    center_space::T1

    # TODO: geo_type is here only for compatibility with the non-orographic gravity wave
    # interface. We probably want to remove it.
    geo_type::T2
end

Base.eltype(domain::AtmosDomain) = Spaces.undertype(domain.center_space)

# The face_space of a given center_space is just a different view, so let's provide a getter
# that returns that on the fly
function Base.getproperty(domain::AtmosDomain, v::Symbol)
    if v == :face_space
        return Spaces.FaceExtrudedFiniteDifferenceSpace(domain.center_space)
    elseif v == :comms_ctx
        return ClimaComms.context(domain.center_space)
    elseif v == :float_type
        return Spaces.undertype(domain.center_space)
    else
        return getfield(domain, v)
    end
end

"""
function ExponentiallyStretchedColumn(; z_elem,
                                        dz_bottom,
                                        dz_top,
                                        z_max,
                                        comms_ctx=ClimaComms.context(),
                                        float_type=Float64)

Construct a `AtmosDomain` for a column with varying resolution according to
`ClimaCore.GeneralizedExponentialStretching`.

Keyword arguments
=================

- `z_elem`: Number of spectral elements.

- `dz_bottom`: Resolution at the lower end of the column (in meters).

- `dz_top`: Resolution at the top end of the column (in meters).

- `z_max`: Height of the column (in meters).

- `comms_ctx`: Context of the computational environment where the simulation should be run,
               as defined in ClimaComms. By default, the CLIMACOMMS_DEVICE environment
               variable is read for one of "CPU", "CPUSingleThreaded", "CPUMultiThreaded",
               "CUDA". If none is found, the fallback is to use a GPU (if available), or a
               single threaded CPU (if not).

- `float_type`: Floating point type. Typically, Float32 or Float64 (default).

"""
function ExponentiallyStretchedColumn(;
    z_elem,
    z_max,
    dz_bottom,
    dz_top,
    comms_ctx = ClimaComms.context(),
    float_type = Float64,
)

    FT = float_type

    isa(comms_ctx, ClimaComms.SingletonCommsContext) ||
        error("columns cannot be run with MPI")

    # Vertical
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zero(FT)),
        Geometry.ZPoint{FT}(FT(z_max));
        boundary_tags = (:bottom, :top),
    )
    stretch = Meshes.GeneralizedExponentialStretching(FT(dz_bottom), FT(dz_top))
    z_mesh = Meshes.IntervalMesh(z_domain, stretch; nelems = z_elem)
    z_topology = Topologies.IntervalTopology(comms_ctx, z_mesh)
    z_space = Spaces.CenterFiniteDifferenceSpace(z_topology)

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
    h_topology = Topologies.DistributedTopology2D(
        comms_ctx,
        h_mesh,
        Topologies.spacefillingcurve(h_mesh),
    )
    h_space = Spaces.SpectralElementSpace2D(h_topology, h_quadrature;)

    return AtmosDomain(
        Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space),
        SingleColumnModel(),
    )
end
