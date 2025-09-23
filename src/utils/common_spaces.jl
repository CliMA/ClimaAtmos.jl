using ClimaCore:
    Geometry, Domains, Meshes, Topologies, Spaces, Grids, Hypsography, Fields
using ClimaComms
using ClimaUtilities: SpaceVaryingInputs.SpaceVaryingInput

# h_elem is the number of elements per side of every panel (6 panels in total)
function cubed_sphere_mesh(; radius, h_elem)
    domain = Domains.SphereDomain(radius)
    return Meshes.EquiangularCubedSphere(domain, h_elem)
end

function make_horizontal_space(
    mesh,
    quad,
    comms_ctx::ClimaComms.SingletonCommsContext,
    bubble,
)

    space = if mesh isa Meshes.AbstractMesh1D
        topology = Topologies.IntervalTopology(comms_ctx, mesh)
        Spaces.SpectralElementSpace1D(topology, quad)
    elseif mesh isa Meshes.AbstractMesh2D
        topology = Topologies.Topology2D(
            comms_ctx,
            mesh,
            Topologies.spacefillingcurve(mesh),
        )
        Spaces.SpectralElementSpace2D(topology, quad; enable_bubble = bubble)
    end
    return space
end

function make_horizontal_space(mesh, quad, comms_ctx, bubble)
    if mesh isa Meshes.AbstractMesh1D
        error("Distributed mode does not work with 1D horizontal spaces.")
    elseif mesh isa Meshes.AbstractMesh2D
        topology = Topologies.DistributedTopology2D(
            comms_ctx,
            mesh,
            Topologies.spacefillingcurve(mesh),
        )
        space = Spaces.SpectralElementSpace2D(
            topology,
            quad;
            enable_bubble = bubble,
        )
    end
    return space
end
