using ClimaCore:
    Geometry, Domains, Meshes, Topologies, Spaces, Grids, Hypsography, Fields
using ClimaComms
using ClimaUtilities: SpaceVaryingInputs.SpaceVaryingInput

function periodic_line_mesh(; x_max, x_elem)
    domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(x_max)),
        Geometry.XPoint(x_max);
        periodic = true,
    )
    return Meshes.IntervalMesh(domain; nelems = x_elem)
end

function periodic_rectangle_mesh(; x_max, y_max, x_elem, y_elem)
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
    domain = Domains.RectangleDomain(x_domain, y_domain)
    return Meshes.RectilinearMesh(domain, x_elem, y_elem)
end

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
    if mesh isa Meshes.AbstractMesh1D
        topology = Topologies.IntervalTopology(comms_ctx, mesh)
        space = Spaces.SpectralElementSpace1D(topology, quad)
    elseif mesh isa Meshes.AbstractMesh2D
        topology = Topologies.Topology2D(
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

function make_hybrid_spaces(
    h_space,
    z_max,
    z_elem,
    z_stretch;
    deep = false,
    parsed_args = nothing,
)
    FT = eltype(z_max)
    h_grid = Spaces.grid(h_space)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, z_stretch; nelems = z_elem)
    @info "z heights" z_mesh.faces
    device = ClimaComms.device(h_space)
    z_topology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        z_mesh,
    )
    z_grid = Grids.FiniteDifferenceGrid(z_topology)

    topography = parsed_args["topography"]
    @assert topography in ("NoWarp", "DCMIP200", "Earth", "Agnesi", "Schar")
    if topography == "DCMIP200"
        z_surface = SpaceVaryingInput(topography_dcmip200, h_space)
        @info "Computing DCMIP200 orography on spectral horizontal space"
    elseif topography == "Agnesi"
        z_surface = SpaceVaryingInput(topography_agnesi, h_space)
        @info "Computing Agnesi orography on spectral horizontal space"
    elseif topography == "Schar"
        z_surface = SpaceVaryingInput(topography_schar, h_space)
        @info "Computing Schar orography on spectral horizontal space"
    elseif topography == "NoWarp"
        z_surface = zeros(h_space)
        @info "No surface orography warp applied"
    elseif topography == "Earth"
        z_surface = SpaceVaryingInput(
            AA.earth_orography_file_path(;
                context = ClimaComms.context(h_space),
            ),
            "z",
            h_space,
        )
        @info "Remapping Earth orography from ETOPO2022 data onto horizontal space"
    end

    topo_smoothing = parsed_args["topo_smoothing"]
    if topography == "NoWarp"
        hypsography = Hypsography.Flat()
    elseif topography == "Earth"
        mask(x::FT) where {FT} = x * FT(x > 0)
        z_surface = @. mask(z_surface)
        # Coefficient for horizontal diffusion of surface orography
        # is obtained from the V1, V2 Topography documentation from 
        # E3SM
        # https://acme-climate.atlassian.net/wiki/spaces/DOC/pages/1456603764/V1+Topography+GLL+grids
        nelem = Meshes.n_elements_per_panel_direction(axes(z_surface))
        Hypsography.diffuse_surface_elevation!(
            z_surface;
            κ = FT(28e7 * (30 / nelem)^2),
            dt = FT(1),
            maxiter = 128,
        )
        z_surface = @. mask(z_surface)
        if parsed_args["mesh_warp_type"] == "SLEVE"
            @info "SLEVE mesh warp"
            hypsography = Hypsography.SLEVEAdaption(
                Geometry.ZPoint.(z_surface),
                FT(parsed_args["sleve_eta"]),
                FT(parsed_args["sleve_s"]),
            )
        elseif parsed_args["mesh_warp_type"] == "Linear"
            @info "Linear mesh warp"
            hypsography =
                Hypsography.LinearAdaption(Geometry.ZPoint.(z_surface))
        else
            @error "Undefined mesh-warping option"
        end
    else
        if topo_smoothing
            Hypsography.diffuse_surface_elevation!(z_surface)
        end
        if parsed_args["mesh_warp_type"] == "SLEVE"
            @info "SLEVE mesh warp"
            hypsography = Hypsography.SLEVEAdaption(
                Geometry.ZPoint.(z_surface),
                FT(parsed_args["sleve_eta"]),
                FT(parsed_args["sleve_s"]),
            )
        elseif parsed_args["mesh_warp_type"] == "Linear"
            @info "Linear mesh warp"
            hypsography =
                Hypsography.LinearAdaption(Geometry.ZPoint.(z_surface))
        end
    end

    grid = Grids.ExtrudedFiniteDifferenceGrid(h_grid, z_grid, hypsography; deep)
    center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)
    return center_space, face_space
end
