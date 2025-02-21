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
    @assert topography in (
        "NoWarp",
        "Earth",
        "DCMIP200",
        "Hughes2023",
        "Agnesi",
        "Schar",
        "Cosine2D",
        "Cosine3D",
    )
    if topography == "NoWarp"
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
    else
        topography_function = if topography == "DCMIP200"
            topography_dcmip200
        elseif topography == "Hughes2023"
            topography_hughes2023
        elseif topography == "Agnesi"
            topography_agnesi
        elseif topography == "Schar"
            topography_schar
        elseif topography == "Cosine2D"
            topography_cosine_2d
        elseif topography == "Cosine3D"
            topography_cosine_3d
        end
        z_surface = SpaceVaryingInput(topography_function, h_space)
        @info "Using $topography orography"
    end

    if topography == "NoWarp"
        hypsography = Hypsography.Flat()
    elseif topography == "Earth"
        mask(x::FT) where {FT} = x * FT(x > 0)
        z_surface = @. mask(z_surface)
        # diff_cfl = νΔt/Δx²
        diff_courant = 0.05 # Arbitrary example value.
        Δh_scale = Spaces.node_horizontal_length_scale(h_space)
        κ = FT(diff_courant * Δh_scale^2)
        n_attenuation = parsed_args["topography_damping_factor"]
        maxiter = Int(round(log(n_attenuation) / diff_courant))
        Hypsography.diffuse_surface_elevation!(
            z_surface;
            κ,
            dt = FT(1),
            maxiter,
        )
        # Coefficient for horizontal diffusion may alternatively be
        # determined from the empirical parameters suggested by
        # E3SM  v1/v2 Topography documentation found here: 
        # https://acme-climate.atlassian.net/wiki/spaces/DOC/pages/1456603764/V1+Topography+GLL+grids
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
        if parsed_args["topo_smoothing"]
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
