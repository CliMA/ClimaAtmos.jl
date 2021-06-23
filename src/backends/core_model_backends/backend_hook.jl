function create_grid(backend::CoreBackend)

    domain = Domains.RectangleDomain(
        -2π..2π,
        -2π..2π,
        x1periodic = true,
        x2periodic = boundary_name != "noslip",
    )

    n1, n2 = 16, 16 # number of elements in each direction
    Nq = 4          # polynomial order in each direction
    mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
    grid_topology = Topologies.GridTopology(mesh)
    quadrature = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quadrature)

    return space
end

function create_rhs(model::ModelSetup, backend::CoreBackend; grid = nothing)

    # Barotropic Fluid, bam!
    function flux(state, p)
        @unpack ρ, ρu, ρθ = state
        u = ρu ./ ρ
        return (ρ = ρu, ρu = ((ρu ⊗ u) + (p.g * ρ^2 / 2) * I), ρθ = ρθ .* u)
    end

    function rhs!(dydt, y, (parameters, numflux), t)

        Nh = Topologies.nlocalelems(y)

        F = flux.(y, Ref(parameters))
        dydt .= Operators.slab_weak_divergence(F)

        Operators.add_numerical_flux_internal!(numflux, dydt, y, parameters)

        Operators.add_numerical_flux_boundary!(
            dydt,
            y,
            parameters,
        ) do normal, (y⁻, parameters)
            y⁺ = (ρ = y⁻.ρ, ρu = y⁻.ρu .- dot(y⁻.ρu, normal) .* normal, ρθ = y⁻.ρθ)
            numflux(normal, (y⁻, parameters), (y⁺, parameters))
        end

        # 6. Solve for final result
        dydt_data = Fields.field_values(dydt)
        dydt_data .= rdiv.(dydt_data, space.local_geometry.WJ)

        M = Spaces.Quadratures.cutoff_filter_matrix(
            Float64,
            space.quadrature_style,
            3,
        )
        Operators.tensor_product!(dydt_data, M)

        return dydt
    end

    
    return rhs!
end

function create_init_state(model::ModelSetup, backend::CoreBackend; rhs = nothing, grid = nothing)

    if grid==nothing
        space = create_grid(backend)
    end

    grid = Fields.coordinate_field(space)

    function core_state(x, p)
        @unpack x1, x2 = x
        # set initial state
        ρ = p.ρ₀

        # set initial velocity
        U₁ = cosh(x2)^(-2)

        # Ψ′ = exp(-(x2 + p.l / 10)^2 / 2p.l^2) * cos(p.k * x1) * cos(p.k * x2)
        # Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
        gaussian = exp(-(x2 + p.l / 10)^2 / 2p.l^2)
        u₁′  = gaussian * (x2 + p.l / 10) / p.l^2 * cos(p.k * x1) * cos(p.k * x2)
        u₁′ += p.k * gaussian * cos(p.k * x1) * sin(p.k * x2)
        u₂′  = -p.k * gaussian * sin(p.k * x1) * cos(p.k * x2)


        u = Cartesian12Vector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
        # set initial tracer
        θ = sin(p.k * x2)

        return (ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
    end

    y0 = core_state.(grid, Ref(parameters))

    return y0
end

function create_boundary_conditions(model::ModelSetup, backend::CoreBackend)

    nothing
end