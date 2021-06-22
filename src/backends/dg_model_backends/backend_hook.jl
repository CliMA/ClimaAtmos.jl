function create_grid(backend::DiscontinuousGalerkinBackend)
    domain = backend.grid.domain
    grid = backend.grid
    elements = get_elements(domain, grid)
    polynomial_order = get_polynomial_order(domain, grid)

    return create_dg_grid(
        domain, 
        elements = elements,
        polynomial_order = polynomial_order,
    )
end

function create_rhs(model::ModelSetup, backend::DiscontinuousGalerkinBackend; grid = nothing)
    balance_law = create_balance_law(model)
    if grid === nothing
        grid = create_grid(backend)
    end
    numerical_flux = create_numerical_flux(backend.numerics.flux)

    # TODO!: Change to hybrid model (final DG version with all components)
    rhs = ESDGModel(
        balance_law, 
        grid,
        surface_numerical_flux_first_order = numerical_flux,
        volume_numerical_flux_first_order = KGVolumeFlux(),
    )

    return rhs
end

function create_init_state(model::ModelSetup, backend::DiscontinuousGalerkinBackend; rhs = nothing)
    if rhs === nothing
        rhs = create_rhs(model, backend)
    end
    FT = eltype(rhs.grid.vgeo)
    state_init = init_ode_state(rhs, FT(0); init_on_cpu = true)

    return state_init
end

function create_boundary_conditions(model::ModelSetup, backend::DiscontinuousGalerkinBackend)
    boundary_conditions = model.boundary_conditions
    
    
    nothing
end

# utils
function get_elements(::ProductDomain, grid)
    return grid.discretization.elements
end

function get_elements(::SphericalShell, grid)
    horizontal_elements = grid.discretization.horizontal.elements
    vertical_elements = grid.discretization.vertical.elements
    return (vertical = vertical_elements, horizontal = horizontal_elements)
end

function get_polynomial_order(::ProductDomain, grid)
    return grid.discretization.polynomial_order
end

function get_polynomial_order(::SphericalShell, grid)
    horizontal_poly_order = grid.discretization.horizontal.polynomial_order
    vertical_poly_order = grid.discretization.vertical.polynomial_order
    return (vertical = vertical_poly_order, horizontal = horizontal_poly_order)
end