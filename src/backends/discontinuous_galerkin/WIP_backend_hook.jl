
function create_grid(::DiscontinuousGalerkinBackend, domain)
    elements = get_elements(domain)
    polynomial_order = get_polynomial_order(domain)

    return create_dg_grid(
        domain, 
        elements = elements,
        polynomial_order = polynomial_order,
    )
end

function create_esdg_rhs(model::AbstractModel, backend::DiscontinuousGalerkinBackend; grid)
    balance_law = create_balance_law(model)
    numerical_flux = create_numerical_flux(backend.numerics.flux)

    rhs = ESDGModel(
        balance_law, 
        grid,
        surface_numerical_flux_first_order = numerical_flux,
        volume_numerical_flux_first_order = KGVolumeFlux(),
    )

    return rhs
end

function create_rhs(::NoSplitting, model::AbstractModel, backend::DiscontinuousGalerkinBackend; grid)
    rhs = create_esdg_rhs(model, backend, grid = grid)
    return rhs 
end


function create_init_state(::AbstractModel, ::DiscontinuousGalerkinBackend; rhs)
    FT = eltype(rhs.grid.vgeo)
    state_init = init_ode_state(rhs, FT(0); init_on_cpu = true)

    return state_init
end

# utils
function get_elements(domain::SphericalShell)
    horizontal_elements = domain.nelements.horizontal
    vertical_elements = domain.nelements.vertical
    return (vertical = vertical_elements, horizontal = horizontal_elements)
end

function get_polynomial_order(domain::SphericalShell)
    horizontal_poly_order = domain.npolynomial.horizontal
    vertical_poly_order = domain.vertical_discretization.vpolynomial
    return (vertical = vertical_poly_order, horizontal = horizontal_poly_order)
end

