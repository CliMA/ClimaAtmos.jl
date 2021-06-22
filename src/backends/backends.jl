abstract type AbstractBackend end

Base.@kwdef struct DiscontinuousGalerkinBackend{𝒜} <: AbstractBackend
    numerics::𝒜
end

function create_grid(::DiscontinuousGalerkinBackend, discretized_domain)
    elements = get_elements(discretized_domain)
    polynomial_order = get_polynomial_order(discretized_domain)

    return create_dg_grid(
        discretized_domain.domain, 
        elements = elements,
        polynomial_order = polynomial_order,
    )
end

function create_rhs(model::ModelSetup, grid, backend::DiscontinuousGalerkinBackend)
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

function create_init_state(model::ModelSetup, backend::DiscontinuousGalerkinBackend; rhs = nothing)
    if rhs === nothing
        rhs = create_rhs(model, backend)
    end
    FT = eltype(rhs.grid.vgeo)
    state_init = init_ode_state(rhs, FT(0); init_on_cpu = true)

    return state_init
end

# function create_boundary_conditions(model::ModelSetup, backend::DiscontinuousGalerkinBackend)
#     boundary_conditions = model.boundary_conditions
    
#     nothing
# end

# utils
function get_elements(discretized_domain::DiscretizedDomain{<:ProductDomain})
    return discretized_domain.discretization.elements
end

function get_elements(discretized_domain::DiscretizedDomain{<:SphericalShell})
    horizontal_elements =discretized_domain.discretization.horizontal.elements
    vertical_elements = discretized_domain.discretization.vertical.elements
    return (vertical = vertical_elements, horizontal = horizontal_elements)
end

function get_polynomial_order(discretized_domain::DiscretizedDomain{<:ProductDomain})
    return discretized_domain.discretization.polynomial_order
end

function get_polynomial_order(discretized_domain::DiscretizedDomain{<:SphericalShell})
    horizontal_poly_order = discretized_domain.discretization.horizontal.polynomial_order
    vertical_poly_order = discretized_domain.discretization.vertical.polynomial_order
    return (vertical = vertical_poly_order, horizontal = horizontal_poly_order)
end