
function create_grid(::DiscontinuousGalerkinBackend, discretized_domain)
    elements = get_elements(discretized_domain)
    polynomial_order = get_polynomial_order(discretized_domain)

    return create_dg_grid(
        discretized_domain.domain, 
        elements = elements,
        polynomial_order = polynomial_order,
    )
end

function create_esdg_rhs(model::AbstractModel, backend::DiscontinuousGalerkinBackend; domain, grid)
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

#=
function create_linear_rhs(model::AbstractModel, splitting::IMEXSplitting; domain, grid)
    balance_law = create_balance_law(model, domain)
    if splitting.linear_model == :linear
        linear_balance_law = linearize_balance_law(balance_law)
        volume_flux = LinearKGVolumeFlux()
        @info "linear model created"
    elseif splitting.linear_model == :verylinear
        linear_balance_law = verylinearize_balance_law(balance_law)
        volume_flux = VeryLinearKGVolumeFlux()
        @info "verylinear model created"
    else
        @warn "invalid linear model specification for IMEXSplitting, set to :linear"
        linear_balance_law = linearize_balance_law(balance_law)
        volume_flux = LinearKGVolumeFlux()
    end 

    rhs = VESDGModel(
        linear_balance_law,
        grid,
        surface_numerical_flux_first_order = RefanovFlux(),
        volume_numerical_flux_first_order = volume_flux,
    )

    return rhs
end
=#

function create_rhs(::NoSplitting, model::AbstractModel, backend::DiscontinuousGalerkinBackend; domain, grid)
    rhs = create_esdg_rhs(model, backend, domain = domain, grid = grid)
    return rhs 
end

function create_rhs(splitting::IMEXSplitting, model::AbstractModel, backend::DiscontinuousGalerkinBackend; domain, grid)
    rhs = []
    # create explicit model and push to rhs
    tmp = #Explicit(
        create_esdg_rhs(model, backend, domain = domain, grid = grid)
    #)
    push!(rhs, tmp)
    # create implicit model and push to rhs
    tmp = #Implicit(
        create_linear_rhs(model, splitting, domain = domain, grid = grid)
    # )
    push!(rhs, tmp)
    return Tuple(rhs)
end

function create_init_state(model::AbstractModel, backend::DiscontinuousGalerkinBackend; rhs = nothing)
    if rhs === nothing
        # TODO: this looks wrong
        rhs = create_rhs(model, backend)
    end
    FT = eltype(rhs.grid.vgeo)
    state_init = init_ode_state(rhs, FT(0); init_on_cpu = true)

    return state_init
end

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