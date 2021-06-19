abstract type AbstractBackend end

Base.@kwdef struct DiscontinuousGalerkinBackend{𝒜,ℬ} <: AbstractBackend
    grid::𝒜
    numerics::ℬ
end

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

# utils
function get_elements(::ProductDomain, grid)
    return grid.discretization.elements
end

function get_polynomial_order(::ProductDomain, grid)
    return grid.discretization.polynomial_order
end