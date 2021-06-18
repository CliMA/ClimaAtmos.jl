abstract type AbstractBackend end

Base.@kwdef struct DiscontinuousGalerkinBackend{𝒜,ℬ} <: AbstractBackend
    grid::𝒜
    numerics::ℬ
end

function instantiate_simulation_state(model::ModelSetup, backend::DiscontinuousGalerkinBackend)
    # translate the backend and model into the corresponding
    # concrete grid and model for this backend
    backend_grid = create_backend_grid(backend)
    backend_model = create_backend_model(model, backend)(
        physics = model.physics,
        boundary_conditions = model.boundary_conditions,
        initial_conditions = model.initial_conditions,
        numerics = backend.numerics,
    )
    
    # initialize the right-hand side
    rhs = ESDGModel(
        balance_law, 
        backend.grid,
        backend.numerics.flux,
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    ) 

    # initialize the state vector
    FT = eltype(rhs.grid.vgeo)
    state_init = init_ode_state(rhs, FT(0); init_on_cpu = true)

    return rhs, state_init
end

function create_backend_grid(::AbstractModel, ::AbstractBackend)
    error("The backend does not support this model or equation set.")
end

function create_backend_model(::AbstractModel, ::AbstractBackend)
    error("The backend does not support this model or equation set.")
end

function create_backend_model(model::ModelSetup{𝒜}, backend::AbstractBackend) where 
    {𝒜 <: ThreeDimensionalNavierStokes{TotalEnergy, DryIdealGas, Compressible}}

    return DryCompressibleThreeDimensionalNavierStokesEquationsWithTotalEnergy
end