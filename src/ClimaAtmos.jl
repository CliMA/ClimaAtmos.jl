module ClimaAtmos
    include("Domains/Domains.jl")
    include("BoundaryConditions/BoundaryConditions.jl")
    include("Timesteppers/Timesteppers.jl")
    include("Models/Models.jl")
    include("Simulations/Simulations.jl")
    include("Utils/Utils.jl")
end