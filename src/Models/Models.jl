module Models

using ClimaCore: Geometry, Spaces

export AbstractModel,
    AbstractSingleColumnModel,
    AbstractNonhydrostatic2DModel,
    AbstractNonhydrostatic3DModel,
    components,
    variable_names,
    variable_types,
    variable_spaces,
    default_initial_conditions,
    make_ode_function

export AbstractModelStyle,
    AbstractBaseModelStyle,
    AbstractThermodynamicsStyle,
    AbstractMoistureStyle,
    AdvectiveForm,
    ConservativeForm,
    PotentialTemperature,
    TotalEnergy,
    Dry,
    EquilibriumMoisture,
    NonEquilibriumMoisture,
    NoPrecipitation,
    PrecipitationRemoval,
    OneMoment,
    get_velocities

export SingleColumnModel, Nonhydrostatic2DModel, Nonhydrostatic3DModel

"""
Supertypes for all models.
"""
abstract type AbstractModel end
abstract type AbstractSingleColumnModel <: AbstractModel end
abstract type AbstractNonhydrostatic2DModel <: AbstractModel end
abstract type AbstractNonhydrostatic3DModel <: AbstractModel end

"""
Supertype for all model styles.
"""
abstract type AbstractModelStyle end

"""
    components(model::AbstractModel)

Return the components of `model`, e.g. `base`, `thermodynamics`, `moisture`, etc.
"""
components(model::AbstractModel) =
    error("components not implemented for given model type")

"""
    variables_names(styles::AbstractModelStyle)

Return the state variable names for `style`.
# Example
```jldoctest; setup = :(using ClimaAtmos.Models)
julia> Models.variable_names(NonEquilibriumMoisture())
(:ρq_tot, :ρq_liq, :ρq_ice)
```
"""
variable_names(style::AbstractModelStyle) =
    error("components not implemented for given model style $style")

variable_names(model::AbstractModel) =
    map(Models.variable_names, Models.components(model))

"""
    variables_types(style::AbstractModelStyle, model::AbstractModel, FT)

Return the state variable types for `style` and `model` with float type `FT`.
"""
variable_types(style::AbstractModelStyle, model::AbstractModel) =
    error("components not implemented for given model style $style & $model")

"""
    variables_spaces(style::AbstractModelStyle)

Return the state variables for `style`.
# Example
```jldoctest; setup = :(using ClimaAtmos.Models)
julia> Models.variable_spaces(TotalEnergy())
(ρe_tot = ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace{ClimaCore.Spaces.CellCenter},)
```
"""
variable_spaces(style::AbstractModelStyle) =
    error("components not implemented for given model style $style")

"""
    default_initial_conditions(model)

Construct the initial conditions for `model`.
"""
default_initial_conditions(model::AbstractModel) =
    error("default_initial_conditions not implemented for given model type")

"""
    make_ode_function(model)

Construct the ordinary differential equations for `model`.
"""
make_ode_function(model::AbstractModel) =
    error("make_ode_function not implemented for given model type")

"""
    get_velocities(Y, model)

Construct the initial conditions for `model`.
"""
get_velocities(Y, model::AbstractModel) =
    error("No method to get velocities implemented for given model type")

# model styles
include("style_base_model.jl")
include("style_thermodynamics.jl")
include("style_moisture.jl")
include("style_precipitation.jl")

# models
include("SingleColumnModels/SingleColumnModels.jl")
include("Nonhydrostatic2DModels/Nonhydrostatic2DModels.jl")
include("Nonhydrostatic3DModels/Nonhydrostatic3DModels.jl")

end # module
