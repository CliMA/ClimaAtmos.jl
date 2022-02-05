
"""
Supertype for all model moisture styles.
"""
abstract type AbstractMoistureStyle <: AbstractModelStyle end

struct Dry <: AbstractMoistureStyle end
struct EquilibriumMoisture <: AbstractMoistureStyle end
struct NonEquilibriumMoisture <: AbstractMoistureStyle end

Models.variable_names(::Dry) = nothing
Models.variable_names(::EquilibriumMoisture) = (:ρq_tot,)
Models.variable_names(::NonEquilibriumMoisture) = (:ρq_tot, :ρq_liq, :ρq_ice)

Models.variable_types(::Dry, ::AbstractModel, FT) = Nothing
Models.variable_types(::EquilibriumMoisture, ::AbstractModel, FT) =
    (ρq_tot = FT,)
Models.variable_types(::NonEquilibriumMoisture, ::AbstractModel, FT) =
    (ρq_tot = FT, ρq_liq = FT, ρq_ice = FT)

Models.variable_spaces(::Dry, ::AbstractSingleColumnModel) = Nothing
Models.variable_spaces(::EquilibriumMoisture, ::AbstractSingleColumnModel) =
    (ρq_tot = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},)
Models.variable_spaces(::NonEquilibriumMoisture, ::AbstractSingleColumnModel) =
    (
        ρq_tot = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},
        ρq_liq = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},
        ρq_ice = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},
    )

Models.variable_spaces(
    ::Dry,
    ::Union{AbstractNonhydrostatic2DModel, AbstractNonhydrostatic3DModel},
) = Nothing
Models.variable_spaces(
    ::EquilibriumMoisture,
    ::Union{AbstractNonhydrostatic2DModel, AbstractNonhydrostatic3DModel},
) = (ρq_tot = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},)
Models.variable_spaces(
    ::NonEquilibriumMoisture,
    ::Union{AbstractNonhydrostatic2DModel, AbstractNonhydrostatic3DModel},
) = (
    ρq_tot = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
    ρq_liq = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
    ρq_ice = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
)
