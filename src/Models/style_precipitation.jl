
"""
Supertype for all model precipitation styles.
"""
abstract type AbstractPrecipitationStyle <: AbstractModelStyle end

struct NoPrecipitation <: AbstractPrecipitationStyle end
struct PrecipitationRemoval <: AbstractPrecipitationStyle end
struct OneMoment <: AbstractPrecipitationStyle end

Models.variable_names(::NoPrecipitation) = nothing
Models.variable_names(::PrecipitationRemoval) = nothing
Models.variable_names(::OneMoment) = (:ρq_rai, :ρq_sno)

Models.variable_types(::NoPrecipitation, ::AbstractModel, FT) = Nothing
Models.variable_types(::PrecipitationRemoval, ::AbstractModel, FT) = Nothing
Models.variable_types(::OneMoment, ::AbstractModel, FT) =
    (ρq_rai = FT, ρq_sno = FT)

Models.variable_spaces(::NoPrecipitation) = Nothing
Models.variable_spaces(::PrecipitationRemoval) = Nothing
Models.variable_spaces(::OneMoment) = (
    ρq_rai = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
    ρq_sno = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
)
