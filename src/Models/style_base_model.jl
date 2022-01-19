"""
Supertype for all base model styles.
"""
abstract type AbstractBaseModelStyle <: AbstractModelStyle end

struct AdvectiveForm <: AbstractBaseModelStyle end
struct ConservativeForm <: AbstractBaseModelStyle end

Models.variable_names(::AdvectiveForm) = (:ρ, :uh, :w)
Models.variable_names(::ConservativeForm) = (:ρ, :ρuh, :ρw)

Models.variable_types(
    ::AdvectiveForm,
    ::AbstractNonhydrostatic2DModel,
    ::Type{FT},
) where {FT} = (ρ = FT, uh = Geometry.UVector{FT}, w = Geometry.WVector{FT})
Models.variable_types(
    ::ConservativeForm,
    ::AbstractNonhydrostatic2DModel,
    ::Type{FT},
) where {FT} = (ρ = FT, ρuh = Geometry.UVector{FT}, ρw = Geometry.WVector{FT})
Models.variable_types(
    ::AdvectiveForm,
    ::AbstractNonhydrostatic3DModel,
    ::Type{FT},
) where {FT} = (
    ρ = FT,
    uh = Geometry.Covariant12Vector{FT},
    w = Geometry.Covariant3Vector{FT},
)
Models.variable_types(
    ::ConservativeForm,
    ::AbstractNonhydrostatic3DModel,
    ::Type{FT},
) where {FT} = (
    ρ = FT,
    ρuh = Geometry.Covariant12Vector{FT},
    ρw = Geometry.Covariant3Vector{FT},
)

Models.variable_spaces(::AdvectiveForm) = (
    ρ = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
    uh = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
    w = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellFace},
)
Models.variable_spaces(::ConservativeForm) = (
    ρ = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
    ρuh = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
    ρw = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellFace},
)
