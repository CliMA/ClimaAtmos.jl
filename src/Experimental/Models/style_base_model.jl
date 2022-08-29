"""
Supertype for all base model styles.
"""
abstract type AbstractBaseModelStyle <: AbstractModelStyle end

struct AdvectiveForm <: AbstractBaseModelStyle end
struct ConservativeForm <: AbstractBaseModelStyle end
struct AnelasticAdvectiveForm <: AbstractBaseModelStyle end

Models.variable_names(::AdvectiveForm) = (:ρ, :uh, :w)
Models.variable_names(::ConservativeForm) = (:ρ, :ρuh, :ρw)
Models.variable_names(::AnelasticAdvectiveForm) = (:ρ, :uh)

Models.variable_types(
    ::AdvectiveForm,
    ::AbstractSingleColumnModel,
    ::Type{FT},
) where {FT} = (ρ = FT, uh = Geometry.UVVector{FT}, w = Geometry.WVector{FT})
Models.variable_types(
    ::ConservativeForm,
    ::AbstractSingleColumnModel,
    ::Type{FT},
) where {FT} = (ρ = FT, ρuh = Geometry.UVVector{FT}, ρw = Geometry.WVector{FT})
Models.variable_types(
    ::AnelasticAdvectiveForm,
    ::AbstractSingleColumnModel,
    ::Type{FT},
) where {FT} = (ρ = FT, uh = Geometry.UVVector{FT})

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

Models.variable_spaces(::AdvectiveForm, ::AbstractSingleColumnModel) = (
    ρ = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},
    uh = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},
    w = Spaces.FiniteDifferenceSpace{Spaces.CellFace},
)

Models.variable_spaces(
    ::AdvectiveForm,
    ::Union{AbstractNonhydrostatic2DModel, AbstractNonhydrostatic3DModel},
) = (
    ρ = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
    uh = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
    w = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellFace},
)

Models.variable_spaces(::ConservativeForm, ::AbstractSingleColumnModel) = (
    ρ = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},
    ρuh = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},
    ρw = Spaces.FiniteDifferenceSpace{Spaces.CellFace},
)

Models.variable_spaces(
    ::ConservativeForm,
    ::Union{AbstractNonhydrostatic2DModel, AbstractNonhydrostatic3DModel},
) = (
    ρ = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
    ρuh = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
    ρw = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellFace},
)

Models.variable_spaces(::AnelasticAdvectiveForm, ::AbstractSingleColumnModel) =
    (
        ρ = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},
        uh = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},
    )
