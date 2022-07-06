# A complementary struct to ClimaCore's `PlusHalf` type.
struct Cent{I <: Integer}
    i::I
end

Base.:+(h::Cent) = h
Base.:-(h::Cent) = Cent(-h.i - one(h.i))
Base.:+(i::Integer, h::Cent) = Cent(i + h.i)
Base.:+(h::Cent, i::Integer) = Cent(h.i + i)
Base.:+(h1::Cent, h2::Cent) = h1.i + h2.i + one(h1.i)
Base.:-(i::Integer, h::Cent) = Cent(i - h.i - one(h.i))
Base.:-(h::Cent, i::Integer) = Cent(h.i - i)
Base.:-(h1::Cent, h2::Cent) = h1.i - h2.i

Base.:<=(h1::Cent, h2::Cent) = h1.i <= h2.i
Base.:<(h1::Cent, h2::Cent) = h1.i < h2.i
Base.max(h1::Cent, h2::Cent) = Cent(max(h1.i, h2.i))
Base.min(h1::Cent, h2::Cent) = Cent(min(h1.i, h2.i))

toscalar(x::CCG.Covariant3Vector) = x.u₃

const FDFields = Union{CC.Fields.ExtrudedFiniteDifferenceField, CC.Fields.FiniteDifferenceField}

const FaceFields = Union{CC.Fields.FaceExtrudedFiniteDifferenceField, CC.Fields.FaceFiniteDifferenceField}

const CenterFields = Union{CC.Fields.CenterExtrudedFiniteDifferenceField, CC.Fields.CenterFiniteDifferenceField}

Base.@propagate_inbounds Base.getindex(field::FDFields, i::Integer) = Base.getproperty(field, i)

Base.@propagate_inbounds Base.getindex(field::CenterFields, i::Cent) = Base.getindex(CC.Fields.field_values(field), i.i)
Base.@propagate_inbounds Base.setindex!(field::CenterFields, v, i::Cent) =
    Base.setindex!(CC.Fields.field_values(field), v, i.i)

Base.@propagate_inbounds Base.getindex(field::FaceFields, i::CCO.PlusHalf) =
    Base.getindex(CC.Fields.field_values(field), i.i)
Base.@propagate_inbounds Base.setindex!(field::FaceFields, v, i::CCO.PlusHalf) =
    Base.setindex!(CC.Fields.field_values(field), v, i.i)

Base.@propagate_inbounds Base.getindex(field::FaceFields, ::Cent) =
    error("Attempting to getindex with a center index (Cent) into a Face field")
Base.@propagate_inbounds Base.getindex(field::CenterFields, ::CCO.PlusHalf) =
    error("Attempting to getindex with a face index (PlusHalf) into a Center field")

Base.@propagate_inbounds Base.setindex!(field::FaceFields, v, ::Cent) =
    error("Attempting to setindex with a center index (Cent) into a Face field")
Base.@propagate_inbounds Base.setindex!(field::CenterFields, v, ::CCO.PlusHalf) =
    error("Attempting to setindex with a face index (PlusHalf) into a Center field")

# TODO: deprecate, we should not overload getindex/setindex for ordinary arrays.
Base.@propagate_inbounds Base.getindex(arr::AbstractArray, i::Cent) = Base.getindex(arr, i.i)
Base.@propagate_inbounds Base.setindex!(arr::AbstractArray, v, i::Cent) = Base.setindex!(arr, v, i.i)
Base.@propagate_inbounds Base.getindex(arr::AbstractArray, i::CCO.PlusHalf) = Base.getindex(arr, i.i)
Base.@propagate_inbounds Base.setindex!(arr::AbstractArray, v, i::CCO.PlusHalf) = Base.setindex!(arr, v, i.i)
Base.@propagate_inbounds Base.getindex(arr::AbstractArray, i::Int, j::Cent) = Base.getindex(arr, i, j.i)
Base.@propagate_inbounds Base.setindex!(arr::AbstractArray, v, i::Int, j::Cent) = Base.setindex!(arr, v, i, j.i)
Base.@propagate_inbounds Base.getindex(arr::AbstractArray, i::Int, j::CCO.PlusHalf) = Base.getindex(arr, i, j.i)
Base.@propagate_inbounds Base.setindex!(arr::AbstractArray, v, i::Int, j::CCO.PlusHalf) = Base.setindex!(arr, v, i, j.i)

# Constant field
function FieldFromNamedTuple(space, nt::NamedTuple)
    cmv(z) = nt
    return cmv.(CC.Fields.coordinate_field(space))
end
# Non-constant field
function FieldFromNamedTuple(space, initial_conditions::Function, ::Type{FT}, params...) where {FT}
    local_geometry = CC.Fields.local_geometry_field(space)
    return initial_conditions.(FT, local_geometry, params...)
end

function Base.cumsum(field::CC.Fields.FiniteDifferenceField)
    Base.cumsum(parent(CC.Fields.weighted_jacobian(field)) .* vec(field), dims = 1)
end
function Base.cumsum!(fieldout::CC.Fields.FiniteDifferenceField, fieldin::CC.Fields.FiniteDifferenceField)
    Base.cumsum!(fieldout, parent(CC.Fields.weighted_jacobian(fieldin)) .* vec(fieldin), dims = 1)
end

get_Δz(field::CC.Fields.FiniteDifferenceField) = parent(CC.Fields.weighted_jacobian(field))

# TODO: move these things into ClimaCore

isa_center_space(space) = false
isa_center_space(::CC.Spaces.CenterFiniteDifferenceSpace) = true
isa_center_space(::CC.Spaces.CenterExtrudedFiniteDifferenceSpace) = true

isa_face_space(space) = false
isa_face_space(::CC.Spaces.FaceFiniteDifferenceSpace) = true
isa_face_space(::CC.Spaces.FaceExtrudedFiniteDifferenceSpace) = true

const CallableZType = Union{Function, Dierckx.Spline1D}

function set_z!(field::CC.Fields.Field, u::CallableZType = x -> x, v::CallableZType = y -> y)
    z = CC.Fields.coordinate_field(axes(field)).z
    @. field = CCG.Covariant12Vector(CCG.UVVector(u(z), v(z)))
end

function set_z!(field::CC.Fields.Field, u::Real, v::Real)
    lg = CC.Fields.local_geometry_field(axes(field))
    uconst(coord) = u
    vconst(coord) = v
    @. field = CCG.Covariant12Vector(CCG.UVVector(uconst(lg), vconst(lg)))
end
