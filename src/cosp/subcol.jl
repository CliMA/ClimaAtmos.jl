module COSPSubcolumns

import ClimaCore: Fields, Operators

const _maximum = 1
const _random = 2
const _maximum_random = 3

struct _RandomPoint{S, I}
    random_seed::S
    isubcolumn::I
end

@inline function (random_point::_RandomPoint)(cloud_fraction, coords)
    return (
        cloud_fraction,
        _rand_for_point(
            random_point.random_seed,
            coords,
            random_point.isubcolumn,
        ),
    )
end

"""
    scops_subcolumn!(cloud_s, threshold_s, cloud_fraction, isubcolumn,
                     nsubcolumns, random_seed; overlap = :maximum_random)

Generate one deterministic cloud subcolumn. Vertical level 1 is the bottom
model level, so the threshold recurrence uses `reverse = true` to proceed from
the top of the atmosphere toward the surface.
"""
function scops_subcolumn!(
    cloud_s::Fields.Field,
    threshold_s::Fields.Field,
    cloud_fraction::Fields.Field,
    isubcolumn::Integer,
    nsubcolumns::Integer,
    random_seed::Integer;
    overlap::Symbol = :maximum_random,
)
    nsubcolumns > 0 || throw(ArgumentError("nsubcolumns must be positive"))
    1 <= isubcolumn <= nsubcolumns ||
        throw(ArgumentError("isubcolumn must be in 1:nsubcolumns"))
    _check_field_axes((cloud_s, threshold_s), cloud_fraction, "output")

    FT = eltype(cloud_fraction)
    random_seed_uint = _seed_uint64(random_seed)
    overlap_code = _overlap_code(overlap)
    box_position = _box_position(FT, isubcolumn, nsubcolumns)
    coords = Fields.coordinate_field(axes(cloud_fraction))
    input = Base.broadcasted(
        _RandomPoint(random_seed_uint, isubcolumn),
        cloud_fraction,
        coords,
    )

    Operators.column_accumulate!(
        threshold_s,
        input;
        init = (; threshold = box_position, previous_cloud = zero(FT)),
        transform = state -> state.threshold,
        reverse = true,
    ) do state, (cloud_fraction_level, random_number)
        total_cloud = _clamp_fraction(cloud_fraction_level)
        new_threshold = _new_threshold_from_random(
            box_position,
            total_cloud,
            state.previous_cloud,
            zero(total_cloud),
            state.threshold,
            random_number,
            overlap_code,
        )
        return (; threshold = new_threshold, previous_cloud = total_cloud)
    end

    out_clear = zero(eltype(cloud_s))
    out_stratiform = one(eltype(cloud_s))
    @. cloud_s = _mask_value(
        cloud_fraction,
        threshold_s,
        out_clear,
        out_stratiform,
    )
    return nothing
end

"""
    shift_up!(output, input)

Set each center level to the input at the level immediately below it. Level 1
is the bottom model level and is set to zero.
"""
function shift_up!(output, input)
    axes(output) == axes(input) ||
        throw(DimensionMismatch("output and input must have matching axes"))
    FT = eltype(output)
    Operators.column_accumulate!(
        output,
        input;
        init = (; current = zero(FT), shifted = zero(FT)),
        transform = state -> state.shifted,
    ) do state, value
        (; current = value, shifted = state.current)
    end
    return nothing
end

"""
Set every level to one if any input level in that column is nonzero.
"""
function column_any!(output, input, scratch)
    axes(output) == axes(input) == axes(scratch) ||
        throw(
            DimensionMismatch(
                "output, input, and scratch must have matching axes",
            ),
        )
    FT = eltype(output)
    indicator = Base.broadcasted(_nonzero_indicator, input, one(FT), zero(FT))
    Operators.column_accumulate!(max, scratch, indicator; init = zero(FT))
    Operators.column_accumulate!(
        max,
        output,
        scratch;
        init = zero(FT),
        reverse = true,
    )
    return nothing
end

@inline _nonzero_indicator(x, one_value, zero_value) =
    ifelse(x != zero(x), one_value, zero_value)

"""
Construct shared precipitation selectors from the actual sampled masks.
"""
function set_scops_selectors!(
    selectors,
    cloud_s,
    threshold_s,
    cloud_fraction,
    nsubcolumns,
    random_seed,
    overlap,
    column_any_scratch,
)
    FT = eltype(cloud_fraction)
    nsubcolumns > 0 || throw(ArgumentError("nsubcolumns must be positive"))
    _check_field_axes(Base.values(selectors), cloud_fraction, "selectors")
    _check_field_axes((column_any_scratch,), cloud_fraction, "column_any_scratch")
    @. selectors.has_cloud = zero(FT)
    for isubcolumn in 1:nsubcolumns
        scops_subcolumn!(
            cloud_s,
            threshold_s,
            cloud_fraction,
            isubcolumn,
            nsubcolumns,
            random_seed;
            overlap,
        )
        @. selectors.has_cloud = max(
            selectors.has_cloud,
            ifelse(cloud_s > zero(FT), one(FT), zero(FT)),
        )
    end
    shift_up!(selectors.has_cloud_below, selectors.has_cloud)
    column_any!(
        selectors.has_cloud_anywhere,
        selectors.has_cloud,
        column_any_scratch,
    )
    return nothing
end

@inline _box_position(::Type{FT}, isubcolumn, nsubcolumns) where {FT} =
    (FT(isubcolumn) - FT(0.5)) / FT(nsubcolumns)

function _new_threshold_from_random(
    box_position,
    total_cloud,
    previous_total_cloud,
    convective_cloud,
    old_threshold,
    random_number,
    overlap_code,
)
    in_convective_region = box_position <= convective_cloud
    if overlap_code == _maximum
        return box_position
    elseif overlap_code == _random
        threshold_min = convective_cloud
        return in_convective_region ? box_position :
               threshold_min + (one(threshold_min) - threshold_min) * random_number
    else
        common_cloud = min(previous_total_cloud, total_cloud)
        threshold_min = max(convective_cloud, common_cloud)
        maximally_overlap_stratiform =
            old_threshold < common_cloud && old_threshold > convective_cloud
        return in_convective_region ? box_position :
               maximally_overlap_stratiform ? old_threshold :
               threshold_min + (one(threshold_min) - threshold_min) * random_number
    end
end

@inline _mask_value(total_cloud, threshold, out_clear, out_stratiform) =
    _clamp_fraction(total_cloud) > threshold ? out_stratiform : out_clear

function _seed_uint64(seed::Integer)
    seed isa Signed && seed < 0 &&
        throw(ArgumentError("random_seed must be non-negative"))
    return UInt64(seed)
end

@inline _clamp_fraction(x) = clamp(x, zero(x), one(x))

@inline function _mix_uint64(x::UInt64)
    x = xor(x, x >> 30)
    x *= 0xbf58476d1ce4e5b9
    x = xor(x, x >> 27)
    x *= 0x94d049bb133111eb
    return xor(x, x >> 31)
end

@inline _coord_bits(x::Float64) = reinterpret(UInt64, x)
@inline _coord_bits(x::Float32) = UInt64(reinterpret(UInt32, x))

@generated function _coord_hash(
    coords,
    ::Val{name},
    ::Type{FT},
    salt::UInt64,
) where {name, FT}
    if name in fieldnames(coords)
        return :(
            _mix_uint64(
            _coord_bits(FT(getfield(coords, $(QuoteNode(name))))) * salt,
        )
        )
    else
        return :(zero(UInt64))
    end
end

# Use the number of random bits that each floating-point type can represent.
@inline _uint64_to_unit_interval(::Type{Float64}, x::UInt64) =
    Float64(x >> 11) * 0x1.0p-53

@inline _uint64_to_unit_interval(::Type{Float32}, x::UInt64) =
    Float32(x >> 40) * Float32(0x1.0p-24)

@inline function _rand_for_point(random_seed::UInt64, coords, isubcolumn)
    FT = typeof(coords.z)
    horizontal_key = _coord_hash(
        coords,
        Val(:x),
        FT,
        UInt64(0x9e3779b97f4a7c15),
    )
    horizontal_key = xor(
        horizontal_key,
        _coord_hash(coords, Val(:y), FT, UInt64(0xbf58476d1ce4e5b9)),
    )
    horizontal_key = xor(
        horizontal_key,
        _coord_hash(coords, Val(:lat), FT, UInt64(0x94d049bb133111eb)),
    )
    horizontal_key = xor(
        horizontal_key,
        _coord_hash(coords, Val(:long), FT, UInt64(0xd2b74407b1ce6e93)),
    )
    vertical_key = _coord_bits(coords.z)
    x = random_seed
    x = xor(x, horizontal_key * 0x9e3779b97f4a7c15)
    x = xor(x, vertical_key * 0xbf58476d1ce4e5b9)
    x = xor(x, UInt64(isubcolumn) * 0x94d049bb133111eb)
    return _uint64_to_unit_interval(FT, _mix_uint64(x))
end

function _check_field_axes(fields, reference, name)
    for field in fields
        axes(field) == axes(reference) ||
            throw(
                DimensionMismatch(
                    "$name field must have the same axes as cloud_fraction",
                ),
            )
    end
end

function _overlap_code(overlap::Symbol)
    overlap === :maximum && return _maximum
    overlap === :random && return _random
    overlap === :maximum_random && return _maximum_random
    throw(ArgumentError("unknown overlap option"))
end

end
