module COSPSubcolumns

import ClimaCore: Fields, Operators, Quadratures, Spaces

export scops!

const _maximum = 1
const _random = 2
const _maximum_random = 3

"""
Generate cloud subcolumn masks from grid-mean cloud fraction.

Inputs

  - `cloud_fraction`
  - `convective_cloud_fraction` is assumed to be zero
  - `random_seed` is used for random overlap sampling

Outputs

  - writes cloud masks in-place into `frac_out`
  - `0`: clear sky
  - `1`: stratiform cloud
  - `2`: convective cloud (not used for now, but reserved for future use)

Overlap assumptions are `:maximum`, `:random`, and `:maximum_random`
"""
function scops!(
    frac_out::NTuple{N},
    threshold::NTuple{N},
    cloud_fraction::Fields.Field,
    random_seed::Integer;
    overlap::Symbol = :maximum_random,
) where {N}
    N > 0 || throw(ArgumentError("frac_out must contain at least one subcolumn"))

    _check_field_axes(frac_out, cloud_fraction, "frac_out")
    _check_field_axes(threshold, cloud_fraction, "threshold")

    overlap_code = _overlap_code(overlap)
    _scops_fields!(
        frac_out,
        cloud_fraction,
        _seed_uint64(random_seed),
        threshold,
        overlap_code,
    )

    return nothing
end

function _scops_fields!(
    frac_out,
    cloud_fraction,
    random_seed,
    thresholds,
    overlap_code,
)
    nsubcolumns = length(frac_out)
    out_clear = zero(eltype(frac_out[1]))
    out_stratiform = one(eltype(frac_out[1]))
    FT = eltype(cloud_fraction)
    rand_field = similar(cloud_fraction)

    for isubcolumn in 1:nsubcolumns
        _fill_random_field!(rand_field, random_seed, isubcolumn)

        box_position = _box_position(FT, isubcolumn, nsubcolumns)
        subcolumn_frac_out = frac_out[isubcolumn]
        subcolumn_threshold = thresholds[isubcolumn]
        input = Base.broadcasted(tuple, cloud_fraction, rand_field)

        Operators.column_accumulate!(
            subcolumn_threshold,
            input;
            init = (; threshold = box_position, previous_cloud = zero(FT)),
            transform = state -> state.threshold,
            reverse = true,
        ) do state, (cloud_fraction_level, random_number)
            total_cloud = _clamp_fraction(cloud_fraction_level)
            convective_cloud = zero(total_cloud)

            new_threshold = _new_threshold_from_random(
                box_position,
                total_cloud,
                state.previous_cloud,
                convective_cloud,
                state.threshold,
                random_number,
                overlap_code,
            )

            return (; threshold = new_threshold, previous_cloud = total_cloud)
        end

        @. subcolumn_frac_out = _mask_value(
            cloud_fraction,
            subcolumn_threshold,
            out_clear,
            out_stratiform,
        )
    end

    return nothing
end

@inline _box_position(::Type{FT}, isubcolumn, nsubcolumns) where {FT} =
    (FT(isubcolumn) - FT(0.5)) / FT(nsubcolumns)

function _fill_random_field!(rand_field, random_seed, isubcolumn)
    nlev = Spaces.nlevels(axes(rand_field))
    field_space = axes(rand_field)
    FT = eltype(rand_field)

    for column_index in _column_indices(rand_field)
        column_id = _column_id(field_space, column_index...)
        rand_column = Fields.column(rand_field, column_index...)
        @inbounds for ilev in 1:nlev
            Fields.level(rand_column, ilev)[] =
                _rand_for_point(
                    random_seed,
                    column_id,
                    nlev - ilev + 1,
                    isubcolumn,
                    FT,
                )
        end
    end

    return nothing
end

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
               threshold_min +
               (one(threshold_min) - threshold_min) * random_number
    else
        common_cloud = min(previous_total_cloud, total_cloud)
        threshold_min = max(convective_cloud, common_cloud)
        maximally_overlap_stratiform =
            old_threshold < common_cloud && old_threshold > convective_cloud

        return in_convective_region ? box_position :
               maximally_overlap_stratiform ? old_threshold :
               threshold_min +
               (one(threshold_min) - threshold_min) * random_number
    end
end

function _column_indices(field::Fields.Field)
    space = axes(field)
    space isa Spaces.FiniteDifferenceSpace && return ((1, 1, 1),)

    horizontal_space = Spaces.horizontal_space(space)
    quadrature_points =
        1:Quadratures.degrees_of_freedom(
            Spaces.quadrature_style(horizontal_space),
        )
    slab_indices = Spaces.eachslabindex(horizontal_space)

    return horizontal_space isa Spaces.SpectralElementSpace1D ?
           Iterators.product(quadrature_points, slab_indices) :
           Iterators.product(quadrature_points, quadrature_points, slab_indices)
end

@inline _column_id(::Spaces.FiniteDifferenceSpace, ::Int, ::Int, ::Int) = 1

@inline function _column_id(
    space::Spaces.ExtrudedFiniteDifferenceSpace,
    i::Int,
    h::Int,
)
    return _column_id(Spaces.horizontal_space(space), i, h)
end

@inline function _column_id(
    space::Spaces.ExtrudedFiniteDifferenceSpace,
    i::Int,
    j::Int,
    h::Int,
)
    return _column_id(Spaces.horizontal_space(space), i, j, h)
end

@inline function _column_id(space::Spaces.SpectralElementSpace1D, i::Int, h::Int)
    ni = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    return i + (h - 1) * ni
end

@inline function _column_id(
    space::Spaces.SpectralElementSpace2D,
    i::Int,
    j::Int,
    h::Int,
)
    ni = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    return i + ni * ((j - 1) + ni * (h - 1))
end

@inline function _mask_value(
    total_cloud,
    threshold,
    out_clear,
    out_stratiform,
)
    return _clamp_fraction(total_cloud) > threshold ? out_stratiform : out_clear
end

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

@inline function _rand_for_point(
    random_seed::UInt64,
    ipoint,
    ilev,
    isubcolumn,
    ::Type{FT},
) where {FT}
    x = random_seed
    x = xor(x, UInt64(ipoint) * 0x9e3779b97f4a7c15)
    x = xor(x, UInt64(ilev) * 0xbf58476d1ce4e5b9)
    x = xor(x, UInt64(isubcolumn) * 0x94d049bb133111eb)
    return FT(_mix_uint64(x) >> 11) * FT(0x1.0p-53)
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
