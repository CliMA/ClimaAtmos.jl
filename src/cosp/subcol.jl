module COSPSubcolumns

import ClimaCore: Fields, Quadratures, Spaces

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

Overlap assumptions are `:maximum`, `:random`, and `:maximum_random`.
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
    threshold,
    overlap_code,
)
    nlev = Spaces.nlevels(axes(cloud_fraction))
    nsubcolumns = length(frac_out)
    field_space = axes(cloud_fraction)

    for column_index in _column_indices(cloud_fraction)
        column_id = _column_id(field_space, column_index...)
        cloud_column =
            Fields.column(cloud_fraction, column_index...)
        frac_columns =
            ntuple(i -> Fields.column(frac_out[i], column_index...), nsubcolumns)
        threshold_columns =
            ntuple(i -> Fields.column(threshold[i], column_index...), nsubcolumns)

        _scops_field_column!(
            frac_columns,
            cloud_column,
            random_seed,
            threshold_columns,
            overlap_code,
            nlev,
            column_id,
        )
    end

    return nothing
end

function _scops_field_column!(
    frac_out,
    cloud_fraction,
    random_seed,
    threshold,
    overlap_code,
    nlev,
    icolumn,
)
    nsubcolumns = length(frac_out)
    out_clear = zero(eltype(frac_out[1]))
    out_stratiform = one(eltype(frac_out[1]))

    @inbounds for ilev in 1:nlev
        total_cloud =
            _clamp_fraction(_level_value(cloud_fraction, ilev))
        previous_total_cloud =
            ilev == 1 ? zero(total_cloud) :
            _clamp_fraction(_level_value(cloud_fraction, ilev - 1))
        convective_cloud = zero(total_cloud)

        for isubcolumn in 1:nsubcolumns
            FT = typeof(total_cloud)
            box_position = (FT(isubcolumn) - FT(0.5)) / FT(nsubcolumns)
            subcolumn_frac_out = frac_out[isubcolumn]
            subcolumn_threshold = threshold[isubcolumn]

            old_threshold =
                ilev == 1 ? box_position : _level_value(subcolumn_threshold, ilev - 1)

            new_threshold = _new_threshold(
                box_position,
                total_cloud,
                previous_total_cloud,
                convective_cloud,
                old_threshold,
                random_seed,
                icolumn,
                ilev,
                isubcolumn,
                overlap_code,
            )

            _set_level_value!(subcolumn_threshold, ilev, new_threshold)

            mask_value =
                total_cloud > new_threshold ? out_stratiform : out_clear

            _set_level_value!(subcolumn_frac_out, ilev, mask_value)
        end
    end

    return nothing
end

function _new_threshold(
    box_position,
    total_cloud,
    previous_total_cloud,
    convective_cloud,
    old_threshold,
    random_seed,
    ipoint,
    ilev,
    isubcolumn,
    overlap_code,
)
    in_convective_region = box_position <= convective_cloud

    if overlap_code == _maximum
        return box_position
    elseif overlap_code == _random
        threshold_min = convective_cloud

        return in_convective_region ? box_position :
               threshold_min +
               (one(threshold_min) - threshold_min) *
               _rand_for_point(
            random_seed,
            ipoint,
            ilev,
            isubcolumn,
            typeof(threshold_min),
        )
    else
        common_cloud = min(previous_total_cloud, total_cloud)
        threshold_min = max(convective_cloud, common_cloud)
        maximally_overlap_stratiform =
            old_threshold < common_cloud && old_threshold > convective_cloud

        return in_convective_region ? box_position :
               maximally_overlap_stratiform ? old_threshold :
               threshold_min +
               (one(threshold_min) - threshold_min) *
               _rand_for_point(
            random_seed,
            ipoint,
            ilev,
            isubcolumn,
            typeof(threshold_min),
        )
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

@inline _level_value(field, ilev) = Fields.level(field, ilev)[]
@inline _set_level_value!(field, ilev, value) = (Fields.level(field, ilev)[] = value)



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
                DimensionMismatch("$name field must have the same axes as cloud_fraction"),
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
