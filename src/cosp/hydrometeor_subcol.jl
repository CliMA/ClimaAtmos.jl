module COSPHydrometeorSubcolumns

export sampled_cloud_fraction!,
    sampled_precip_fraction!,
    slice_hydrometeor_subcolumns!

const CLOUD_HYDROMETEORS = (:q_lcl, :q_icl)
const PRECIP_HYDROMETEORS = (:q_rai, :q_sno)

function sampled_cloud_fraction!(sampled_fraction, cloud_mask::NTuple{N}) where {N}
    N > 0 || throw(ArgumentError("cloud_mask must contain at least one subcolumn"))
    _check_field_axes(cloud_mask, sampled_fraction, "cloud_mask")

    zero_value = zero(eltype(sampled_fraction))
    @. sampled_fraction = zero_value
    for mask in cloud_mask
        @. sampled_fraction += _cloud_indicator(mask)
    end
    @. sampled_fraction = sampled_fraction / N

    return nothing
end

function sampled_precip_fraction!(sampled_fraction, precip_mask::NTuple{N}) where {N}
    N > 0 || throw(ArgumentError("precip_mask must contain at least one subcolumn"))
    _check_field_axes(precip_mask, sampled_fraction, "precip_mask")

    zero_value = zero(eltype(sampled_fraction))
    @. sampled_fraction = zero_value
    for mask in precip_mask
        @. sampled_fraction += _precip_indicator(mask)
    end
    @. sampled_fraction = sampled_fraction / N

    return nothing
end

"""
    slice_hydrometeor_subcolumns!(
        subcolumns,
        cloud_mask,
        precip_mask,
        grid_mean,
        sampled_cloud_fraction,
        sampled_precip_fraction,
    )

Populate diagnostic hydrometeor subcolumns from grid-mean hydrometeor fields and
precomputed COSP cloud and precipitation masks.

Cloud hydrometeors are placed only where the cloud mask is non-clear (`1` or
`2`). Precipitating hydrometeors are placed only where the precipitation mask is
non-zero (`1`, `2`, or `3`). In each grid cell, the grid-mean value is divided by
the sampled mask fraction so that the mean over subcolumns recovers the original
grid-mean value when the sampled fraction is non-zero. The sampled-fraction
fields are scratch storage and are overwritten.
"""
function slice_hydrometeor_subcolumns!(
    subcolumns::NamedTuple,
    cloud_mask::NTuple{N},
    precip_mask::NTuple{N},
    grid_mean::NamedTuple,
    sampled_cloud_fraction,
    sampled_precip_fraction,
) where {N}
    N > 0 || throw(ArgumentError("masks must contain at least one subcolumn"))
    keys(subcolumns) == keys(grid_mean) ||
        throw(ArgumentError("subcolumns and grid_mean must have matching keys"))

    reference = first(Base.values(grid_mean))
    _check_field_axes(cloud_mask, reference, "cloud_mask")
    _check_field_axes(precip_mask, reference, "precip_mask")
    axes(sampled_cloud_fraction) == axes(reference) ||
        throw(DimensionMismatch("sampled_cloud_fraction must have matching axes"))
    axes(sampled_precip_fraction) == axes(reference) ||
        throw(DimensionMismatch("sampled_precip_fraction must have matching axes"))
    _check_subcolumn_output_axes(subcolumns, grid_mean, N)

    sampled_cloud_fraction!(sampled_cloud_fraction, cloud_mask)
    sampled_precip_fraction!(sampled_precip_fraction, precip_mask)

    for name in keys(grid_mean)
        name in CLOUD_HYDROMETEORS && _slice_cloud_field!(
            getproperty(subcolumns, name),
            getproperty(grid_mean, name),
            cloud_mask,
            sampled_cloud_fraction,
        )
        name in PRECIP_HYDROMETEORS && _slice_precip_field!(
            getproperty(subcolumns, name),
            getproperty(grid_mean, name),
            precip_mask,
            sampled_precip_fraction,
        )
        name in CLOUD_HYDROMETEORS || name in PRECIP_HYDROMETEORS ||
            throw(ArgumentError("unknown hydrometeor field name: $name"))
    end

    return nothing
end

function _slice_cloud_field!(
    subcolumns::NTuple{N},
    grid_mean,
    cloud_mask,
    fraction,
) where {N}
    for isubcolumn in 1:N
        subcolumn = subcolumns[isubcolumn]
        mask = cloud_mask[isubcolumn]
        @. subcolumn = _sliced_cloud_value(grid_mean, fraction, mask)
    end
    return nothing
end

function _slice_precip_field!(
    subcolumns::NTuple{N},
    grid_mean,
    precip_mask,
    fraction,
) where {N}
    for isubcolumn in 1:N
        subcolumn = subcolumns[isubcolumn]
        mask = precip_mask[isubcolumn]
        @. subcolumn = _sliced_precip_value(grid_mean, fraction, mask)
    end
    return nothing
end

@inline _cloud_indicator(mask) =
    _is_cloudy(mask) ? one(mask) : zero(mask)

@inline _precip_indicator(mask) =
    _is_precipitating(mask) ? one(mask) : zero(mask)

# For now, all cloud hydrometeors use any cloudy subcolumn, and all precipitation
# hydrometeors use any precipitating subcolumn. Future large-scale and convective
# separation should branch on mask values 1, 2, and 3.
@inline _is_cloudy(mask) = mask == one(mask) || mask == 2 * one(mask)
@inline _is_precipitating(mask) =
    mask == one(mask) || mask == 2 * one(mask) || mask == 3 * one(mask)

@inline function _sliced_cloud_value(q, fraction, mask)
    return _is_cloudy(mask) && fraction > zero(fraction) ? q / fraction : zero(q)
end

@inline function _sliced_precip_value(q, fraction, mask)
    return _is_precipitating(mask) && fraction > zero(fraction) ? q / fraction : zero(q)
end

function _check_subcolumn_output_axes(subcolumns, grid_mean, nsubcolumns)
    for name in keys(grid_mean)
        output = getproperty(subcolumns, name)
        input = getproperty(grid_mean, name)
        length(output) == nsubcolumns ||
            throw(DimensionMismatch("$name must contain $nsubcolumns subcolumns"))
        _check_field_axes(output, input, "$name subcolumns")
    end
end

function _check_field_axes(fields, reference, name)
    for field in fields
        axes(field) == axes(reference) ||
            throw(DimensionMismatch("$name field must have matching axes"))
    end
end

end
