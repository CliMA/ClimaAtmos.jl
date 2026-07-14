module COSPHydrometeorSubcolumns

export accumulate_sampled_cloud_fraction!,
    accumulate_sampled_precip_fraction!,
    slice_hydrometeor_subcolumn!

const CLOUD_HYDROMETEORS = (:q_lcl, :q_icl)
const PRECIP_HYDROMETEORS = (:q_rai, :q_sno)

function accumulate_sampled_cloud_fraction!(sampled_fraction, cloud_mask, nsubcolumns)
    FT = eltype(sampled_fraction)
    @. sampled_fraction += _cloud_indicator(cloud_mask) / FT(nsubcolumns)
    return nothing
end

function accumulate_sampled_precip_fraction!(sampled_fraction, precip_mask, nsubcolumns)
    FT = eltype(sampled_fraction)
    @. sampled_fraction += _precip_indicator(precip_mask) / FT(nsubcolumns)
    return nothing
end

"""
Populate one stored hydrometeor subcolumn from streamed mask fields.
"""
function slice_hydrometeor_subcolumn!(
    subcolumn::NamedTuple,
    cloud_mask,
    precip_mask,
    grid_mean::NamedTuple,
    sampled_cloud_fraction,
    sampled_precip_fraction,
)
    keys(subcolumn) == keys(grid_mean) ||
        throw(ArgumentError("subcolumn and grid_mean must have matching keys"))
    for name in keys(grid_mean)
        output = getproperty(subcolumn, name)
        input = getproperty(grid_mean, name)
        name in CLOUD_HYDROMETEORS &&
            (@. output = _sliced_cloud_value(input, sampled_cloud_fraction, cloud_mask))
        name in PRECIP_HYDROMETEORS &&
            (@. output = _sliced_precip_value(input, sampled_precip_fraction, precip_mask))
        name in CLOUD_HYDROMETEORS || name in PRECIP_HYDROMETEORS ||
            throw(ArgumentError("unknown hydrometeor field name: $name"))
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
    return _sliced_mask_value(q, fraction, _is_cloudy(mask))
end

@inline function _sliced_precip_value(q, fraction, mask)
    return _sliced_mask_value(q, fraction, _is_precipitating(mask))
end

@inline function _sliced_mask_value(q, fraction, is_selected)
    if fraction > zero(fraction)
        return is_selected ? q / fraction : zero(q)
    else
        return q > zero(q) ? q : zero(q)
    end
end

end
