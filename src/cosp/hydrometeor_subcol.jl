module COSPHydrometeorSubcolumns

import LazyBroadcast: lazy

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
The returned broadcasts must be consumed before either mask or the grid-mean
scratch fields are overwritten.
"""
function lazy_hydrometeor_subcolumn(
    grid_mean,
    cloud_mask,
    precip_mask,
    sampled_cloud_fraction,
    sampled_precip_fraction,
)
    return (;
        q_lcl = lazy.(
            _sliced_cloud_value.(
                grid_mean.q_lcl,
                sampled_cloud_fraction,
                cloud_mask,
            ),
        ),
        q_icl = lazy.(
            _sliced_cloud_value.(
                grid_mean.q_icl,
                sampled_cloud_fraction,
                cloud_mask,
            ),
        ),
        q_rai = lazy.(
            _sliced_precip_value.(
                grid_mean.q_rai,
                sampled_precip_fraction,
                precip_mask,
            ),
        ),
        q_sno = lazy.(
            _sliced_precip_value.(
                grid_mean.q_sno,
                sampled_precip_fraction,
                precip_mask,
            ),
        ),
    )
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
        return zero(q)
    end
end

end
