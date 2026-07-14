module COSPPrecipSubcolumns

import ClimaCore: Operators
import ..COSPSubcolumns

export scops_subcolumn_precip!

"""
Generate one large-scale precipitation subcolumn from shared selectors.
"""
function scops_subcolumn_precip!(precip_subcol, cloud_s, flux, selectors, scratch)
    _check_axes(precip_subcol, cloud_s, "cloud_s")
    _check_axes(precip_subcol, flux, "flux")
    for field in values(selectors)
        _check_axes(precip_subcol, field, "selector")
    end
    for field in values(scratch)
        _check_axes(precip_subcol, field, "scratch")
    end
    FT = eltype(precip_subcol)
    cloud_one = one(eltype(cloud_s))
    output_one = one(FT)
    output_zero = zero(FT)

    @. scratch.cloud = ifelse(cloud_s == cloud_one, output_one, output_zero)
    COSPSubcolumns.shift_up!(scratch.cloud_below, scratch.cloud)
    COSPSubcolumns.column_any!(
        scratch.any_cloud,
        scratch.cloud,
        scratch.column_any,
    )

    input = Base.broadcasted(
        tuple,
        scratch.cloud,
        scratch.cloud_below,
        selectors.has_cloud,
        selectors.has_cloud_below,
        selectors.has_cloud_anywhere,
        flux,
        scratch.any_cloud,
    )
    Operators.column_accumulate!(
        precip_subcol,
        input;
        init = (; precip = zero(FT), flux_above = zero(FT)),
        transform = state -> state.precip,
        reverse = true,
    ) do state, (c, c_below, hc, hc_below, hc_any, fx, any_s)
        if !(fx > zero(FT))
            return (; precip = zero(FT), flux_above = fx)
        end
        primary_rule_active = (hc > zero(FT)) | (state.flux_above > zero(FT))
        use_cloud_below = (!primary_rule_active) & (hc_below > zero(FT))
        precip =
            primary_rule_active ? max(c, state.precip) :
            use_cloud_below ? c_below :
            (hc_any > zero(FT)) ? any_s : one(FT)
        return (; precip, flux_above = fx)
    end
    return nothing
end

function _check_axes(reference, field, name)
    axes(field) == axes(reference) ||
        throw(DimensionMismatch("$name must have matching axes"))
end

end
