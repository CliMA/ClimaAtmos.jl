module COSPCloudSatCloudFraction

import ClimaCore: Operators
import LazyBroadcast: lazy

export cloudsat_cloud_fraction!,
    initialize_cloudsat_cloud_fraction!,
    accumulate_cloudsat_cloud_fraction!

initialize_cloudsat_cloud_fraction!(cloudsat_tcc) =
    (cloudsat_tcc .= zero(eltype(cloudsat_tcc)); nothing)

function accumulate_cloudsat_cloud_fraction!(
    cloudsat_tcc,
    detected_column_scratch,
    DBZe_subcolumn,
    contribution;
    detection_limit = -30.0,
    maximum_detection_limit = 10.0,
)
    FT = eltype(DBZe_subcolumn)
    typed_detection_limit = FT(detection_limit)
    typed_maximum_detection_limit = FT(maximum_detection_limit)
    _accumulate_cloudsat_tcc!(
        cloudsat_tcc,
        detected_column_scratch,
        DBZe_subcolumn,
        contribution,
        typed_detection_limit,
        typed_maximum_detection_limit,
    )
    return nothing
end

function accumulate_cloudsat_cloud_fraction!(
    cloudsat_tcc,
    cloudsat_tcc2,
    detected_column_scratch,
    DBZe_subcolumn,
    height_km,
    surface_height_km,
    contribution;
    detection_limit = -30.0,
    maximum_detection_limit = 10.0,
    excluded_near_surface_depth_km = 1.0,
)
    FT = eltype(DBZe_subcolumn)
    typed_detection_limit = FT(detection_limit)
    typed_maximum_detection_limit = FT(maximum_detection_limit)
    typed_excluded_depth_km = FT(excluded_near_surface_depth_km)
    _accumulate_cloudsat_tcc!(
        cloudsat_tcc,
        detected_column_scratch,
        DBZe_subcolumn,
        contribution,
        typed_detection_limit,
        typed_maximum_detection_limit,
    )
    minimum_height_km = @. lazy(surface_height_km + typed_excluded_depth_km)
    # COSPv2 omits approximately the lowest kilometre on its fixed statistical
    # grid. On the native model grid, apply that cutoff using geometric height
    # above the local surface.
    Operators.column_reduce!(
        max,
        detected_column_scratch,
        (DBZe_subcolumn .>= typed_detection_limit) .&
        (DBZe_subcolumn .<= typed_maximum_detection_limit) .&
        (height_km .> minimum_height_km);
        init = false,
    )
    @. cloudsat_tcc2 += contribution * detected_column_scratch
    return nothing
end

function _accumulate_cloudsat_tcc!(
    cloudsat_tcc,
    detected_column_scratch,
    DBZe_subcolumn,
    contribution,
    detection_limit,
    maximum_detection_limit,
)
    Operators.column_reduce!(
        max,
        detected_column_scratch,
        (DBZe_subcolumn .>= detection_limit) .&
        (DBZe_subcolumn .<= maximum_detection_limit);
        init = false,
    )
    @. cloudsat_tcc += contribution * detected_column_scratch
    return nothing
end

"""
    cloudsat_cloud_fraction!(
        cloudsat_tcc,
        detected_column_scratch,
        DBZe_cloudsat;
        detection_limit = -30.0,
        maximum_detection_limit = 10.0,
    )

Compute CloudSat total cloud cover in percent from attenuated radar
reflectivity. A subcolumn contributes once when at least one of its levels has
reflectivity between `detection_limit` and `maximum_detection_limit`,
inclusive.
"""
function cloudsat_cloud_fraction!(
    cloudsat_tcc,
    detected_column_scratch,
    DBZe_cloudsat::NTuple{N};
    detection_limit = -30.0,
    maximum_detection_limit = 10.0,
) where {N}
    N > 0 ||
        throw(
            ArgumentError(
                "CloudSat cloud fraction needs at least one subcolumn",
            ),
        )

    FT = eltype(DBZe_cloudsat[1])
    typed_detection_limit = FT(detection_limit)
    typed_maximum_detection_limit = FT(maximum_detection_limit)
    contribution = FT(100) / FT(N)

    initialize_cloudsat_cloud_fraction!(cloudsat_tcc)

    for DBZe_subcolumn in DBZe_cloudsat
        accumulate_cloudsat_cloud_fraction!(
            cloudsat_tcc,
            detected_column_scratch,
            DBZe_subcolumn,
            contribution;
            detection_limit = typed_detection_limit,
            maximum_detection_limit = typed_maximum_detection_limit,
        )
    end

    return nothing
end

end
