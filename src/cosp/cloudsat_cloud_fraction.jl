module COSPCloudSatCloudFraction

import ClimaCore: Operators

export cloudsat_cloud_fraction!

"""
    cloudsat_cloud_fraction!(
        cloudsat_tcc,
        detected_column_scratch,
        DBZe_cloudsat;
        detection_limit = -30.0,
    )

Compute CloudSat total cloud cover in percent from attenuated radar
reflectivity. A subcolumn contributes once when at least one of its levels has
reflectivity greater than or equal to `detection_limit`.
"""
function cloudsat_cloud_fraction!(
    cloudsat_tcc,
    detected_column_scratch,
    DBZe_cloudsat::NTuple{N};
    detection_limit = -30.0,
) where {N}
    N > 0 ||
        throw(
            ArgumentError(
                "CloudSat cloud fraction needs at least one subcolumn",
            ),
        )

    FT = eltype(DBZe_cloudsat[1])
    typed_detection_limit = FT(detection_limit)
    contribution = FT(100) / FT(N)

    cloudsat_tcc .= zero(eltype(cloudsat_tcc))

    for DBZe_subcolumn in DBZe_cloudsat
        Operators.column_reduce!(
            max,
            detected_column_scratch,
            DBZe_subcolumn .>= typed_detection_limit;
            init = false,
        )
        @. cloudsat_tcc += contribution * detected_column_scratch
    end

    return nothing
end

end
