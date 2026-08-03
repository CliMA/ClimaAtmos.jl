module COSPCloudSatCFAD

export cloudsat_cfad_bin_edges,
    cloudsat_cfad_bin_centers,
    initialize_cloudsat_cfad!,
    accumulate_cloudsat_cfad!

const CLOUDSAT_CFAD_BIN_EDGES =
    (-100, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 80)

cloudsat_cfad_bin_edges(::Type{FT}) where {FT} =
    map(FT, CLOUDSAT_CFAD_BIN_EDGES)

function cloudsat_cfad_bin_centers(::Type{FT}) where {FT}
    edges = cloudsat_cfad_bin_edges(FT)
    return ntuple(
        index -> (edges[index] + edges[index + 1]) / FT(2),
        length(edges) - 1,
    )
end

function initialize_cloudsat_cfad!(cfad)
    for cfad_bin in cfad
        cfad_bin .= zero(eltype(cfad_bin))
    end
    return nothing
end

"""
    accumulate_cloudsat_cfad!(cfad, DBZe, bin_edges, contribution)

Accumulate one streamed CloudSat subcolumn into the reflectivity CFAD on the
model vertical grid. Bins follow COSPv2 `hist1D`: lower edges are inclusive,
upper edges are exclusive, and each populated bin is normalized by the total
number of subcolumns through `contribution`.
"""
function accumulate_cloudsat_cfad!(
    cfad::NTuple{N},
    DBZe,
    bin_edges::NTuple{NPlusOne},
    contribution,
) where {N, NPlusOne}
    NPlusOne == N + 1 ||
        throw(ArgumentError("CloudSat CFAD needs one more edge than bins"))

    no_contribution = zero(contribution)
    for index in 1:N
        cfad_bin = cfad[index]
        lower_edge = bin_edges[index]
        upper_edge = bin_edges[index + 1]
        @. cfad_bin +=
            ifelse(
                (DBZe >= lower_edge) & (DBZe < upper_edge),
                contribution,
                no_contribution,
            )
    end
    return nothing
end

end
