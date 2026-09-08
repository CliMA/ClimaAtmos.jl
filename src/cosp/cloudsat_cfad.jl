module COSPCloudSatCFAD

import ClimaCore: Fields

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

@inline _zero_cfad(current::T) where {T <: Tuple} =
    ntuple(index -> zero(current[index]), Val(fieldcount(T)))

function initialize_cloudsat_cfad!(cfad::Fields.Field)
    @. cfad = _zero_cfad(cfad)
    return nothing
end

struct CFADAccumulator{E, C}
    bin_edges::E
    contribution::C
end

@inline function (accumulator::CFADAccumulator)(
    current::T,
    dbze,
) where {T <: Tuple}
    return ntuple(Val(fieldcount(T))) do index
        lower_edge = accumulator.bin_edges[index]
        upper_edge = accumulator.bin_edges[index + 1]
        increment = ifelse(
            (dbze >= lower_edge) & (dbze < upper_edge),
            accumulator.contribution,
            zero(accumulator.contribution),
        )
        current[index] + increment
    end
end

"""
    accumulate_cloudsat_cfad!(cfad, DBZe, bin_edges, contribution)

Accumulate one streamed CloudSat subcolumn into the reflectivity CFAD on the
model vertical grid. Bins follow COSPv2 `hist1D`: lower edges are inclusive,
upper edges are exclusive, and each populated bin is normalized by the total
number of subcolumns through `contribution`.
"""
function accumulate_cloudsat_cfad!(
    cfad::Fields.Field,
    DBZe,
    bin_edges::NTuple{NPlusOne},
    contribution,
) where {NPlusOne}
    n_bins = fieldcount(eltype(cfad))
    NPlusOne == n_bins + 1 ||
        throw(ArgumentError("CloudSat CFAD needs one more edge than bins"))

    accumulator = CFADAccumulator(bin_edges, contribution)
    @. cfad = accumulator(cfad, DBZe)
    return nothing
end

end
