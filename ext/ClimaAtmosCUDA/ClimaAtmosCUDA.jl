module ClimaAtmosCUDA

import ClimaAtmos
import ClimaComms
import ClimaCore: DataLayouts, Fields
import ClimaCore.DataLayouts: column, level
import ClimaCore.Operators
using CUDA: @cuda, blockDim, blockIdx, threadIdx

const COSPSubcolumns = ClimaAtmos.COSPSubcolumns

@inline COSPSubcolumns._level_value(field::DataLayouts.DataColumn, ilev) =
    level(field, ilev)[]

@inline COSPSubcolumns._set_level_value!(
    field::DataLayouts.DataColumn,
    ilev,
    value,
) = (level(field, ilev)[] = value)

function COSPSubcolumns._scops_fields!(
    ::ClimaComms.CUDADevice,
    frac_out::NTuple{N},
    cloud_fraction,
    random_seed::UInt64,
    threshold::NTuple{N},
    overlap_code,
) where {N}
    parent_space = axes(cloud_fraction)
    cloud_data =
        Fields.field_values(Operators.strip_space(cloud_fraction, parent_space))
    frac_data = ntuple(
        i -> Fields.field_values(Operators.strip_space(frac_out[i], parent_space)),
        Val(N),
    )
    threshold_data = ntuple(
        i -> Fields.field_values(Operators.strip_space(threshold[i], parent_space)),
        Val(N),
    )

    us = DataLayouts.UniversalSize(cloud_data)
    (Ni, Nj, _, _, Nh) = DataLayouts.universal_size(us)
    ncolumns = Ni * Nj * Nh
    threads = min(256, ncolumns)
    blocks = cld(ncolumns, threads)

    @cuda always_inline = true threads = threads blocks = blocks scops_kernel!(
        frac_data,
        cloud_data,
        random_seed,
        threshold_data,
        overlap_code,
        us,
    )

    return nothing
end

function scops_kernel!(
    frac_out_data::NTuple{N},
    cloud_fraction_data,
    random_seed::UInt64,
    threshold_data::NTuple{N},
    overlap_code,
    us::DataLayouts.UniversalSize,
) where {N}
    tidx =
        Int(threadIdx().x) + Int(blockDim().x) * (Int(blockIdx().x) - 1)
    (Ni, Nj, _, nlev, Nh) = DataLayouts.universal_size(us)
    ncolumns = Ni * Nj * Nh

    @inbounds if tidx <= ncolumns
        linear_index = tidx - 1
        i = mod(linear_index, Ni) + 1
        jh = fld(linear_index, Ni)
        j = mod(jh, Nj) + 1
        h = fld(jh, Nj) + 1

        cloud_column = column(cloud_fraction_data, i, j, h)
        frac_columns =
            ntuple(n -> column(frac_out_data[n], i, j, h), Val(N))
        threshold_columns =
            ntuple(n -> column(threshold_data[n], i, j, h), Val(N))

        COSPSubcolumns._scops_field_column!(
            frac_columns,
            cloud_column,
            random_seed,
            threshold_columns,
            overlap_code,
            nlev,
            tidx,
        )
    end

    return nothing
end

end # module ClimaAtmosCUDA
