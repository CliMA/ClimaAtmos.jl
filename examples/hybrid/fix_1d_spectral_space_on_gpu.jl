import ClimaCore:
    ClimaCore,
    Utilities,
    RecursiveApply,
    DataLayouts,
    Geometry,
    Meshes,
    Topologies,
    Quadratures,
    Grids,
    Spaces,
    Fields,
    Operators,
    Hypsography,
    Remapping,
    slab
import ClimaCore.RecursiveApply: ⊞, ⊟, ⊠
import ClimaComms
import CUDA
import Adapt
import StaticArrays: SMatrix

ClimaCoreCUDAExt = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)

##########
## ClimaCore.jl/ext/cuda/data_layouts.jl
##########

function Base.copyto!(
    dest::DataLayouts.IFH{S, Ni},
    bc::DataLayouts.BroadcastedUnionIFH{S, Ni},
    ::ClimaCoreCUDAExt.ToCUDA,
) where {S, Ni}
    _, _, _, _, Nh = size(bc)
    if Nh > 0
        ClimaCoreCUDAExt.auto_launch!(
            ClimaCoreCUDAExt.knl_copyto!,
            (dest, bc),
            dest;
            threads_s = (Ni, 1),
            blocks_s = (Nh, 1),
        )
    end
    return dest
end

function Base.copyto!(
    dest::DataLayouts.VIFH{S, Nv, Ni},
    bc::DataLayouts.BroadcastedUnionVIFH{S, Nv, Ni},
    ::ClimaCoreCUDAExt.ToCUDA,
) where {S, Nv, Ni}
    _, _, _, _, Nh = size(bc)
    if Nv > 0 && Nh > 0
        Nv_per_block = min(Nv, fld(256, Ni))
        Nv_blocks = cld(Nv, Nv_per_block)
        ClimaCoreCUDAExt.auto_launch!(
            ClimaCoreCUDAExt.knl_copyto!,
            (dest, bc),
            dest;
            threads_s = (Ni, 1, Nv_per_block),
            blocks_s = (Nh, Nv_blocks),
        )
    end
    return dest
end

##########
## ClimaCore.jl/ext/cuda/fields.jl
##########

function ClimaCoreCUDAExt.mapreduce_cuda(
    f,
    op,
    field::Fields.Field{V};
    weighting = false,
    opargs...,
) where {
    S,
    V <: Union{
        DataLayouts.VF{S},
        DataLayouts.IFH{S},
        DataLayouts.IJFH{S},
        DataLayouts.VIFH{S},
        DataLayouts.VIJFH{S},
    },
}
    data = Fields.field_values(field)
    pdata = parent(data)
    T = eltype(pdata)
    (Ni, Nj, Nk, Nv, Nh) = size(data)
    Nf = div(length(pdata), prod(size(data))) # length of field dimension
    wt = Spaces.weighted_jacobian(axes(field))
    pwt = parent(wt)

    nitems = Nv * Ni * Nj * Nk * Nh
    max_threads = 256# 512 1024
    nthreads = min(max_threads, nitems)
    # perform n ops during loading to shmem (this is a tunable parameter)
    n_ops_on_load = cld(nitems, nthreads) == 1 ? 0 : 7
    effective_blksize = nthreads * (n_ops_on_load + 1)
    nblocks = cld(nitems, effective_blksize)

    reduce_cuda = CUDA.CuArray{T}(undef, nblocks, Nf)
    shmemsize = nthreads
    # place each field on a different block
    CUDA.@cuda always_inline = true threads = (nthreads) blocks = (nblocks, Nf) ClimaCoreCUDAExt.mapreduce_cuda_kernel!(
        reduce_cuda,
        f,
        op,
        pdata,
        pwt,
        weighting,
        n_ops_on_load,
        Val(shmemsize),
        nitems * Nf,
        ClimaCoreCUDAExt._dataview,
        ClimaCoreCUDAExt._get_gidx,
        ClimaCoreCUDAExt._cuda_intrablock_reduce!,
    )
    # reduce block data
    if nblocks > 1
        nthreads = min(32, nblocks)
        shmemsize = nthreads
        CUDA.@cuda always_inline = true threads = (nthreads) blocks = (Nf) ClimaCoreCUDAExt.reduce_cuda_blocks_kernel!(
            reduce_cuda,
            op,
            Val(shmemsize),
        )
    end
    return DataLayouts.DataF{S}(Array(Array(reduce_cuda)[1, :]))
end

function ClimaCoreCUDAExt.mapreduce_cuda_kernel!(
    reduce_cuda::AbstractArray{T, 2},
    f,
    op,
    pdata::AbstractArray{T, N},
    pwt::AbstractArray{T, N},
    weighting::Bool,
    n_ops_on_load::Int,
    ::Val{shmemsize},
    nitems,
    _dataview,
    _get_gidx,
    _cuda_intrablock_reduce!,
) where {T, N, shmemsize}
    blksize = CUDA.blockDim().x
    nblk = CUDA.gridDim().x
    tidx = CUDA.threadIdx().x
    bidx = CUDA.blockIdx().x
    fidx = CUDA.blockIdx().y
    dataview = _dataview(pdata, fidx)
    effective_blksize = blksize * (n_ops_on_load + 1)
    gidx = _get_gidx(tidx, bidx, effective_blksize)
    reduction = CUDA.CuStaticSharedArray(T, shmemsize)
    reduction[tidx] = 0

    # load shmem
    if gidx ≤ nitems
        if weighting
            reduction[tidx] = f(dataview[gidx]) * pwt[gidx]
            for n_ops in 1:n_ops_on_load
                gidx2 =
                    _get_gidx(tidx + blksize * n_ops, bidx, effective_blksize)
                if gidx2 ≤ nitems
                    reduction[tidx] =
                        op(reduction[tidx], f(dataview[gidx2]) * pwt[gidx2])
                end
            end
        else
            reduction[tidx] = f(dataview[gidx])
            for n_ops in 1:n_ops_on_load
                gidx2 =
                    _get_gidx(tidx + blksize * n_ops, bidx, effective_blksize)
                if gidx2 ≤ nitems
                    reduction[tidx] = op(reduction[tidx], f(dataview[gidx2]))
                end
            end
        end
    end
    CUDA.sync_threads()
    _cuda_intrablock_reduce!(op, reduction, tidx, blksize)

    tidx == 1 && (reduce_cuda[bidx, fidx] = reduction[1])
    return nothing
end

@inline ClimaCoreCUDAExt._dataview(
    pdata::AbstractArray{FT, 3},
    fidx,
) where {FT} = view(pdata, :, fidx:fidx, :)

##########
## ClimaCore.jl/ext/cuda/remapping_distributed.jl
##########

function ClimaCoreCUDAExt.set_interpolated_values_kernel!(
    out::AbstractArray,
    (I,)::NTuple{1},
    local_horiz_indices,
    vert_interpolation_weights,
    vert_bounding_indices,
    field_values,
)
    # TODO: Check the memory access pattern. This was not optimized and likely inefficient!
    num_horiz = length(local_horiz_indices)
    num_vert = length(vert_bounding_indices)
    num_fields = length(field_values)

    hindex =
        (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
    vindex =
        (CUDA.blockIdx().y - Int32(1)) * CUDA.blockDim().y + CUDA.threadIdx().y
    findex =
        (CUDA.blockIdx().z - Int32(1)) * CUDA.blockDim().z + CUDA.threadIdx().z

    totalThreadsX = CUDA.gridDim().x * CUDA.blockDim().x
    totalThreadsY = CUDA.gridDim().y * CUDA.blockDim().y
    totalThreadsZ = CUDA.gridDim().z * CUDA.blockDim().z

    _, Nq = size(I)

    for i in hindex:totalThreadsX:num_horiz
        h = local_horiz_indices[i]
        for j in vindex:totalThreadsY:num_vert
            v_lo, v_hi = vert_bounding_indices[j]
            A, B = vert_interpolation_weights[j]
            for k in findex:totalThreadsZ:num_fields
                if i ≤ num_horiz && j ≤ num_vert && k ≤ num_fields
                    out[i, j, k] = 0
                    for t in 1:Nq
                        out[i, j, k] +=
                            I[i, t] * (
                                A *
                                field_values[k][t, nothing, nothing, v_lo, h] +
                                B *
                                field_values[k][t, nothing, nothing, v_hi, h]
                            )
                    end
                end
            end
        end
    end
    return nothing
end

##########
## ClimaCore.jl/src/Remapping/distributed_remapping.jl
##########

function Remapping._set_interpolated_values!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    _scratch_field_values,
    local_horiz_indices,
    local_horiz_interpolation_weights,
    ::Nothing,
    ::Nothing,
)
    CUDA.@allowscalar Remapping._set_interpolated_values_device!(
        out,
        fields,
        _scratch_field_values,
        local_horiz_indices,
        local_horiz_interpolation_weights,
        nothing,
        nothing,
        ClimaComms.CPUSingleThreaded(),
    ) # TODO: Fix the kernel function above and remove the @allowscalar.
end

##########
## ClimaCore.jl/ext/cuda/operators_sem_shmem.jl
##########

Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_shmem(
    space,
    ::Val{Nvt},
    op::Operators.Divergence{(1,)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    RT = Operators.operator_return_eltype(op, eltype(arg))
    Jv¹ = CUDA.CuStaticSharedArray(RT, (Nq, Nvt))
    return (Jv¹,)
end
Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_fill_shmem!(
    op::Operators.Divergence{(1,)},
    (Jv¹,),
    space,
    ij,
    slabidx,
    arg,
)
    vt = CUDA.threadIdx().z
    local_geometry = Operators.get_local_geometry(space, ij, slabidx)
    i, _ = ij.I
    Jv¹[i, vt] =
        local_geometry.J ⊠ RecursiveApply.rmap(
            v -> Geometry.contravariant1(v, local_geometry),
            arg,
        )
end

Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_shmem(
    space,
    ::Val{Nvt},
    op::Operators.WeakDivergence{(1,)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    RT = Operators.operator_return_eltype(op, eltype(arg))
    Nf = DataLayouts.typesize(FT, RT)
    WJv¹ = CUDA.CuStaticSharedArray(RT, (Nq, Nvt))
    return (WJv¹,)
end
Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_fill_shmem!(
    op::Operators.WeakDivergence{(1,)},
    (WJv¹,),
    space,
    ij,
    slabidx,
    arg,
)
    vt = CUDA.threadIdx().z
    local_geometry = Operators.get_local_geometry(space, ij, slabidx)
    i, _ = ij.I
    WJv¹[i, vt] =
        local_geometry.WJ ⊠ RecursiveApply.rmap(
            v -> Geometry.contravariant1(v, local_geometry),
            arg,
        )
end

Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_shmem(
    space,
    ::Val{Nvt},
    op::Operators.Gradient{(1,)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    IT = eltype(arg)
    input = CUDA.CuStaticSharedArray(IT, (Nq, Nvt))
    return (input,)
end
Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_fill_shmem!(
    op::Operators.Gradient{(1,)},
    (input,),
    space,
    ij,
    slabidx,
    arg,
)
    vt = CUDA.threadIdx().z
    i, _ = ij.I
    input[i, vt] = arg
end

Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_shmem(
    space,
    ::Val{Nvt},
    op::Operators.WeakGradient{(1,)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    # allocate temp output
    IT = eltype(arg)
    Wf = CUDA.CuStaticSharedArray(IT, (Nq, Nvt))
    return (Wf,)
end
Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_fill_shmem!(
    op::Operators.WeakGradient{(1,)},
    (Wf,),
    space,
    ij,
    slabidx,
    arg,
)
    vt = CUDA.threadIdx().z
    local_geometry = Operators.get_local_geometry(space, ij, slabidx)
    W = local_geometry.WJ * local_geometry.invJ
    i, _ = ij.I
    Wf[i, vt] = W ⊠ arg
end

Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_shmem(
    space,
    ::Val{Nvt},
    op::Operators.Curl{(1,)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    IT = eltype(arg)
    ET = eltype(IT)
    RT = Operators.operator_return_eltype(op, IT)
    # allocate temp output
    if RT <: Geometry.Contravariant3Vector
        v₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        return (nothing, v₂)
    elseif RT <: Geometry.Contravariant2Vector
        v₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        return (v₃,)
    elseif RT <: Geometry.Contravariant23Vector
        v₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        v₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        return (nothing, v₂, v₃)
    else
        error("invalid return type")
    end
end
Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_fill_shmem!(
    op::Operators.Curl{(1,)},
    work,
    space,
    ij,
    slabidx,
    arg,
)
    vt = CUDA.threadIdx().z
    i, _ = ij.I
    local_geometry = Operators.get_local_geometry(space, ij, slabidx)
    RT = Operators.operator_return_eltype(op, typeof(arg))
    if RT <: Geometry.Contravariant3Vector
        _, v₂ = work
        v₂[i, vt] = Geometry.covariant2(arg, local_geometry)
    elseif RT <: Geometry.Contravariant2Vector
        (v₃,) = work
        v₃[i, vt] = Geometry.covariant3(arg, local_geometry)
    else
        _, v₂, v₃ = work
        v₂[i, vt] = Geometry.covariant2(arg, local_geometry)
        v₃[i, vt] = Geometry.covariant3(arg, local_geometry)
    end
end

Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_shmem(
    space,
    ::Val{Nvt},
    op::Operators.WeakCurl{(1,)},
    arg,
) where {Nvt}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    IT = eltype(arg)
    ET = eltype(IT)
    RT = Operators.operator_return_eltype(op, IT)
    # allocate temp output
    if RT <: Geometry.Contravariant3Vector
        Wv₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        return (nothing, Wv₂)
    elseif RT <: Geometry.Contravariant2Vector
        Wv₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        return (Wv₃,)
    elseif RT <: Geometry.Contravariant23Vector
        Wv₂ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        Wv₃ = CUDA.CuStaticSharedArray(ET, (Nq, Nvt))
        return (nothing, Wv₂, Wv₃)
    else
        error("invalid return type")
    end
end
Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_fill_shmem!(
    op::Operators.WeakCurl{(1,)},
    work,
    space,
    ij,
    slabidx,
    arg,
)
    vt = CUDA.threadIdx().z
    i, _ = ij.I
    local_geometry = Operators.get_local_geometry(space, ij, slabidx)
    W = local_geometry.WJ * local_geometry.invJ
    RT = Operators.operator_return_eltype(op, typeof(arg))
    if RT <: Geometry.Contravariant3Vector
        _, Wv₂ = work
        Wv₂[i, vt] = W ⊠ Geometry.covariant2(arg, local_geometry)
    elseif RT <: Geometry.Contravariant2Vector
        (Wv₃,) = work
        Wv₃[i, vt] = W ⊠ Geometry.covariant3(arg, local_geometry)
    else
        _, Wv₂, Wv₃ = work
        Wv₂[i, vt] = W ⊠ Geometry.covariant2(arg, local_geometry)
        Wv₃[i, vt] = W ⊠ Geometry.covariant3(arg, local_geometry)
    end
end

##########
## ClimaCore.jl/ext/cuda/operators_spectral_element.jl
##########

function Base.copyto!(
    out::Fields.Field,
    sbc::Union{
        Operators.SpectralBroadcasted{ClimaCoreCUDAExt.CUDASpectralStyle},
        Base.Broadcast.Broadcasted{ClimaCoreCUDAExt.CUDASpectralStyle},
    },
)
    space = axes(out)
    (Ni, Nj, _, Nv, Nh) = size(Fields.field_values(out))
    max_threads = 256
    @assert Ni * Nj ≤ max_threads
    Nvthreads = fld(max_threads, Ni * Nj)
    Nvblocks = cld(Nv, Nvthreads)
    # executed
    args = (
        Operators.strip_space(out, space),
        Operators.strip_space(sbc, space),
        space,
        Val(Nvthreads),
    )
    ClimaCoreCUDAExt.auto_launch!(
        ClimaCoreCUDAExt.copyto_spectral_kernel!,
        args,
        out;
        threads_s = (Ni, Nj, Nvthreads),
        blocks_s = (Nh, Nvblocks),
    )
    return out
end

Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_evaluate(
    op::Operators.Divergence{(1,)},
    (Jv¹,),
    space,
    ij,
    slabidx,
)
    vt = CUDA.threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = Operators.get_local_geometry(space, ij, slabidx)

    DJv = D[i, 1] ⊠ Jv¹[1, vt]
    for k in 2:Nq
        DJv = DJv ⊞ D[i, k] ⊠ Jv¹[k, vt]
    end
    return RecursiveApply.rmul(DJv, local_geometry.invJ)
end

Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_evaluate(
    op::Operators.WeakDivergence{(1,)},
    (WJv¹,),
    space,
    ij,
    slabidx,
)
    vt = CUDA.threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = Operators.get_local_geometry(space, ij, slabidx)

    Dᵀ₁WJv¹ = D[1, i] ⊠ WJv¹[1, vt]
    for k in 2:Nq
        Dᵀ₁WJv¹ = Dᵀ₁WJv¹ ⊞ D[k, i] ⊠ WJv¹[k, vt]
    end
    return ⊟(RecursiveApply.rdiv(Dᵀ₁WJv¹, local_geometry.WJ))
end

Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_evaluate(
    op::Operators.Gradient{(1,)},
    (input,),
    space,
    ij,
    slabidx,
)
    vt = CUDA.threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    ∂f∂ξ₁ = D[i, 1] * input[1, vt]
    for k in 2:Nq
        ∂f∂ξ₁ += D[i, k] * input[k, vt]
    end
    return Geometry.Covariant1Vector(∂f∂ξ₁)
end

Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_evaluate(
    op::Operators.WeakGradient{(1,)},
    (Wf,),
    space,
    ij,
    slabidx,
)
    vt = CUDA.threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)

    local_geometry = Operators.get_local_geometry(space, ij, slabidx)
    W = local_geometry.WJ * local_geometry.invJ

    Dᵀ₁Wf = D[1, i] ⊠ Wf[1, vt]
    for k in 2:Nq
        Dᵀ₁Wf = Dᵀ₁Wf ⊞ D[k, i] ⊠ Wf[k, vt]
    end
    return Geometry.Covariant1Vector(⊟(RecursiveApply.rdiv(Dᵀ₁Wf, W)))
end

Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_evaluate(
    op::Operators.Curl{(1,)},
    work,
    space,
    ij,
    slabidx,
)
    vt = CUDA.threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    local_geometry = Operators.get_local_geometry(space, ij, slabidx)

    if length(work) == 2
        _, v₂ = work
        D₁v₂ = D[i, 1] ⊠ v₂[1, vt]
        for k in 2:Nq
            D₁v₂ = D₁v₂ ⊞ D[i, k] ⊠ v₂[k, vt]
        end
        return Geometry.Contravariant3Vector(
            RecursiveApply.rmul(D₁v₂, local_geometry.invJ),
        )
    elseif length(work) == 1
        (v₃,) = work
        D₁v₃ = D[i, 1] ⊠ v₃[1, vt]
        for k in 2:Nq
            D₁v₃ = D₁v₃ ⊞ D[i, k] ⊠ v₃[k, vt]
        end
        return Geometry.Contravariant2Vector(
            ⊟(RecursiveApply.rmul(D₁v₃, local_geometry.invJ)),
        )
    else
        _, v₂, v₃ = work
        D₁v₂ = D[i, 1] ⊠ v₂[1, vt]
        D₁v₃ = D[i, 1] ⊠ v₃[1, vt]
        @simd for k in 2:Nq
            D₁v₂ = D₁v₂ ⊞ D[i, k] ⊠ v₂[k, vt]
            D₁v₃ = D₁v₃ ⊞ D[i, k] ⊠ v₃[k, vt]
        end
        return Geometry.Contravariant23Vector(
            ⊟(RecursiveApply.rmul(D₁v₃, local_geometry.invJ)),
            RecursiveApply.rmul(D₁v₂, local_geometry.invJ),
        )
    end
end

Base.@propagate_inbounds function ClimaCoreCUDAExt.operator_evaluate(
    op::Operators.WeakCurl{(1,)},
    work,
    space,
    ij,
    slabidx,
)
    vt = CUDA.threadIdx().z
    i, _ = ij.I

    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    local_geometry = Operators.get_local_geometry(space, ij, slabidx)

    if length(work) == 2
        _, Wv₂ = work
        Dᵀ₁Wv₂ = D[1, i] ⊠ Wv₂[1, vt]
        for k in 2:Nq
            Dᵀ₁Wv₂ = Dᵀ₁Wv₂ ⊞ D[k, i] ⊠ Wv₂[k, vt]
        end
        return Geometry.Contravariant3Vector(
            RecursiveApply.rdiv(⊟(Dᵀ₁Wv₂), local_geometry.WJ),
        )
    elseif length(work) == 1
        (Wv₃,) = work
        Dᵀ₁Wv₃ = D[1, i] ⊠ Wv₃[1, vt]
        for k in 2:Nq
            Dᵀ₁Wv₃ = Dᵀ₁Wv₃ ⊞ D[k, i] ⊠ Wv₃[k, vt]
        end
        return Geometry.Contravariant2Vector(
            RecursiveApply.rdiv(Dᵀ₁Wv₃, local_geometry.WJ),
        )
    else
        _, Wv₂, Wv₃ = work
        Dᵀ₁Wv₂ = D[1, i] ⊠ Wv₂[1, vt]
        Dᵀ₁Wv₃ = D[1, i] ⊠ Wv₃[1, vt]
        @simd for k in 2:Nq
            Dᵀ₁Wv₂ = Dᵀ₁Wv₂ ⊞ D[k, i] ⊠ Wv₂[k, vt]
            Dᵀ₁Wv₃ = Dᵀ₁Wv₃ ⊞ D[k, i] ⊠ Wv₃[k, vt]
        end
        return Geometry.Contravariant23Vector(
            RecursiveApply.rdiv(Dᵀ₁Wv₃, local_geometry.WJ),
            RecursiveApply.rdiv(⊟(Dᵀ₁Wv₂), local_geometry.WJ),
        )
    end
end

##########
## ClimaCore.jl/ext/cuda/operators_finite_difference.jl
##########

function Base.copyto!(
    out::Fields.Field,
    bc::Union{
        Operators.StencilBroadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle},
        Base.Broadcast.Broadcasted{ClimaCoreCUDAExt.CUDAColumnStencilStyle},
    },
)
    space = axes(out)
    (Ni, Nj, _, Nv, Nh) = size(Fields.field_values(out))
    (li, lw, rw, ri) = bounds = Operators.window_bounds(space, bc)
    @assert Nv == ri - li + 1
    max_threads = 256
    nitems = Nv * Ni * Nj * Nh # # of independent items
    (nthreads, nblocks) =
        ClimaCoreCUDAExt._configure_threadblock(max_threads, nitems)
    args = (
        Operators.strip_space(out, space),
        Operators.strip_space(bc, space),
        axes(out),
        bounds,
        Ni,
        Nj,
        Nh,
        Nv,
    )
    ClimaCoreCUDAExt.auto_launch!(
        ClimaCoreCUDAExt.copyto_stencil_kernel!,
        args,
        out;
        threads_s = (nthreads,),
        blocks_s = (nblocks,),
    )
    return out
end

function ClimaCoreCUDAExt.copyto_stencil_kernel!(
    out,
    bc,
    space,
    bds,
    Ni,
    Nj,
    Nh,
    Nv,
)
    gid = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if gid ≤ Nv * Ni * Nj * Nh
        (li, lw, rw, ri) = bds
        (v, i, j, h) = Utilities.cart_ind((Nv, Ni, Nj, Nh), gid).I
        hidx = (i, j, h)
        idx = v - 1 + li
        window =
            idx < lw ?
            Operators.LeftBoundaryWindow{Spaces.left_boundary_name(space)}() :
            (
                idx > rw ?
                Operators.RightBoundaryWindow{
                    Spaces.right_boundary_name(space),
                }() : Operators.Interior()
            )
        Operators.setidx!(
            space,
            out,
            idx,
            hidx,
            Operators.getidx(space, bc, window, idx, hidx),
        )
    end
    return nothing
end

##########
## ClimaCore.jl/ext/cuda/topologies_dss.jl
##########

function Topologies.dss_1d!(
    ::ClimaComms.CUDADevice,
    data::Union{DataLayouts.VIFH, DataLayouts.IFH},
    topology::Topologies.IntervalTopology,
    lg = nothing,
    weight = nothing,
)
    (_, _, _, Nv, Nh) = size(data)
    Nfaces = Topologies.isperiodic(topology) ? Nh : Nh - 1
    if Nfaces > 0
        nthreads, nblocks = ClimaCoreCUDAExt._configure_threadblock(Nv * Nfaces)
        ClimaCoreCUDAExt.auto_launch!(
            dss_1d_kernel!,
            (data, lg, weight, Nfaces),
            data;
            threads_s = nthreads,
            blocks_s = nblocks,
        )
    end
    return nothing
end

function dss_1d_kernel!(data, lg, weight, Nfaces)
    T = eltype(data)
    (Ni, _, _, Nv, Nh) = size(data)
    gidx = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if gidx ≤ Nv * Nfaces
        left_face_elem = cld(gidx, Nv)
        level = gidx - (left_face_elem - 1) * Nv
        right_face_elem = (left_face_elem % Nh) + 1
        left_idx = CartesianIndex(Ni, 1, 1, level, left_face_elem)
        right_idx = CartesianIndex(1, 1, 1, level, right_face_elem)
        val =
            Topologies.dss_transform(data, lg, weight, left_idx) ⊞
            Topologies.dss_transform(data, lg, weight, right_idx)
        data[left_idx] = Topologies.dss_untransform(T, val, lg, left_idx)
        data[right_idx] = Topologies.dss_untransform(T, val, lg, right_idx)
    end
    return nothing
end

##########
## ClimaCore.jl/src/Topologies/dss.jl
##########

function Topologies.dss_1d!(
    ::ClimaComms.AbstractCPUDevice,
    data::Union{DataLayouts.VIFH, DataLayouts.IFH},
    topology::Topologies.IntervalTopology,
    lg = nothing,
    weight = nothing,
)
    T = eltype(data)
    (Ni, _, _, Nv, Nh) = size(data)
    Nfaces = Topologies.isperiodic(topology) ? Nh : Nh - 1
    @inbounds for left_face_elem in 1:Nfaces, level in 1:Nv
        right_face_elem = (left_face_elem % Nh) + 1
        left_idx = CartesianIndex(Ni, 1, 1, level, left_face_elem)
        right_idx = CartesianIndex(1, 1, 1, level, right_face_elem)
        val =
            Topologies.dss_transform(data, lg, weight, left_idx) ⊞
            Topologies.dss_transform(data, lg, weight, right_idx)
        data[left_idx] = Topologies.dss_untransform(T, val, lg, left_idx)
        data[right_idx] = Topologies.dss_untransform(T, val, lg, right_idx)
    end
end

##########
## ClimaCore.jl/src/Spaces/dss.jl
##########

function Spaces.weighted_dss_internal!(
    data::Union{
        DataLayouts.IFH,
        DataLayouts.VIFH,
        DataLayouts.IJFH,
        DataLayouts.VIJFH,
    },
    space::Union{
        Spaces.AbstractSpectralElementSpace,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
    hspace::Spaces.AbstractSpectralElementSpace,
    dss_buffer::Union{Topologies.DSSBuffer, Nothing},
)
    Spaces.assert_same_eltype(data, dss_buffer)
    length(parent(data)) == 0 && return nothing
    device = ClimaComms.device(Spaces.topology(hspace))
    if hspace isa Spaces.SpectralElementSpace1D
        Topologies.dss_1d!(
            device,
            data,
            Spaces.topology(hspace),
            Spaces.local_geometry_data(space),
            Spaces.local_dss_weights(space),
        )
    else
        Topologies.dss_transform!(
            device,
            dss_buffer,
            data,
            Spaces.local_geometry_data(space),
            Spaces.local_dss_weights(space),
            Spaces.perimeter(hspace),
            dss_buffer.internal_elems,
        )
        Topologies.dss_local!(
            device,
            dss_buffer.perimeter_data,
            Spaces.perimeter(hspace),
            Spaces.topology(hspace),
        )
        Topologies.dss_untransform!(
            device,
            dss_buffer,
            data,
            Spaces.local_geometry_data(space),
            Spaces.perimeter(hspace),
            dss_buffer.internal_elems,
        )
    end
    return nothing
end

##########
## ClimaCore.jl/src/Grids/spectralelement.jl
##########

function Grids._SpectralElementGrid1D(
    topology::Topologies.IntervalTopology,
    quadrature_style::Quadratures.QuadratureStyle,
)
    DA = ClimaComms.array_type(topology)
    global_geometry = Geometry.CartesianGlobalGeometry()
    CoordType = Topologies.coordinate_type(topology)
    AIdx = Geometry.coordinate_axis(CoordType)
    FT = eltype(CoordType)
    nelements = Topologies.nlocalelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    LG = Geometry.LocalGeometry{AIdx, CoordType, FT, SMatrix{1, 1, FT, 1}}
    quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)

    # Initialize the local_geometry with Array{FT} as the backing array type.
    local_geometry_cpu = DataLayouts.IFH{LG, Nq}(Array{FT}, nelements)

    for elem in 1:nelements
        local_geometry_slab = slab(local_geometry_cpu, elem)
        for i in 1:Nq
            ξ = quad_points[i]
            vcoords = Topologies.vertex_coordinates(topology, elem)
            x = Geometry.linear_interpolate(vcoords, ξ)
            ∂x∂ξ =
                (
                    Geometry.component(vcoords[2], 1) -
                    Geometry.component(vcoords[1], 1)
                ) / 2
            J = abs(∂x∂ξ)
            WJ = J * quad_weights[i]
            local_geometry_slab[i] = Geometry.LocalGeometry(
                x,
                J,
                WJ,
                Geometry.AxisTensor(
                    (
                        Geometry.LocalAxis{AIdx}(),
                        Geometry.CovariantAxis{AIdx}(),
                    ),
                    ∂x∂ξ,
                ),
            )
        end
    end

    # If needed, move the local_geometry onto the GPU.
    local_geometry = DataLayouts.rebuild(local_geometry_cpu, DA)

    # TODO: Why was this code setting dss_weights = 1 / DSS(1)? That's just 1??
    # dss_weights = copy(local_geometry.J)
    # dss_weights .= one(FT)
    # Topologies.dss_1d!(topology, dss_weights)
    # dss_weights = one(FT) ./ dss_weights

    dss_weights = copy(local_geometry.J)
    Topologies.dss_1d!(ClimaComms.device(topology), dss_weights, topology)
    dss_weights .= local_geometry.J ./ dss_weights

    return Grids.SpectralElementGrid1D(
        topology,
        quadrature_style,
        global_geometry,
        local_geometry,
        dss_weights,
    )
end

##########
## ClimaCore.jl/src/Hypsography/Hypsography.jl
##########

function Grids._ExtrudedFiniteDifferenceGrid(
    horizontal_grid::Grids.AbstractGrid,
    vertical_grid::Grids.FiniteDifferenceGrid,
    adaption::Grids.HypsographyAdaption,
    global_geometry::Geometry.AbstractGlobalGeometry,
)
    @assert Spaces.grid(axes(adaption.surface)) == horizontal_grid
    z_surface = Fields.field_values(adaption.surface)

    face_z_ref =
        Grids.local_geometry_data(vertical_grid, Grids.CellFace()).coordinates
    vertical_domain = Topologies.domain(vertical_grid)
    z_top = vertical_domain.coord_max

    face_z =
        modified_ref_z_to_physical_z.(
            Ref(typeof(adaption)),
            face_z_ref,
            z_surface,
            Ref(z_top),
            Ref(adaption_parameters(adaption)),
        )
    face_z_cpu =
        modified_ref_z_to_physical_z.(
            Ref(typeof(adaption)),
            Adapt.adapt(Array, face_z_ref),
            Adapt.adapt(Array, z_surface),
            Ref(z_top),
            Ref(adaption_parameters(adaption)),
        )

    return Grids._ExtrudedFiniteDifferenceGrid(
        horizontal_grid,
        vertical_grid,
        adaption,
        global_geometry,
        face_z,
    )
end

adaption_parameters(::Grids.Flat) = (;)
adaption_parameters(::Hypsography.LinearAdaption) = (;)
adaption_parameters((; ηₕ, s)::Hypsography.SLEVEAdaption) = (; ηₕ, s)

function modified_ref_z_to_physical_z(
    ::Type{<:Grids.Flat},
    z_ref::Geometry.ZPoint,
    z_surface::Geometry.ZPoint,
    z_top::Geometry.ZPoint,
    _,
)
    return z_ref
end
function modified_ref_z_to_physical_z(
    ::Type{<:Hypsography.LinearAdaption},
    z_ref::Geometry.ZPoint,
    z_surface::Geometry.ZPoint,
    z_top::Geometry.ZPoint,
    _,
)
    Geometry.ZPoint(z_ref.z + (1 - z_ref.z / z_top.z) * z_surface.z)
end
function modified_ref_z_to_physical_z(
    ::Type{<:Hypsography.SLEVEAdaption},
    z_ref::Geometry.ZPoint,
    z_surface::Geometry.ZPoint,
    z_top::Geometry.ZPoint,
    (; ηₕ, s),
)
    if s * z_top.z <= z_surface.z
        error("Decay scale (s*z_top) must be higher than max surface elevation")
    end

    η = z_ref.z / z_top.z
    if η <= ηₕ
        return Geometry.ZPoint(
            η * z_top.z +
            z_surface.z * (sinh((ηₕ - η) / s / ηₕ)) / (sinh(1 / s)),
        )
    else
        return Geometry.ZPoint(η * z_top.z)
    end
end
