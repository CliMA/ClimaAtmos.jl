#=
       .-.      Welcome to ClimaAtmos!
      (   ).    ----------------------
     (___(__)   A state-of-the-art Julia model for
    ⌜^^^^^^^⌝   simulating atmospheric dynamics.
   ⌜  ~  ~  ⌝
  ⌜ ~  ~  ~  ⌝  This example: *Dry Baroclinic Wave*
 ⌜  ~   ~  ~ ⌝
⌜~~~~~~~~~~~~~⌝  ⚡ Harnessing GPU acceleration with CUDA.jl
    “““““““      🌎 Pushing the frontiers of climate science!

Run with
```
julia +1.11 --project=.buildkite
ENV["CLIMACOMMS_DEVICE"] = "CUDA";
ENV["CLIMACOMMS_DEVICE"] = "CPU";
using Revise; include("examples/dry_baro_wave_kernel.jl")
=#
ENV["CLIMACOMMS_DEVICE"] = "CUDA";
using CUDA
import ClimaComms
ClimaComms.@import_required_backends
using ClimaCore.CommonSpaces
import ClimaAtmos as CA
using LazyBroadcast: lazy
using LinearAlgebra: ×, dot, norm
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD
import SciMLBase
import ClimaCore.Grids
import ClimaCore
using KernelAbstractions
import KernelAbstractions as KA
import ClimaTimeSteppers as CTS
import ClimaCore.Geometry
import ClimaCore.MatrixFields: @name, ⋅
import ClimaCore.MatrixFields: DiagonalMatrixRow, BidiagonalMatrixRow
import LinearAlgebra: Adjoint
import LinearAlgebra: adjoint
import LinearAlgebra as LA
import ClimaCore: Operators, Topologies, DataLayouts
import ClimaCore.MatrixFields
import ClimaCore.Spaces
import ClimaCore.Fields
# import KernelAbstractions as KA
# using KernelAbstractions

# Unless the kernel here fills the shmem, we cannot use the shmem getidx.
# So, let's disable for now. Maybe we can write a simple shmem transformation
# of the broadcasted object + the state.
const cccuda_ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt);
# LazyBroadcast calls instantiate on intermediate broadcast expressions, so we
# should be determining AbstractStencilStyle locally.
# ClimaCore bug: stencil style should be locally determined at the top level, not using the recursive Operators.any_fd_shmem_supported(bc)
Operators.AbstractStencilStyle(bc, ::ClimaComms.CUDADevice) =
    cccuda_ext.CUDAColumnStencilStyle
Operators.fd_shmem_is_supported(bc::Base.Broadcast.Broadcasted) = false
ClimaCore.Operators.use_fd_shmem() = false

@info "Arch: $(ClimaComms.device())"
if get(ENV, "CLIMACOMMS_DEVICE", "CPU") == "CUDA"
    using CUDA
    using CUDA.CUDAKernels
    CUDA.allowscalar(false)
else
end

import ClimaAtmos: C1, C2, C12, C3, C123, CT1, CT2, CT12, CT3, CT123, UVW
import ClimaAtmos:
    divₕ, wdivₕ, gradₕ, wgradₕ, curlₕ, wcurlₕ, ᶜinterp, ᶜdivᵥ, ᶜgradᵥ
import ClimaAtmos: ᶠinterp, ᶠgradᵥ, ᶠcurlᵥ, ᶜinterp_matrix, ᶠgradᵥ_matrix
import ClimaAtmos: ᶜadvdivᵥ, ᶜadvdivᵥ_matrix, ᶠwinterp, ᶠinterp_matrix

Fields.local_geometry_field(bc::Base.Broadcast.Broadcasted) =
    Fields.local_geometry_field(axes(bc))

ᶜtendencies(ρ, uₕ, ρe_tot) = (; ρ, uₕ, ρe_tot)
ᶠtendencies(u₃) = (; u₃)

@inline is_valid_index(us, I) = 1 ≤ I ≤ DataLayouts.get_N(us)
@inline function is_valid_index_md(us, Nv, I)
    v = vindex()
    return 1 ≤ v ≤ Nv && is_valid_index(us, I)
end

function implicit_tendency_bc!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    set_precomputed_quantities!(Y, p, t)
    (; rayleigh_sponge, params, dt) = p
    (; ᶜh_tot, ᶠu³, ᶜp) = p.precomputed
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠz = Fields.coordinate_field(Y.f).z
    grav = FT(CAP.grav(params))
    zmax = CA.z_max(axes(Y.f))

    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠu³)
    # Central advection of active tracers (e_tot and q_tot)
    Yₜ.c.ρe_tot .+= CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, Val(:none))
    @. Yₜ.f.u₃ -= ᶠgradᵥ(ᶜp) / ᶠinterp(Y.c.ρ) + ᶠgradᵥ(Φ(grav, ᶜz))

    @. Yₜ.f.u₃ -= CA.β_rayleigh_w(rayleigh_sponge, ᶠz, zmax) * Y.f.u₃
    return nothing
end

# Define tendency functions
# implicit_tendency!(Yₜ, Y, p, t) = implicit_tendency_cuda!(Yₜ, Y, p, t)
# implicit_tendency!(Yₜ, Y, p, t) = implicit_tendency_KA!(Yₜ, Y, p, t)
implicit_tendency!(Yₜ, Y, p, t) = implicit_tendency_cuda_md_launch!(Yₜ, Y, p, t)
function implicit_tendency_cuda_linear_launch!(Yₜ, Y, p, t)
    ᶜspace = axes(Y.c)
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶠNv = Spaces.nlevels(ᶠspace)
    ᶜcf = Fields.coordinate_field(ᶜspace)
    us = DataLayouts.UniversalSize(Fields.field_values(ᶜcf))
    (Ni, Nj, _, _, Nh) = DataLayouts.universal_size(us)
    nitems = Ni * Nj * 1 * ᶠNv * Nh
    ᶜYₜ = Yₜ.c
    ᶠYₜ = Yₜ.f
    ᶜY = Y.c
    ᶠY = Y.f
    (; rayleigh_sponge, params, dt) = p
    p_kernel = (; rayleigh_sponge, params, dt)
    zmax = Spaces.z_max(axes(ᶠY)) # DeviceIntervalTopology does not have mesh, and therefore cannot compute zmax
    
    kernel = CUDA.@cuda(
        always_inline = true,
        launch = false,
        implicit_tendency_kernel_cuda!(ᶜYₜ, ᶠYₜ, ᶜY, ᶠY, p_kernel, t, us, zmax)
    )
    (;threads, blocks) = CUDA.launch_configuration(kernel.fun)
    @show ᶠNv, nitems, blocks, threads, blocks * threads
    threads = min(nitems, threads)
    blocks = cld(nitems, threads)
    @show ᶠNv, nitems, blocks, threads, blocks * threads
    kernel(ᶜYₜ, ᶠYₜ, ᶜY, ᶠY, p_kernel, t, us, zmax; threads, blocks)
end

function implicit_tendency_cuda_md_launch!(Yₜ, Y, p, t)
    ᶜspace = axes(Y.c)
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶠNv = Spaces.nlevels(ᶠspace)
    ᶜcf = Fields.coordinate_field(ᶜspace)
    us = DataLayouts.UniversalSize(Fields.field_values(ᶜcf))
    (Ni, Nj, _, _, Nh) = DataLayouts.universal_size(us)
    nitems = Ni * Nj * 1 * ᶠNv * Nh
    ᶜYₜ = Yₜ.c
    ᶠYₜ = Yₜ.f
    ᶜY = Y.c
    ᶠY = Y.f
    (; rayleigh_sponge, params, dt) = p
    p_kernel = (; rayleigh_sponge, params, dt)
    zmax = Spaces.z_max(axes(ᶠY)) # DeviceIntervalTopology does not have mesh, and therefore cannot compute zmax
    
    kernel = CUDA.@cuda(
        always_inline = true,
        launch = false,
        implicit_tendency_kernel_cuda_md_launch!(ᶜYₜ, ᶠYₜ, ᶜY, ᶠY, p_kernel, t, us, zmax)
    )
    threads = (ᶠNv, )
    blocks = (Nh, 1, Ni * Nj)
    kernel(ᶜYₜ, ᶠYₜ, ᶜY, ᶠY, p_kernel, t, us, zmax; threads, blocks)
end

function implicit_tendency_KA!(Yₜ, Y, p, t)
    ᶜspace = axes(Y.c)
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶠNv = Spaces.nlevels(ᶠspace)
    ᶜcf = Fields.coordinate_field(ᶜspace)
    us = DataLayouts.UniversalSize(Fields.field_values(ᶜcf))
    (Ni, Nj, _, _, Nh) = DataLayouts.universal_size(us)
    nitems = Ni * Nj * 1 * ᶠNv * Nh
    ᶜYₜ = Yₜ.c
    ᶠYₜ = Yₜ.f
    ᶜY = Y.c
    ᶠY = Y.f
    (; rayleigh_sponge, params, dt) = p
    p_kernel = (; rayleigh_sponge, params, dt)
    zmax = Spaces.z_max(axes(ᶠY)) # DeviceIntervalTopology does not have mesh, and therefore cannot compute zmax

    backend = if ClimaComms.device(ᶜspace) isa ClimaComms.CUDADevice
        CUDABackend()
    else
        KA.CPU()
    end
    kernel = implicit_tendency_kernel_KA!(backend)
    kernel(ᶜYₜ, ᶠYₜ, ᶜY, ᶠY, p_kernel, t, us, zmax, ndrange = nitems)
end

# allow on-device use of lazy broadcast objects
DataLayouts.parent_array_type(::Type{<:CUDA.CuDeviceArray{T, N, A} where {N}}) where {T, A} =
    CUDA.CuDeviceArray{T, N, A} where {N}

# allow on-device use of lazy broadcast objects
DataLayouts.promote_parent_array_type(
    ::Type{CUDA.CuDeviceArray{T1, N, B} where {N}},
    ::Type{CUDA.CuDeviceArray{T2, N, B} where {N}},
) where {T1, T2, B} = CUDA.CuDeviceArray{promote_type(T1, T2), N, B} where {N}

# allow on-device use of lazy broadcast objects with different type params
DataLayouts.promote_parent_array_type(
    ::Type{CUDA.CuDeviceArray{T1, N, B1} where {N}},
    ::Type{CUDA.CuDeviceArray{T2, N, B2} where {N}},
) where {T1, T2, B1, B2} = CUDA.CuDeviceArray{promote_type(T1, T2), N, B} where {N, B}

# allow on-device use of lazy broadcast objects with different type params
DataLayouts.promote_parent_array_type(
    ::Type{CUDA.CuDeviceArray{T1}},
    ::Type{CUDA.CuDeviceArray{T2, N, B2} where {N}},
) where {T1, T2, B2} = CUDA.CuDeviceArray{promote_type(T1, T2), N, B} where {N, B}

DataLayouts.promote_parent_array_type(
    ::Type{CUDA.CuDeviceArray{T1, N, B1} where {N}},
    ::Type{CUDA.CuDeviceArray{T2} where {N}},
) where {T1, T2, B1} = CUDA.CuDeviceArray{promote_type(T1, T2), N, B} where {N, B}

# Specialize to allow on-device call of `device` for `DeviceExtrudedFiniteDifferenceGrid`
ClimaComms.device(grid::Grids.DeviceExtrudedFiniteDifferenceGrid) =
    ClimaComms.device(Grids.vertical_topology(grid))

# The existing implementation limits our ability to apply the same expressions from within kernels
ClimaComms.device(topology::Topologies.DeviceIntervalTopology) = ClimaComms.CUDADevice()

Fields.error_mismatched_spaces(::Type, ::Type) = nothing # causes unsupported dynamic function invocation

@inline function operator_inds(space, I)
    li = Operators.left_idx(space)
    (i, j, _, v, h) = I.I
    hidx = (i, j, h)
    idx = v - 1 + li
    return (idx, hidx)
end

@inline cartesian_indices(field::Fields.Field) =
    cartesian_indices(Fields.field_values(field))
@inline cartesian_indices(data::DataLayouts.AbstractData) =
    cartesian_indices(DataLayouts.UniversalSize(data))
@inline cartesian_indices(us::DataLayouts.UniversalSize) =
    CartesianIndices(map(Base.OneTo, DataLayouts.universal_size(us)))
@inline universal_index(x) = cartesian_indices(x)


function thermo_state(thermo_params, ᶜρ, ᶜρe_tot, ᶜK, grav, ᶜz)
    return @. lazy(TD.PhaseDry_ρe(
            thermo_params,
            ᶜρ,
            ᶜρe_tot / ᶜρ - ᶜK - Φ(grav, ᶜz),
        ))
end

# Drop everything except Nv and S:
@inline column_type_params(data::DataLayouts.AbstractData) = column_type_params(typeof(data))
@inline column_type_params(::Type{DataLayouts.IJFH{S, Nij, A}}) where {S, Nij, A} = (S, )
@inline column_type_params(::Type{DataLayouts.IJHF{S, Nij, A}}) where {S, Nij, A} = (S, )
@inline column_type_params(::Type{DataLayouts.IFH{S, Ni, A}}) where {S, Ni, A} = (S, )
@inline column_type_params(::Type{DataLayouts.IHF{S, Ni, A}}) where {S, Ni, A} = (S, )
@inline column_type_params(::Type{DataLayouts.DataF{S, A}}) where {S, A} = (S,)
@inline column_type_params(::Type{DataLayouts.IJF{S, Nij, A}}) where {S, Nij, A} = (S, )
@inline column_type_params(::Type{DataLayouts.IF{S, Ni, A}}) where {S, Ni, A} = (S, )
@inline column_type_params(::Type{DataLayouts.VF{S, Nv, A}}) where {S, Nv, A} = (S, Nv)
@inline column_type_params(::Type{DataLayouts.VIJFH{S, Nv, Nij, A}}) where {S, Nv, Nij, A} = (S, Nv)
@inline column_type_params(::Type{DataLayouts.VIJHF{S, Nv, Nij, A}}) where {S, Nv, Nij, A} = (S, Nv)
@inline column_type_params(::Type{DataLayouts.VIFH{S, Nv, Ni, A}}) where {S, Nv, Ni, A} = (S, Nv)
@inline column_type_params(::Type{DataLayouts.VIHF{S, Nv, Ni, A}}) where {S, Nv, Ni, A} = (S, Nv)

# Drop everything except V and F:
@inline column_singleton(::DataLayouts.IJFH) = DataLayouts.DataFSingleton()
@inline column_singleton(::DataLayouts.IJHF) = DataLayouts.DataFSingleton()
@inline column_singleton(::DataLayouts.IFH) = DataLayouts.DataFSingleton()
@inline column_singleton(::DataLayouts.IHF) = DataLayouts.DataFSingleton()
@inline column_singleton(::DataLayouts.DataF) = DataLayouts.DataFSingleton()
@inline column_singleton(::DataLayouts.IJF) = DataLayouts.DataFSingleton()
@inline column_singleton(::DataLayouts.IF) = DataLayouts.DataFSingleton()
@inline column_singleton(::DataLayouts.VF) = DataLayouts.VFSingleton()
@inline column_singleton(::DataLayouts.VIJFH) = DataLayouts.VFSingleton()
@inline column_singleton(::DataLayouts.VIJHF) = DataLayouts.VFSingleton()
@inline column_singleton(::DataLayouts.VIFH) = DataLayouts.VFSingleton()
@inline column_singleton(::DataLayouts.VIHF) = DataLayouts.VFSingleton()

function rebuild_column(data, array::AbstractArray)
    s_column = column_singleton(data)
    return DataLayouts.union_all(s_column){column_type_params(data)...}(array)
end

function column_lg_shmem(f, ui)
    (i, j, _, _, h) = ui.I
    colidx = Grids.ColumnIndex((i, j), h)
    lg = Spaces.local_geometry_data(axes(f))
    lg_col = Spaces.column(lg, colidx)
    FT = Spaces.undertype(axes(f))
    Nv = Spaces.nlevels(axes(f))
    TS = DataLayouts.typesize(FT, eltype(lg_col))
    lg_arr = CUDA.CuStaticSharedArray(FT, (Nv, TS))
    return rebuild_column(lg_col, lg_arr)
end

function column_spaces(ᶜY, ᶠY, ui)
    (i, j, _, _, h) = ui.I
    colidx = Grids.ColumnIndex((i, j), h)
    ᶜlg_col = column_lg_shmem(ᶜY, ui)
    ᶠlg_col = column_lg_shmem(ᶠY, ui)
    col_space = Spaces.column(axes(ᶜY), colidx)
    col_grid = Spaces.grid(col_space)
    if col_grid isa Grids.ColumnGrid && col_grid.full_grid isa Grids.DeviceExtrudedFiniteDifferenceGrid
        (; full_grid) = col_grid
        (; vertical_topology, global_geometry) = full_grid
        col_grid_shmem = Grids.DeviceFiniteDifferenceGrid(vertical_topology, global_geometry, ᶜlg_col, ᶠlg_col)
        ᶜspace_col = Spaces.space(col_grid_shmem, Grids.CellCenter())
        ᶠspace_col = Spaces.space(col_grid_shmem, Grids.CellFace())
    else
        ᶜspace_col = nothing
        ᶠspace_col = nothing
    end
    return (ᶜspace_col, ᶠspace_col)
end

function column_states(ᶜY, ᶠY, ᶜdata_col, ᶠdata_col, ui, ᶜspace_col, ᶠspace_col)
    ᶜY_col = Fields.Field(ᶜdata_col, ᶜspace_col)
    ᶠY_col = Fields.Field(ᶠdata_col, ᶠspace_col)
    return (ᶜY_col, ᶠY_col)
end

@kernel function implicit_tendency_kernel_KA!(ᶜYₜ, ᶠYₜ, _ᶜY, _ᶠY, p, t, us, zmax)
    # gid = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
    gid = @index(Global)
    ᶜY_fv = @uniform Fields.field_values(_ᶜY)
    ᶠY_fv = @uniform Fields.field_values(_ᶠY)
    ᶜus = @uniform DataLayouts.UniversalSize(ᶜY_fv)
    ᶠus = @uniform DataLayouts.UniversalSize(ᶠY_fv)
    FT = @uniform Spaces.undertype(axes(_ᶜY))
    ᶜNv = @uniform Spaces.nlevels(axes(_ᶜY))
    ᶠNv = @uniform Spaces.nlevels(axes(_ᶠY))
    ᶜui = is_valid_index(ᶜus, gid) ? universal_index(ᶜus)[gid] : CartesianIndex((-1, -1, -1, -1, -1))
    ᶠui = is_valid_index(ᶠus, gid) ? universal_index(ᶠus)[gid] : CartesianIndex((-1, -1, -1, -1, -1))

    ᶜTS = @uniform DataLayouts.typesize(FT, eltype(ᶜY_fv))
    ᶠTS = @uniform DataLayouts.typesize(FT, eltype(ᶠY_fv))
    ᶜY_arr = @localmem FT (ᶜNv, ᶜTS)
    ᶠY_arr = @localmem FT (ᶠNv, ᶠTS)
    ᶜdata_col = @uniform rebuild_column(ᶜY_fv, ᶜY_arr)
    ᶠdata_col = @uniform rebuild_column(ᶠY_fv, ᶠY_arr)
    is_valid_index(ᶜus, gid) && (ᶜdata_col[ᶜui] = ᶜY_fv[ᶜui])
    is_valid_index(ᶠus, gid) && (ᶠdata_col[ᶠui] = ᶠY_fv[ᶠui])

    @synchronize

    ᶜui = is_valid_index(ᶜus, gid) ? universal_index(ᶜus)[gid] : CartesianIndex((-1, -1, -1, -1, -1))
    ᶠui = is_valid_index(ᶠus, gid) ? universal_index(ᶠus)[gid] : CartesianIndex((-1, -1, -1, -1, -1))
    # ᶜY = _ᶜY
    # ᶠY = _ᶠY

    if is_valid_index(ᶜus, gid)
        (ᶜY, ᶠY) = column_states(_ᶜY, _ᶠY, ᶜdata_col, ᶠdata_col, ᶠui)
        ᶜbc = ᶜimplicit_tendency_bc(ᶜY, ᶠY, p, t, zmax)
        (ᶜidx, ᶜhidx) = operator_inds(axes(ᶜY), ᶜui)
        Fields.field_values(ᶜYₜ)[ᶜui] = Operators.getidx(axes(ᶜY), ᶜbc, ᶜidx, ᶜhidx)
        # ᶜYₜ[ᶜui] = ᶜimplicit_tendency_bc(ᶜY, ᶠY, p, t)[ᶜui] # might be possible?
    end
    if is_valid_index(ᶠus, gid)
        (ᶜY, ᶠY) = column_states(_ᶜY, _ᶠY, ᶜdata_col, ᶠdata_col, ᶠui)
        ᶠbc = ᶠimplicit_tendency_bc(ᶜY, ᶠY, p, t, zmax)
        (ᶠidx, ᶠhidx) = operator_inds(axes(ᶠY), ᶠui)
        Fields.field_values(ᶠYₜ)[ᶠui] = Operators.getidx(axes(ᶠY), ᶠbc, ᶠidx, ᶠhidx)
        # ᶠYₜ[ᶠui] = ᶠimplicit_tendency_bc(ᶜY, ᶠY, p, t)[ᶠui] # might be possible?
    end
end

function vindex()
    (tv,) = CUDA.threadIdx()
    (h, bv, ij) = CUDA.blockIdx()
    v = tv + (bv - 1) * CUDA.blockDim().x
    return v
end

function implicit_tendency_kernel_cuda_md_launch!(ᶜYₜ, ᶠYₜ, _ᶜY, _ᶠY, p, t, us, zmax)
    gid = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
    ᶜY_fv = Fields.field_values(_ᶜY)
    ᶠY_fv = Fields.field_values(_ᶠY)
    ᶜus = DataLayouts.UniversalSize(ᶜY_fv)
    ᶠus = DataLayouts.UniversalSize(ᶠY_fv)
    FT = Spaces.undertype(axes(_ᶜY))
    ᶜNv = Spaces.nlevels(axes(_ᶜY))
    ᶠNv = Spaces.nlevels(axes(_ᶠY))
    ᶜui = cccuda_ext.fd_shmem_stencil_universal_index(axes(_ᶜY), ᶜus)
    ᶠui = cccuda_ext.fd_shmem_stencil_universal_index(axes(_ᶠY), ᶠus)

    (i, j, _, _, h) = ᶜui.I
    colidx = Grids.ColumnIndex((i, j), h)
    (ᶜspace_col, ᶠspace_col) = column_spaces(_ᶜY, _ᶠY, ᶠui)

    ᶜTS = DataLayouts.typesize(FT, eltype(DataLayouts.column(ᶜY_fv, colidx)))
    ᶠTS = DataLayouts.typesize(FT, eltype(DataLayouts.column(ᶠY_fv, colidx)))
    ᶜY_arr = CUDA.CuStaticSharedArray(FT, (ᶜNv, ᶜTS))
    ᶠY_arr = CUDA.CuStaticSharedArray(FT, (ᶠNv, ᶠTS))
    ᶜdata_col = rebuild_column(ᶜY_fv, ᶜY_arr)
    ᶠdata_col = rebuild_column(ᶠY_fv, ᶠY_arr)
    is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜdata_col[ᶜui] = ᶜY_fv[ᶜui])
    is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠdata_col[ᶠui] = ᶠY_fv[ᶠui])

    ᶜlg = Spaces.local_geometry_data(axes(_ᶜY))
    ᶠlg = Spaces.local_geometry_data(axes(_ᶠY))
    ᶜlg_col = Spaces.local_geometry_data(ᶜspace_col)
    ᶠlg_col = Spaces.local_geometry_data(ᶠspace_col)
    is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.coordinates.z[ᶜui] = ᶜlg.coordinates.z[ᶜui]) # needed
    is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.coordinates.z[ᶠui] = ᶠlg.coordinates.z[ᶠui]) # needed
    is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.J[ᶜui] = ᶜlg.J[ᶜui]) # needed
    is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.J[ᶠui] = ᶠlg.J[ᶠui]) # needed
    # is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.WJ[ᶜui] = ᶜlg.WJ[ᶜui]) # not needed
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.WJ[ᶠui] = ᶠlg.WJ[ᶠui]) # not needed
    is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.invJ[ᶜui] = ᶜlg.invJ[ᶜui]) # needed
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.invJ[ᶠui] = ᶠlg.invJ[ᶠui]) # not needed
    # is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.∂x∂ξ[ᶜui] = ᶜlg.∂x∂ξ[ᶜui]) # not needed
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.∂x∂ξ[ᶠui] = ᶠlg.∂x∂ξ[ᶠui]) # not needed
    # is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.∂ξ∂x[ᶜui] = ᶜlg.∂ξ∂x[ᶜui]) # not needed
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.∂ξ∂x[ᶠui] = ᶠlg.∂ξ∂x[ᶠui]) # not needed
    # is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.gᵢⱼ[ᶜui] = ᶜlg.gᵢⱼ[ᶜui]) # not needed
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.gᵢⱼ[ᶠui] = ᶠlg.gᵢⱼ[ᶠui]) # not needed

    is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.gⁱʲ.components.data.:1[ᶜui] = ᶜlg.gⁱʲ.components.data.:1[ᶜui]) # needed
    is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.gⁱʲ.components.data.:2[ᶜui] = ᶜlg.gⁱʲ.components.data.:2[ᶜui]) # needed
    is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.gⁱʲ.components.data.:3[ᶜui] = ᶜlg.gⁱʲ.components.data.:3[ᶜui]) # needed
    is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.gⁱʲ.components.data.:4[ᶜui] = ᶜlg.gⁱʲ.components.data.:4[ᶜui]) # needed
    is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.gⁱʲ.components.data.:5[ᶜui] = ᶜlg.gⁱʲ.components.data.:5[ᶜui]) # needed
    is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.gⁱʲ.components.data.:6[ᶜui] = ᶜlg.gⁱʲ.components.data.:6[ᶜui]) # needed
    # is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.gⁱʲ.components.data.:7[ᶜui] = ᶜlg.gⁱʲ.components.data.:7[ᶜui]) # not needed
    # is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.gⁱʲ.components.data.:8[ᶜui] = ᶜlg.gⁱʲ.components.data.:8[ᶜui]) # not needed
    # is_valid_index_md(ᶜus, ᶜNv, gid) && (ᶜlg_col.gⁱʲ.components.data.:9[ᶜui] = ᶜlg.gⁱʲ.components.data.:9[ᶜui]) # not needed
    
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.gⁱʲ.components.data.:1[ᶠui] = ᶠlg.gⁱʲ.components.data.:1[ᶠui]) # not needed
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.gⁱʲ.components.data.:2[ᶠui] = ᶠlg.gⁱʲ.components.data.:2[ᶠui]) # not needed
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.gⁱʲ.components.data.:3[ᶠui] = ᶠlg.gⁱʲ.components.data.:3[ᶠui]) # not needed
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.gⁱʲ.components.data.:4[ᶠui] = ᶠlg.gⁱʲ.components.data.:4[ᶠui]) # not needed
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.gⁱʲ.components.data.:5[ᶠui] = ᶠlg.gⁱʲ.components.data.:5[ᶠui]) # not needed
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.gⁱʲ.components.data.:6[ᶠui] = ᶠlg.gⁱʲ.components.data.:6[ᶠui]) # not needed
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.gⁱʲ.components.data.:7[ᶠui] = ᶠlg.gⁱʲ.components.data.:7[ᶠui]) # not needed
    # is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.gⁱʲ.components.data.:8[ᶠui] = ᶠlg.gⁱʲ.components.data.:8[ᶠui]) # not needed
    is_valid_index_md(ᶠus, ᶠNv, gid) && (ᶠlg_col.gⁱʲ.components.data.:9[ᶠui] = ᶠlg.gⁱʲ.components.data.:9[ᶠui]) # needed

    CUDA.sync_threads()

    if is_valid_index_md(ᶜus, ᶜNv, gid)
        (ᶜY, ᶠY) = column_states(_ᶜY, _ᶠY, ᶜdata_col, ᶠdata_col, ᶠui, ᶜspace_col, ᶠspace_col)
        ᶜbc = ᶜimplicit_tendency_bc(ᶜY, ᶠY, p, t, zmax)
        (ᶜidx, ᶜhidx) = operator_inds(axes(ᶜY), ᶜui)
        # Fields.field_values(ᶜYₜ)[ᶜui] = Operators.getidx(axes(ᶜY), ᶜbc, ᶜidx, ᶜhidx)
        Fields.field_values(ᶜYₜ)[ᶜui] = Operators.getidx(ᶜspace_col, ᶜbc, ᶜidx, ᶜhidx)
    #     # ᶜYₜ[ᶜui] = ᶜimplicit_tendency_bc(ᶜY, ᶠY, p, t)[ᶜui] # might be possible?
    end
    if is_valid_index_md(ᶠus, ᶠNv, gid)
        (ᶜY, ᶠY) = column_states(_ᶜY, _ᶠY, ᶜdata_col, ᶠdata_col, ᶠui, ᶜspace_col, ᶠspace_col)
        ᶠbc = ᶠimplicit_tendency_bc(ᶜY, ᶠY, p, t, zmax)
        (ᶠidx, ᶠhidx) = operator_inds(axes(ᶠY), ᶠui)
        # Fields.field_values(ᶠYₜ)[ᶠui] = Operators.getidx(axes(ᶠY), ᶠbc, ᶠidx, ᶠhidx)
        Fields.field_values(ᶠYₜ)[ᶠui] = Operators.getidx(ᶠspace_col, ᶠbc, ᶠidx, ᶠhidx)
    #     # ᶠYₜ[ᶠui] = ᶠimplicit_tendency_bc(ᶜY, ᶠY, p, t)[ᶠui] # might be possible?
    end
    return nothing
end

function implicit_tendency_kernel_cuda_linear_launch!(ᶜYₜ, ᶠYₜ, _ᶜY, _ᶠY, p, t, us, zmax)
    gid = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
    ᶜY_fv = Fields.field_values(_ᶜY)
    ᶠY_fv = Fields.field_values(_ᶠY)
    ᶜus = DataLayouts.UniversalSize(ᶜY_fv)
    ᶠus = DataLayouts.UniversalSize(ᶠY_fv)
    FT = Spaces.undertype(axes(_ᶜY))
    ᶜNv = Spaces.nlevels(axes(_ᶜY))
    ᶠNv = Spaces.nlevels(axes(_ᶠY))
    ᶜui = is_valid_index(ᶜus, gid) ? universal_index(ᶜus)[gid] : CartesianIndex((-1, -1, -1, -1, -1))
    ᶠui = is_valid_index(ᶠus, gid) ? universal_index(ᶠus)[gid] : CartesianIndex((-1, -1, -1, -1, -1))

    ᶜTS = DataLayouts.typesize(FT, eltype(ᶜY_fv))
    ᶠTS = DataLayouts.typesize(FT, eltype(ᶠY_fv))
    ᶜY_arr = CUDA.CuStaticSharedArray(FT, (ᶜNv, ᶜTS))
    ᶠY_arr = CUDA.CuStaticSharedArray(FT, (ᶠNv, ᶠTS))
    ᶜdata_col = rebuild_column(ᶜY_fv, ᶜY_arr)
    ᶠdata_col = rebuild_column(ᶠY_fv, ᶠY_arr)
    is_valid_index(ᶜus, gid) && (ᶜdata_col[ᶜui] = ᶜY_fv[ᶜui])
    is_valid_index(ᶠus, gid) && (ᶠdata_col[ᶠui] = ᶠY_fv[ᶠui])
    CUDA.sync_threads()

    if is_valid_index(ᶜus, gid)
        (ᶜY, ᶠY) = column_states(_ᶜY, _ᶠY, ᶜdata_col, ᶠdata_col, ᶠui)
        ᶜbc = ᶜimplicit_tendency_bc(ᶜY, ᶠY, p, t, zmax)
        (ᶜidx, ᶜhidx) = operator_inds(axes(ᶜY), ᶜui)
        Fields.field_values(ᶜYₜ)[ᶜui] = Operators.getidx(axes(ᶜY), ᶜbc, ᶜidx, ᶜhidx)
        # ᶜYₜ[ᶜui] = ᶜimplicit_tendency_bc(ᶜY, ᶠY, p, t)[ᶜui] # might be possible?
    end
    if is_valid_index(ᶠus, gid)
        (ᶜY, ᶠY) = column_states(_ᶜY, _ᶠY, ᶜdata_col, ᶠdata_col, ᶠui)
        ᶠbc = ᶠimplicit_tendency_bc(ᶜY, ᶠY, p, t, zmax)
        (ᶠidx, ᶠhidx) = operator_inds(axes(ᶠY), ᶠui)
        Fields.field_values(ᶠYₜ)[ᶠui] = Operators.getidx(axes(ᶠY), ᶠbc, ᶠidx, ᶠhidx)
        # ᶠYₜ[ᶠui] = ᶠimplicit_tendency_bc(ᶜY, ᶠY, p, t)[ᶠui] # might be possible?
    end
    return nothing
end

function ᶜimplicit_tendency_bc(ᶜY, ᶠY, p, t, zmax)
    (; rayleigh_sponge, params, dt) = p
    ᶜz = Fields.coordinate_field(ᶜY).z
    ᶜJ = Fields.local_geometry_field(ᶜY).J
    ᶠz = Fields.coordinate_field(ᶠY).z
    FT = Spaces.undertype(axes(ᶜY))
    grav = FT(CAP.grav(params))
    thermo_params = CAP.thermodynamics_params(params)
    ᶜρ = ᶜY.ρ
    ᶜρe_tot = ᶜY.ρe_tot
    ᶜuₕ = ᶜY.uₕ
    ᶠu₃ = ᶠY.u₃

    ᶜK = CA.compute_kinetic(ᶜuₕ, ᶠu₃)
    ᶜts = thermo_state(thermo_params, ᶜρ, ᶜρe_tot, ᶜK, grav, ᶜz)
    ᶜp = @. lazy(TD.air_pressure(thermo_params, ᶜts))
    ᶜh_tot = @. lazy(TD.total_specific_enthalpy(thermo_params, ᶜts, ᶜρe_tot / ᶜρ))
    # Central advection of active tracers (e_tot and q_tot)
    ᶠuₕ³ = @. lazy(ᶠwinterp(ᶜρ * ᶜJ, CT3(ᶜuₕ)))
    ᶠu³ = @. lazy(ᶠuₕ³ + CT3(ᶠu₃))
    tend_ρ_1 = @. lazy(ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠuₕ³))
    # tend_ᶠu₃_1 = @. lazy(ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) + ᶠgradᵥ(Φ(grav, ᶜz)))
    tend_ᶠu₃_2 = @. lazy(CA.β_rayleigh_w(rayleigh_sponge, ᶠz, zmax) * ᶠu₃)
    tend_ρe_tot_1 = CA.vertical_transport(ᶜρ, ᶠu³, ᶜh_tot, dt, Val(:none))

    ᶜuₕ₀ = (zero(eltype(ᶜuₕ)),)

    return @. lazy(ᶜtendencies(
        - tend_ρ_1,
        # - tend_ᶠu₃_2,
        - ᶜuₕ₀,
        tend_ρe_tot_1,
    ))
end

function ᶠimplicit_tendency_bc(ᶜY, ᶠY, p, t, zmax)
    (; rayleigh_sponge, params) = p
    ᶜz = Fields.coordinate_field(ᶜY).z
    ᶠz = Fields.coordinate_field(ᶠY).z
    FT = Spaces.undertype(axes(ᶜY))
    grav = FT(CAP.grav(params))
    thermo_params = CAP.thermodynamics_params(params)
    ᶜρ = ᶜY.ρ
    ᶜρe_tot = ᶜY.ρe_tot
    ᶜuₕ = ᶜY.uₕ
    ᶠu₃ = ᶠY.u₃
    ᶜK = CA.compute_kinetic(ᶜuₕ, ᶠu₃)
    ᶜts = thermo_state(thermo_params, ᶜρ, ᶜρe_tot, ᶜK, grav, ᶜz)
    ᶜp = @. lazy(TD.air_pressure(thermo_params, ᶜts))
    bc1 = @. lazy(- (ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) + ᶠgradᵥ(Φ(grav, ᶜz))))
    bc2 = @. lazy(- CA.β_rayleigh_w(rayleigh_sponge, ᶠz, zmax) * ᶠu₃)
    return @. lazy(ᶠtendencies(bc1 + bc2))
end

function ImplicitEquationJacobian(
    Y::Fields.FieldVector;
    approximate_solve_iters = 1,
    transform_flag = false,
)
    FT = Spaces.undertype(axes(Y.c))
    CTh = CA.CTh_vector_type(axes(Y.c))

    BidiagonalRow_C3 = MatrixFields.BidiagonalMatrixRow{CA.C3{FT}}
    BidiagonalRow_ACT3 =
        MatrixFields.BidiagonalMatrixRow{LA.Adjoint{FT, CA.CT3{FT}}}
    BidiagonalRow_C3xACTh = MatrixFields.BidiagonalMatrixRow{
        typeof(zero(CA.C3{FT}) * zero(CTh{FT})'),
    }
    TridiagonalRow_C3xACT3 = MatrixFields.TridiagonalMatrixRow{
        typeof(zero(CA.C3{FT}) * zero(CA.CT3{FT})'),
    }

    is_in_Y(name) = MatrixFields.has_field(Y, name)

    sfc_if_available = is_in_Y(@name(sfc)) ? (@name(sfc),) : ()


    # Note: We have to use FT(-1) * I instead of -I because inv(-1) == -1.0,
    # which means that multiplying inv(-1) by a Float32 will yield a Float64.
    identity_blocks = MatrixFields.unrolled_map(
        name -> (name, name) => FT(-1) * LA.I,
        (@name(c.ρ), sfc_if_available...),
    )

    active_scalar_names = (@name(c.ρ), @name(c.ρe_tot))
    advection_blocks = (
        MatrixFields.unrolled_map(
            name -> (name, @name(f.u₃)) => similar(Y.c, BidiagonalRow_ACT3),
            active_scalar_names,
        )...,
        MatrixFields.unrolled_map(
            name -> (@name(f.u₃), name) => similar(Y.f, BidiagonalRow_C3),
            active_scalar_names,
        )...,
        (@name(f.u₃), @name(c.uₕ)) => similar(Y.f, BidiagonalRow_C3xACTh),
        (@name(f.u₃), @name(f.u₃)) => similar(Y.f, TridiagonalRow_C3xACT3),
    )

    diffused_scalar_names = (@name(c.ρe_tot),)
    diffusion_blocks = MatrixFields.unrolled_map(
        name -> (name, name) => FT(-1) * LA.I,
        (diffused_scalar_names..., @name(c.uₕ)),
    )

    matrix = MatrixFields.FieldMatrix(
        identity_blocks...,
        advection_blocks...,
        diffusion_blocks...,
    )

    names₁_group₁ = (@name(c.ρ), sfc_if_available...)
    names₁_group₃ = (@name(c.ρe_tot),)
    names₁ = (names₁_group₁..., names₁_group₃...)

    alg₂ = MatrixFields.BlockLowerTriangularSolve(@name(c.uₕ))
    alg = MatrixFields.BlockArrowheadSolve(names₁...; alg₂)

    return CA.ImplicitEquationJacobian(
        matrix,
        MatrixFields.FieldMatrixSolver(alg, matrix, Y),
        CA.IgnoreDerivative(), # diffusion_flag
        CA.IgnoreDerivative(), # topography_flag
        CA.IgnoreDerivative(), # sgs_advection_flag
        CA.IgnoreDerivative(), # sgs_entr_detr_flag
        CA.IgnoreDerivative(), # sgs_nh_pressure_flag
        CA.IgnoreDerivative(), # sgs_mass_flux_flag
        similar(Y),
        similar(Y),
        transform_flag,
        Ref{FT}(),
    )
end

function Wfact!(A, Y, p, dtγ, t)
    FT = Spaces.undertype(axes(Y.c))
    dtγ′ = FT(float(dtγ))
    A.dtγ_ref[] = dtγ′
    update_implicit_equation_jacobian!(A, Y, p, dtγ′)
end

Φ(grav, z) = grav * z

function update_implicit_equation_jacobian!(A, Y, p, dtγ)
    (; matrix) = A
    (; ᶜK, ᶜts, ᶜp, ᶜh_tot) = p.precomputed
    (; ∂ᶜK_∂ᶜuₕ, ∂ᶜK_∂ᶠu₃, ᶠp_grad_matrix, ᶜadvection_matrix) = p
    (; params) = p

    FT = Spaces.undertype(axes(Y.c))
    CTh = CA.CTh_vector_type(axes(Y.c))
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'
    rs = p.rayleigh_sponge
    ᶠz = Fields.coordinate_field(Y.f).z
    zmax = CA.z_max(axes(Y.f))

    T_0 = FT(CAP.T_0(params))
    cp_d = FT(CAP.cp_d(params))
    thermo_params = CAP.thermodynamics_params(params)
    ᶜz = Fields.coordinate_field(Y.c).z
    grav = FT(CAP.grav(params))

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠgⁱʲ = Fields.local_geometry_field(Y.f).gⁱʲ

    ᶜkappa_m = p.ᶜtemp_scalar
    @. ᶜkappa_m =
        TD.gas_constant_air(thermo_params, ᶜts) / TD.cv_m(thermo_params, ᶜts)

    @. ∂ᶜK_∂ᶜuₕ = DiagonalMatrixRow(adjoint(CTh(ᶜuₕ)))
    @. ∂ᶜK_∂ᶠu₃ =
        ᶜinterp_matrix() ⋅ DiagonalMatrixRow(adjoint(CT3(ᶠu₃))) +
        DiagonalMatrixRow(adjoint(CT3(ᶜuₕ))) ⋅ ᶜinterp_matrix()

    @. ᶠp_grad_matrix = DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix()

    @. ᶜadvection_matrix =
        -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠwinterp(ᶜJ, ᶜρ))

    ∂ᶜρ_err_∂ᶠu₃ = matrix[@name(c.ρ), @name(f.u₃)]
    @. ∂ᶜρ_err_∂ᶠu₃ = dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(CA.g³³(ᶠgⁱʲ))

    ∂ᶜρχ_err_∂ᶠu₃ = matrix[@name(c.ρe_tot), @name(f.u₃)]
    @. ∂ᶜρχ_err_∂ᶠu₃ =
        dtγ * ᶜadvection_matrix ⋅
        DiagonalMatrixRow(ᶠinterp(ᶜh_tot) * CA.g³³(ᶠgⁱʲ))

    ∂ᶠu₃_err_∂ᶜρ = matrix[@name(f.u₃), @name(c.ρ)]
    ∂ᶠu₃_err_∂ᶜρe_tot = matrix[@name(f.u₃), @name(c.ρe_tot)]

    @. ∂ᶠu₃_err_∂ᶜρ =
        dtγ * (
            ᶠp_grad_matrix ⋅
            DiagonalMatrixRow(ᶜkappa_m * (T_0 * cp_d - ᶜK - Φ(grav, ᶜz))) +
            DiagonalMatrixRow(ᶠgradᵥ(ᶜp) / abs2(ᶠinterp(ᶜρ))) ⋅
            ᶠinterp_matrix()
        )
    @. ∂ᶠu₃_err_∂ᶜρe_tot = dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜkappa_m)

    ∂ᶠu₃_err_∂ᶜuₕ = matrix[@name(f.u₃), @name(c.uₕ)]
    ∂ᶠu₃_err_∂ᶠu₃ = matrix[@name(f.u₃), @name(f.u₃)]
    I_u₃ = DiagonalMatrixRow(one_C3xACT3)
    @. ∂ᶠu₃_err_∂ᶜuₕ =
        dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅ ∂ᶜK_∂ᶜuₕ

    @. ∂ᶠu₃_err_∂ᶠu₃ =
        dtγ * (
            ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅ ∂ᶜK_∂ᶠu₃ +
            DiagonalMatrixRow(-CA.β_rayleigh_w(rs, ᶠz, zmax) * (one_C3xACT3,))
        ) - (I_u₃,)

end

function set_precomputed_quantities!(Y, p, t)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜu, ᶠu³, ᶠu, ᶜK, ᶜts, ᶜp) = p.precomputed

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶜz = Fields.coordinate_field(Y.c).z
    grav = FT(CAP.grav(params))
    ᶠu₃ = Y.f.u₃
    @. ᶜu = C123(ᶜuₕ) + ᶜinterp(C123(ᶠu₃))
    ᶠu³ .= CA.compute_ᶠuₕ³(ᶜuₕ, ᶜρ) .+ CT3.(ᶠu₃)
    ᶜK .= CA.compute_kinetic(ᶜuₕ, ᶠu₃)

    @. ᶜts = TD.PhaseDry_ρe(
        thermo_params,
        Y.c.ρ,
        Y.c.ρe_tot / Y.c.ρ - ᶜK - Φ(grav, ᶜz),
    )
    @. ᶜp = TD.air_pressure(thermo_params, ᶜts)

    (; ᶜh_tot) = p.precomputed
    @. ᶜh_tot =
        TD.total_specific_enthalpy(thermo_params, ᶜts, Y.c.ρe_tot / Y.c.ρ)
    return nothing
end

function dss!(Y, p, t)
    Spaces.weighted_dss!(Y.c => p.ghost_buffer.c, Y.f => p.ghost_buffer.f)
    return nothing
end

function remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    # Yₜ_lim .= zero(eltype(Yₜ_lim))
    Yₜ .= zero(eltype(Yₜ))
    (; dt, params, rayleigh_sponge) = p
    (; ᶜh_tot) = p.precomputed
    (; ᶠu³, ᶜu, ᶜK, ᶜp) = p.precomputed
    (; ᶜf³, ᶠf¹²) = p.precomputed
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜJ = Fields.local_geometry_field(Y.c).J
    grav = FT(CAP.grav(params))
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜρ = Y.c.ρ

    @. Yₜ.c.ρ -= wdivₕ(ᶜρ * ᶜu)
    @. Yₜ.c.ρe_tot -= wdivₕ(ᶜρ * ᶜh_tot * ᶜu)
    @. Yₜ.c.uₕ -= C12(gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + Φ(grav, ᶜz)))

    ᶜω³ = p.scratch.ᶜtemp_CT3
    ᶠω¹² = p.scratch.ᶠtemp_CT12

    point_type = eltype(Fields.coordinate_field(Y.c))
    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = curlₕ(ᶜuₕ)
    elseif point_type <: Geometry.Abstract2DPoint
        @. ᶜω³ = zero(ᶜω³)
    end

    @. ᶠω¹² = ᶠcurlᵥ(ᶜuₕ)
    @. ᶠω¹² += CT12(curlₕ(ᶠu₃))
    # Without the CT12(), the right-hand side would be a CT1 or CT2 in 2D space.

    ᶠω¹²′ = if isnothing(ᶠf¹²)
        ᶠω¹² # shallow atmosphere
    else
        @. lazy(ᶠf¹² + ᶠω¹²) # deep atmosphere
    end

    @. Yₜ.c.uₕ -=
        ᶜinterp(ᶠω¹²′ × (ᶠinterp(ᶜρ * ᶜJ) * ᶠu³)) / (ᶜρ * ᶜJ) +
        (ᶜf³ + ᶜω³) × CT12(ᶜu)
    @. Yₜ.f.u₃ -= ᶠω¹²′ × ᶠinterp(CT12(ᶜu)) + ᶠgradᵥ(ᶜK)

    Yₜ.c.uₕ .+= CA.rayleigh_sponge_tendency_uₕ(ᶜuₕ, rayleigh_sponge)

    return Yₜ
end

# This block:
# @time if !@isdefined(integrator)
    FT = Float64;
    ᶜspace = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 63,
        z_min = 0,
        z_max = 30000.0,
        radius = 6.371e6,
        h_elem = 30,
        n_quad_points = 4,
        staggering = CellCenter(),
    );
    ᶠspace = Spaces.face_space(ᶜspace);
    cnt = (; ρ = zero(FT), uₕ = zero(CA.C12{FT}), ρe_tot = zero(FT));
    Yc = Fields.fill(cnt, ᶜspace);
    fill!(parent(Yc.ρ), 1)
    fill!(parent(Yc.uₕ), 0.01)
    fill!(parent(Yc.ρe_tot), 1000.0)
    Yf = Fields.fill((; u₃ = zero(CA.C3{FT})), ᶠspace);
    Y = Fields.FieldVector(; c = Yc, f = Yf);

    A = ImplicitEquationJacobian(
        Y;
        approximate_solve_iters = 2,
        transform_flag = false, # assumes use_transform returns false
    )

    implicit_func = SciMLBase.ODEFunction(
        implicit_tendency!;
        jac_prototype = A,
        Wfact = Wfact!, # assumes use_transform returns false
        tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= 0),
    )

    func = CTS.ClimaODEFunction(;
        T_exp_T_lim! = remaining_tendency!,
        T_imp! = implicit_func,
        # Can we just pass implicit_tendency! and jac_prototype etc.?
        lim! = (Y, p, t, ref_Y) -> nothing, # limiters_func!
        dss!,
        cache! = set_precomputed_quantities!,
        cache_imp! = set_precomputed_quantities!,
    )

    newtons_method = CTS.NewtonsMethod(; max_iters = 2)
    params = CA.ClimaAtmosParameters(FT)
    ᶠcoord = Fields.coordinate_field(ᶠspace);
    ᶜcoord = Fields.coordinate_field(ᶜspace);
    (; ᶜf³, ᶠf¹²) = CA.compute_coriolis(ᶜcoord, ᶠcoord, params);
    scratch = (;
        ᶜtemp_CT3 = Fields.Field(CT3{FT}, ᶜspace),
        ᶠtemp_CT12 = Fields.Field(CT12{FT}, ᶠspace),
    )
    precomputed = (;
        ᶜh_tot = Fields.Field(FT, ᶜspace),
        ᶠu³ = Fields.Field(CA.CT3{FT}, ᶠspace),
        ᶜf³,
        ᶠf¹²,
        ᶜp = Fields.Field(FT, ᶜspace),
        ᶜK = Fields.Field(FT, ᶜspace),
        ᶜts = Fields.Field(TD.PhaseDry{FT}, ᶜspace),
        ᶠu = Fields.Field(C123{FT}, ᶠspace),
        ᶜu = Fields.Field(C123{FT}, ᶜspace),
    )
    dt = FT(0.1)

    ghost_buffer =
        !CA.do_dss(axes(Y.c)) ? (;) :
        (; c = Spaces.create_dss_buffer(Y.c), f = Spaces.create_dss_buffer(Y.f))

    CTh = CA.CTh_vector_type(axes(Y.c))
    p = (;
        rayleigh_sponge = CA.RayleighSponge{FT}(;
            zd = params.zd_rayleigh,
            α_uₕ = params.alpha_rayleigh_uh,
            α_w = params.alpha_rayleigh_w,
        ),
        params,
        ∂ᶜK_∂ᶜuₕ = Fields.Field(DiagonalMatrixRow{Adjoint{FT, CTh{FT}}}, ᶜspace),
        ∂ᶜK_∂ᶠu₃ = Fields.Field(BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}, ᶜspace),
        ᶜadvection_matrix = Fields.Field(
            BidiagonalMatrixRow{Adjoint{FT, C3{FT}}},
            ᶜspace,
        ),
        ᶜtemp_scalar = Fields.Field(FT, ᶜspace),
        ᶠp_grad_matrix = Fields.Field(BidiagonalMatrixRow{C3{FT}}, ᶠspace),
        scratch,
        ghost_buffer,
        dt,
        precomputed,
    )
    ode_algo = CTS.IMEXAlgorithm(CTS.ARS343(), newtons_method)
    problem = SciMLBase.ODEProblem(func, Y, (FT(0), FT(1)), p)
    integrator = SciMLBase.init(problem, ode_algo; dt)
    Yₜ = similar(integrator.u);
# end

function main!(integrator, Yₜ, n)
    for _ in 1:n
        # @time SciMLBase.step!(integrator)
        @time implicit_tendency!(Yₜ, integrator.u, integrator.p, integrator.t)
    end
    return nothing
end
using Test
if ClimaComms.device() isa ClimaComms.CUDADevice
    Yₜ_bc = similar(Yₜ);
    @. Yₜ_bc = 0
    @. Yₜ = 0
    Yc = integrator.u.c;
    Yf = integrator.u.f;
    fill!(parent(Yc.ρ), 1);
    zc = Fields.coordinate_field(Yc).z;
    zf = Fields.coordinate_field(Yf).z;
    @. Yc.ρ += 0.1*sin(zc);
    parent(Yf.u₃) .+= 0.001 .* sin.(parent(zf));
    fill!(parent(Yc.uₕ), 0.01);
    fill!(parent(Yc.ρe_tot), 100000.0);

    implicit_tendency!(Yₜ, integrator.u, integrator.p, integrator.t)
    implicit_tendency_bc!(Yₜ_bc, integrator.u, integrator.p, integrator.t)
    abs_err_c = maximum(Array(abs.(parent(Yₜ.c) .- parent(Yₜ_bc.c))))
    abs_err_f = maximum(Array(abs.(parent(Yₜ.f) .- parent(Yₜ_bc.f))))
    results_match = abs_err_c < 6e-9 && abs_err_c < 6e-9
    if !results_match
        @show norm(Array(parent(Yₜ_bc.c))), norm(Array(parent(Yₜ.c)))
        @show norm(Array(parent(Yₜ_bc.f))), norm(Array(parent(Yₜ.f)))
        @show abs_err_c
        @show abs_err_f
    end
    @test results_match
    println(CUDA.@profile begin
        # SciMLBase.step!(integrator)
        implicit_tendency!(Yₜ, integrator.u, integrator.p, integrator.t)
        implicit_tendency!(Yₜ, integrator.u, integrator.p, integrator.t)
        implicit_tendency!(Yₜ, integrator.u, integrator.p, integrator.t)
        implicit_tendency!(Yₜ, integrator.u, integrator.p, integrator.t)
    end)
    println(CUDA.@profile begin
        # SciMLBase.step!(integrator)
        @. Yₜ += 1
        @. Yₜ += 1
        @. Yₜ += 1
        @. Yₜ += 1
    end)
else
    @info "Compiling main loop"
    @time main!(integrator, Yₜ, 1)
    @info "Running main loop"
    @time main!(integrator, Yₜ, 3)
end


nothing
