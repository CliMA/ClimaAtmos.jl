module ClimaAtmosCUDSSExt

import ClimaAtmos
import ClimaAtmos:
    GPUSparseHelmholtz2DCache,
    build_sparse_helmholtz_2d_dof_map,
    wdivₕ, gradₕ, C12

using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using CUDSS
using SparseArrays
using LinearAlgebra

import ClimaCore: Spaces, Topologies, Quadratures

# ============================================================================
# GPU kernels for gather/scatter (avoid CPU round-trips)
# ============================================================================

"""
GPU gather kernel: field parent (VIJFH layout) → DOF vector for one level.
Each thread handles one (i, j, e) node. Atomic add for shared DOFs (DSS-like).
"""
function _gpu_gather_kernel!(
    rhs, field_parent, dof_map, scale, k, Nq,
)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    nelem = size(dof_map, 3)
    Nq_sq = Nq * Nq
    total = Nq_sq * nelem
    if idx <= total
        e = div(idx - Int32(1), Nq_sq) + Int32(1)
        rem = mod(idx - Int32(1), Nq_sq)
        j = div(rem, Nq) + Int32(1)
        i = mod(rem, Nq) + Int32(1)
        h_dof = dof_map[i, j, e]
        CUDA.@atomic rhs[h_dof] += scale * field_parent[k, i, j, Int32(1), e]
    end
    return nothing
end

"""
GPU scatter kernel: DOF vector → field parent (VIJFH layout) for one level.
Each thread handles one (i, j, e) node. Shared DOFs get the same value.
"""
function _gpu_scatter_kernel!(
    field_parent, sol, dof_map, k, Nq,
)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    nelem = size(dof_map, 3)
    Nq_sq = Nq * Nq
    total = Nq_sq * nelem
    if idx <= total
        e = div(idx - Int32(1), Nq_sq) + Int32(1)
        rem = mod(idx - Int32(1), Nq_sq)
        j = div(rem, Nq) + Int32(1)
        i = mod(rem, Nq) + Int32(1)
        h_dof = dof_map[i, j, e]
        field_parent[k, i, j, Int32(1), e] = sol[h_dof]
    end
    return nothing
end

"""
GPU field-to-vec kernel: extract all levels from field parent → flat DOF vector.
Layout: vec[(k-1)*N_h + h_dof]. Each thread handles one (i, j, e, k).
"""
function _gpu_field_to_vec_kernel!(
    vec, field_parent, dof_map, N_h, N_v, Nq,
)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    nelem = size(dof_map, 3)
    Nq_sq = Nq * Nq
    total = N_v * Nq_sq * nelem
    if idx <= total
        # Decompose: idx → (k, i, j, e) with e outermost
        e = div(idx - Int32(1), N_v * Nq_sq) + Int32(1)
        rem1 = mod(idx - Int32(1), N_v * Nq_sq)
        j = div(rem1, N_v * Nq) + Int32(1)
        rem2 = mod(rem1, N_v * Nq)
        i = div(rem2, N_v) + Int32(1)
        k = mod(rem2, N_v) + Int32(1)
        h_dof = dof_map[i, j, e]
        vec[(k - Int32(1)) * N_h + h_dof] = field_parent[k, i, j, Int32(1), e]
    end
    return nothing
end

"""
GPU vec-to-field kernel: write flat DOF vector → field parent (all levels).
Layout: vec[(k-1)*N_h + h_dof]. Each thread handles one (i, j, e, k).
"""
function _gpu_vec_to_field_kernel!(
    field_parent, vec, dof_map, N_h, N_v, Nq,
)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    nelem = size(dof_map, 3)
    Nq_sq = Nq * Nq
    total = N_v * Nq_sq * nelem
    if idx <= total
        e = div(idx - Int32(1), N_v * Nq_sq) + Int32(1)
        rem1 = mod(idx - Int32(1), N_v * Nq_sq)
        j = div(rem1, N_v * Nq) + Int32(1)
        rem2 = mod(rem1, N_v * Nq)
        i = div(rem2, N_v) + Int32(1)
        k = mod(rem2, N_v) + Int32(1)
        h_dof = dof_map[i, j, e]
        field_parent[k, i, j, Int32(1), e] = vec[(k - Int32(1)) * N_h + h_dof]
    end
    return nothing
end

# ============================================================================
# Launcher helpers
# ============================================================================

const _GPU_THREADS = 256

"""
GPU gather: zero `rhs`, then gather field parent → rhs for level `k` with scale.
"""
function _gpu_gather!(rhs, field_parent, dof_map_gpu, scale, k, Nq, nelem)
    rhs .= zero(eltype(rhs))
    total = Nq * Nq * nelem
    nblocks = cld(total, _GPU_THREADS)
    CUDA.@cuda threads=_GPU_THREADS blocks=nblocks _gpu_gather_kernel!(
        rhs, field_parent, dof_map_gpu, scale, Int32(k), Int32(Nq),
    )
    return nothing
end

"""
GPU scatter: sol → field parent for level `k`.
"""
function _gpu_scatter!(field_parent, sol, dof_map_gpu, k, Nq, nelem)
    total = Nq * Nq * nelem
    nblocks = cld(total, _GPU_THREADS)
    CUDA.@cuda threads=_GPU_THREADS blocks=nblocks _gpu_scatter_kernel!(
        field_parent, sol, dof_map_gpu, Int32(k), Int32(Nq),
    )
    return nothing
end

"""
GPU field-to-vec: extract all levels from field → flat GPU vector.
"""
function _gpu_field_to_vec!(gpu_vec, field_parent, dof_map_gpu, N_h, N_v, Nq, nelem)
    total = N_v * Nq * Nq * nelem
    nblocks = cld(total, _GPU_THREADS)
    CUDA.@cuda threads=_GPU_THREADS blocks=nblocks _gpu_field_to_vec_kernel!(
        gpu_vec, field_parent, dof_map_gpu, Int32(N_h), Int32(N_v), Int32(Nq),
    )
    return nothing
end

"""
GPU vec-to-field: write flat GPU vector → field parent (all levels).
"""
function _gpu_vec_to_field!(field_parent, gpu_vec, dof_map_gpu, N_h, N_v, Nq, nelem)
    total = N_v * Nq * Nq * nelem
    nblocks = cld(total, _GPU_THREADS)
    CUDA.@cuda threads=_GPU_THREADS blocks=nblocks _gpu_vec_to_field_kernel!(
        field_parent, gpu_vec, dof_map_gpu, Int32(N_h), Int32(N_v), Int32(Nq),
    )
    return nothing
end

# ============================================================================
# Public GPU field↔vec wrappers
# ============================================================================

function ClimaAtmos.gpu_sparse_helmholtz_2d_field_to_vec!(
    gpu_vec::CuVector{FT}, field, shc::GPUSparseHelmholtz2DCache{FT},
) where {FT}
    _gpu_field_to_vec!(
        gpu_vec, parent(field), shc.dof_map_gpu,
        shc.N_h, shc.N_v, shc.Nq, shc.nelem,
    )
    return gpu_vec
end

function ClimaAtmos.gpu_sparse_helmholtz_2d_vec_to_field!(
    field, gpu_vec::CuVector{FT}, shc::GPUSparseHelmholtz2DCache{FT},
) where {FT}
    _gpu_vec_to_field!(
        parent(field), gpu_vec, shc.dof_map_gpu,
        shc.N_h, shc.N_v, shc.Nq, shc.nelem,
    )
    return field
end

# ============================================================================
# Builder
# ============================================================================

"""
    build_gpu_sparse_helmholtz_2d_cache(::Type{FT}, Y; backsub_alpha)

Build a `GPUSparseHelmholtz2DCache` with CUDSS solvers for GPU-native
sparse direct Helmholtz solves. Assembly happens on CPU; factorization
and solve happen on-device via `CuSparseMatrixCSR` + `CudssSolver`.
Gather/scatter use GPU kernels with a GPU-resident DOF map — zero CPU
round-trips in the per-GMRES-iteration correction path.
"""
function ClimaAtmos.build_gpu_sparse_helmholtz_2d_cache(
    ::Type{FT},
    Y;
    backsub_alpha::FT = FT(0),
) where {FT}
    cspace = axes(Y.c)
    hspace = Spaces.horizontal_space(cspace)
    topology = Spaces.topology(hspace)
    quad = Spaces.quadrature_style(hspace)

    Nq = Quadratures.degrees_of_freedom(quad)
    nelem = Topologies.nlocalelems(topology)
    N_v = Spaces.nlevels(cspace)

    # Build DOF mapping (reuse existing)
    dof_map, N_h = build_sparse_helmholtz_2d_dof_map(topology, Nq)
    N_dof = N_h * N_v

    @info "GPUSparseHelmholtz2D (CUDSS): nelem=$nelem, Nq=$Nq, N_h=$N_h, N_v=$N_v, N_dof=$N_dof"

    # GPU copy of DOF map (Int32 for GPU efficiency)
    dof_map_gpu = CuArray{Int32}(dof_map)

    # GLL quadrature data
    points, w = Quadratures.quadrature_points(FT, quad)
    D = Matrix{FT}(Quadratures.differentiation_matrix(FT, quad))
    weights = Vector{FT}(w)
    K1D_ref = D' * Diagonal(weights) * D

    # Build sparsity pattern (same COO logic as CPU path)
    I_idx = Int[]
    J_idx = Int[]
    V_val = FT[]

    for e in 1:nelem
        for j1 in 1:Nq, i1 in 1:Nq
            h_row = dof_map[i1, j1, e]
            for i2 in 1:Nq
                h_col = dof_map[i2, j1, e]
                push!(I_idx, h_row)
                push!(J_idx, h_col)
                push!(V_val, FT(0))
            end
            for j2 in 1:Nq
                h_col = dof_map[i1, j2, e]
                push!(I_idx, h_row)
                push!(J_idx, h_col)
                push!(V_val, FT(0))
            end
        end
    end

    # Create CPU template and per-level copies
    H_template = sparse(I_idx, J_idx, V_val, N_h, N_h)
    H_levels_cpu = [copy(H_template) for _ in 1:N_v]

    # Extract WJ values
    WJ_vec = zeros(FT, N_h)
    lg_data = Spaces.local_geometry_data(hspace)
    WJ_arr = Array(parent(lg_data.WJ))
    @inbounds for e in 1:nelem
        for j in 1:Nq, i in 1:Nq
            h_dof = dof_map[i, j, e]
            WJ_vec[h_dof] += FT(WJ_arr[i, j, 1, e])
        end
    end

    # Convert CSC → CSR for CUDSS: create GPU matrices and solvers
    H_levels_gpu = Vector{CuSparseMatrixCSR{FT}}(undef, N_v)
    H_solvers = Vector{CudssSolver}(undef, N_v)

    for k in 1:N_v
        # Initialize with identity-like values to get valid structure for analysis
        Hk_cpu = H_levels_cpu[k]
        for h in 1:N_h
            Hk_cpu[h, h] += FT(1)
        end
        # Convert to GPU CSR
        Hk_gpu = CuSparseMatrixCSR(Hk_cpu)
        H_levels_gpu[k] = Hk_gpu

        # Create CUDSS solver and perform symbolic analysis (one-time)
        structure = "G"  # general (not symmetric)
        view_type = 'F'  # full matrix
        solver = CudssSolver(Hk_gpu, structure, view_type)
        cudss("analysis", solver, CuVector{FT}(undef, N_h), CuVector{FT}(undef, N_h))
        H_solvers[k] = solver
    end

    # GPU work vectors
    rhs_gpu = CuVector{FT}(undef, N_h)
    sol_gpu = CuVector{FT}(undef, N_h)

    # CPU gather/scatter buffers (used only in assembly path)
    rhs_cpu = zeros(FT, N_h)
    sol_cpu = zeros(FT, N_h)

    # GPU state vectors for GPU-native field extraction
    cs2_vec_gpu = CuVector{FT}(undef, N_dof)
    ρ_vec_gpu = CuVector{FT}(undef, N_dof)
    e_tot_vec_gpu = CuVector{FT}(undef, N_dof)

    # Scratch fields
    _scratch_ρ = similar(Y.c, FT)
    _scratch_div = similar(Y.c, FT)
    _scratch_e_tot = similar(Y.c, FT)

    return GPUSparseHelmholtz2DCache{FT, CuSparseMatrixCSR{FT}, CudssSolver, CuVector{FT}}(
        H_levels_cpu,
        H_levels_gpu,
        H_solvers,
        rhs_gpu,
        sol_gpu,
        rhs_cpu,
        sol_cpu,
        zeros(FT, N_dof),        # cs2_vec (CPU, for assembly)
        zeros(FT, N_dof),        # ρ_vec (CPU, for assembly)
        zeros(FT, N_dof),        # e_tot_vec (CPU, for assembly)
        cs2_vec_gpu,
        ρ_vec_gpu,
        e_tot_vec_gpu,
        Ref(FT(0)),              # dtγ
        backsub_alpha,
        dof_map,
        dof_map_gpu,
        D,
        weights,
        Matrix{FT}(K1D_ref),
        WJ_vec,
        Nq,
        nelem,
        N_h,
        N_v,
        N_dof,
        _scratch_ρ,
        _scratch_div,
        _scratch_e_tot,
    )
end

# ============================================================================
# Assembly (CPU assembly + GPU factorization — unchanged)
# ============================================================================

"""
    assemble_gpu_sparse_helmholtz_2d!(shc::GPUSparseHelmholtz2DCache)

Assemble per-level Helmholtz matrices on CPU, then copy nzvals to GPU
and refactorize via CUDSS.
"""
function ClimaAtmos.assemble_gpu_sparse_helmholtz_2d!(
    shc::GPUSparseHelmholtz2DCache{FT},
) where {FT}
    (; H_levels_cpu, H_levels_gpu, H_solvers,
        cs2_vec, WJ_vec, K1D_ref, weights, dtγ,
        dof_map, Nq, nelem, N_h, N_v,
        rhs_gpu, sol_gpu) = shc

    δtγ² = dtγ[]^2

    @inbounds for k in 1:N_v
        Hk_cpu = H_levels_cpu[k]
        Hk_cpu.nzval .= FT(0)

        # Mass matrix (diagonal)
        for h in 1:N_h
            Hk_cpu[h, h] += FT(WJ_vec[h])
        end

        # Stiffness matrix — assembled per element
        for e in 1:nelem
            for j in 1:Nq, i in 1:Nq
                h_row = dof_map[i, j, e]
                cs2_row = cs2_vec[(k - 1) * N_h + h_row]

                # Direction 1
                w_j = weights[j]
                for i2 in 1:Nq
                    h_col = dof_map[i2, j, e]
                    k_val = K1D_ref[i, i2]
                    Hk_cpu[h_row, h_col] += δtγ² * cs2_row * w_j * k_val
                end

                # Direction 2
                w_i = weights[i]
                for j2 in 1:Nq
                    j2 == j && continue
                    h_col = dof_map[i, j2, e]
                    k_val = K1D_ref[j, j2]
                    Hk_cpu[h_row, h_col] += δtγ² * cs2_row * w_i * k_val
                end
            end
        end

        # Copy nzvals to GPU CSR matrix
        Hk_gpu = H_levels_gpu[k]
        Hk_csr = CuSparseMatrixCSR(Hk_cpu)
        copyto!(Hk_gpu.nzVal, Hk_csr.nzVal)

        # Numerical factorization on GPU
        cudss("factorization", H_solvers[k], sol_gpu, rhs_gpu)
    end

    return nothing
end

# ============================================================================
# GPU-native correction (zero CPU round-trips)
# ============================================================================

"""
    gpu_sparse_helmholtz_2d_correction!(shc::GPUSparseHelmholtz2DCache, ΔY)

Apply additive 2D sparse Helmholtz correction with fully GPU-native pipeline.
The entire gather → CUDSS solve → scatter path stays on GPU with no CPU
round-trips per GMRES iteration.
"""
function ClimaAtmos.gpu_sparse_helmholtz_2d_correction!(
    shc::GPUSparseHelmholtz2DCache{FT},
    ΔY,
) where {FT}
    (; H_solvers, rhs_gpu, sol_gpu,
        dtγ, ρ_vec_gpu, backsub_alpha, cs2_vec_gpu, e_tot_vec_gpu,
        dof_map_gpu, Nq, nelem, N_h, N_v,
        _scratch_ρ, _scratch_div, _scratch_e_tot) = shc

    # Step 1: Scatter ρ state vector to scratch field (GPU kernel, all levels)
    _gpu_vec_to_field!(
        parent(_scratch_ρ), ρ_vec_gpu, dof_map_gpu, N_h, N_v, Nq, nelem,
    )

    # Step 2: Form wdivₕ(ρ · z.uₕ) into scratch field (GPU broadcast + DSS)
    @. _scratch_div = wdivₕ(_scratch_ρ * ΔY.c.uₕ)
    Spaces.weighted_dss!(_scratch_div)

    δtγ_val = dtγ[]
    div_parent = parent(_scratch_div)
    ρ_parent = parent(_scratch_ρ)

    # Step 3: Per-level GPU gather → CUDSS solve → GPU scatter
    @inbounds for k in 1:N_v
        # Gather RHS: rhs_gpu[h_dof] += -δtγ * div_parent[k,i,j,1,e]
        _gpu_gather!(rhs_gpu, div_parent, dof_map_gpu, FT(-δtγ_val), k, Nq, nelem)

        # CUDSS solve on GPU (already factorized)
        cudss("solve", H_solvers[k], sol_gpu, rhs_gpu)

        # Scatter solution → scratch_ρ parent for this level
        _gpu_scatter!(ρ_parent, sol_gpu, dof_map_gpu, k, Nq, nelem)
    end

    # Step 4: DSS the solution and add to ΔY (GPU broadcasts)
    Spaces.weighted_dss!(_scratch_ρ)
    @. ΔY.c.ρ += _scratch_ρ

    # Step 5: Damped uₕ and ρe_tot back-substitution (GPU broadcasts + GPU vec-to-field)
    if backsub_alpha > FT(0)
        α = backsub_alpha

        # cs²/ρ → _scratch_div
        _gpu_vec_to_field!(
            parent(_scratch_div), cs2_vec_gpu, dof_map_gpu, N_h, N_v, Nq, nelem,
        )
        _gpu_vec_to_field!(
            parent(_scratch_e_tot), ρ_vec_gpu, dof_map_gpu, N_h, N_v, Nq, nelem,
        )
        @. _scratch_div = _scratch_div / max(_scratch_e_tot, FT(1e-6))

        @. ΔY.c.uₕ -= C12(FT(α * δtγ_val) * _scratch_div * gradₕ(_scratch_ρ))

        # e_tot → _scratch_e_tot
        _gpu_vec_to_field!(
            parent(_scratch_e_tot), e_tot_vec_gpu, dof_map_gpu, N_h, N_v, Nq, nelem,
        )
        @. ΔY.c.ρe_tot += FT(α) * _scratch_e_tot * _scratch_ρ
    end

    return nothing
end

end # module
