import LinearAlgebra: I, Adjoint, Diagonal, lu
using SparseArrays

using ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name
import ClimaCore.Topologies as Topologies
import ClimaCore.Quadratures as Quadratures

abstract type DerivativeFlag end
struct UseDerivative <: DerivativeFlag end
struct IgnoreDerivative <: DerivativeFlag end

DerivativeFlag(value) = value ? UseDerivative() : IgnoreDerivative()
DerivativeFlag(mode::AbstractTimesteppingMode) =
    DerivativeFlag(mode == Implicit())

use_derivative(::UseDerivative) = true
use_derivative(::IgnoreDerivative) = false

"""
    ManualSparseJacobian(
        topography_flag,
        diffusion_flag,
        sgs_advection_flag,
        sgs_entr_detr_flag,
        sgs_mass_flux_flag,
        sgs_nh_pressure_flag,
        sgs_vertdiff_flag,
        acoustic_diagonal_flag,
        approximate_solve_iters,
        sparse_helmholtz_flag,
        sparse_helmholtz_2d_flag,
    )

A [`JacobianAlgorithm`](@ref) that approximates the Jacobian using analytically
derived tendency derivatives and inverts it using a specialized nested linear
solver. Certain groups of derivatives can be toggled on or off by setting their
`DerivativeFlag`s to either `UseDerivative` or `IgnoreDerivative`.

# Arguments

- `topography_flag::DerivativeFlag`: whether the derivative of vertical
  contravariant velocity with respect to horizontal covariant velocity should be
  computed
- `diffusion_flag::DerivativeFlag`: whether the derivatives of the grid-scale
  diffusion tendency should be computed
- `sgs_advection_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale advection tendency should be computed
- `sgs_entr_detr_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale entrainment and detrainment tendencies should be computed
- `sgs_mass_flux_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale mass flux tendency should be computed
- `sgs_nh_pressure_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale non-hydrostatic pressure drag tendency should be computed
- `sgs_vertdiff_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale vertical diffusion tendency should be computed
- `acoustic_diagonal_flag::DerivativeFlag`: whether to add a diagonal
  approximation of the horizontal acoustic Schur complement to the `(ρ,ρ)`,
  `(uₕ,uₕ)`, and `(ρe_tot,ρe_tot)` blocks, improving convergence for fully
  implicit solves where horizontal acoustic/gravity wave stiffness dominates
- `approximate_solve_iters::Int`: number of iterations to take for the
  approximate linear solve required when the `diffusion_flag` is `UseDerivative`
- `n_helmholtz_iters::Int`: number of Jacobi-preconditioned Richardson
  iterations for the horizontal Helmholtz solve in the preconditioner (only
  used when `acoustic_diagonal_flag` is `UseDerivative`). 0 = diagonal only.
- `sparse_helmholtz_flag::DerivativeFlag`: whether to use a sparse direct
  Helmholtz preconditioner for horizontal acoustics (plane topology only).
  Independent of `acoustic_diagonal_flag` and `n_helmholtz_iters`. Uses
  LU-factorized sparse matrix solve instead of iterative Chebyshev. Requires
  FGMRES.
- `sparse_helmholtz_2d_flag::DerivativeFlag`: whether to use a 2D sparse direct
  Helmholtz preconditioner for horizontal acoustics on sphere grids. Assembles
  the SE stiffness matrix using cubed-sphere topology connectivity with proper
  face/vertex DOF sharing. LU-factorized and applied level-by-level. Requires
  FGMRES.
- `helmholtz_2d_solver::String`: `"direct"` (sparse LU) or `"chebyshev"` (matrix-free).
- `n_helmholtz_2d_iters::Int`: Chebyshev iterations for matrix-free 2D Helmholtz.
"""
struct ManualSparseJacobian{F1, F2, F3, F4, F5, F6, F7, F8, F9, F10} <: SparseJacobian
    topography_flag::F1
    diffusion_flag::F2
    sgs_advection_flag::F3
    sgs_entr_detr_flag::F4
    sgs_mass_flux_flag::F5
    sgs_nh_pressure_flag::F6
    sgs_vertdiff_flag::F7
    acoustic_diagonal_flag::F8
    approximate_solve_iters::Int
    n_helmholtz_iters::Int
    sparse_helmholtz_flag::F9
    sparse_helmholtz_2d_flag::F10
    helmholtz_backsub_alpha::Float64
    helmholtz_2d_solver::String
    n_helmholtz_2d_iters::Int
end

"""
    HelmholtzPreconditionerState

Stores state needed by `helmholtz_correction!` during `invert_jacobian!`.
These quantities are captured from `update_jacobian!` so that the Helmholtz
solve in the preconditioner has access to the current Jacobian parameters.
"""
mutable struct HelmholtzPreconditionerState{FT, F, GB}
    dtγ::FT
    α_acoustic_max::FT   # max(ᶜα_acoustic) for Chebyshev eigenvalue bounds
    ᶜα_acoustic::F       # diagonal: 1 + dtγ²·cs²·2π²/Δx²
    ᶜcs²::F              # sound speed squared: γ_d·ᶜp/ᶜρ
    ᶜρ::F                # density
    ᶜe_tot::F            # specific total energy (for tracer correction)
    ᶜh_tot::F            # total specific enthalpy: (ρe_tot + p)/ρ (for energy Helmholtz)
    ghost_buffer_c::GB   # DSS buffer for center vector fields (uₕ)
    n_helmholtz_iters::Int
    call_counter::Int     # tracks invert_jacobian! calls since last Jacobian update
end

"""
    SparseHelmholtzCache

Stores the sparse Helmholtz matrix and work vectors for the sparse direct
Helmholtz preconditioner. The Helmholtz operator is:
    H = M_h + δtγ² · diag(cs²) · K_h
where M_h is the GLL mass matrix and K_h is the SE stiffness matrix
(horizontal only, no vertical coupling).
"""
struct SparseHelmholtzCache{FT}
    H_sparse::SparseMatrixCSC{FT, Int}
    H_factorization::Base.RefValue{Any}
    rhs_vec::Vector{FT}
    sol_vec::Vector{FT}
    cs2_vec::Vector{FT}
    ρ_vec::Vector{FT}
    dtγ::Base.RefValue{FT}
    D_matrix::Matrix{FT}
    weights::Vector{FT}
    elem_Δx::FT
    helem::Int
    npoly::Int
    Nq::Int
    N_h::Int
    N_v::Int
    N_dof::Int
    _scratch_ρ::Any    # center scalar scratch field
    _scratch_div::Any  # center scalar scratch field
    # UMFPACK `lu` on sparse matrices uses Float64; use these for ldiv! when FT === Float32
    rhs_work64::Union{Nothing, Vector{Float64}}
    sol_work64::Union{Nothing, Vector{Float64}}
end

"""
Global DOF index from horizontal DOF i_h and vertical level k.
"""
@inline function _sparse_helmholtz_global_dof(i_h::Int, k::Int, N_h::Int)
    return (k - 1) * N_h + i_h
end

"""
Map (element e, local node q) to unique horizontal DOF index.
Periodic: node Nq of element e shares with node 1 of element e+1.
"""
@inline function _sparse_helmholtz_h_dof(e::Int, q::Int, npoly::Int, Nq::Int, helem::Int)
    if q == Nq
        next_e = (e == helem) ? 1 : e + 1
        return (next_e - 1) * npoly + 1
    else
        return (e - 1) * npoly + q
    end
end

function build_sparse_helmholtz_cache(::Type{FT}, Y) where {FT}
    cspace = axes(Y.c)
    hspace = Spaces.horizontal_space(cspace)
    topology = Spaces.topology(hspace)
    quad = Spaces.quadrature_style(hspace)

    helem = Topologies.nlocalelems(topology)
    Nq = Quadratures.degrees_of_freedom(quad)
    npoly = Nq - 1
    N_h = helem * npoly  # unique horizontal DOFs (periodic)
    N_v = Spaces.nlevels(cspace)
    N_dof = N_h * N_v

    # GLL quadrature data
    points, w = Quadratures.quadrature_points(FT, quad)
    D = Matrix{FT}(Quadratures.differentiation_matrix(FT, quad))
    weights = Vector{FT}(w)

    # Element physical width: node spacing × nodes per element
    Δx_node = FT(Spaces.node_horizontal_length_scale(hspace))
    elem_Δx = Δx_node * npoly

    J_e = elem_Δx / 2

    # Build sparsity pattern (horizontal stiffness only)
    I_idx = Int[]
    J_idx = Int[]
    V_val = FT[]

    for k in 1:N_v
        for e in 1:helem
            for qi in 1:Nq
                i_h = _sparse_helmholtz_h_dof(e, qi, npoly, Nq, helem)
                row = _sparse_helmholtz_global_dof(i_h, k, N_h)
                for qj in 1:Nq
                    j_h = _sparse_helmholtz_h_dof(e, qj, npoly, Nq, helem)
                    col = _sparse_helmholtz_global_dof(j_h, k, N_h)
                    push!(I_idx, row)
                    push!(J_idx, col)
                    push!(V_val, FT(1))
                end
            end
        end
    end

    H_sparse = sparse(I_idx, J_idx, V_val, N_dof, N_dof)

    _scratch_ρ = similar(Y.c, FT)
    _scratch_div = similar(Y.c, FT)

    rhs_work64 = FT === Float32 ? zeros(Float64, N_dof) : nothing
    sol_work64 = FT === Float32 ? zeros(Float64, N_dof) : nothing

    return SparseHelmholtzCache{FT}(
        H_sparse,
        Ref{Any}(nothing),
        zeros(FT, N_dof),
        zeros(FT, N_dof),
        zeros(FT, N_dof),
        zeros(FT, N_dof),
        Ref(FT(0)),
        D,
        weights,
        elem_Δx,
        helem,
        npoly,
        Nq,
        N_h,
        N_v,
        N_dof,
        _scratch_ρ,
        _scratch_div,
        rhs_work64,
        sol_work64,
    )
end

"""
Extract a center scalar field to a flat vector using the DOF mapping.
Parent layout: (Nv, Ni, Nf, Nh) = (N_v, Nq, 1, helem)
"""
function _sparse_helmholtz_field_to_vec!(
    vec::Vector{FT}, field, helem, npoly, N_v, N_h,
) where {FT}
    arr = Array(parent(field))  # copy to CPU for scalar iteration
    @inbounds for e in 1:helem
        for q in 1:npoly
            i_h = (e - 1) * npoly + q
            for k in 1:N_v
                vec[_sparse_helmholtz_global_dof(i_h, k, N_h)] = arr[k, q, 1, e]
            end
        end
    end
    return vec
end

"""
Write a flat vector back to a center scalar field.
"""
function _sparse_helmholtz_vec_to_field!(
    field, vec::Vector{FT}, helem, npoly, Nq, N_v, N_h,
) where {FT}
    arr = Array(parent(field))  # copy to CPU for scalar iteration
    @inbounds for e in 1:helem
        for q in 1:npoly
            i_h = (e - 1) * npoly + q
            for k in 1:N_v
                arr[k, q, 1, e] = vec[_sparse_helmholtz_global_dof(i_h, k, N_h)]
            end
        end
    end
    # Copy shared boundary: q=Nq of element e = q=1 of element e+1
    @inbounds for e in 1:helem
        next_e = (e == helem) ? 1 : e + 1
        for k in 1:N_v
            arr[k, Nq, 1, e] = arr[k, 1, 1, next_e]
        end
    end
    copyto!(parent(field), arr)  # copy back to GPU
    return field
end

"""
    assemble_sparse_helmholtz!(shc::SparseHelmholtzCache)

Assemble the horizontal Helmholtz matrix H = M_h + δtγ² · diag(cs²) · K_h
and compute its LU factorization.
"""
function assemble_sparse_helmholtz!(shc::SparseHelmholtzCache{FT}) where {FT}
    (; H_sparse, cs2_vec, D_matrix, weights, elem_Δx, dtγ,
        helem, npoly, Nq, N_h, N_v) = shc

    J_e = elem_Δx / 2
    δtγ² = dtγ[]^2

    K_elem = (1 / J_e) * (D_matrix' * Diagonal(weights) * D_matrix)
    M_elem = weights .* J_e

    H_sparse.nzval .= FT(0)

    @inbounds for k in 1:N_v
        for e in 1:helem
            for qi in 1:Nq
                i_h = _sparse_helmholtz_h_dof(e, qi, npoly, Nq, helem)
                row = _sparse_helmholtz_global_dof(i_h, k, N_h)
                cs2_ik = cs2_vec[row]

                for qj in 1:Nq
                    j_h = _sparse_helmholtz_h_dof(e, qj, npoly, Nq, helem)
                    col = _sparse_helmholtz_global_dof(j_h, k, N_h)
                    m_val = (qi == qj) ? M_elem[qi] : FT(0)
                    k_val = K_elem[qi, qj]
                    H_sparse[row, col] += m_val + δtγ² * cs2_ik * k_val
                end
            end
        end
    end

    shc.H_factorization[] = lu(H_sparse)
    return nothing
end

"""
    sparse_helmholtz_correction!(shc::SparseHelmholtzCache, ΔY)

Apply additive sparse Helmholtz correction after column-local solve.
1. RHS = -δtγ · M · wdivₕ(ρ · z.uₕ)
2. Solve (M_h + δtγ² · cs² · K_h) · Δρ_h = RHS
3. ΔY.c.ρ += Δρ_h (additive, ρ-only)
"""
function sparse_helmholtz_correction!(shc::SparseHelmholtzCache{FT}, ΔY) where {FT}
    (; H_factorization, rhs_vec, sol_vec, dtγ, ρ_vec,
        weights, elem_Δx, helem, npoly, Nq, N_h, N_v, N_dof,
        _scratch_ρ, _scratch_div, rhs_work64, sol_work64) = shc

    J_e = elem_Δx / 2
    M_elem = weights .* J_e

    # Step 1: Form RHS = -δtγ · M · wdivₕ(ρ · z.uₕ)
    _sparse_helmholtz_vec_to_field!(_scratch_ρ, ρ_vec, helem, npoly, Nq, N_v, N_h)
    @. _scratch_div = wdivₕ(_scratch_ρ * ΔY.c.uₕ)
    Spaces.weighted_dss!(_scratch_div)

    rhs_vec .= FT(0)
    arr_div = Array(parent(_scratch_div))  # copy to CPU for scalar gather
    @inbounds for e in 1:helem
        for q in 1:Nq
            i_h = _sparse_helmholtz_h_dof(e, q, npoly, Nq, helem)
            M_q = M_elem[q]
            for k in 1:N_v
                idx = _sparse_helmholtz_global_dof(i_h, k, N_h)
                rhs_vec[idx] += -dtγ[] * M_q * arr_div[k, q, 1, e]
            end
        end
    end

    # Step 2: Solve H · Δρ_h = rhs
    # Sparse UMFPACK factorization is Float64; rhs/sol are FT (often Float32)
    Fac = H_factorization[]
    if rhs_work64 !== nothing
        @. rhs_work64 = rhs_vec
        LinearAlgebra.ldiv!(sol_work64, Fac, rhs_work64)
        @. sol_vec = sol_work64
    else
        LinearAlgebra.ldiv!(sol_vec, Fac, rhs_vec)
    end

    # Step 3: Write correction to scratch field, DSS, add to ΔY
    _sparse_helmholtz_vec_to_field!(_scratch_ρ, sol_vec, helem, npoly, Nq, N_v, N_h)
    Spaces.weighted_dss!(_scratch_ρ)
    @. ΔY.c.ρ += _scratch_ρ

    return nothing
end

# ============================================================================
# 2D Sparse Helmholtz for sphere (cubed-sphere topology)
# ============================================================================

# --- Union-Find helpers for DOF merging ---
function _uf_find!(parent::Vector{Int}, i::Int)
    while parent[i] != i
        parent[i] = parent[parent[i]]  # path compression
        i = parent[i]
    end
    return i
end

function _uf_union!(parent::Vector{Int}, rank::Vector{Int}, a::Int, b::Int)
    ra = _uf_find!(parent, a)
    rb = _uf_find!(parent, b)
    ra == rb && return
    if rank[ra] < rank[rb]
        parent[ra] = rb
    elseif rank[ra] > rank[rb]
        parent[rb] = ra
    else
        parent[rb] = ra
        rank[ra] += 1
    end
    return
end

"""
    SparseHelmholtz2DCache

Stores per-level sparse Helmholtz matrices and work vectors for the 2D sparse
direct Helmholtz preconditioner on sphere (cubed-sphere) grids. Each vertical
level has an independent N_h × N_h Helmholtz system:
    H_k = M_h + δtγ² · diag(cs²_k) · K_h
where M_h is the GLL mass matrix (using actual WJ values) and K_h is the 2D SE
stiffness matrix using an isotropic metric approximation. DOFs are mapped using
cubed-sphere topology connectivity (shared face/vertex nodes merged).

Level-decoupled storage avoids the O(N_v²) fill-in cost of a monolithic
N_dof × N_dof LU and enables per-level parallelism.
"""
struct SparseHelmholtz2DCache{FT}
    # Per-level Helmholtz matrices and factorizations (length N_v)
    H_levels::Vector{SparseMatrixCSC{FT, Int}}
    H_level_factors::Vector{Base.RefValue{Any}}
    # Per-level work vectors (length N_h each)
    rhs_lev::Vector{FT}
    sol_lev::Vector{FT}
    # State vectors (length N_h * N_v, level-major: idx = (k-1)*N_h + h)
    cs2_vec::Vector{FT}
    ρ_vec::Vector{FT}
    e_tot_vec::Vector{FT}       # specific total energy e_tot = ρe_tot/ρ (for back-sub)
    dtγ::Base.RefValue{FT}
    backsub_alpha::FT           # damping for uₕ/ρe_tot back-sub (0 = ρ-only)
    # DOF mapping: dof_map[i, j, e] → unique horizontal DOF index (1..N_h)
    dof_map::Array{Int, 3}
    # 1D GLL quadrature
    D_matrix::Matrix{FT}        # Nq × Nq differentiation matrix
    weights::Vector{FT}         # Nq quadrature weights
    K1D_ref::Matrix{FT}         # Nq × Nq reference 1D stiffness: Σ_q w_q D[q,a]D[q,b]
    # Per-DOF mass (length N_h): WJ accumulated across shared elements
    WJ_vec::Vector{FT}
    # Mesh metadata
    Nq::Int
    nelem::Int
    N_h::Int                    # unique horizontal DOFs
    N_v::Int                    # vertical levels
    N_dof::Int                  # N_h * N_v
    _scratch_ρ::Any             # center scalar scratch field (reused as Δρ output)
    _scratch_div::Any           # center scalar scratch field (reused as cs²/ρ for back-sub)
    _scratch_e_tot::Any         # center scalar scratch field (e_tot for ρe_tot back-sub)
    # Float32 → Float64 temporaries for UMFPACK (length N_h)
    rhs_work64::Union{Nothing, Vector{Float64}}
    sol_work64::Union{Nothing, Vector{Float64}}
end

"""
    GPUSparseHelmholtz2DCache

GPU-native sparse Helmholtz preconditioner using CUDSS.jl for direct solves.
Same mathematical formulation as `SparseHelmholtz2DCache` but factorization
and solve happen on-device via `CuSparseMatrixCSR` + `CudssSolver`.

Assembly is done on CPU (into `H_levels_cpu`), then nzvals are copied to GPU
and refactorized per Newton step. The solve is GPU-native per vertical level.
"""
struct GPUSparseHelmholtz2DCache{FT, GM, SV, GV}
    # Per-level Helmholtz matrices: CPU for assembly, GPU for solve
    H_levels_cpu::Vector{SparseMatrixCSC{FT, Int}}
    H_levels_gpu::Vector{GM}       # CuSparseMatrixCSR per level
    H_solvers::Vector{SV}          # CudssSolver per level
    # GPU work vectors (length N_h)
    rhs_gpu::GV
    sol_gpu::GV
    # CPU gather/scatter buffers (length N_h)
    rhs_cpu::Vector{FT}
    sol_cpu::Vector{FT}
    # State vectors (length N_h * N_v, level-major)
    cs2_vec::Vector{FT}
    ρ_vec::Vector{FT}
    e_tot_vec::Vector{FT}
    dtγ::Base.RefValue{FT}
    backsub_alpha::FT
    # DOF mapping and quadrature
    dof_map::Array{Int, 3}
    D_matrix::Matrix{FT}
    weights::Vector{FT}
    K1D_ref::Matrix{FT}
    WJ_vec::Vector{FT}
    # Mesh metadata
    Nq::Int
    nelem::Int
    N_h::Int
    N_v::Int
    N_dof::Int
    # Scratch fields for ClimaCore broadcasts
    _scratch_ρ::Any
    _scratch_div::Any
    _scratch_e_tot::Any
end

# Stubs for GPU sparse Helmholtz — implemented by ext/ClimaAtmosCUDSSExt.jl
const _CUDSS_EXT_MSG =
    "helmholtz_2d_solver=\"gpu_direct\" requires CUDA.jl and CUDSS.jl. " *
    "Add `using CUDA, CUDSS` before loading ClimaAtmos."

function build_gpu_sparse_helmholtz_2d_cache(::Type{FT}, Y; kwargs...) where {FT}
    error(_CUDSS_EXT_MSG)
end
function assemble_gpu_sparse_helmholtz_2d!(shc)
    error(_CUDSS_EXT_MSG)
end
function gpu_sparse_helmholtz_2d_correction!(shc, ΔY)
    error(_CUDSS_EXT_MSG)
end

"""
    build_sparse_helmholtz_2d_dof_map(topology, Nq)

Build the unique horizontal DOF mapping for a 2D spectral element topology
(cubed-sphere or 2D box). Uses union-find to merge shared face and vertex nodes.

Returns `(dof_map, N_h)` where `dof_map[i, j, e]` gives the unique horizontal
DOF index and `N_h` is the total number of unique DOFs.
"""
function build_sparse_helmholtz_2d_dof_map(topology, Nq)
    nelem = Topologies.nlocalelems(topology)
    N_raw = nelem * Nq * Nq

    # Initialize union-find
    uf_parent = collect(1:N_raw)
    uf_rank = zeros(Int, N_raw)

    # Raw index from (i, j, e)
    @inline raw_idx(i, j, e) = (e - 1) * Nq * Nq + (j - 1) * Nq + i

    # Merge shared face DOFs using interior_faces
    # Copy to CPU for scalar iteration (interior_faces is a GPU array on CUDADevice)
    for (e1, f1, e2, f2, reversed) in Array(Topologies.interior_faces(topology))
        for q in 1:Nq
            i1, j1 = Topologies.face_node_index(f1, Nq, q, false)
            i2, j2 = Topologies.face_node_index(f2, Nq, q, reversed)
            r1 = raw_idx(i1, j1, e1)
            r2 = raw_idx(i2, j2, e2)
            _uf_union!(uf_parent, uf_rank, r1, r2)
        end
    end

    # Compact: assign sequential DOF indices
    root_to_dof = Dict{Int, Int}()
    next_dof = 0
    dof_map = Array{Int, 3}(undef, Nq, Nq, nelem)
    @inbounds for e in 1:nelem
        for j in 1:Nq, i in 1:Nq
            root = _uf_find!(uf_parent, raw_idx(i, j, e))
            if !haskey(root_to_dof, root)
                next_dof += 1
                root_to_dof[root] = next_dof
            end
            dof_map[i, j, e] = root_to_dof[root]
        end
    end
    N_h = next_dof

    return dof_map, N_h
end

"""
    build_sparse_helmholtz_2d_cache(::Type{FT}, Y) where {FT}

Build the 2D sparse Helmholtz cache for sphere grids. Computes the DOF mapping
from cubed-sphere topology connectivity, builds the sparsity pattern for the
2D SE Helmholtz operator, and allocates work vectors.
"""
function build_sparse_helmholtz_2d_cache(
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

    # Build DOF mapping
    dof_map, N_h = build_sparse_helmholtz_2d_dof_map(topology, Nq)
    N_dof = N_h * N_v

    @info "SparseHelmholtz2D: nelem=$nelem, Nq=$Nq, N_h=$N_h, N_v=$N_v, N_dof=$N_dof (level-decoupled)"

    # GLL quadrature data
    points, w = Quadratures.quadrature_points(FT, quad)
    D = Matrix{FT}(Quadratures.differentiation_matrix(FT, quad))
    weights = Vector{FT}(w)

    # Reference 1D stiffness: K1D[a,b] = Σ_q w_q * D[q,a] * D[q,b]
    K1D_ref = D' * Diagonal(weights) * D

    # Build N_h × N_h sparsity pattern (shared by all levels, values differ).
    I_idx = Int[]
    J_idx = Int[]
    V_val = FT[]

    for e in 1:nelem
        for j1 in 1:Nq, i1 in 1:Nq
            h_row = dof_map[i1, j1, e]
            # Direction 1: coupling to (i2, j1) for all i2
            for i2 in 1:Nq
                h_col = dof_map[i2, j1, e]
                push!(I_idx, h_row)
                push!(J_idx, h_col)
                push!(V_val, FT(0))
            end
            # Direction 2: coupling to (i1, j2) for all j2
            for j2 in 1:Nq
                h_col = dof_map[i1, j2, e]
                push!(I_idx, h_row)
                push!(J_idx, h_col)
                push!(V_val, FT(0))
            end
        end
    end

    # Create N_v independent copies of the horizontal sparsity pattern
    H_template = sparse(I_idx, J_idx, V_val, N_h, N_h)
    H_levels = [copy(H_template) for _ in 1:N_v]
    H_level_factors = [Ref{Any}(nothing) for _ in 1:N_v]

    # Extract WJ values from horizontal local geometry, store by DOF
    WJ_vec = zeros(FT, N_h)
    lg_data = Spaces.local_geometry_data(hspace)
    WJ_arr = Array(parent(lg_data.WJ))  # shape (Nq, Nq, 1, nelem) for IJFH; copy to CPU

    @inbounds for e in 1:nelem
        for j in 1:Nq, i in 1:Nq
            h_dof = dof_map[i, j, e]
            # Accumulate WJ for shared DOFs (DSS-like)
            WJ_vec[h_dof] += FT(WJ_arr[i, j, 1, e])
        end
    end

    _scratch_ρ = similar(Y.c, FT)
    _scratch_div = similar(Y.c, FT)
    _scratch_e_tot = similar(Y.c, FT)

    # Float32 → Float64 temporaries sized per-level (N_h) for UMFPACK
    rhs_work64 = FT === Float32 ? zeros(Float64, N_h) : nothing
    sol_work64 = FT === Float32 ? zeros(Float64, N_h) : nothing

    return SparseHelmholtz2DCache{FT}(
        H_levels,
        H_level_factors,
        zeros(FT, N_h),          # rhs_lev
        zeros(FT, N_h),          # sol_lev
        zeros(FT, N_dof),        # cs2_vec (all levels)
        zeros(FT, N_dof),        # ρ_vec (all levels)
        zeros(FT, N_dof),        # e_tot_vec (all levels)
        Ref(FT(0)),
        backsub_alpha,
        dof_map,
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
        rhs_work64,
        sol_work64,
    )
end

"""
Extract a center scalar field (VIJFH layout) to a flat DOF vector.
Parent layout: (Nv, Ni, Nj, Nf, Nh) = (N_v, Nq, Nq, 1, nelem).
For shared DOFs, the last-written value wins (should be identical post-DSS).
"""
function _sparse_helmholtz_2d_field_to_vec!(
    vec::Vector{FT}, field, dof_map, Nq, nelem, N_v, N_h,
) where {FT}
    arr = Array(parent(field))  # copy to CPU for scalar iteration
    @inbounds for e in 1:nelem
        for j in 1:Nq, i in 1:Nq
            h_dof = dof_map[i, j, e]
            for k in 1:N_v
                vec[(k - 1) * N_h + h_dof] = arr[k, i, j, 1, e]
            end
        end
    end
    return vec
end

"""
Write a flat DOF vector back to a center scalar field (VIJFH layout).
All (i,j,e) mapping to the same DOF receive the same value.
"""
function _sparse_helmholtz_2d_vec_to_field!(
    field, vec::Vector{FT}, dof_map, Nq, nelem, N_v, N_h,
) where {FT}
    arr = Array(parent(field))  # copy to CPU for scalar iteration
    @inbounds for e in 1:nelem
        for j in 1:Nq, i in 1:Nq
            h_dof = dof_map[i, j, e]
            for k in 1:N_v
                arr[k, i, j, 1, e] = vec[(k - 1) * N_h + h_dof]
            end
        end
    end
    copyto!(parent(field), arr)  # copy back to GPU
    return field
end

"""
    assemble_sparse_helmholtz_2d!(shc::SparseHelmholtz2DCache)

Assemble per-level 2D horizontal Helmholtz matrices:
    H_k = M_h + δtγ² · diag(cs²_k) · K_h
where M_h uses the actual WJ values (diagonal) and K_h uses the isotropic
metric approximation (J·g^{αβ} ≈ δ^{αβ}), giving a tensor-product stiffness:
    K[(i1,j1),(i2,j2)] = δ(j1,j2)·w[j1]·K1D[i1,i2] + δ(i1,i2)·w[i1]·K1D[j1,j2]

Each level is assembled and LU-factorized independently (N_h × N_h).
"""
function assemble_sparse_helmholtz_2d!(shc::SparseHelmholtz2DCache{FT}) where {FT}
    (; H_levels, H_level_factors, cs2_vec, WJ_vec, K1D_ref, weights, dtγ,
        dof_map, Nq, nelem, N_h, N_v) = shc

    δtγ² = dtγ[]^2

    @inbounds for k in 1:N_v
        Hk = H_levels[k]
        Hk.nzval .= FT(0)

        # Mass matrix (diagonal) — once per unique DOF
        for h in 1:N_h
            Hk[h, h] += FT(WJ_vec[h])
        end

        # Stiffness matrix — assembled per element via direct stiffness summation
        for e in 1:nelem
            for j in 1:Nq, i in 1:Nq
                h_row = dof_map[i, j, e]
                cs2_row = cs2_vec[(k - 1) * N_h + h_row]

                # Direction 1: K_h coupling along i-direction (fixed j)
                w_j = weights[j]
                for i2 in 1:Nq
                    h_col = dof_map[i2, j, e]
                    k_val = K1D_ref[i, i2]
                    Hk[h_row, h_col] += δtγ² * cs2_row * w_j * k_val
                end

                # Direction 2: K_h coupling along j-direction (fixed i)
                w_i = weights[i]
                for j2 in 1:Nq
                    j2 == j && continue  # diagonal already counted in direction 1
                    h_col = dof_map[i, j2, e]
                    k_val = K1D_ref[j, j2]
                    Hk[h_row, h_col] += δtγ² * cs2_row * w_i * k_val
                end
            end
        end

        H_level_factors[k][] = lu(Hk)
    end

    return nothing
end

"""
    sparse_helmholtz_2d_correction!(shc::SparseHelmholtz2DCache, ΔY)

Apply additive 2D sparse Helmholtz correction after column-local solve.
1. RHS_k = -δtγ · wdivₕ(ρ · z.uₕ)  (per level k)
2. Solve H_k · Δρ_h_k = RHS_k  (per level k, independent N_h × N_h systems)
3. ΔY.c.ρ += Δρ_h (additive)
4. If backsub_alpha > 0, apply damped uₕ and ρe_tot back-substitution:
   ΔY.c.uₕ  -= α · dtγ · (cs²/ρ) · C12(gradₕ(Δρ_h))
   ΔY.c.ρe_tot += α · e_tot · Δρ_h
"""
function sparse_helmholtz_2d_correction!(shc::SparseHelmholtz2DCache{FT}, ΔY) where {FT}
    (; H_level_factors, rhs_lev, sol_lev, dtγ, ρ_vec,
        backsub_alpha, cs2_vec, e_tot_vec,
        dof_map, Nq, nelem, N_h, N_v,
        _scratch_ρ, _scratch_div, _scratch_e_tot, rhs_work64, sol_work64) = shc

    # Step 1: Form wdivₕ(ρ · z.uₕ) into scratch field
    _sparse_helmholtz_2d_vec_to_field!(_scratch_ρ, ρ_vec, dof_map, Nq, nelem, N_v, N_h)
    @. _scratch_div = wdivₕ(_scratch_ρ * ΔY.c.uₕ)
    Spaces.weighted_dss!(_scratch_div)

    arr_div = Array(parent(_scratch_div))  # copy to CPU for scalar gather
    arr_out = Array(parent(_scratch_ρ))    # CPU buffer for scatter

    δtγ_val = dtγ[]

    # Step 2: Solve each level independently
    @inbounds for k in 1:N_v
        # Gather RHS for this level
        rhs_lev .= FT(0)
        for e in 1:nelem
            for j in 1:Nq, i in 1:Nq
                h_dof = dof_map[i, j, e]
                rhs_lev[h_dof] += -δtγ_val * arr_div[k, i, j, 1, e]
            end
        end

        # Solve H_k · sol = rhs
        Fac = H_level_factors[k][]
        if rhs_work64 !== nothing
            @inbounds for h in 1:N_h
                ;
                rhs_work64[h] = rhs_lev[h];
            end
            LinearAlgebra.ldiv!(sol_work64, Fac, rhs_work64)
            @inbounds for h in 1:N_h
                ;
                sol_lev[h] = FT(sol_work64[h]);
            end
        else
            LinearAlgebra.ldiv!(sol_lev, Fac, rhs_lev)
        end

        # Scatter solution back to field for this level
        for e in 1:nelem
            for j in 1:Nq, i in 1:Nq
                h_dof = dof_map[i, j, e]
                arr_out[k, i, j, 1, e] = sol_lev[h_dof]
            end
        end
    end

    # Copy scattered result back to GPU and DSS
    copyto!(parent(_scratch_ρ), arr_out)
    Spaces.weighted_dss!(_scratch_ρ)
    @. ΔY.c.ρ += _scratch_ρ

    # Step 4: Damped uₕ and ρe_tot back-substitution
    # _scratch_ρ now holds the DSS'd Δρ_h correction field
    if backsub_alpha > FT(0)
        α = backsub_alpha

        # Reconstruct cs²/ρ into _scratch_div (free after step 1)
        _sparse_helmholtz_2d_vec_to_field!(
            _scratch_div,
            cs2_vec,
            dof_map,
            Nq,
            nelem,
            N_v,
            N_h,
        )
        # _scratch_e_tot temporarily holds ρ field for division
        _sparse_helmholtz_2d_vec_to_field!(
            _scratch_e_tot,
            ρ_vec,
            dof_map,
            Nq,
            nelem,
            N_v,
            N_h,
        )
        @. _scratch_div = _scratch_div / max(_scratch_e_tot, FT(1e-6))

        # uₕ back-sub: ΔY.c.uₕ -= α · dtγ · (cs²/ρ) · C12(gradₕ(Δρ_h))
        @. ΔY.c.uₕ -= C12(FT(α * δtγ_val) * _scratch_div * gradₕ(_scratch_ρ))

        # Reconstruct e_tot into _scratch_e_tot
        _sparse_helmholtz_2d_vec_to_field!(
            _scratch_e_tot,
            e_tot_vec,
            dof_map,
            Nq,
            nelem,
            N_v,
            N_h,
        )

        # ρe_tot back-sub: ΔY.c.ρe_tot += α · e_tot · Δρ_h
        @. ΔY.c.ρe_tot += FT(α) * _scratch_e_tot * _scratch_ρ
    end

    return nothing
end

"""
    MatrixFreeHelmholtz2DCache

Matrix-free Chebyshev-accelerated 2D Helmholtz preconditioner for sphere grids.
Uses ClimaCore's `wdivₕ(gradₕ(...))` operator instead of assembling sparse matrices.
Solved via Chebyshev semi-iteration with Jacobi preconditioning.
"""
struct MatrixFreeHelmholtz2DCache{FT, F, GB}
    dtγ::Base.RefValue{FT}
    ᶜcs²::F                       # sound speed squared field
    ᶜρ::F                         # density field
    ᶜe_tot::F                     # specific total energy field (back-sub)
    ᶜα_acoustic::F                # Jacobi preconditioner diagonal
    α_acoustic_max::Base.RefValue{FT}
    backsub_alpha::FT
    # Chebyshev scratch (4 center scalar fields)
    ᶜhelm_x::F                   # iterate
    ᶜhelm_dir::F                 # direction
    ᶜhelm_rhs::F                 # saved RHS
    ᶜhelm_lap::F                 # wdivₕ(gradₕ(x)) scratch
    dss_buffer::GB
    n_iters::Int
end

"""
    build_matfree_helmholtz_2d_cache(::Type{FT}, Y; backsub_alpha, n_iters)

Build a matrix-free Chebyshev Helmholtz 2D cache. Allocates scalar fields
for the Chebyshev iteration and a DSS buffer.
"""
function build_matfree_helmholtz_2d_cache(
    ::Type{FT}, Y;
    backsub_alpha::FT = FT(0),
    n_iters::Int = 10,
) where {FT}
    ᶜscalar() = similar(Y.c, FT)
    dss_buf = Spaces.create_dss_buffer(ᶜscalar())

    @info "MatrixFreeHelmholtz2D: n_iters=$n_iters, backsub_alpha=$backsub_alpha"

    return MatrixFreeHelmholtz2DCache{FT, typeof(ᶜscalar()), typeof(dss_buf)}(
        Ref(FT(0)),          # dtγ
        ᶜscalar(),           # ᶜcs²
        ᶜscalar(),           # ᶜρ
        ᶜscalar(),           # ᶜe_tot
        ᶜscalar(),           # ᶜα_acoustic
        Ref(FT(0)),          # α_acoustic_max
        backsub_alpha,
        ᶜscalar(),           # ᶜhelm_x
        ᶜscalar(),           # ᶜhelm_dir
        ᶜscalar(),           # ᶜhelm_rhs
        ᶜscalar(),           # ᶜhelm_lap
        dss_buf,
        n_iters,
    )
end

"""
    matfree_helmholtz_2d_correction!(mfc::MatrixFreeHelmholtz2DCache, ΔY)

Apply matrix-free Chebyshev 2D Helmholtz correction after column-local solve.
Uses ClimaCore's `wdivₕ(gradₕ(...))` for the Laplacian operator.

1. RHS = -dtγ · wdivₕ(ρ · z.uₕ)
2. Chebyshev semi-iteration to solve (I - dtγ²·cs²·∇²h)·Δρ = RHS
3. ΔY.c.ρ += Δρ
4. Optional back-sub: uₕ, ρe_tot corrections
"""
function matfree_helmholtz_2d_correction!(
    mfc::MatrixFreeHelmholtz2DCache{FT},
    ΔY,
) where {FT}
    (; dtγ, ᶜcs², ᶜρ, ᶜe_tot, ᶜα_acoustic, α_acoustic_max, backsub_alpha,
        ᶜhelm_x, ᶜhelm_dir, ᶜhelm_rhs, ᶜhelm_lap, dss_buffer, n_iters) = mfc

    δtγ_val = dtγ[]
    α = δtγ_val^2

    # Step 1: Form RHS = -dtγ · wdivₕ(ρ · z.uₕ)
    @. ᶜhelm_rhs = wdivₕ(ᶜρ * ΔY.c.uₕ)
    Spaces.weighted_dss!(ᶜhelm_rhs => dss_buffer)
    @. ᶜhelm_rhs = -δtγ_val * ᶜhelm_rhs

    # Step 2: Chebyshev semi-iteration
    # Operator: A·x = x - α·cs²·wdivₕ(gradₕ(x))
    # Preconditioner: M = diag(1 + α_acoustic)
    # Eigenvalues of M⁻¹A ∈ [1/(1+α_max), 1]
    α_max = α_acoustic_max[]
    a = FT(1) / (FT(1) + α_max)
    b = FT(1)
    θ = (a + b) / 2
    δ = (b - a) / 2
    σ₁ = θ / δ

    @. ᶜhelm_x = ᶜhelm_rhs  # initial guess

    if n_iters >= 1
        # Chebyshev semi-iteration WITHOUT intermediate DSS.
        # The element-local operator is used throughout; with the safety-factor-
        # augmented eigenvalue bounds, the element-local eigenvalues are safely
        # within [a, b]. The final DSS projects the result to the continuous space.
        # This avoids the O(n_iters) DSS cost that would destroy performance.

        # Step 1: d₀ = (1/θ) · M⁻¹·r₀
        @. ᶜhelm_lap = wdivₕ(gradₕ(ᶜhelm_x))
        @. ᶜhelm_dir =
            (ᶜhelm_rhs - ᶜhelm_x + α * ᶜcs² * ᶜhelm_lap) /
            ((FT(1) + ᶜα_acoustic) * θ)
        @. ᶜhelm_x += ᶜhelm_dir

        # Steps 2..N: Chebyshev three-term recurrence
        ρ_prev = FT(1) / σ₁
        for _ in 2:n_iters
            ρ_new = FT(1) / (2 * σ₁ - FT(1) / ρ_prev)
            @. ᶜhelm_lap = wdivₕ(gradₕ(ᶜhelm_x))
            @. ᶜhelm_dir =
                2 * ρ_new * σ₁ / θ *
                (ᶜhelm_rhs - ᶜhelm_x + α * ᶜcs² * ᶜhelm_lap) /
                (FT(1) + ᶜα_acoustic) +
                ρ_new * ρ_prev * ᶜhelm_dir
            @. ᶜhelm_x += ᶜhelm_dir
            ρ_prev = ρ_new
        end
    end

    # DSS only the final iterate (project to continuous space)
    Spaces.weighted_dss!(ᶜhelm_x => dss_buffer)

    # Step 3: Additive ρ correction
    @. ΔY.c.ρ += ᶜhelm_x

    # Step 4: Optional back-substitution
    if backsub_alpha > FT(0)
        α_bs = backsub_alpha

        # uₕ back-sub: ΔY.c.uₕ -= α_bs · dtγ · (cs²/ρ) · C12(gradₕ(Δρ_h))
        @. ΔY.c.uₕ -= C12(
            FT(α_bs * δtγ_val) * (ᶜcs² / max(ᶜρ, FT(1e-6))) * gradₕ(ᶜhelm_x),
        )

        # ρe_tot back-sub: ΔY.c.ρe_tot += α_bs · e_tot · Δρ_h
        @. ΔY.c.ρe_tot += FT(α_bs) * ᶜe_tot * ᶜhelm_x
    end

    return nothing
end

function jacobian_cache(alg::ManualSparseJacobian, Y, atmos)
    (;
        topography_flag,
        diffusion_flag,
        sgs_advection_flag,
        sgs_mass_flux_flag,
        acoustic_diagonal_flag,
        approximate_solve_iters,
    ) = alg
    FT = Spaces.undertype(axes(Y.c))

    DiagonalRow = DiagonalMatrixRow{FT}
    TridiagonalRow = TridiagonalMatrixRow{FT}
    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    TridiagonalRow_ACT12 = TridiagonalMatrixRow{Adjoint{FT, CT12{FT}}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    BidiagonalRow_C3xACT12 =
        BidiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT12{FT})')}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT3{FT})')}

    is_in_Y(name) = MatrixFields.has_field(Y, name)

    ρq_tot_if_available = is_in_Y(@name(c.ρq_tot)) ? (@name(c.ρq_tot),) : ()
    ρtke_if_available =
        is_in_Y(@name(c.ρtke)) ? (@name(c.ρtke),) : ()
    sfc_if_available = is_in_Y(@name(sfc)) ? (@name(sfc),) : ()

    condensate_mass_names = (
        @name(c.ρq_liq),
        @name(c.ρq_ice),
        @name(c.ρq_rai),
        @name(c.ρq_sno),
    )
    available_condensate_mass_names =
        MatrixFields.unrolled_filter(is_in_Y, condensate_mass_names)
    condensate_names = (
        condensate_mass_names...,
        @name(c.ρn_liq),
        @name(c.ρn_rai),
        # P3 frozen
        @name(c.ρn_ice), @name(c.ρq_rim), @name(c.ρb_rim),
    )
    available_condensate_names =
        MatrixFields.unrolled_filter(is_in_Y, condensate_names)
    available_tracer_names =
        (ρq_tot_if_available..., available_condensate_names...)

    # we define the list of condensate masses separately because ρa and q_tot
    # depend on the masses via sedimentation
    sgs_condensate_mass_names = (
        @name(c.sgsʲs.:(1).q_liq),
        @name(c.sgsʲs.:(1).q_ice),
        @name(c.sgsʲs.:(1).q_rai),
        @name(c.sgsʲs.:(1).q_sno),
    )
    available_sgs_condensate_mass_names =
        MatrixFields.unrolled_filter(is_in_Y, sgs_condensate_mass_names)

    sgs_condensate_names =
        (sgs_condensate_mass_names..., @name(c.sgsʲs.:(1).n_liq), @name(c.sgsʲs.:(1).n_rai))
    available_sgs_condensate_names =
        MatrixFields.unrolled_filter(is_in_Y, sgs_condensate_names)

    sgs_scalar_names =
        (
            sgs_condensate_names...,
            @name(c.sgsʲs.:(1).q_tot),
            @name(c.sgsʲs.:(1).mse),
            @name(c.sgsʲs.:(1).ρa)
        )
    available_sgs_scalar_names =
        MatrixFields.unrolled_filter(is_in_Y, sgs_scalar_names)

    sgs_u³_if_available =
        is_in_Y(@name(f.sgsʲs.:(1).u₃)) ? (@name(f.sgsʲs.:(1).u₃),) : ()

    # Note: We have to use FT(-1) * I instead of -I because inv(-1) == -1.0,
    # which means that multiplying inv(-1) by a Float32 will yield a Float64.
    identity_blocks = if use_derivative(acoustic_diagonal_flag)
        (
            (@name(c.ρ), @name(c.ρ)) => similar(Y.c, DiagonalRow),
            MatrixFields.unrolled_map(
                name -> (name, name) => FT(-1) * I,
                (sfc_if_available...,),
            )...,
        )
    else
        MatrixFields.unrolled_map(
            name -> (name, name) => FT(-1) * I,
            (@name(c.ρ), sfc_if_available...),
        )
    end

    active_scalar_names = (@name(c.ρ), @name(c.ρe_tot), ρq_tot_if_available...)
    advection_blocks = (
        (
            use_derivative(topography_flag) ?
            MatrixFields.unrolled_map(
                name ->
                    (name, @name(c.uₕ)) =>
                        similar(Y.c, TridiagonalRow_ACT12),
                active_scalar_names,
            ) : ()
        )...,
        MatrixFields.unrolled_map(
            name -> (name, @name(f.u₃)) => similar(Y.c, BidiagonalRow_ACT3),
            active_scalar_names,
        )...,
        MatrixFields.unrolled_map(
            name -> (@name(f.u₃), name) => similar(Y.f, BidiagonalRow_C3),
            active_scalar_names,
        )...,
        MatrixFields.unrolled_map(
            name -> (@name(f.u₃), name) => similar(Y.f, BidiagonalRow_C3),
            available_condensate_mass_names,
        )...,
        (@name(f.u₃), @name(c.uₕ)) => similar(Y.f, BidiagonalRow_C3xACT12),
        (@name(f.u₃), @name(f.u₃)) => similar(Y.f, TridiagonalRow_C3xACT3),
    )

    diffused_scalar_names = (@name(c.ρe_tot), available_tracer_names...)
    diffusion_blocks = if use_derivative(diffusion_flag)
        (
            MatrixFields.unrolled_map(
                name -> (name, @name(c.ρ)) => similar(Y.c, TridiagonalRow),
                (diffused_scalar_names..., ρtke_if_available...),
            )...,
            MatrixFields.unrolled_map(
                name -> (name, name) => similar(Y.c, TridiagonalRow),
                (diffused_scalar_names..., ρtke_if_available...),
            )...,
            (
                is_in_Y(@name(c.ρq_tot)) ?
                (
                    (@name(c.ρe_tot), @name(c.ρq_tot)) =>
                        similar(Y.c, TridiagonalRow),
                ) : ()
            )...,
            MatrixFields.unrolled_map(
                name -> (@name(c.ρe_tot), name) => similar(Y.c, TridiagonalRow),
                available_condensate_mass_names,
            )...,
            MatrixFields.unrolled_map(
                name -> (@name(c.ρq_tot), name) => similar(Y.c, TridiagonalRow),
                available_condensate_mass_names,
            )...,
            (@name(c.uₕ), @name(c.uₕ)) =>
                !isnothing(atmos.turbconv_model) ||
                    !disable_momentum_vertical_diffusion(
                        atmos.vertical_diffusion,
                    ) ? similar(Y.c, TridiagonalRow) :
                use_derivative(acoustic_diagonal_flag) ?
                similar(Y.c, DiagonalRow) : FT(-1) * I,
        )
    elseif atmos.microphysics_model isa DryModel
        if use_derivative(acoustic_diagonal_flag)
            (
                (@name(c.ρe_tot), @name(c.ρe_tot)) => similar(Y.c, DiagonalRow),
                MatrixFields.unrolled_map(
                    name -> (name, name) => FT(-1) * I,
                    (ρtke_if_available...,),
                )...,
                (@name(c.uₕ), @name(c.uₕ)) => similar(Y.c, DiagonalRow),
            )
        else
            MatrixFields.unrolled_map(
                name -> (name, name) => FT(-1) * I,
                (diffused_scalar_names..., ρtke_if_available..., @name(c.uₕ)),
            )
        end
    else
        (
            MatrixFields.unrolled_map(
                name -> (name, name) => similar(Y.c, TridiagonalRow),
                diffused_scalar_names,
            )...,
            MatrixFields.unrolled_map(
                name -> (@name(c.ρe_tot), name) => similar(Y.c, TridiagonalRow),
                available_condensate_mass_names,
            )...,
            MatrixFields.unrolled_map(
                name -> (@name(c.ρq_tot), name) => similar(Y.c, TridiagonalRow),
                available_condensate_mass_names,
            )...,
            (@name(c.ρe_tot), @name(c.ρq_tot)) =>
                similar(Y.c, TridiagonalRow),
            MatrixFields.unrolled_map(
                name -> (name, name) => FT(-1) * I,
                (ρtke_if_available...,),
            )...,
            (@name(c.uₕ), @name(c.uₕ)) =>
                use_derivative(acoustic_diagonal_flag) ?
                similar(Y.c, DiagonalRow) : FT(-1) * I,
        )
    end

    sgs_advection_blocks = if atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_advection_flag)
            (
                MatrixFields.unrolled_map(
                    name -> (name, name) => similar(Y.c, TridiagonalRow),
                    available_sgs_scalar_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (@name(c.sgsʲs.:(1).q_tot), name) =>
                            similar(Y.c, TridiagonalRow),
                    available_sgs_condensate_mass_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (@name(c.sgsʲs.:(1).ρa), name) => similar(Y.c, TridiagonalRow),
                    available_sgs_condensate_mass_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (@name(c.sgsʲs.:(1).mse), name) => similar(Y.c, DiagonalRow),
                    available_sgs_condensate_mass_names,
                )...,
                (@name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.c, DiagonalRow),
                (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).mse)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) => FT(-1) * I,
            )
        else
            # When implicit microphysics is active, some SGS scalar entries
            # need a DiagonalRow so that update_microphysics_jacobian! can
            # increment them.  UniformScaling is not incrementable in-place.
            needs_implicit_micro =
                atmos.microphysics_tendency_timestepping == Implicit()
            # 0M EDMF writes to q_tot and ρa; 1M EDMF writes to
            # condensate species (q_liq, q_ice, q_rai, q_sno).
            sgs_micro_names =
                needs_implicit_micro ?
                (
                    (
                        atmos.microphysics_model isa EquilibriumMicrophysics0M ?
                        (
                            @name(c.sgsʲs.:(1).q_tot),
                            @name(c.sgsʲs.:(1).ρa),
                        ) : ()
                    )...,
                    (
                        atmos.microphysics_model isa NonEquilibriumMicrophysics ?
                        sgs_condensate_mass_names : ()
                    )...,
                ) : ()
            (
                MatrixFields.unrolled_map(
                    name ->
                        (name, name) =>
                            name in sgs_micro_names ?
                            similar(Y.c, DiagonalRow) : FT(-1) * I,
                    available_sgs_scalar_names,
                )...,
                (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) => FT(-1) * I,
            )
        end
    else
        ()
    end

    sgs_massflux_blocks = if atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_mass_flux_flag)
            (
                MatrixFields.unrolled_map(
                    name ->
                        (name, get_χʲ_name_from_ρχ_name(name)) =>
                            similar(Y.c, TridiagonalRow),
                    available_tracer_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (name, @name(c.sgsʲs.:(1).ρa)) =>
                            similar(Y.c, TridiagonalRow),
                    available_tracer_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (name, @name(f.u₃)) =>
                            similar(Y.c, BidiagonalRow_ACT3),
                    available_condensate_names,
                )...,
                (@name(c.ρe_tot), @name(c.sgsʲs.:(1).mse)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)) =>
                    similar(Y.c, TridiagonalRow),
            )
        else
            ()
        end
    else
        ()
    end

    matrix = MatrixFields.FieldMatrix(
        identity_blocks...,
        sgs_advection_blocks...,
        advection_blocks...,
        diffusion_blocks...,
        sgs_massflux_blocks...,
    )

    mass_and_surface_names = (@name(c.ρ), sfc_if_available...)
    available_scalar_names = (
        mass_and_surface_names...,
        available_tracer_names...,
        @name(c.ρe_tot),
        ρtke_if_available...,
        available_sgs_scalar_names...,
    )

    velocity_alg = MatrixFields.BlockLowerTriangularSolve(
        @name(c.uₕ),
        sgs_u³_if_available...,
    )
    full_alg =
        if use_derivative(diffusion_flag) ||
           use_derivative(sgs_advection_flag) ||
           !(atmos.microphysics_model isa DryModel)
            gs_scalar_subalg = if !(atmos.microphysics_model isa DryModel)
                MatrixFields.BlockLowerTriangularSolve(
                    available_condensate_mass_names...,
                    alg₂ = MatrixFields.BlockLowerTriangularSolve(
                        @name(c.ρq_tot),
                    ),
                )
            else
                MatrixFields.BlockDiagonalSolve()
            end
            scalar_subalg =
                if atmos.turbconv_model isa PrognosticEDMFX &&
                   use_derivative(sgs_advection_flag)
                    MatrixFields.BlockLowerTriangularSolve(
                        available_sgs_condensate_names...;
                        alg₂ = MatrixFields.BlockLowerTriangularSolve(
                            @name(c.sgsʲs.:(1).q_tot);
                            alg₂ = MatrixFields.BlockLowerTriangularSolve(
                                @name(c.sgsʲs.:(1).mse);
                                alg₂ = MatrixFields.BlockLowerTriangularSolve(
                                    @name(c.sgsʲs.:(1).ρa);
                                    alg₂ = gs_scalar_subalg,
                                ),
                            ),
                        ),
                    )
                else
                    gs_scalar_subalg
                end
            scalar_alg = MatrixFields.BlockLowerTriangularSolve(
                mass_and_surface_names...;
                alg₂ = scalar_subalg,
            )
            MatrixFields.ApproximateBlockArrowheadIterativeSolve(
                available_scalar_names...;
                alg₁ = scalar_alg,
                alg₂ = velocity_alg,
                P_alg₁ = MatrixFields.MainDiagonalPreconditioner(),
                n_iters = approximate_solve_iters,
            )
        else
            MatrixFields.BlockArrowheadSolve(
                available_scalar_names...;
                alg₂ = velocity_alg,
            )
        end

    matrix_cache = MatrixFields.FieldMatrixWithSolver(matrix, Y, full_alg)

    # Helmholtz preconditioner state and scratch fields
    if use_derivative(acoustic_diagonal_flag)
        ᶜscalar_field() = similar(Y.c, FT)
        helmholtz_state = HelmholtzPreconditionerState(
            FT(0),                  # dtγ (updated in update_jacobian!)
            FT(0),                  # α_acoustic_max (updated in update_jacobian!)
            ᶜscalar_field(),        # ᶜα_acoustic
            ᶜscalar_field(),        # ᶜcs²
            ᶜscalar_field(),        # ᶜρ
            ᶜscalar_field(),        # ᶜe_tot
            ᶜscalar_field(),        # ᶜh_tot
            nothing,                # ghost_buffer_c (set in update_jacobian!)
            alg.n_helmholtz_iters,  # from config
            0,                      # call_counter
        )
        helmholtz_scratch = (;
            ᶜhelmholtz_ρ = ᶜscalar_field(),
            ᶜhelmholtz_rhs = ᶜscalar_field(),
            ᶜhelmholtz_laplacian = ᶜscalar_field(),
            ᶜhelmholtz_ρe = ᶜscalar_field(),
            ᶜhelmholtz_dir = ᶜscalar_field(),  # Chebyshev direction vector
            ᶜhelmholtz_dss_buffer = Spaces.create_dss_buffer(
                ᶜscalar_field(),
            ),
        )
        return (;
            matrix = matrix_cache,
            helmholtz_state,
            helmholtz_scratch,
            sparse_helmholtz = nothing,
            sparse_helmholtz_2d = nothing,
        )
    else
        # Sparse direct Helmholtz preconditioner (independent of Chebyshev path)
        sparse_helmholtz = if use_derivative(alg.sparse_helmholtz_flag)
            build_sparse_helmholtz_cache(FT, Y)
        else
            nothing
        end
        # 2D sparse Helmholtz for sphere grids
        sparse_helmholtz_2d = if use_derivative(alg.sparse_helmholtz_2d_flag)
            if alg.helmholtz_2d_solver == "chebyshev"
                build_matfree_helmholtz_2d_cache(FT, Y;
                    backsub_alpha = FT(alg.helmholtz_backsub_alpha),
                    n_iters = alg.n_helmholtz_2d_iters,
                )
            elseif alg.helmholtz_2d_solver == "gpu_direct"
                build_gpu_sparse_helmholtz_2d_cache(
                    FT,
                    Y;
                    backsub_alpha = FT(alg.helmholtz_backsub_alpha),
                )
            else
                build_sparse_helmholtz_2d_cache(
                    FT,
                    Y;
                    backsub_alpha = FT(alg.helmholtz_backsub_alpha),
                )
            end
        else
            nothing
        end
        return (;
            matrix = matrix_cache,
            helmholtz_state = nothing,
            helmholtz_scratch = nothing,
            sparse_helmholtz,
            sparse_helmholtz_2d,
        )
    end
end

# TODO: There are a few for loops in this function. This is because
# using unrolled_foreach allocates (breaks the flame tests)
function update_jacobian!(alg::ManualSparseJacobian, cache, Y, p, dtγ, t)
    (;
        topography_flag,
        diffusion_flag,
        sgs_advection_flag,
        sgs_entr_detr_flag,
        sgs_mass_flux_flag,
        sgs_vertdiff_flag,
        acoustic_diagonal_flag,
    ) = alg
    (; matrix) = cache
    (; params) = p
    (; ᶜΦ) = p.core
    (; ᶜu, ᶠu³, ᶜK, ᶜp, ᶜT, ᶜh_tot) = p.precomputed
    (; ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    (;
        ∂ᶜK_∂ᶜuₕ,
        ∂ᶜK_∂ᶠu₃,
        ᶠp_grad_matrix,
        ᶜadvection_matrix,
        ᶜdiffusion_h_matrix,
        ᶜdiffusion_u_matrix,
        ᶜtridiagonal_matrix_scalar,
        ᶠbidiagonal_matrix_ct3,
        ᶠbidiagonal_matrix_ct3_2,
        ᶠsed_tracer_advection,
        ᶜtracer_advection_matrix,
        ᶜtridiagonal_matrix,
    ) = p.scratch
    rs = p.atmos.rayleigh_sponge

    FT = Spaces.undertype(axes(Y.c))
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'

    cv_d = FT(CAP.cv_d(params))
    Δcv_v = FT(CAP.cv_v(params)) - cv_d
    T_0 = FT(CAP.T_0(params))
    R_d = FT(CAP.R_d(params))
    R_v = FT(CAP.R_v(params))
    ΔR_v = R_v - R_d
    cp_d = FT(CAP.cp_d(params))
    Δcp_v = FT(CAP.cp_v(params)) - cp_d
    e_int_v0 = FT(CAP.e_int_v0(params))
    LH_v0 = FT(CAP.LH_v0(params))
    LH_s0 = FT(CAP.LH_s0(params))
    Δcp_l = FT(CAP.cp_l(params) - CAP.cp_v(params))
    Δcp_i = FT(CAP.cp_i(params) - CAP.cp_v(params))
    Δcv_l = FT(CAP.cp_l(params) - CAP.cv_v(params))
    Δcv_i = FT(CAP.cp_i(params) - CAP.cv_v(params))
    e_int_v0 = FT(CAP.e_int_v0(params))
    e_int_s0 = FT(CAP.e_int_i0(params)) + e_int_v0
    thermo_params = CAP.thermodynamics_params(params)

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜgⁱʲ = Fields.local_geometry_field(Y.c).gⁱʲ
    ᶠgⁱʲ = Fields.local_geometry_field(Y.f).gⁱʲ
    ᶠz = Fields.coordinate_field(Y.f).z
    zmax = z_max(axes(Y.f))

    ᶜkappa_m = p.scratch.ᶜtemp_scalar
    @. ᶜkappa_m =
        TD.gas_constant_air(thermo_params, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) /
        TD.cv_m(thermo_params, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno)

    ᶜ∂p∂ρq_tot = p.scratch.ᶜtemp_scalar_2
    @. ᶜ∂p∂ρq_tot = ᶜkappa_m * (-e_int_v0 - R_d * T_0 - Δcv_v * (ᶜT - T_0)) + ΔR_v * ᶜT

    if use_derivative(topography_flag)
        @. ∂ᶜK_∂ᶜuₕ = DiagonalMatrixRow(
            adjoint(CT12(ᶜuₕ)) + adjoint(ᶜinterp(ᶠu₃)) * g³ʰ(ᶜgⁱʲ),
        )
    else
        @. ∂ᶜK_∂ᶜuₕ = DiagonalMatrixRow(adjoint(CT12(ᶜuₕ)))
    end
    @. ∂ᶜK_∂ᶠu₃ =
        ᶜinterp_matrix() ⋅ DiagonalMatrixRow(adjoint(CT3(ᶠu₃))) +
        DiagonalMatrixRow(adjoint(CT3(ᶜuₕ))) ⋅ ᶜinterp_matrix()

    @. ᶠp_grad_matrix = DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix()

    @. ᶜadvection_matrix =
        -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ)
    @. p.scratch.ᶠbidiagonal_matrix_ct3xct12 =
        ᶠwinterp_matrix(ᶜJ * ᶜρ) ⋅ DiagonalMatrixRow(g³ʰ(ᶜgⁱʲ))
    if use_derivative(topography_flag)
        ∂ᶜρ_err_∂ᶜuₕ = matrix[@name(c.ρ), @name(c.uₕ)]
        @. ∂ᶜρ_err_∂ᶜuₕ =
            dtγ * ᶜadvection_matrix ⋅ p.scratch.ᶠbidiagonal_matrix_ct3xct12
    end
    ∂ᶜρ_err_∂ᶠu₃ = matrix[@name(c.ρ), @name(f.u₃)]
    @. ∂ᶜρ_err_∂ᶠu₃ = dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(g³³(ᶠgⁱʲ))

    tracer_info = (@name(c.ρe_tot), @name(c.ρq_tot))

    MatrixFields.unrolled_foreach(tracer_info) do ρχ_name
        MatrixFields.has_field(Y, ρχ_name) || return
        ᶜχ = ρχ_name === @name(c.ρe_tot) ? ᶜh_tot : (@. lazy(specific(Y.c.ρq_tot, Y.c.ρ)))

        if use_derivative(topography_flag)
            ∂ᶜρχ_err_∂ᶜuₕ = matrix[ρχ_name, @name(c.uₕ)]
            @. ∂ᶜρχ_err_∂ᶜuₕ =
                dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(ᶠinterp(ᶜχ)) ⋅
                p.scratch.ᶠbidiagonal_matrix_ct3xct12
        end

        ∂ᶜρχ_err_∂ᶠu₃ = matrix[ρχ_name, @name(f.u₃)]
        @. ∂ᶜρχ_err_∂ᶠu₃ =
            dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(ᶠinterp(ᶜχ) * g³³(ᶠgⁱʲ))
    end

    ∂ᶠu₃_err_∂ᶜρ = matrix[@name(f.u₃), @name(c.ρ)]
    ∂ᶠu₃_err_∂ᶜρe_tot = matrix[@name(f.u₃), @name(c.ρe_tot)]

    ᶜθ_v = p.scratch.ᶜtemp_scalar_3
    @. ᶜθ_v = theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno)
    ᶜΠ = @. lazy(TD.exner_given_pressure(thermo_params, ᶜp))
    # In implicit tendency, we use the new pressure-gradient formulation (PGF) and gravitational acceleration:
    #              grad(p) / ρ + grad(Φ)  =  cp_d * θ_v * grad(Π) + grad(Φ).
    # Here below, we use the old formulation of (grad(Φ) + grad(p) / ρ).
    # This is because the new formulation would require computing the derivative of θ_v.
    # The only exception is:
    # We are rewriting grad(p) / ρ from the expansion of ∂ᶠu₃_err_∂ᶜρ with the new PGF.
    @. ∂ᶠu₃_err_∂ᶜρ =
        dtγ * (
            ᶠp_grad_matrix ⋅
            DiagonalMatrixRow(
                ᶜkappa_m * (T_0 * cp_d - ᶜK - ᶜΦ) + (R_d - ᶜkappa_m * cv_d) * ᶜT,
            ) +
            DiagonalMatrixRow(cp_d * ᶠinterp(ᶜθ_v) * ᶠgradᵥ(ᶜΠ) / ᶠinterp(ᶜρ)) ⋅
            ᶠinterp_matrix()
        )
    @. ∂ᶠu₃_err_∂ᶜρe_tot = dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜkappa_m)

    if MatrixFields.has_field(Y, @name(c.ρq_tot))
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
        ∂ᶠu₃_err_∂ᶜρq_tot = matrix[@name(f.u₃), @name(c.ρq_tot)]
        @. ∂ᶠu₃_err_∂ᶜρq_tot =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜ∂p∂ρq_tot)
    end

    microphysics_tracers =
        p.atmos.microphysics_model isa Union{
            NonEquilibriumMicrophysics1M,
            NonEquilibriumMicrophysics2M,
        } ?
        (
            (@name(c.ρq_liq), e_int_v0, Δcv_l),
            (@name(c.ρq_ice), e_int_s0, Δcv_i),
            (@name(c.ρq_rai), e_int_v0, Δcv_l),
            (@name(c.ρq_sno), e_int_s0, Δcv_i),
        ) : (;)

    for (q_name, e_int_q, ∂cv∂q) in microphysics_tracers
        MatrixFields.has_field(Y, q_name) || continue
        ∂ᶠu₃_err_∂ᶜρq = matrix[@name(f.u₃), q_name]
        @. ∂ᶠu₃_err_∂ᶜρq =
            dtγ * ᶠp_grad_matrix ⋅
            DiagonalMatrixRow(ᶜkappa_m * (e_int_q - ∂cv∂q * (ᶜT - T_0)) - R_v * ᶜT)
    end

    ∂ᶠu₃_err_∂ᶜuₕ = matrix[@name(f.u₃), @name(c.uₕ)]
    ∂ᶠu₃_err_∂ᶠu₃ = matrix[@name(f.u₃), @name(f.u₃)]
    I_u₃ = DiagonalMatrixRow(one_C3xACT3)
    if rs isa RayleighSponge
        @. ∂ᶠu₃_err_∂ᶜuₕ =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅ ∂ᶜK_∂ᶜuₕ
        @. ∂ᶠu₃_err_∂ᶠu₃ =
            dtγ * (
                ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅
                ∂ᶜK_∂ᶠu₃ +
                DiagonalMatrixRow(-β_rayleigh_u₃(rs, ᶠz, zmax) * (one_C3xACT3,))
            ) - (I_u₃,)
    else
        @. ∂ᶠu₃_err_∂ᶜuₕ =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅ ∂ᶜK_∂ᶜuₕ
        @. ∂ᶠu₃_err_∂ᶠu₃ =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅
            ∂ᶜK_∂ᶠu₃ - (I_u₃,)
    end

    α_vert_diff_tracer = CAP.α_vert_diff_tracer(params)
    tracer_info = (
        (@name(c.ρq_liq), @name(ᶜwₗ), FT(1)),
        (@name(c.ρq_ice), @name(ᶜwᵢ), FT(1)),
        (@name(c.ρq_rai), @name(ᶜwᵣ), α_vert_diff_tracer),
        (@name(c.ρq_sno), @name(ᶜwₛ), α_vert_diff_tracer),
        (@name(c.ρn_liq), @name(ᶜwₙₗ), FT(1)),
        (@name(c.ρn_rai), @name(ᶜwₙᵣ), α_vert_diff_tracer),
        (@name(c.ρn_ice), @name(ᶜwnᵢ), FT(1)),
        (@name(c.ρq_rim), @name(ᶜwᵢ), FT(1)),
        (@name(c.ρb_rim), @name(ᶜwᵢ), FT(1)),
    )
    internal_energy_func(name) =
        (name == @name(c.ρq_liq) || name == @name(c.ρq_rai)) ? TD.internal_energy_liquid :
        (name == @name(c.ρq_ice) || name == @name(c.ρq_sno)) ? TD.internal_energy_ice :
        nothing
    if !(p.atmos.microphysics_model isa DryModel) || use_derivative(diffusion_flag)
        ∂ᶜρe_tot_err_∂ᶜρe_tot = matrix[@name(c.ρe_tot), @name(c.ρe_tot)]
        @. ∂ᶜρe_tot_err_∂ᶜρe_tot = zero(typeof(∂ᶜρe_tot_err_∂ᶜρe_tot)) - (I,)
    end

    if !(p.atmos.microphysics_model isa DryModel)
        ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
        @. ∂ᶜρe_tot_err_∂ᶜρq_tot = zero(typeof(∂ᶜρe_tot_err_∂ᶜρq_tot))

        ∂ᶜρq_tot_err_∂ᶜρq_tot = matrix[@name(c.ρq_tot), @name(c.ρq_tot)]
        @. ∂ᶜρq_tot_err_∂ᶜρq_tot = zero(typeof(∂ᶜρq_tot_err_∂ᶜρq_tot)) - (I,)

        # This scratch variable computation could be skipped if no tracers are present
        @. p.scratch.ᶜbidiagonal_adjoint_matrix_c3 =
            dtγ * (-ClimaAtmos.ᶜprecipdivᵥ_matrix()) ⋅
            DiagonalMatrixRow(ClimaAtmos.ᶠinterp(ᶜρ * ᶜJ) / ᶠJ)

        MatrixFields.unrolled_foreach(tracer_info) do (ρχₚ_name, wₚ_name, _)
            MatrixFields.has_field(Y, ρχₚ_name) || return

            ∂ᶜρχₚ_err_∂ᶜρχₚ = matrix[ρχₚ_name, ρχₚ_name]
            ᶜwₚ = MatrixFields.get_field(p.precomputed, wₚ_name)
            # TODO: come up with read-able names for the intermediate computations...
            @. p.scratch.ᶠband_matrix_wvec =
                ClimaAtmos.ᶠright_bias_matrix() ⋅
                DiagonalMatrixRow(ClimaCore.Geometry.WVector(-(ᶜwₚ) / ᶜρ))
            @. ∂ᶜρχₚ_err_∂ᶜρχₚ =
                p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅
                p.scratch.ᶠband_matrix_wvec - (I,)

            if ρχₚ_name in
               (@name(c.ρq_liq), @name(c.ρq_ice), @name(c.ρq_rai), @name(c.ρq_sno))
                ∂ᶜρq_tot_err_∂ᶜρq = matrix[@name(c.ρq_tot), ρχₚ_name]
                @. ∂ᶜρq_tot_err_∂ᶜρq =
                    p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅
                    p.scratch.ᶠband_matrix_wvec

                ∂ᶜρe_tot_err_∂ᶜρq = matrix[@name(c.ρe_tot), ρχₚ_name]
                e_int_func = internal_energy_func(ρχₚ_name)
                @. ∂ᶜρe_tot_err_∂ᶜρq =
                    p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅
                    p.scratch.ᶠband_matrix_wvec ⋅
                    DiagonalMatrixRow(
                        e_int_func(thermo_params, ᶜT) + ᶜΦ + $(Kin(ᶜwₚ, ᶜu)),
                    )
            end
        end

    end

    if use_derivative(diffusion_flag)
        (; turbconv_model) = p.atmos
        turbconv_params = CAP.turbconv_params(params)
        FT = eltype(params)
        (; vertical_diffusion, smagorinsky_lilly) = p.atmos
        (; ᶜp) = p.precomputed
        ᶜK_u = p.scratch.ᶜtemp_scalar_4
        ᶜK_h = p.scratch.ᶜtemp_scalar_6
        if vertical_diffusion isa DecayWithHeightDiffusion
            ᶜK_h .= ᶜcompute_eddy_diffusivity_coefficient(Y.c.ρ, vertical_diffusion)
            ᶜK_u = ᶜK_h
        elseif vertical_diffusion isa VerticalDiffusion
            ᶜK_h .= ᶜcompute_eddy_diffusivity_coefficient(Y.c.uₕ, ᶜp, vertical_diffusion)
            ᶜK_u = ᶜK_h
        elseif is_smagorinsky_vertical(smagorinsky_lilly)
            set_smagorinsky_lilly_precomputed_quantities!(Y, p, smagorinsky_lilly)
            ᶜK_u = p.precomputed.ᶜνₜ_v
            ᶜK_h = p.precomputed.ᶜD_v
        elseif turbconv_model isa AbstractEDMF
            (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
            ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
            ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_3
            ᶜmixing_length_field .= ᶜmixing_length(Y, p)
            ᶜK_u = p.scratch.ᶜtemp_scalar_4
            @. ᶜK_u = eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field)
            ᶜprandtl_nvec = @. lazy(
                turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm),
            )
            ᶜK_h = p.scratch.ᶜtemp_scalar_6
            @. ᶜK_h = eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec)
        end

        ∂ᶠρχ_dif_flux_∂ᶜχ = ᶠp_grad_matrix
        @. ∂ᶠρχ_dif_flux_∂ᶜχ =
            DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_h)) ⋅ ᶠgradᵥ_matrix()
        @. ᶜdiffusion_h_matrix = ᶜadvdivᵥ_matrix() ⋅ ∂ᶠρχ_dif_flux_∂ᶜχ
        if (
            MatrixFields.has_field(Y, @name(c.ρtke)) ||
            !isnothing(p.atmos.turbconv_model) ||
            !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        )
            @. ∂ᶠρχ_dif_flux_∂ᶜχ =
                DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_u)) ⋅ ᶠgradᵥ_matrix()
            @. ᶜdiffusion_u_matrix = ᶜadvdivᵥ_matrix() ⋅ ∂ᶠρχ_dif_flux_∂ᶜχ
        end

        ∂ᶜρe_tot_err_∂ᶜρ = matrix[@name(c.ρe_tot), @name(c.ρ)]
        @. ∂ᶜρe_tot_err_∂ᶜρ = zero(typeof(∂ᶜρe_tot_err_∂ᶜρ))
        @. ∂ᶜρe_tot_err_∂ᶜρe_tot +=
            dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow((1 + ᶜkappa_m) / ᶜρ)

        if MatrixFields.has_field(Y, @name(c.ρq_tot))
            ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
            ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
            ∂ᶜρq_tot_err_∂ᶜρ = matrix[@name(c.ρq_tot), @name(c.ρ)]
            @. ∂ᶜρe_tot_err_∂ᶜρq_tot +=
                dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(ᶜ∂p∂ρq_tot / ᶜρ)
            @. ∂ᶜρq_tot_err_∂ᶜρ = zero(typeof(∂ᶜρq_tot_err_∂ᶜρ))
            @. ∂ᶜρq_tot_err_∂ᶜρq_tot +=
                dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(1 / ᶜρ)
        end

        for (q_name, e_int_q, ∂cv∂q) in microphysics_tracers
            MatrixFields.has_field(Y, q_name) || continue
            ∂ᶜρe_tot_err_∂ᶜρq = matrix[@name(c.ρe_tot), q_name]
            @. ∂ᶜρe_tot_err_∂ᶜρq +=
                dtγ * ᶜdiffusion_h_matrix ⋅
                DiagonalMatrixRow(
                    (ᶜkappa_m * (e_int_q - ∂cv∂q * (ᶜT - T_0)) - R_v * ᶜT) / ᶜρ,
                )
        end

        MatrixFields.unrolled_foreach(tracer_info) do (ρχ_name, _, α)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρ = matrix[ρχ_name, @name(c.ρ)]
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            @. ∂ᶜρχ_err_∂ᶜρ = zero(typeof(∂ᶜρχ_err_∂ᶜρ))
            @. ∂ᶜρχ_err_∂ᶜρχ +=
                dtγ * α * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(1 / ᶜρ)
        end

        if MatrixFields.has_field(Y, @name(c.ρtke))
            turbconv_params = CAP.turbconv_params(params)
            c_d = CAP.tke_diss_coeff(turbconv_params)
            (; dt) = p
            turbconv_model = p.atmos.turbconv_model
            ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
            ᶜρtke = Y.c.ρtke

            # scratch to prevent GPU Kernel parameter memory error
            ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_3
            ᶜmixing_length_field .= ᶜmixing_length(Y, p)

            @inline tke_dissipation_rate_tendency(tke, mixing_length) =
                tke >= 0 ? c_d * sqrt(tke) / mixing_length : 1 / typeof(tke)(dt)
            @inline ∂tke_dissipation_rate_tendency_∂tke(tke, mixing_length) =
                tke > 0 ? c_d / (2 * mixing_length * sqrt(tke)) :
                typeof(tke)(0)

            ᶜdissipation_matrix_diagonal = p.scratch.ᶜtemp_scalar
            @. ᶜdissipation_matrix_diagonal =
                ᶜρtke * ∂tke_dissipation_rate_tendency_∂tke(
                    ᶜtke,
                    ᶜmixing_length_field,
                )

            ∂ᶜρtke_err_∂ᶜρ = matrix[@name(c.ρtke), @name(c.ρ)]
            ∂ᶜρtke_err_∂ᶜρtke =
                matrix[@name(c.ρtke), @name(c.ρtke)]
            @. ∂ᶜρtke_err_∂ᶜρ =
                dtγ * (
                    DiagonalMatrixRow(ᶜdissipation_matrix_diagonal)
                ) ⋅ DiagonalMatrixRow(ᶜtke / Y.c.ρ)
            @. ∂ᶜρtke_err_∂ᶜρtke =
                dtγ * (
                    (
                        ᶜdiffusion_u_matrix -
                        DiagonalMatrixRow(ᶜdissipation_matrix_diagonal)
                    ) ⋅ DiagonalMatrixRow(1 / Y.c.ρ) - DiagonalMatrixRow(
                        tke_dissipation_rate_tendency(
                            ᶜtke,
                            ᶜmixing_length_field,
                        ),
                    )
                ) - (I,)
        end

        if (
            !isnothing(p.atmos.turbconv_model) ||
            !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        )
            ∂ᶜuₕ_err_∂ᶜuₕ = matrix[@name(c.uₕ), @name(c.uₕ)]
            @. ∂ᶜuₕ_err_∂ᶜuₕ =
                dtγ * DiagonalMatrixRow(1 / ᶜρ) ⋅ ᶜdiffusion_u_matrix - (I,)
        end

    end

    # Acoustic diagonal shift: approximate the Schur complement of the
    # coupled (ρ, uₕ) horizontal acoustic system. The Schur complement
    # gives a Helmholtz operator -I - dtγ² c_s² ∇²ₕ on the diagonal.
    # We approximate the diagonal of -∇²ₕ as 2π²/Δx² (2D spectral element).
    if use_derivative(acoustic_diagonal_flag)
        hspace = Spaces.horizontal_space(axes(Y.c))
        Δx = FT(Spaces.node_horizontal_length_scale(hspace))
        γ_d = cp_d / cv_d
        ᶜα_acoustic = p.scratch.ᶜtemp_scalar_2
        @. ᶜα_acoustic = FT(dtγ)^2 * γ_d * ᶜp / ᶜρ * FT(2 * π^2) / Δx^2

        ∂ᶜρ_err_∂ᶜρ = matrix[@name(c.ρ), @name(c.ρ)]
        @. ∂ᶜρ_err_∂ᶜρ = DiagonalMatrixRow(-(FT(1) + ᶜα_acoustic))

        ∂ᶜρe_tot_err_∂ᶜρe_tot = matrix[@name(c.ρe_tot), @name(c.ρe_tot)]
        ρe_tot_already_initialized =
            !(p.atmos.microphysics_model isa DryModel) || use_derivative(diffusion_flag)
        if ρe_tot_already_initialized
            @. ∂ᶜρe_tot_err_∂ᶜρe_tot += DiagonalMatrixRow(FT(-1) * ᶜα_acoustic)
        else
            @. ∂ᶜρe_tot_err_∂ᶜρe_tot = DiagonalMatrixRow(-(FT(1) + ᶜα_acoustic))
        end

        uₕ_already_initialized =
            use_derivative(diffusion_flag) && (
                !isnothing(p.atmos.turbconv_model) ||
                !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
            )
        ∂ᶜuₕ_err_∂ᶜuₕ = matrix[@name(c.uₕ), @name(c.uₕ)]
        if uₕ_already_initialized
            @. ∂ᶜuₕ_err_∂ᶜuₕ += DiagonalMatrixRow(FT(-1) * ᶜα_acoustic)
        else
            @. ∂ᶜuₕ_err_∂ᶜuₕ = DiagonalMatrixRow(-(FT(1) + ᶜα_acoustic))
        end

        # Store state for Helmholtz solve in invert_jacobian!
        cache.helmholtz_state.call_counter = 0
        cache.helmholtz_state.dtγ = dtγ
        cache.helmholtz_state.α_acoustic_max = maximum(ᶜα_acoustic)
        @. cache.helmholtz_state.ᶜα_acoustic = ᶜα_acoustic
        @. cache.helmholtz_state.ᶜcs² = γ_d * ᶜp / ᶜρ
        @. cache.helmholtz_state.ᶜρ = ᶜρ
        @. cache.helmholtz_state.ᶜe_tot = Y.c.ρe_tot / ᶜρ
        @. cache.helmholtz_state.ᶜh_tot = (Y.c.ρe_tot + ᶜp) / ᶜρ
        if do_dss(axes(Y.c))
            cache.helmholtz_state.ghost_buffer_c = p.ghost_buffer.c
        end
    end

    # Sparse direct Helmholtz: extract state and assemble matrix
    if use_derivative(alg.sparse_helmholtz_flag) && !isnothing(cache.sparse_helmholtz)
        shc = cache.sparse_helmholtz
        shc.dtγ[] = FT(dtγ)
        γ_d_sh = cp_d / cv_d
        ᶜcs²_tmp = p.scratch.ᶜtemp_scalar_2
        @. ᶜcs²_tmp = γ_d_sh * ᶜp / ᶜρ
        _sparse_helmholtz_field_to_vec!(
            shc.cs2_vec, ᶜcs²_tmp, shc.helem, shc.npoly, shc.N_v, shc.N_h,
        )
        _sparse_helmholtz_field_to_vec!(
            shc.ρ_vec, ᶜρ, shc.helem, shc.npoly, shc.N_v, shc.N_h,
        )
        assemble_sparse_helmholtz!(shc)
    end

    # 2D Sparse Helmholtz for sphere: extract state and assemble/update
    if use_derivative(alg.sparse_helmholtz_2d_flag) && !isnothing(cache.sparse_helmholtz_2d)
        shc2 = cache.sparse_helmholtz_2d
        γ_d_sh2 = cp_d / cv_d
        ᶜcs²_tmp2 = p.scratch.ᶜtemp_scalar_2
        @. ᶜcs²_tmp2 = γ_d_sh2 * ᶜp / ᶜρ

        if shc2 isa MatrixFreeHelmholtz2DCache
            # Matrix-free: copy fields and compute α_acoustic
            shc2.dtγ[] = FT(dtγ)
            @. shc2.ᶜcs² = ᶜcs²_tmp2
            @. shc2.ᶜρ = ᶜρ
            if shc2.backsub_alpha > 0
                @. shc2.ᶜe_tot = Y.c.ρe_tot / ᶜρ
            end
            # α_acoustic = dtγ²·cs²·λ_diag/Δx² where λ_diag is the Jacobi
            # diagonal estimate. The 2π²/Δx² formula used in HEVI underestimates
            # the true max eigenvalue of the GLL spectral element Laplacian by
            # ~1.5-2× (due to non-uniform GLL spacing at element boundaries).
            # At large dt, this causes Chebyshev eigenvalues to exceed b=1 and
            # the iteration diverges. Safety factor 2.5 ensures the Chebyshev
            # interval [a,b] contains all eigenvalues of M⁻¹A.
            hspace = Spaces.horizontal_space(axes(Y.c))
            Δx_ref = Spaces.node_horizontal_length_scale(hspace)
            chebyshev_safety = FT(1.5)
            @. shc2.ᶜα_acoustic =
                chebyshev_safety * FT(dtγ)^2 * ᶜcs²_tmp2 * FT(2 * π^2) / FT(Δx_ref)^2
            shc2.α_acoustic_max[] = maximum(parent(shc2.ᶜα_acoustic))
        elseif shc2 isa GPUSparseHelmholtz2DCache
            # GPU sparse direct: extract to vectors and assemble/factorize on GPU
            shc2.dtγ[] = FT(dtγ)
            _sparse_helmholtz_2d_field_to_vec!(
                shc2.cs2_vec, ᶜcs²_tmp2, shc2.dof_map, shc2.Nq, shc2.nelem, shc2.N_v,
                shc2.N_h,
            )
            _sparse_helmholtz_2d_field_to_vec!(
                shc2.ρ_vec, ᶜρ, shc2.dof_map, shc2.Nq, shc2.nelem, shc2.N_v, shc2.N_h,
            )
            if shc2.backsub_alpha > 0
                @. ᶜcs²_tmp2 = Y.c.ρe_tot / ᶜρ
                _sparse_helmholtz_2d_field_to_vec!(
                    shc2.e_tot_vec, ᶜcs²_tmp2, shc2.dof_map, shc2.Nq, shc2.nelem, shc2.N_v,
                    shc2.N_h,
                )
            end
            assemble_gpu_sparse_helmholtz_2d!(shc2)
        else
            # CPU sparse direct: extract to vectors and assemble matrix
            shc2.dtγ[] = FT(dtγ)
            _sparse_helmholtz_2d_field_to_vec!(
                shc2.cs2_vec, ᶜcs²_tmp2, shc2.dof_map, shc2.Nq, shc2.nelem, shc2.N_v,
                shc2.N_h,
            )
            _sparse_helmholtz_2d_field_to_vec!(
                shc2.ρ_vec, ᶜρ, shc2.dof_map, shc2.Nq, shc2.nelem, shc2.N_v, shc2.N_h,
            )
            if shc2.backsub_alpha > 0
                @. ᶜcs²_tmp2 = Y.c.ρe_tot / ᶜρ
                _sparse_helmholtz_2d_field_to_vec!(
                    shc2.e_tot_vec, ᶜcs²_tmp2, shc2.dof_map, shc2.Nq, shc2.nelem, shc2.N_v,
                    shc2.N_h,
                )
            end
            assemble_sparse_helmholtz_2d!(shc2)
        end
    end

    if p.atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_advection_flag)
            (; ᶜgradᵥ_ᶠΦ) = p.core
            (;
                ᶜρʲs,
                ᶠu³ʲs,
                ᶜTʲs,
                ᶜq_tot_safeʲs,
                ᶜq_liq_raiʲs,
                ᶜq_ice_snoʲs,
                ᶜKʲs,
            ) = p.precomputed

            # upwinding options for q_tot and mse
            is_third_order =
                p.atmos.numerics.edmfx_mse_q_tot_upwinding == Val(:third_order)
            ᶠupwind = is_third_order ? ᶠupwind3 : ᶠupwind1
            ᶠset_upwind_bcs = Operators.SetBoundaryOperator(;
                top = Operators.SetValue(zero(CT3{FT})),
                bottom = Operators.SetValue(zero(CT3{FT})),
            ) # Need to wrap ᶠupwind in this for well-defined boundaries.
            UpwindMatrixRowType =
                is_third_order ? QuaddiagonalMatrixRow : BidiagonalMatrixRow
            ᶠupwind_matrix = is_third_order ? ᶠupwind3_matrix : ᶠupwind1_matrix
            ᶠset_upwind_matrix_bcs = Operators.SetBoundaryOperator(;
                top = Operators.SetValue(zero(UpwindMatrixRowType{CT3{FT}})),
                bottom = Operators.SetValue(zero(UpwindMatrixRowType{CT3{FT}})),
            ) # Need to wrap ᶠupwind_matrix in this for well-defined boundaries.

            # upwinding options for other tracers
            is_tracer_upwinding_third_order =
                p.atmos.numerics.edmfx_tracer_upwinding == Val(:third_order)
            ᶠtracer_upwind = is_tracer_upwinding_third_order ? ᶠupwind3 : ᶠupwind1
            ᶠset_tracer_upwind_bcs = Operators.SetBoundaryOperator(;
                top = Operators.SetValue(zero(CT3{FT})),
                bottom = Operators.SetValue(zero(CT3{FT})),
            ) # Need to wrap ᶠtracer_upwind in this for well-defined boundaries.
            TracerUpwindMatrixRowType =
                is_tracer_upwinding_third_order ? QuaddiagonalMatrixRow :
                BidiagonalMatrixRow
            ᶠtracer_upwind_matrix =
                is_tracer_upwinding_third_order ? ᶠupwind3_matrix : ᶠupwind1_matrix
            ᶠset_tracer_upwind_matrix_bcs = Operators.SetBoundaryOperator(;
                top = Operators.SetValue(zero(TracerUpwindMatrixRowType{CT3{FT}})),
                bottom = Operators.SetValue(zero(TracerUpwindMatrixRowType{CT3{FT}})),
            ) # Need to wrap ᶠtracer_upwind_matrix in this for well-defined boundaries.

            ᶠu³ʲ_data = ᶠu³ʲs.:(1).components.data.:1

            ᶜkappa_mʲ = p.scratch.ᶜtemp_scalar
            @. ᶜkappa_mʲ =
                TD.gas_constant_air(
                    thermo_params,
                    ᶜq_tot_safeʲs.:(1),
                    ᶜq_liq_raiʲs.:(1),
                    ᶜq_ice_snoʲs.:(1),
                ) /
                TD.cv_m(
                    thermo_params,
                    ᶜq_tot_safeʲs.:(1),
                    ᶜq_liq_raiʲs.:(1),
                    ᶜq_ice_snoʲs.:(1),
                )

            ∂ᶜq_totʲ_err_∂ᶜq_totʲ =
                matrix[@name(c.sgsʲs.:(1).q_tot), @name(c.sgsʲs.:(1).q_tot)]
            @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ =
                dtγ * (
                    DiagonalMatrixRow(ᶜadvdivᵥ(ᶠu³ʲs.:(1))) -
                    ᶜadvdivᵥ_matrix() ⋅
                    ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1)))
                ) - (I,)

            ∂ᶜmseʲ_err_∂ᶜmseʲ =
                matrix[@name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).mse)]
            @. ∂ᶜmseʲ_err_∂ᶜmseʲ =
                dtγ * (
                    DiagonalMatrixRow(ᶜadvdivᵥ(ᶠu³ʲs.:(1))) -
                    ᶜadvdivᵥ_matrix() ⋅
                    ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) -
                    DiagonalMatrixRow(
                        adjoint(ᶜinterp(ᶠu³ʲs.:(1))) *
                        ᶜgradᵥ_ᶠΦ *
                        Y.c.ρ *
                        ᶜkappa_mʲ / ((ᶜkappa_mʲ + 1) * ᶜp),
                    )
                ) - (I,)

            ∂ᶜρaʲ_err_∂ᶜρaʲ =
                matrix[@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).ρa)]
            @. ᶜadvection_matrix =
                -(ᶜadvdivᵥ_matrix()) ⋅
                DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ)
            @. ∂ᶜρaʲ_err_∂ᶜρaʲ =
                dtγ * ᶜadvection_matrix ⋅
                ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) ⋅
                DiagonalMatrixRow(1 / ᶜρʲs.:(1)) - (I,)

            # contribution of ρʲ variations in vertical transport of ρa and updraft buoyancy eq
            ∂ᶜρaʲ_err_∂ᶜmseʲ =
                matrix[@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).mse)]
            @. ᶠbidiagonal_matrix_ct3 =
                DiagonalMatrixRow(
                    ᶠset_upwind_bcs(
                        ᶠupwind(
                            ᶠu³ʲs.:(1),
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ),
                    ) / ᶠJ,
                ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(
                    ᶜJ * ᶜkappa_mʲ * (ᶜρʲs.:(1))^2 / ((ᶜkappa_mʲ + 1) * ᶜp),
                )
            @. ᶠbidiagonal_matrix_ct3_2 =
                DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) ⋅
                ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) ⋅
                DiagonalMatrixRow(
                    Y.c.sgsʲs.:(1).ρa * ᶜkappa_mʲ / ((ᶜkappa_mʲ + 1) * ᶜp),
                )
            @. ∂ᶜρaʲ_err_∂ᶜmseʲ =
                dtγ * ᶜadvdivᵥ_matrix() ⋅
                (ᶠbidiagonal_matrix_ct3 - ᶠbidiagonal_matrix_ct3_2)

            turbconv_params = CAP.turbconv_params(params)
            α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
            ᶜ∂RmT∂qʲ = p.scratch.ᶜtemp_scalar_2
            sgs_microphysics_tracers =
                p.atmos.microphysics_model isa Union{
                    NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M,
                } ?
                (
                    (@name(c.sgsʲs.:(1).q_tot), -LH_v0, Δcp_v, ΔR_v),
                    (@name(c.sgsʲs.:(1).q_liq), LH_v0, Δcp_l, -R_v),
                    (@name(c.sgsʲs.:(1).q_ice), LH_s0, Δcp_i, -R_v),
                    (@name(c.sgsʲs.:(1).q_rai), LH_v0, Δcp_l, -R_v),
                    (@name(c.sgsʲs.:(1).q_sno), LH_s0, Δcp_i, -R_v),
                ) : (
                    (@name(c.sgsʲs.:(1).q_tot), -LH_v0, Δcp_v, ΔR_v),
                )

            for (qʲ_name, LH, ∂cp∂q, ∂Rm∂q) in sgs_microphysics_tracers
                MatrixFields.has_field(Y, qʲ_name) || continue

                @. ᶜ∂RmT∂qʲ =
                    ᶜkappa_mʲ / (ᶜkappa_mʲ + 1) * (LH - ∂cp∂q * (ᶜTʲs.:(1) - T_0)) +
                    ∂Rm∂q * ᶜTʲs.:(1)

                # ∂ᶜρaʲ_err_∂ᶜqʲ through ρʲ variations in vertical transport of ρa
                ∂ᶜρaʲ_err_∂ᶜqʲ = matrix[@name(c.sgsʲs.:(1).ρa), qʲ_name]
                @. ᶠbidiagonal_matrix_ct3 =
                    DiagonalMatrixRow(
                        ᶠset_upwind_bcs(
                            ᶠupwind(
                                ᶠu³ʲs.:(1),
                                draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                            ),
                        ) / ᶠJ,
                    ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(
                        ᶜJ * (ᶜρʲs.:(1))^2 / ᶜp * ᶜ∂RmT∂qʲ,
                    )
                @. ᶠbidiagonal_matrix_ct3_2 =
                    DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) ⋅
                    ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) ⋅
                    DiagonalMatrixRow(
                        Y.c.sgsʲs.:(1).ρa / ᶜp * ᶜ∂RmT∂qʲ,
                    )
                @. ∂ᶜρaʲ_err_∂ᶜqʲ =
                    dtγ * ᶜadvdivᵥ_matrix() ⋅
                    (ᶠbidiagonal_matrix_ct3 - ᶠbidiagonal_matrix_ct3_2)

                # ∂ᶜmseʲ_err_∂ᶜqʲ through ρʲ variations in buoyancy term in mse eq
                ∂ᶜmseʲ_err_∂ᶜqʲ = matrix[@name(c.sgsʲs.:(1).mse), qʲ_name]
                @. ∂ᶜmseʲ_err_∂ᶜqʲ =
                    dtγ * (
                        -DiagonalMatrixRow(
                            adjoint(ᶜinterp(ᶠu³ʲs.:(1))) * ᶜgradᵥ_ᶠΦ * Y.c.ρ / ᶜp *
                            ᶜ∂RmT∂qʲ,
                        )
                    )
            end

            # advection and sedimentation of microphysics tracers
            if p.atmos.microphysics_model isa Union{
                NonEquilibriumMicrophysics1M,
                NonEquilibriumMicrophysics2M,
            }

                ᶜa = (@. lazy(draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1))))
                ᶜ∂a∂z = p.scratch.ᶜtemp_scalar_7
                @. ᶜ∂a∂z = ᶜprecipdivᵥ(ᶠinterp(ᶜJ) / ᶠJ * ᶠright_bias(Geometry.WVector(ᶜa)))
                ᶜinv_ρ̂ = (@. lazy(
                    specific(
                        FT(1),
                        Y.c.sgsʲs.:(1).ρa,
                        FT(0),
                        ᶜρʲs.:(1),
                        p.atmos.turbconv_model,
                    ),
                ))
                sgs_microphysics_tracers = (
                    (@name(c.sgsʲs.:(1).q_liq), @name(ᶜwₗʲs.:(1))),
                    (@name(c.sgsʲs.:(1).q_ice), @name(ᶜwᵢʲs.:(1))),
                    (@name(c.sgsʲs.:(1).q_rai), @name(ᶜwᵣʲs.:(1))),
                    (@name(c.sgsʲs.:(1).q_sno), @name(ᶜwₛʲs.:(1))),
                    (@name(c.sgsʲs.:(1).n_liq), @name(ᶜwₙₗʲs.:(1))),
                    (@name(c.sgsʲs.:(1).n_rai), @name(ᶜwₙᵣʲs.:(1))),
                )
                MatrixFields.unrolled_foreach(
                    sgs_microphysics_tracers,
                ) do (χʲ_name, wʲ_name)
                    MatrixFields.has_field(Y, χʲ_name) || return
                    ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                    ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)

                    # advection
                    ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χʲ_name, χʲ_name]
                    @. ∂ᶜχʲ_err_∂ᶜχʲ =
                        dtγ * (
                            DiagonalMatrixRow(ᶜadvdivᵥ(ᶠu³ʲs.:(1))) -
                            ᶜadvdivᵥ_matrix() ⋅
                            ᶠset_tracer_upwind_matrix_bcs(
                                ᶠtracer_upwind_matrix(ᶠu³ʲs.:(1)),
                            )
                        ) - (I,)

                    # sedimentation
                    # (pull out common subexpression for performance)
                    @. ᶠsed_tracer_advection =
                        DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) ⋅
                        ᶠright_bias_matrix() ⋅
                        DiagonalMatrixRow(-Geometry.WVector(ᶜwʲ))
                    @. ᶜtridiagonal_matrix_scalar =
                        dtγ * ifelse(ᶜ∂a∂z < 0,
                            -(ᶜprecipdivᵥ_matrix()) ⋅ ᶠsed_tracer_advection *
                            DiagonalMatrixRow(ᶜa),
                            -DiagonalMatrixRow(ᶜa) ⋅ ᶜprecipdivᵥ_matrix() ⋅
                            ᶠsed_tracer_advection,
                        )

                    @. ∂ᶜχʲ_err_∂ᶜχʲ +=
                        DiagonalMatrixRow(ᶜinv_ρ̂) ⋅ ᶜtridiagonal_matrix_scalar

                    if χʲ_name in (
                        @name(c.sgsʲs.:(1).q_liq),
                        @name(c.sgsʲs.:(1).q_ice),
                        @name(c.sgsʲs.:(1).q_rai),
                        @name(c.sgsʲs.:(1).q_sno),
                    )
                        ∂ᶜq_totʲ_err_∂ᶜχʲ =
                            matrix[@name(c.sgsʲs.:(1).q_tot), χʲ_name]
                        @. ∂ᶜq_totʲ_err_∂ᶜχʲ =
                            DiagonalMatrixRow(ᶜinv_ρ̂) ⋅ ᶜtridiagonal_matrix_scalar
                    end

                end
            end

            # vertical diffusion of updrafts
            if use_derivative(sgs_vertdiff_flag)
                α_vert_diff_tracer = CAP.α_vert_diff_tracer(params)
                @. p.scratch.ᶜbidiagonal_adjoint_matrix_c3 =
                    ᶜadvdivᵥ_matrix() ⋅
                    DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1)) * ᶠinterp(ᶜK_h))
                @. ᶜdiffusion_h_matrix =
                    p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅ ᶠgradᵥ_matrix()

                @. ∂ᶜmseʲ_err_∂ᶜmseʲ +=
                    dtγ * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅ ᶜdiffusion_h_matrix
                @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ +=
                    dtγ * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅ ᶜdiffusion_h_matrix
                @. ∂ᶜρaʲ_err_∂ᶜρaʲ +=
                    dtγ * DiagonalMatrixRow(1 / (1 - Y.c.sgsʲs.:(1).q_tot) / ᶜρʲs.:(1)) ⋅
                    ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(Y.c.sgsʲs.:(1).q_tot)
                ∂ᶜρaʲ_err_∂ᶜq_totʲ =
                    matrix[@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).q_tot)]
                @. ∂ᶜρaʲ_err_∂ᶜq_totʲ +=
                    dtγ * DiagonalMatrixRow(
                        Y.c.sgsʲs.:(1).ρa / (1 - Y.c.sgsʲs.:(1).q_tot) / ᶜρʲs.:(1),
                    ) ⋅
                    ᶜdiffusion_h_matrix
                @. ∂ᶜρaʲ_err_∂ᶜq_totʲ +=
                    dtγ * DiagonalMatrixRow(
                        Y.c.sgsʲs.:(1).ρa / (1 - Y.c.sgsʲs.:(1).q_tot)^2 / ᶜρʲs.:(1),
                    ) ⋅
                    ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(Y.c.sgsʲs.:(1).q_tot)
                if p.atmos.microphysics_model isa Union{
                    NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M,
                }
                    sgs_microphysics_tracers = (
                        (@name(c.sgsʲs.:(1).q_liq), FT(1)),
                        (@name(c.sgsʲs.:(1).q_ice), FT(1)),
                        (@name(c.sgsʲs.:(1).q_rai), α_vert_diff_tracer),
                        (@name(c.sgsʲs.:(1).q_sno), α_vert_diff_tracer),
                        (@name(c.sgsʲs.:(1).n_liq), FT(1)),
                        (@name(c.sgsʲs.:(1).n_rai), α_vert_diff_tracer),
                    )
                    MatrixFields.unrolled_foreach(
                        sgs_microphysics_tracers,
                    ) do (χʲ_name, α)
                        MatrixFields.has_field(Y, χʲ_name) || return
                        ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χʲ_name, χʲ_name]
                        @. ∂ᶜχʲ_err_∂ᶜχʲ +=
                            dtγ * α * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅
                            ᶜdiffusion_h_matrix
                    end
                end
            end
            # entrainment and detrainment (rates are treated explicitly)
            if use_derivative(sgs_entr_detr_flag)
                (; ᶜentrʲs, ᶜdetrʲs, ᶜturb_entrʲs) = p.precomputed
                @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ -=
                    dtγ * DiagonalMatrixRow(ᶜentrʲs.:(1) + ᶜturb_entrʲs.:(1))
                @. ∂ᶜmseʲ_err_∂ᶜmseʲ -=
                    dtγ * DiagonalMatrixRow(ᶜentrʲs.:(1) + ᶜturb_entrʲs.:(1))
                @. ∂ᶜρaʲ_err_∂ᶜρaʲ +=
                    dtγ * DiagonalMatrixRow(ᶜentrʲs.:(1) - ᶜdetrʲs.:(1))
                if p.atmos.microphysics_model isa Union{
                    NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M,
                }
                    sgs_microphysics_tracers = (
                        (@name(c.sgsʲs.:(1).q_liq)),
                        (@name(c.sgsʲs.:(1).q_ice)),
                        (@name(c.sgsʲs.:(1).q_rai)),
                        (@name(c.sgsʲs.:(1).q_sno)),
                    )
                    MatrixFields.unrolled_foreach(
                        sgs_microphysics_tracers,
                    ) do (qʲ_name)
                        MatrixFields.has_field(Y, qʲ_name) || return

                        ∂ᶜqʲ_err_∂ᶜqʲ = matrix[qʲ_name, qʲ_name]
                        @. ∂ᶜqʲ_err_∂ᶜqʲ -=
                            dtγ * DiagonalMatrixRow(ᶜentrʲs.:(1) + ᶜturb_entrʲs.:(1))
                    end
                end
            end

            # add updraft mass flux contributions to grid-mean
            if use_derivative(sgs_mass_flux_flag)
                # Jacobian contributions of updraft massflux to grid-mean
                ∂ᶜupdraft_mass_flux_∂ᶜscalar = ᶠbidiagonal_matrix_ct3
                @. ∂ᶜupdraft_mass_flux_∂ᶜscalar =
                    DiagonalMatrixRow(
                        (ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) * (ᶠu³ʲs.:(1) - ᶠu³),
                    ) ⋅ ᶠinterp_matrix() ⋅
                    DiagonalMatrixRow(Y.c.sgsʲs.:(1).ρa / ᶜρʲs.:(1))
                @. p.scratch.ᶜtridiagonal_matrix_scalar =
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar

                # Derivative of total energy tendency with respect to updraft MSE
                ## grid-mean ρe_tot
                ᶜkappa_m = p.scratch.ᶜtemp_scalar
                @. ᶜkappa_m =
                    TD.gas_constant_air(
                        thermo_params,
                        ᶜq_tot_safe,
                        ᶜq_liq_rai,
                        ᶜq_ice_sno,
                    ) /
                    TD.cv_m(thermo_params, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno)


                ᶜ∂p∂ρq_tot = p.scratch.ᶜtemp_scalar_2
                @. ᶜ∂p∂ρq_tot =
                    ᶜkappa_m * (-e_int_v0 - R_d * T_0 - Δcv_v * (ᶜT - T_0)) + ΔR_v * ᶜT

                ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
                @. ∂ᶜρe_tot_err_∂ᶜρ +=
                    p.scratch.ᶜtridiagonal_matrix_scalar ⋅
                    DiagonalMatrixRow(
                        (
                            -(ᶜh_tot) +
                            ᶜkappa_m * (T_0 * cp_d - ᶜK - ᶜΦ) +
                            (R_d - ᶜkappa_m * cv_d) * ᶜT
                        ) / ᶜρ,
                    )

                @. ∂ᶜρe_tot_err_∂ᶜρq_tot +=
                    p.scratch.ᶜtridiagonal_matrix_scalar ⋅
                    DiagonalMatrixRow(ᶜ∂p∂ρq_tot / ᶜρ)

                for (q_name, e_int_q, ∂cv∂q) in microphysics_tracers
                    MatrixFields.has_field(Y, q_name) || continue
                    ∂ᶜρe_tot_err_∂ᶜρq = matrix[@name(c.ρe_tot), q_name]
                    @. ∂ᶜρe_tot_err_∂ᶜρq +=
                        p.scratch.ᶜtridiagonal_matrix_scalar ⋅
                        DiagonalMatrixRow(
                            (ᶜkappa_m * (e_int_q - ∂cv∂q * (ᶜT - T_0)) - R_v * ᶜT) / ᶜρ,
                        )
                end

                @. ∂ᶜρe_tot_err_∂ᶜρe_tot +=
                    p.scratch.ᶜtridiagonal_matrix_scalar ⋅
                    DiagonalMatrixRow((1 + ᶜkappa_m) / ᶜρ)

                ∂ᶜρe_tot_err_∂ᶜmseʲ =
                    matrix[@name(c.ρe_tot), @name(c.sgsʲs.:(1).mse)]
                @. ∂ᶜρe_tot_err_∂ᶜmseʲ =
                    -(p.scratch.ᶜtridiagonal_matrix_scalar)

                ## grid-mean ρq_tot
                @. ∂ᶜρq_tot_err_∂ᶜρ +=
                    p.scratch.ᶜtridiagonal_matrix_scalar ⋅
                    DiagonalMatrixRow(-(ᶜq_tot) / ᶜρ)

                @. ∂ᶜρq_tot_err_∂ᶜρq_tot +=
                    p.scratch.ᶜtridiagonal_matrix_scalar ⋅
                    DiagonalMatrixRow(1 / ᶜρ)

                ∂ᶜρq_tot_err_∂ᶜq_totʲ =
                    matrix[@name(c.ρq_tot), @name(c.sgsʲs.:(1).q_tot)]
                @. ∂ᶜρq_tot_err_∂ᶜq_totʲ =
                    -(p.scratch.ᶜtridiagonal_matrix_scalar)

                # grid-mean ∂/∂(u₃ʲ)
                ∂ᶜρe_tot_err_∂ᶠu₃ = matrix[@name(c.ρe_tot), @name(f.u₃)]
                @. ∂ᶜρe_tot_err_∂ᶠu₃ +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(
                        ᶠinterp(
                            (Y.c.sgsʲs.:(1).mse + ᶜKʲs.:(1) - ᶜh_tot) *
                            ᶜρʲs.:(1) *
                            ᶜJ *
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ) / ᶠJ * (g³³(ᶠgⁱʲ)),
                    )

                @. p.scratch.ᶠdiagonal_matrix_ct3xct3 = DiagonalMatrixRow(
                    ᶠinterp(
                        (Y.c.sgsʲs.:(1).q_tot - ᶜq_tot) *
                        ᶜρʲs.:(1) *
                        ᶜJ *
                        draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                    ) / ᶠJ * (g³³(ᶠgⁱʲ)),
                )

                ∂ᶜρq_tot_err_∂ᶠu₃ = matrix[@name(c.ρq_tot), @name(f.u₃)]
                @. ∂ᶜρq_tot_err_∂ᶠu₃ +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ p.scratch.ᶠdiagonal_matrix_ct3xct3

                # grid-mean ∂/∂(rho*a)
                ∂ᶜρe_tot_err_∂ᶜρa =
                    matrix[@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)]
                @. p.scratch.ᶠtemp_CT3_2 =
                    (ᶠu³ʲs.:(1) - ᶠu³) *
                    ᶠinterp((Y.c.sgsʲs.:(1).mse + ᶜKʲs.:(1) - ᶜh_tot)) / ᶠJ
                @. p.scratch.ᶜbidiagonal_matrix_scalar =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(p.scratch.ᶠtemp_CT3_2)
                @. ∂ᶜρe_tot_err_∂ᶜρa =
                    p.scratch.ᶜbidiagonal_matrix_scalar ⋅ ᶠinterp_matrix() ⋅
                    DiagonalMatrixRow(ᶜJ)

                ∂ᶜρq_tot_err_∂ᶜρa =
                    matrix[@name(c.ρq_tot), @name(c.sgsʲs.:(1).ρa)]
                @. p.scratch.ᶠtemp_CT3_2 =
                    (ᶠu³ʲs.:(1) - ᶠu³) *
                    ᶠinterp((Y.c.sgsʲs.:(1).q_tot - ᶜq_tot)) / ᶠJ
                @. p.scratch.ᶜbidiagonal_matrix_scalar =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(p.scratch.ᶠtemp_CT3_2)
                @. ∂ᶜρq_tot_err_∂ᶜρa =
                    p.scratch.ᶜbidiagonal_matrix_scalar ⋅ ᶠinterp_matrix() ⋅
                    DiagonalMatrixRow(ᶜJ)

                # grid-mean tracers
                if p.atmos.microphysics_model isa Union{
                    NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M,
                }

                    microphysics_tracers = (
                        (@name(c.ρq_liq), @name(c.sgsʲs.:(1).q_liq), @name(q_liq)),
                        (@name(c.ρq_ice), @name(c.sgsʲs.:(1).q_ice), @name(q_ice)),
                        (@name(c.ρq_rai), @name(c.sgsʲs.:(1).q_rai), @name(q_rai)),
                        (@name(c.ρq_sno), @name(c.sgsʲs.:(1).q_sno), @name(q_sno)),
                        (@name(c.ρn_liq), @name(c.sgsʲs.:(1).n_liq), @name(n_liq)),
                        (@name(c.ρn_rai), @name(c.sgsʲs.:(1).n_rai), @name(n_rai)),
                    )

                    # add updraft contributions
                    # pull common subexpressions that don't depend on which
                    # tracer out of the tracer loop for performance
                    @. ᶜtracer_advection_matrix =
                        -(ᶜadvdivᵥ_matrix()) ⋅
                        DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ)
                    @. ᶜtridiagonal_matrix =
                        ᶜtracer_advection_matrix ⋅ ᶠset_tracer_upwind_matrix_bcs(
                            ᶠtracer_upwind_matrix(ᶠu³ʲs.:(1)),
                        )
                    MatrixFields.unrolled_foreach(
                        microphysics_tracers,
                    ) do (ρχ_name, χʲ_name, χ_name)
                        MatrixFields.has_field(Y, ρχ_name) || return
                        ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)

                        ∂ᶜρχ_err_∂ᶜχʲ =
                            matrix[ρχ_name, χʲ_name]
                        @. ∂ᶜρχ_err_∂ᶜχʲ =
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)))

                        ∂ᶜρχ_err_∂ᶜρa =
                            matrix[ρχ_name, @name(c.sgsʲs.:(1).ρa)]
                        @. ∂ᶜρχ_err_∂ᶜρa =
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(ᶜχʲ / ᶜρʲs.:(1))

                    end

                    # add env flux contributions
                    (; ᶜp) = p.precomputed
                    (; ᶠu³⁰, ᶜT⁰, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed
                    ᶜρ⁰ = @. lazy(
                        TD.air_density(
                            thermo_params,
                            ᶜT⁰,
                            ᶜp,
                            ᶜq_tot_safe⁰,
                            ᶜq_liq_rai⁰,
                            ᶜq_ice_sno⁰,
                        ),
                    )
                    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
                    ᶠu³⁰_data = ᶠu³⁰.components.data.:1

                    # pull common subexpressions that don't depend on which
                    # tracer out of the tracer loop for performance
                    @. ᶜtracer_advection_matrix =
                        -(ᶜadvdivᵥ_matrix()) ⋅
                        DiagonalMatrixRow(ᶠinterp(ᶜρ⁰ * ᶜJ) / ᶠJ)
                    @. ᶜtridiagonal_matrix =
                        ᶜtracer_advection_matrix ⋅ ᶠset_tracer_upwind_matrix_bcs(
                            ᶠtracer_upwind_matrix(ᶠu³⁰),
                        )
                    MatrixFields.unrolled_foreach(
                        microphysics_tracers,
                    ) do (ρχ_name, χʲ_name, χ_name)
                        MatrixFields.has_field(Y, ρχ_name) || return
                        ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                        ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)

                        ∂ᶜρχ_err_∂ᶜχʲ =
                            matrix[ρχ_name, χʲ_name]
                        @. ∂ᶜρχ_err_∂ᶜχʲ +=
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(-1 * Y.c.sgsʲs.:(1).ρa / ᶜρ⁰)

                        ∂ᶜρχ_err_∂ᶜρa =
                            matrix[ρχ_name, @name(c.sgsʲs.:(1).ρa)]
                        # pull out and store for kernel performance
                        @. ᶠbidiagonal_matrix_ct3_2 =
                            ᶠset_tracer_upwind_matrix_bcs(
                                ᶠtracer_upwind_matrix(CT3(sign(ᶠu³⁰_data))),
                            ) ⋅ DiagonalMatrixRow(ᶜχ⁰ * draft_area(ᶜρa⁰, ᶜρ⁰))
                        @. ∂ᶜρχ_err_∂ᶜρa +=
                            dtγ *
                            ᶜtracer_advection_matrix ⋅
                            DiagonalMatrixRow(
                                (ᶠu³⁰_data - ᶠu³ʲ_data) / ᶠinterp(ᶜρa⁰),
                            ) ⋅ ᶠbidiagonal_matrix_ct3_2

                        @. ∂ᶜρχ_err_∂ᶜρa +=
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(-1 * ᶜχʲ / ᶜρ⁰)

                        ∂ᶜρχ_err_∂ᶜρχ =
                            matrix[ρχ_name, ρχ_name]
                        @. ∂ᶜρχ_err_∂ᶜρχ +=
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(1 / ᶜρ⁰)

                        ∂ᶜρχ_err_∂ᶠu₃ =
                            matrix[ρχ_name, @name(f.u₃)]
                        @. ∂ᶜρχ_err_∂ᶠu₃ =
                            dtγ * ᶜtracer_advection_matrix ⋅
                            DiagonalMatrixRow(
                                ᶠset_tracer_upwind_bcs(
                                    ᶠtracer_upwind(CT3(sign(ᶠu³⁰_data)),
                                        ᶜχ⁰ * draft_area(ᶜρa⁰, ᶜρ⁰),
                                    ),
                                ) * adjoint(C3(sign(ᶠu³⁰_data))) *
                                ᶠinterp(Y.c.ρ / ᶜρa⁰) * g³³(ᶠgⁱʲ),
                            )
                    end
                end
            end
        end
    end

    update_microphysics_jacobian!(matrix, Y, p, dtγ, sgs_advection_flag)

    # NOTE: All velocity tendency derivatives should be set BEFORE this call.
    zero_velocity_jacobian!(matrix, Y, p, t)
end

"""
    update_microphysics_jacobian!(matrix, Y, p, dtγ, sgs_advection_flag)

Add diagonal Jacobian entries for implicit microphysics tendencies (0M, 1M, 2M,
and EDMF updraft species).

Extracted from `update_jacobian!` to keep the parent function below Julia's
optimization threshold — large functions cause the compiler to miss inlining
opportunities in broadcast expressions, resulting in heap allocations.
"""
function update_microphysics_jacobian!(matrix, Y, p, dtγ, sgs_advection_flag)
    p.atmos.microphysics_tendency_timestepping == Implicit() || return nothing

    ᶜρ = Y.c.ρ
    # TODO - do we need a corresponding term for ρe_tot?

    # 0M microphysics: diagonal entry for ρq_tot
    if p.atmos.microphysics_model isa EquilibriumMicrophysics0M
        if MatrixFields.has_field(Y, @name(c.ρq_tot))
            (; ᶜρ_dq_tot_dt) = p.precomputed
            ∂ᶜρq_tot_err_∂ᶜρq_tot = matrix[@name(c.ρq_tot), @name(c.ρq_tot)]
            @. ∂ᶜρq_tot_err_∂ᶜρq_tot +=
                dtγ * DiagonalMatrixRow(_jac_coeff(
                    ᶜρ_dq_tot_dt, Y.c.ρq_tot,
                ))
        end
    end

    # 1M microphysics: diagonal entries for ρq_liq, ρq_ice, ρq_rai, ρq_sno
    if p.atmos.microphysics_model isa NonEquilibriumMicrophysics1M
        (; ᶜmp_derivative) = p.precomputed

        # Cloud condensate (q_lcl, q_icl): use BMT grid-mean derivatives
        # (dominated by the condensation/deposition term -1/τ_relax, which
        # is independent of the SGS distribution)
        cloud_1m_deriv_tracers = (
            (@name(c.ρq_liq), ᶜmp_derivative.∂tendency_∂q_lcl),
            (@name(c.ρq_ice), ᶜmp_derivative.∂tendency_∂q_icl),
        )
        MatrixFields.unrolled_foreach(
            cloud_1m_deriv_tracers,
        ) do (ρχ_name, ᶜ∂S∂q)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            @. ∂ᶜρχ_err_∂ᶜρχ += dtγ * DiagonalMatrixRow(ᶜ∂S∂q)
        end

        # Precipitation (q_rai, q_sno): use S/q from quadrature-integrated
        # tendencies. This makes the Jacobian consistent with the SGS quadrature
        # used in the implicit tendency, preventing Newton solver divergence
        # when the SGS distribution differs from the grid mean.
        if p.atmos.turbconv_model isa PrognosticEDMFX
            # Environment quadrature tendencies
            (; ᶜmp_tendency⁰) = p.precomputed
            precip_1m_sq_tracers = (
                (@name(c.ρq_rai), ᶜmp_tendency⁰.dq_rai_dt, Y.c.ρq_rai),
                (@name(c.ρq_sno), ᶜmp_tendency⁰.dq_sno_dt, Y.c.ρq_sno),
            )
        else
            # Grid-mean quadrature tendencies
            (; ᶜmp_tendency) = p.precomputed
            precip_1m_sq_tracers = (
                (@name(c.ρq_rai), ᶜmp_tendency.dq_rai_dt, Y.c.ρq_rai),
                (@name(c.ρq_sno), ᶜmp_tendency.dq_sno_dt, Y.c.ρq_sno),
            )
        end
        MatrixFields.unrolled_foreach(
            precip_1m_sq_tracers,
        ) do (ρχ_name, ᶜS, ᶜρχ)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            # S/q approximation: ∂(dq/dt)/∂q ≈ (dq/dt) / q
            # Uses the full derivative (including source terms) for an accurate
            # Newton linearization consistent with the quadrature tendencies.
            @. ∂ᶜρχ_err_∂ᶜρχ += dtγ * DiagonalMatrixRow(
                _jac_coeff_from_ratio(ᶜS, ᶜρχ, ᶜρ),
            )
        end
    end

    # 2M microphysics: diagonal entries for ρq_liq, ρq_rai, ρn_liq, ρn_rai
    if p.atmos.microphysics_model isa NonEquilibriumMicrophysics2M
        (; ᶜmp_derivative) = p.precomputed

        # Cloud fields: use BMT grid-mean derivatives
        cloud_2m_deriv_tracers = (
            (@name(c.ρq_liq), ᶜmp_derivative.∂tendency_∂q_lcl),
            (@name(c.ρn_liq), ᶜmp_derivative.∂tendency_∂n_lcl),
        )
        MatrixFields.unrolled_foreach(
            cloud_2m_deriv_tracers,
        ) do (ρχ_name, ᶜ∂S∂q)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            @. ∂ᶜρχ_err_∂ᶜρχ += dtγ * DiagonalMatrixRow(ᶜ∂S∂q)
        end

        # Precipitation: use S/q from quadrature-integrated tendencies
        # _jac_coeff_from_ratio safely returns zero when |q| < ε
        (; ᶜmp_tendency) = p.precomputed
        precip_2m_sq_tracers = (
            (@name(c.ρq_rai), ᶜmp_tendency.dq_rai_dt, Y.c.ρq_rai),
            (@name(c.ρn_rai), ᶜmp_tendency.dn_rai_dt, Y.c.ρn_rai),
        )
        MatrixFields.unrolled_foreach(
            precip_2m_sq_tracers,
        ) do (ρχ_name, ᶜS, ᶜρχ)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            @. ∂ᶜρχ_err_∂ᶜρχ += dtγ * DiagonalMatrixRow(
                _jac_coeff_from_ratio(ᶜS, ᶜρχ, ᶜρ),
            )
        end
    end

    # EDMF microphysics: diagonal entries for updraft variables
    if p.atmos.turbconv_model isa PrognosticEDMFX
        # 0M EDMF
        if p.atmos.microphysics_model isa EquilibriumMicrophysics0M
            if hasproperty(p.precomputed, :ᶜmp_tendencyʲs)
                (; ᶜmp_tendencyʲs) = p.precomputed
                ᶜSq_tot = ᶜmp_tendency.:(1).dq_tot_dt

                q_name = @name(c.sgsʲs.:(1).q_tot)
                if MatrixFields.has_field(Y, q_name)
                    ∂ᶜq_err_∂ᶜq = matrix[q_name, q_name]
                    if !use_derivative(sgs_advection_flag)
                        @. ∂ᶜq_err_∂ᶜq =
                            zero(typeof(∂ᶜq_err_∂ᶜq)) - (I,)
                    end
                    add_microphysics_jacobian_entry!(
                        ∂ᶜq_err_∂ᶜq, dtγ, ᶜSq_tot, Y.c.sgsʲs.:(1).q_tot,
                    )
                end

                ρa_name = @name(c.sgsʲs.:(1).ρa)
                if MatrixFields.has_field(Y, ρa_name)
                    ∂ᶜρa_err_∂ᶜρa = matrix[ρa_name, ρa_name]
                    if !use_derivative(sgs_advection_flag)
                        @. ∂ᶜρa_err_∂ᶜρa =
                            zero(typeof(∂ᶜρa_err_∂ᶜρa)) - (I,)
                    end
                    @. ∂ᶜρa_err_∂ᶜρa += dtγ * DiagonalMatrixRow(ᶜSq_tot)
                end
            end
        end

        # 1M EDMF: diagonal entries for individual condensate species.
        if p.atmos.microphysics_model isa NonEquilibriumMicrophysics1M
            # Cloud (q_liq, q_ice): BMT analytical derivatives precomputed per
            # updraft.  Same pattern as grid-mean (dominated by −1/τ_relax).
            (; ᶜmp_derivativeʲs) = p.precomputed
            ᶜ∂Sq_liq = ᶜmp_derivativeʲs.:(1).∂tendency_∂q_lcl
            ᶜ∂Sq_ice = ᶜmp_derivativeʲs.:(1).∂tendency_∂q_icl
            sgs_cloud_deriv_tracers = (
                (@name(c.sgsʲs.:(1).q_liq), ᶜ∂Sq_liq),
                (@name(c.sgsʲs.:(1).q_ice), ᶜ∂Sq_ice),
            )
            MatrixFields.unrolled_foreach(
                sgs_cloud_deriv_tracers,
            ) do (q_name, ᶜ∂S∂q)
                MatrixFields.has_field(Y, q_name) || return
                ∂ᶜq_err_∂ᶜq = matrix[q_name, q_name]
                if !use_derivative(sgs_advection_flag)
                    @. ∂ᶜq_err_∂ᶜq =
                        zero(typeof(∂ᶜq_err_∂ᶜq)) - (I,)
                end
                @. ∂ᶜq_err_∂ᶜq += dtγ * DiagonalMatrixRow(ᶜ∂S∂q)
            end

            # Precipitation (q_rai, q_sno): S/q computed inline using frozen
            # tendencies and the current iterate.  Matches grid-mean treatment.
            (; ᶜmp_tendencyʲs) = p.precomputed

            sgs_precip_sq_tracers = (
                (
                    @name(c.sgsʲs.:(1).q_rai),
                    ᶜmp_tendencyʲs.:(1).dq_rai_dt,
                    Y.c.sgsʲs.:(1).q_rai,
                ),
                (
                    @name(c.sgsʲs.:(1).q_sno),
                    ᶜmp_tendencyʲs.:(1).dq_sno_dt,
                    Y.c.sgsʲs.:(1).q_sno,
                ),
            )
            MatrixFields.unrolled_foreach(
                sgs_precip_sq_tracers,
            ) do (q_name, ᶜS, ᶜq)
                MatrixFields.has_field(Y, q_name) || return
                ∂ᶜq_err_∂ᶜq = matrix[q_name, q_name]
                if !use_derivative(sgs_advection_flag)
                    @. ∂ᶜq_err_∂ᶜq =
                        zero(typeof(∂ᶜq_err_∂ᶜq)) - (I,)
                end
                @. ∂ᶜq_err_∂ᶜq += dtγ * DiagonalMatrixRow(_jac_coeff(ᶜS, ᶜq))
            end
        end

        # TODO: 2M EDMF updraft Jacobian entries remain to be implemented.
        # This requires extending the Jacobian sparsity pattern to include
        # diagonal blocks for updraft n_liq and n_rai species.
        # Without these entries, 2M microphysics should use explicit
        # timestepping for stability.

    end
    return nothing
end

function invert_jacobian!(alg::ManualSparseJacobian, cache, ΔY, R)
    # Step 1: Column-local solve
    LinearAlgebra.ldiv!(ΔY, cache.matrix, R)

    # Step 2: Horizontal Helmholtz correction (variable preconditioner)
    # Applied every n_helmholtz_iters-th GMRES call to amortize cost.
    # Requires FGMRES (flexible GMRES) which handles variable preconditioning.
    # n_helmholtz_iters controls the application frequency:
    #   0: disabled (diagonal only)
    #   1: every GMRES iteration (expensive but most accurate)
    #   N: every N-th iteration (amortized cost)
    if use_derivative(alg.acoustic_diagonal_flag) &&
       !isnothing(cache.helmholtz_state) &&
       cache.helmholtz_state.n_helmholtz_iters > 0
        hs = cache.helmholtz_state
        hs.call_counter += 1
        if hs.call_counter >= hs.n_helmholtz_iters
            helmholtz_correction!(cache, ΔY)
            hs.call_counter = 0
        end
    end

    # Step 3: Sparse direct Helmholtz correction (independent of Chebyshev path)
    if use_derivative(alg.sparse_helmholtz_flag) && !isnothing(cache.sparse_helmholtz)
        sparse_helmholtz_correction!(cache.sparse_helmholtz, ΔY)
    end

    # Step 4: 2D Helmholtz correction for sphere grids (direct or matrix-free)
    if use_derivative(alg.sparse_helmholtz_2d_flag) && !isnothing(cache.sparse_helmholtz_2d)
        if cache.sparse_helmholtz_2d isa MatrixFreeHelmholtz2DCache
            matfree_helmholtz_2d_correction!(cache.sparse_helmholtz_2d, ΔY)
        elseif cache.sparse_helmholtz_2d isa GPUSparseHelmholtz2DCache
            gpu_sparse_helmholtz_2d_correction!(cache.sparse_helmholtz_2d, ΔY)
        else
            sparse_helmholtz_2d_correction!(cache.sparse_helmholtz_2d, ΔY)
        end
    end
end

"""
    helmholtz_correction!(cache, ΔY)

Apply block Gauss-Seidel horizontal Helmholtz correction after the column-local
solve. Uses Chebyshev semi-iterative acceleration (Saad, Algorithm 12.1) with
Jacobi preconditioning. Eigenvalue bounds of M⁻¹A ∈ [1/(1+α_max), 1] where
α_max = max(dtγ²·cs²·2π²/Δx²). DSS is applied only to the final iterate.

Sequentially updates (ρ, uₕ, ρe_tot, tracers):

1. ρ-block:     Solve (I - dtγ²·cs²·∇²h)·Δρ = z.ρ - dtγ·wdivₕ(ρ·z.uₕ)
2. uₕ-block:    Δuₕ = z.uₕ - dtγ·(cs²/ρ)·gradₕ(Δρ)  [uses updated Δρ]
3. ρe_tot-block: Solve (I - dtγ²·cs²·∇²h)·Δ(ρe_tot) = z.ρe_tot - dtγ·wdivₕ(h_tot·ρ·z.uₕ)
                 where h_tot = (ρe_tot + p)/ρ  [uses updated Δuₕ]
4. tracer-block: Δ(ρq) += q·(Δρ_new - z.ρ_old)  [advective, no Helmholtz]
"""
function helmholtz_correction!(cache, ΔY)
    (; helmholtz_state, helmholtz_scratch) = cache
    (; dtγ, α_acoustic_max, ᶜα_acoustic, ᶜcs², ᶜρ, ᶜe_tot, ᶜh_tot,
        n_helmholtz_iters) = helmholtz_state
    (; ᶜhelmholtz_ρ, ᶜhelmholtz_rhs, ᶜhelmholtz_laplacian,
        ᶜhelmholtz_ρe, ᶜhelmholtz_dir, ᶜhelmholtz_dss_buffer) =
        helmholtz_scratch

    FT = eltype(dtγ)
    α = FT(dtγ)^2

    # Chebyshev parameters: eigenvalues of M⁻¹A ∈ [a, b]
    # where A = I - α·cs²·∇²h, M = diag(1 + α_acoustic)
    a = FT(1) / (FT(1) + α_acoustic_max)
    b = FT(1)
    θ = (a + b) / 2   # center
    δ = (b - a) / 2   # half-width
    σ₁ = θ / δ

    # ── Block 1: ρ-Helmholtz (Chebyshev semi-iterative) ──
    # RHS = z.ρ - dtγ·wdivₕ(ρ·z.uₕ)
    @. ᶜhelmholtz_rhs = wdivₕ(ᶜρ * ΔY.c.uₕ)
    Spaces.weighted_dss!(ᶜhelmholtz_rhs => ᶜhelmholtz_dss_buffer)
    @. ᶜhelmholtz_rhs = ΔY.c.ρ - FT(dtγ) * ᶜhelmholtz_rhs

    @. ᶜhelmholtz_ρ = ᶜhelmholtz_rhs
    if n_helmholtz_iters >= 1
        # Step 1: d₀ = (1/θ) · M⁻¹·r₀
        @. ᶜhelmholtz_laplacian = wdivₕ(gradₕ(ᶜhelmholtz_ρ))
        @. ᶜhelmholtz_dir =
            (ᶜhelmholtz_rhs - ᶜhelmholtz_ρ +
             α * ᶜcs² * ᶜhelmholtz_laplacian) /
            ((FT(1) + ᶜα_acoustic) * θ)
        @. ᶜhelmholtz_ρ += ᶜhelmholtz_dir

        # Steps 2..N: Chebyshev three-term recurrence on direction
        ρ_prev = FT(1) / σ₁
        for _ in 2:n_helmholtz_iters
            ρ_new = FT(1) / (2 * σ₁ - FT(1) / ρ_prev)
            @. ᶜhelmholtz_laplacian = wdivₕ(gradₕ(ᶜhelmholtz_ρ))
            @. ᶜhelmholtz_dir =
                2 * ρ_new * σ₁ / θ *
                (ᶜhelmholtz_rhs - ᶜhelmholtz_ρ +
                 α * ᶜcs² * ᶜhelmholtz_laplacian) /
                (FT(1) + ᶜα_acoustic) +
                ρ_new * ρ_prev * ᶜhelmholtz_dir
            @. ᶜhelmholtz_ρ += ᶜhelmholtz_dir
            ρ_prev = ρ_new
        end
    end
    # DSS only the final ρ iterate
    Spaces.weighted_dss!(ᶜhelmholtz_ρ => ᶜhelmholtz_dss_buffer)

    # Save old z.ρ before overwriting (reuse ᶜhelmholtz_laplacian as scratch)
    @. ᶜhelmholtz_laplacian = ΔY.c.ρ
    ΔY.c.ρ .= ᶜhelmholtz_ρ

    # ── Block 2: uₕ back-substitution (uses updated Δρ) ──
    @. ΔY.c.uₕ -= C12(
        FT(dtγ) * (ᶜcs² / max(ᶜρ, FT(1e-6))) * gradₕ(ᶜhelmholtz_ρ),
    )

    # ── Block 3: ρe_tot-Helmholtz (Chebyshev semi-iterative) ──
    # Energy Schur complement: (I - dtγ²·cs²·∇²h)·Δ(ρe_tot) = z.ρe_tot - dtγ·wdivₕ(h_tot·ρ·z.uₕ)
    @. ᶜhelmholtz_rhs = wdivₕ(ᶜh_tot * ᶜρ * ΔY.c.uₕ)
    Spaces.weighted_dss!(ᶜhelmholtz_rhs => ᶜhelmholtz_dss_buffer)
    @. ᶜhelmholtz_rhs = ΔY.c.ρe_tot - FT(dtγ) * ᶜhelmholtz_rhs

    @. ᶜhelmholtz_ρe = ᶜhelmholtz_rhs
    if n_helmholtz_iters >= 1
        # Step 1: d₀ = (1/θ) · M⁻¹·r₀
        @. ᶜhelmholtz_laplacian = wdivₕ(gradₕ(ᶜhelmholtz_ρe))
        @. ᶜhelmholtz_dir =
            (ᶜhelmholtz_rhs - ᶜhelmholtz_ρe +
             α * ᶜcs² * ᶜhelmholtz_laplacian) /
            ((FT(1) + ᶜα_acoustic) * θ)
        @. ᶜhelmholtz_ρe += ᶜhelmholtz_dir

        # Steps 2..N: Chebyshev three-term recurrence on direction
        ρ_prev = FT(1) / σ₁
        for _ in 2:n_helmholtz_iters
            ρ_new = FT(1) / (2 * σ₁ - FT(1) / ρ_prev)
            @. ᶜhelmholtz_laplacian = wdivₕ(gradₕ(ᶜhelmholtz_ρe))
            @. ᶜhelmholtz_dir =
                2 * ρ_new * σ₁ / θ *
                (ᶜhelmholtz_rhs - ᶜhelmholtz_ρe +
                 α * ᶜcs² * ᶜhelmholtz_laplacian) /
                (FT(1) + ᶜα_acoustic) +
                ρ_new * ρ_prev * ᶜhelmholtz_dir
            @. ᶜhelmholtz_ρe += ᶜhelmholtz_dir
            ρ_prev = ρ_new
        end
    end
    # DSS only the final ρe_tot iterate
    Spaces.weighted_dss!(ᶜhelmholtz_ρe => ᶜhelmholtz_dss_buffer)
    ΔY.c.ρe_tot .= ᶜhelmholtz_ρe

    # ── Block 4: Tracer correction (advective, no Helmholtz) ──
    # Δ(ρq) += q · (Δρ_new - z.ρ_old)
    # ᶜhelmholtz_laplacian holds old z.ρ from Block 1
    # Currently no-op for DryModel; extend for moist tracers when needed.
end
