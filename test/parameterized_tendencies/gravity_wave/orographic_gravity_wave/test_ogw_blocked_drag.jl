#=
Unit tests for the Garner/GFDL blocked (non-propagating) orographic drag.

Each test runs `blocked_layer_start_height!` and `calc_nonpropagating_forcing!` on one
column, with pressure p = p0 exp(-z/H) and a given z_pbl (the center height of the
PBL-top cell). The drag in every cell is compared with an expected value worked out
here from cell indices, face heights, and pressures only, following GFDL `topo_drag`:

  1. Where the blocked layer starts. Its lowest cell is the cell just above the
     PBL-top cell (GFDL kpbl). The phase is counted from that cell's top face (GFDL
     zhalf(kpbl)). If there are no cells above the PBL-top cell, there is no blocked
     drag.
  2. Where it ends. Going up from the start face, each layer between two faces adds
     Delta z * N / V_tau to the phase, using N and V_tau at the layer's upper face.
     N is clamped to [0.7e-2, 1.7e-2] 1/s and V_tau is at least 1 m/s. z_ref is the
     first face where the phase exceeds pi. If the phase never reaches pi, z_ref is
     the model top face.
  3. How the drag is shared. The blocked layer is every cell from its lowest cell up
     to the cell whose top face is z_ref. Each cell k in it gets
       weight:              w_k = (p_bot + p_top)/2 - p(z_ref)
       pressure thickness:  Delta p_k = p_bot - p_top
       tendency:            du/dt = g * tau_x * tau_np / tau_l * w_k / Sigma (Delta p * w)
     where Sigma sums over the blocked layer, and the same with tau_y for dv/dt. So
     the column total Sigma (du/dt) * Delta p / g equals tau_x * tau_np / tau_l
     exactly.

It also tests the two helpers that prepare the inputs, on a stretched grid (cells of
different thickness):
  - `ogw_face_pressure!`: the pressure at the bottom and top faces, from hydrostatic
    balance.
  - `blocked_layer_start_height!`: the top face of the cell above the PBL-top cell.
=#

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: CommonSpaces, Fields, Grids, Meshes, Spaces, Utilities

const half = Utilities.half

const p0 = 1e5
const H = 8_000.0

# Run blocked_layer_start_height! and calc_nonpropagating_forcing! on one column with
# p = p0 exp(-z/H). `z_pbl` is the center height of the PBL-top cell, as from
# get_pbl_z!. `N` and `Vtau` are numbers or functions of the face height.
function blocked_drag_column(
    FT;
    z_min,
    z_max,
    z_elem,
    z_pbl,
    N,
    Vtau,
    tau_np,
    tau_x = 1,
    tau_y = 0,
    tau_l = 1,
    stretch = Meshes.Uniform(),
)
    grav = FT(9.81)

    center_space = CommonSpaces.ColumnSpace(
        FT;
        z_min,
        z_max,
        z_elem,
        stretch,
        staggering = Grids.CellCenter(),
    )
    face_space = Spaces.face_space(center_space)

    f_z = Fields.coordinate_field(face_space).z
    f_dz = Fields.Δz_field(face_space)
    uforcing = Fields.zeros(center_space)
    vforcing = Fields.zeros(center_space)
    mask = Fields.Field(Bool, center_space)
    weights = Fields.zeros(center_space)
    diffs = Fields.zeros(center_space)

    f_N = N isa Number ? Fields.ones(face_space) .* FT(N) : N.(f_z)
    f_Vtau = Vtau isa Number ? Fields.ones(face_space) .* FT(Vtau) : Vtau.(f_z)
    f_p = @. FT(p0) * exp(-f_z / FT(H))

    surf_c = Spaces.level(center_space, 1)
    surf_f = Spaces.level(face_space, half)
    f_z_ref = Fields.zeros(surf_f)
    f_p_ref = Fields.zeros(surf_f)
    wtsum = Fields.zeros(surf_c)
    f_z_start = Fields.zeros(surf_f)
    CA.blocked_layer_start_height!(f_z_start, Fields.ones(surf_c) .* FT(z_pbl), f_z)

    CA.calc_nonpropagating_forcing!(
        uforcing,
        vforcing,
        Fields.ones(surf_c) .* FT(tau_x),
        Fields.ones(surf_c) .* FT(tau_y),
        Fields.ones(surf_c) .* FT(tau_l),
        Fields.ones(surf_c) .* FT(tau_np),
        f_Vtau,
        f_z_start,
        f_z_ref,
        f_p_ref,
        mask,
        weights,
        diffs,
        wtsum,
        f_p,
        f_N,
        f_z,
        f_dz,
        grav,
    )

    p_faces = vec(parent(f_p))
    return (;
        z_ref = Fields.maximum(f_z_ref),
        u = vec(parent(uforcing)),
        v = vec(parent(vforcing)),
        diffs = vec(parent(diffs)),
        Δp = p_faces[1:(end - 1)] .- p_faces[2:end],
        p_faces,
        zf = vec(parent(f_z)),
        zc = vec(parent(Fields.coordinate_field(center_space).z)),
        wtsum = maximum(wtsum),
        grav,
    )
end

# Independent target: the start face, z_ref, the blocked-layer mask, and the per-cell
# tendencies, from the cell indices, face heights, and pressures only. `N` and `Vtau`
# are numbers or functions of z.
function expected_blocked_drag(col; z_pbl, N, Vtau, tau_np, tau_x = 1, tau_y = 0, tau_l = 1)
    Nf = N isa Number ? (_ -> N) : N
    Vf = Vtau isa Number ? (_ -> Vtau) : Vtau
    zf, zc, pf = col.zf, col.zc, col.p_faces
    n = length(zc)
    k_low = findlast(<=(z_pbl), zc) + 1   # lowest blocked cell, GFDL kpbl
    k_low > n && return (; z_start = Inf, z_ref = Inf, in_layer = falses(n),
        u = zeros(n), v = zeros(n))
    i_start = k_low + 1                   # its top face, GFDL zhalf(kpbl)
    phase = 0.0
    i_ref = lastindex(zf)                 # top face if the phase never reaches pi
    for i in (i_start + 1):lastindex(zf)
        phase +=
            (zf[i] - zf[i - 1]) * clamp(Nf(zf[i]), 0.7e-2, 1.7e-2) / max(1.0, Vf(zf[i]))
        if phase > π
            i_ref = i
            break
        end
    end
    z_ref = zf[i_ref]
    in_layer = (1:n .>= k_low) .& (1:n .< i_ref)
    p_bot, p_top = pf[1:(end - 1)], pf[2:end]
    w = in_layer .* ((p_bot .+ p_top) ./ 2 .- pf[i_ref])
    Δp = p_bot .- p_top
    wtsum = sum(Δp .* w)
    scale = col.grav * tau_np / tau_l / wtsum
    return (; z_start = zf[i_start], z_ref, in_layer, u = scale * tau_x .* w,
        v = scale * tau_y .* w)
end

# Every cell matches the target to rtol, relative to the largest tendency.
matches_target(x, x_target; rtol = 1e-10) =
    all(abs.(x .- x_target) .<= rtol * maximum(abs, x_target))

@testset "OGW blocked drag: phase depth, per-cell tendency, column stress" begin
    FT = Float64
    z_max = FT(20_000)
    z_elem = 80
    dz = z_max / z_elem
    z_pbl = FT(875)       # center of the PBL-top cell [750, 1000] m
    z_start = FT(1_250)   # top face of the lowest blocked cell [1000, 1250] m
    N = FT(0.01)          # inside the [0.7e-2, 1.7e-2] clamp
    Vtau = FT(10)
    tau_x, tau_y, tau_l, tau_np = FT(0.6), FT(-0.8), FT(2), FT(0.5)
    args = (; z_pbl, N, Vtau, tau_np, tau_x, tau_y, tau_l)
    col = blocked_drag_column(FT; z_min = FT(0), z_max, z_elem, args...)
    target = expected_blocked_drag(col; args...)

    # z_ref is the first face above z_start + pi * V / N.
    expected_depth = FT(pi) * Vtau / N
    @test target.z_start == z_start
    @test col.z_ref ≈ target.z_ref ≈ z_start + ceil(expected_depth / dz) * dz
    @test col.z_ref > z_start + expected_depth

    # Each cell's own pressure thickness, the weights against p(z_ref), and the
    # GFDL normalization, cell by cell.
    @test all(isapprox.(col.diffs, col.Δp; rtol = 1e-12))
    @test matches_target(col.u, target.u)
    @test matches_target(col.v, target.v)
    @test all(col.u .>= 0) && all(col.v .<= 0)  # signs of tau_x > 0, tau_y < 0
    @test count(target.in_layer) == 14          # 1000 m to 4500 m in 250 m cells

    # Column-integrated blocked stress in each direction.
    @test sum(col.u .* col.Δp) / col.grav ≈ tau_x * tau_np / tau_l rtol = 1e-10
    @test sum(col.v .* col.Δp) / col.grav ≈ tau_y * tau_np / tau_l rtol = 1e-10

    # Where the drag goes: there is no drag in any PBL cell (center at or below
    # z_pbl). The drag is spread over the cells between z_pbl and z_ref, starting
    # with the first cell above the PBL.
    in_pbl = col.zc .<= z_pbl
    @test all(iszero, col.u[in_pbl]) && all(iszero, col.v[in_pbl]) &&
          col.u[findfirst(!, in_pbl)] > 0
    above = col.zc .> col.z_ref
    @test any(above) && all(iszero, col.u[above]) && all(iszero, col.v[above])
end

@testset "OGW blocked drag: PBL top below sea level" begin
    # A surface below sea level (as at the Dead Sea): the phase still counts from the
    # start face. Counting from 0 m instead would put z_ref one face higher here.
    FT = Float64
    z_min = FT(-1_000)
    z_max = FT(19_000)
    z_elem = 80
    dz = (z_max - z_min) / z_elem
    z_pbl = FT(-625)      # center of the PBL-top cell [-750, -500] m
    z_start = FT(-250)    # top face of the lowest blocked cell [-500, -250] m
    args = (; z_pbl, N = FT(0.01), Vtau = FT(10), tau_np = FT(0.5))
    col = blocked_drag_column(FT; z_min, z_max, z_elem, args...)
    target = expected_blocked_drag(col; args...)

    faces = z_min .+ dz .* (0:z_elem)
    expected_z_ref = first(filter(>(z_start + FT(pi) * args.Vtau / args.N), faces))
    @test target.z_start == z_start
    @test col.z_ref ≈ expected_z_ref ≈ target.z_ref
    @test !(col.z_ref ≈ first(filter(>(FT(pi) * args.Vtau / args.N), faces)))
    @test matches_target(col.u, target.u)
    @test sum(col.u .* col.Δp) / col.grav ≈ args.tau_np rtol = 1e-10
end

@testset "OGW blocked drag: clamps and fallbacks" begin
    # Each case: (column, expected z_ref). Every case also matches the per-cell target.
    # With z_pbl = 875 m the blocked layer starts at 1000 m and the phase at 1250 m.
    FT = Float64
    kw = (; z_min = FT(0), z_max = FT(20_000), z_elem = 80)
    base = (; z_pbl = FT(875), tau_np = FT(0.5))
    top_two = (; kw..., z_max = FT(3_000), z_elem = 12)
    cases = [
        # Phase never reaches pi: the 3.1 km half wavelength does not fit in a 3 km
        # column, so z_ref is the top face (and the top cell is in the layer).
        ((; base..., N = FT(0.01), Vtau = FT(10)), top_two, 3_000),
        # N above the clamp, V_tau below its floor: depth pi * 1 / 0.017 = 185 m.
        ((; base..., N = FT(0.05), Vtau = FT(0.5)), kw, 1_500),
        # N below the clamp: depth pi * 10 / 0.007 = 4488 m.
        ((; base..., N = FT(0.001), Vtau = FT(10)), kw, 5_750),
        # N or V_tau varying with height: each layer uses the values at its upper face;
        # the lower face would give 4500 m and 6500 m.
        ((; base..., N = z -> 0.0075 + 2e-6 * (z - 1_250), Vtau = FT(10)), kw, 4_250),
        ((; base..., N = FT(0.01), Vtau = z -> 10 + 0.003 * (z - 1_250)), kw, 6_750),
        # Lowest possible PBL top. If no level above the lowest passes its test,
        # get_pbl_z! returns the lowest center, 125 m (cell 1, 0-250 m). The blocked
        # layer then starts in cell 2 (250-500 m), as GFDL does with
        # kpbl = kdim - 1, and the phase counts from 500 m. The depth is 3142 m, so
        # z_ref is the first face above 3642 m, 3750 m.
        ((; base..., z_pbl = FT(125), N = FT(0.01), Vtau = FT(10)), kw, 3_750),
        # Highest PBL top that leaves a cell above it. The PBL-top cell is cell 11 of
        # 12 (2500-2750 m), so the only cell above it is the top cell (2750-3000 m).
        # The phase would count from that cell's top face, which is the model top, so
        # there is nothing to sum: z_ref is the model top and the blocked layer is the
        # top cell alone.
        ((; base..., z_pbl = FT(2_625), N = FT(0.01), Vtau = FT(10)), top_two, 3_000),
    ]
    results = map(cases) do (args, grid, z_ref_expected)
        col = blocked_drag_column(FT; grid..., args...)
        target = expected_blocked_drag(col; args...)
        (; col, target,
            ok_z_ref = col.z_ref ≈ z_ref_expected && target.z_ref ≈ z_ref_expected,
            ok_u = matches_target(col.u, target.u) &&
                   isapprox(sum(col.u .* col.Δp) / col.grav, args.tau_np; rtol = 1e-10))
    end
    @test all(r.ok_z_ref for r in results)
    @test all(r.ok_u for r in results)
    @test results[1].col.u[end] > 0
    @test findall(!iszero, results[6].col.u) == findall(results[6].target.in_layer) ==
          2:15
    @test findall(!iszero, results[7].col.u) == [12]

    # PBL-top cell at the top: no cell above it, so no blocked drag.
    col = blocked_drag_column(FT; top_two..., z_pbl = FT(2_875), N = FT(0.01),
        Vtau = FT(10), tau_np = FT(0.5))
    @test col.z_ref == Inf && all(iszero, col.u) && all(iszero, col.v) && col.wtsum == 0
end

@testset "OGW blocked drag: stretched grid" begin
    # Cell thicknesses grow from 30 m near the surface, so the layer depths, the
    # pressure thicknesses, and the weights all vary from cell to cell.
    FT = Float64
    stretch = Meshes.GeneralizedExponentialStretching(FT(30), FT(2_000))
    kw = (; z_min = FT(0), z_max = FT(30_000), z_elem = 40, stretch)
    probe = blocked_drag_column(
        FT;
        kw...,
        z_pbl = FT(0),
        N = FT(0.01),
        Vtau = FT(10),
        tau_np = FT(0.5),
    )
    dz = diff(probe.zf)
    @test dz[end] > 10 * dz[1]
    z_pbl = probe.zc[findfirst(>(FT(800)), probe.zc)]   # a center, as in the model
    args = (; z_pbl, N = FT(0.01), Vtau = FT(10), tau_np = FT(0.5), tau_x = FT(0.6),
        tau_y = FT(-0.8), tau_l = FT(2))
    col = blocked_drag_column(FT; kw..., args...)
    target = expected_blocked_drag(col; args...)
    @test col.z_ref ≈ target.z_ref
    @test col.z_ref > target.z_start + FT(pi) * args.Vtau / args.N
    @test all(isapprox.(col.diffs, col.Δp; rtol = 1e-12))
    @test matches_target(col.u, target.u) && matches_target(col.v, target.v)
    stress_x = sum(col.u .* col.Δp) / col.grav
    stress_y = sum(col.v .* col.Δp) / col.grav
    @test isapprox(stress_x, args.tau_x * args.tau_np / args.tau_l; rtol = 1e-10) &&
          isapprox(stress_y, args.tau_y * args.tau_np / args.tau_l; rtol = 1e-10)
end

# Face pressures from ogw_face_pressure! on an isothermal hydrostatic column,
# p = p0 exp(-z/H) and rho = p / (g H), with the face heights and pressures.
function face_pressure_column(FT; stretch, z_max, z_elem, grav)
    center_space = CommonSpaces.ColumnSpace(
        FT;
        z_min = FT(0),
        z_max,
        z_elem,
        stretch,
        staggering = Grids.CellCenter(),
    )
    face_space = Spaces.face_space(center_space)
    c_z = Fields.coordinate_field(center_space).z
    f_z = Fields.coordinate_field(face_space).z
    c_p = @. FT(p0) * exp(-c_z / FT(H))
    c_ρ = @. c_p / (grav * FT(H))
    f_p = Fields.zeros(face_space)
    CA.ogw_face_pressure!(f_p, c_p, c_ρ, c_z, f_z, grav)
    zf = vec(parent(f_z))
    return (;
        center_space,
        face_space,
        f_z,
        zc = vec(parent(c_z)),
        zf,
        pc = vec(parent(c_p)),
        pf = vec(parent(f_p)),
        p_exact = @. p0 * exp(-zf / H)
    )
end

@testset "OGW blocked drag: boundary face pressure and blocked-layer start" begin
    FT = Float64
    grav = FT(9.81)
    stretched = face_pressure_column(
        FT;
        stretch = Meshes.GeneralizedExponentialStretching(FT(30), FT(2_000)),
        z_max = FT(30_000),
        z_elem = 40,
        grav,
    )
    uniform = face_pressure_column(
        FT;
        stretch = Meshes.Uniform(),
        z_max = FT(20_000),
        z_elem = 80,
        grav,
    )

    for col in (stretched, uniform)
        (; zc, zf, pc, pf, p_exact) = col
        # Interior faces are the mean of the adjacent centers.
        @test pf[2:(end - 1)] ≈ (pc[1:(end - 1)] .+ pc[2:end]) ./ 2 rtol = 1e-14
        # Boundary faces follow the half-cell hydrostatic integral, to second order
        # in the half-cell depth over H.
        x = abs.([zf[1] - zc[1], zf[end] - zc[end]]) ./ H
        @test all(abs.(pf[[1, end]] ./ p_exact[[1, end]] .- 1) .< x .^ 2)
    end

    # So the bottom and top cells get their full pressure thickness, which copying the
    # nearest center onto the boundary face would roughly halve. (On a uniform grid,
    # where every face is midway between its centers.)
    (; pc, pf, p_exact) = uniform
    Δp_exact = p_exact[1:(end - 1)] .- p_exact[2:end]
    Δp = pf[1:(end - 1)] .- pf[2:end]
    @test Δp[1] ≈ Δp_exact[1] rtol = 2e-2
    @test Δp[end] ≈ Δp_exact[end] rtol = 2e-2
    @test (pc[1] - pf[2]) < 0.6 * Δp_exact[1]
    @test (pf[end - 1] - pc[end]) < 0.6 * Δp_exact[end]

    # blocked_layer_start_height! on the stretched grid. Cell k lies between faces k
    # and k + 1. If the PBL-top cell is k, the lowest blocked cell is k + 1, and the
    # start face is that cell's top face, face k + 2. Checked with the PBL-top cell
    # low, in the middle, and high in the column, plus two edge cases:
    #   - k = n - 1 (second cell from the top): face n + 1, the model top.
    #   - k = n (top cell): no cell above it, so Inf.
    # The last check makes sure the cells here differ enough in thickness that
    # guessing the face from the PBL-top cell's own thickness would be wrong.
    (; center_space, face_space, f_z, zc, zf) = stretched
    surf_c = Spaces.level(center_space, 1)
    surf_f = Spaces.level(face_space, half)
    n = length(zc)
    start_faces = map([1, 5, 17, 30, n - 1, n]) do k
        f_z_start = Fields.zeros(surf_f)
        CA.blocked_layer_start_height!(f_z_start, Fields.ones(surf_c) .* zc[k], f_z)
        Fields.maximum(f_z_start)
    end
    @test start_faces == [zf[[3, 7, 19, 32, n + 1]]; Inf]
    @test !(zf[19] ≈ zc[17] + 3 * (zf[18] - zf[17]) / 2)
end
