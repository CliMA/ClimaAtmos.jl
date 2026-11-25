#=
Tests of the orographic gravity-wave (OGW) drag and of the turbulent orographic form drag
(TOFD), in one file. Every check compares the code with a value worked out independently
(closed forms, GFDL `topo_drag` numbers, or exact properties such as symmetry), not with
the code's own earlier output, except where a section says it pins a regression.

Sections, in order (the slowest, the Earth-topography sphere, is last):

  1. The damping that keeps the cached gravity-wave drag from reversing the wind
     (`_semi_implicit_gw_drag`; TOFD's own implicit rate is in section 4), and the NaN
     guards in the blocked-drag weights.
  2. Blocked (non-propagating) drag on single columns: where the blocked layer starts and
     ends and how the drag is shared, against values worked out by hand from GFDL
     `topo_drag`'s rules; also the end-cell dT/dz and the cell the base flux is read in.
  3. Single columns against stored GFDL `topo_drag` numbers (data in
     test_ogw_gfdl_reference_data.jl).
  4. TOFD drag coefficient, smoothing, filtered-orography variance, implicit rate.
  5. Preprocessing of the elevation into hmax and the drag tensor, with analytic mountains,
     and the build record of the stored drag files.
  6. One OGW step on a coarse Earth-topography sphere (about 3 minutes, mostly the build).

Each section starts with a comment describing what it checks. Plot-only scripts with no
assertions (Garner 2005 figures, 3D and base-flux plots, the artifact plots) and the
script that builds the `raw_topo` artifacts live in scripts/. The two artifact scripts
run in the Buildkite smoke step; the plot scripts are run by hand.
=#

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore:
    CommonSpaces, Domains, Fields, Geometry, Grids, InputOutput, Meshes, Spaces, Utilities
import Logging

const half = Utilities.half

# ======================================================================================
# 1. Semi-implicit damping and NaN guards
# ======================================================================================

#=
Pure-function unit tests for the orographic gravity wave drag helpers in
`src/parameterized_tendencies/gravity_wave_drag/orographic_gravity_wave.jl`.

Fast (~1 s, no simulation build) so they run in `test/runtests.jl` under the
`parameterizations` group.
=#

@testset "Semi-implicit orographic gravity-wave drag damping" begin
    for FT in (Float32, Float64)
        u_vec = CA.Geometry.UVWVector(FT(10.0), FT(-5.0), FT(0.0))

        # Small dt d / s limit reproduces the explicit drag.
        du_small = FT(-1e-4)
        dv_small = FT(5e-5)
        d_si_small =
            CA._semi_implicit_gw_drag(du_small, dv_small, u_vec, FT(1.0))
        @test d_si_small.u ≈ du_small rtol = FT(1e-4)
        @test d_si_small.v ≈ dv_small rtol = FT(1e-4)

        # Stiff opposing drag (|D_∥| dt ≫ |u|) cannot reverse the parallel
        # wind in one step, while the perpendicular component D_⊥ is unchanged.
        dt_large = FT(360.0)
        # Purely parallel opposing drag (-2, 1) + perpendicular drag (1, 2).
        du_stiff = FT(-1.0)
        dv_stiff = FT(3.0)
        d_si_stiff =
            CA._semi_implicit_gw_drag(du_stiff, dv_stiff, u_vec, dt_large)
        u_next = FT(10.0) + dt_large * d_si_stiff.u
        v_next = FT(-5.0) + dt_large * d_si_stiff.v
        @test u_next * FT(10.0) + v_next * FT(-5.0) > 0
        @test d_si_stiff.u * FT(-5.0) - d_si_stiff.v * FT(10.0) ≈
              du_stiff * FT(-5.0) - dv_stiff * FT(10.0) rtol = FT(1e-5)

        # Weak oblique winds: purely opposing stiff drag (D_⊥ = 0) preserves
        # forward-Euler wind sign down to s = O(1e-3); with stiff D_⊥ ≠ 0,
        # forward-Euler sign holds above the O(eps(FT) * dt * |D_⊥|) rounding
        # of uₕ + dt * D_⊥, while D_⊥ and D_∥ match at every weak speed.
        u_weak = CA.Geometry.UVWVector(FT(1e-3), FT(-5e-4), FT(0.0))
        d_si_opp = CA._semi_implicit_gw_drag(FT(-2.0), FT(1.0), u_weak, dt_large)
        u_opp_next = FT(1e-3) + dt_large * d_si_opp.u
        v_opp_next = FT(-5e-4) + dt_large * d_si_opp.v
        @test u_opp_next * FT(1e-3) + v_opp_next * FT(-5e-4) > 0

        u_mod = CA.Geometry.UVWVector(FT(0.5), FT(-0.25), FT(0.0))
        d_si_mod =
            CA._semi_implicit_gw_drag(du_stiff, dv_stiff, u_mod, dt_large)
        @test (FT(0.5) + dt_large * d_si_mod.u) * FT(0.5) +
              (FT(-0.25) + dt_large * d_si_mod.v) * FT(-0.25) > 0
        @test d_si_mod.u * FT(-0.25) - d_si_mod.v * FT(0.5) ≈
              du_stiff * FT(-0.25) - dv_stiff * FT(0.5) rtol = FT(1e-4)

        d_si_weak =
            CA._semi_implicit_gw_drag(du_stiff, dv_stiff, u_weak, dt_large)
        s_weak = hypot(FT(1e-3), FT(-5e-4))
        d_weak = -(du_stiff * FT(1e-3) + dv_stiff * FT(-5e-4)) / s_weak
        par_weak = (d_si_weak.u * FT(1e-3) + d_si_weak.v * FT(-5e-4)) / s_weak
        @test isapprox(
            par_weak,
            -(d_weak * s_weak) / (s_weak + dt_large * d_weak);
            atol = 4 * eps(FT) * hypot(du_stiff, dv_stiff),
        )
        @test d_si_weak.u * FT(-5e-4) - d_si_weak.v * FT(1e-3) ≈
              du_stiff * FT(-5e-4) - dv_stiff * FT(1e-3) rtol = FT(1e-4)

        # For s ≪ eps(FT) * dt * d, evaluating D_⊥ = (D ⋅ n̂) n̂ in the
        # orthogonal basis avoids O(eps(FT) |D|) cancellation of D against -D_∥,
        # leaving at most O(eps(FT)^2 |D|) unit-vector rounding when D_⊥ = 0.
        s_tiny = FT(1e-20)
        u_tiny = CA.Geometry.UVWVector(s_tiny, FT(0.0), FT(0.0))
        d_si_tiny =
            CA._semi_implicit_gw_drag(FT(-1.0), FT(0.25), u_tiny, dt_large)
        @test d_si_tiny.u ≈ -s_tiny / dt_large rtol = FT(1e-5)
        @test d_si_tiny.v ≈ FT(0.25) rtol = FT(1e-6)

        u_oblique =
            CA.Geometry.UVWVector(FT(6e-21), FT(8e-21), FT(0.0))
        d_si_oblique =
            CA._semi_implicit_gw_drag(FT(-0.6), FT(-0.8), u_oblique, dt_large)
        @test isapprox(
            d_si_oblique.u * FT(0.6) + d_si_oblique.v * FT(0.8),
            -FT(1e-20) / dt_large;
            rtol = FT(1e-5),
            atol = 10 * eps(FT)^2 * hypot(FT(-0.6), FT(-0.8)),
        )
        @test isapprox(
            -d_si_oblique.u * FT(0.8) + d_si_oblique.v * FT(0.6),
            FT(0);
            atol = 2 * eps(FT) * hypot(FT(-0.6), FT(-0.8)),
        )

        # Accelerating component (D ⋅ uₕ ≥ 0) is left unchanged.
        d_acc = CA._semi_implicit_gw_drag(FT(2.0), FT(-1.0), u_vec, dt_large)
        @test d_acc.u == FT(2.0)
        @test d_acc.v == FT(-1.0)

        # Zero and subnormal winds return the explicit drag, finite.
        for w in (FT(0), floatmin(FT) / 4)
            u_zero = CA.Geometry.UVWVector(w, w, FT(0.0))
            d_si_zero =
                CA._semi_implicit_gw_drag(du_stiff, dv_stiff, u_zero, dt_large)
            @test isfinite(d_si_zero.u) && isfinite(d_si_zero.v)
            @test d_si_zero.u == du_stiff
            @test d_si_zero.v == dv_stiff
        end
    end
end

@testset "Orographic non-propagating forcing guards NaN weights" begin
    for FT in (Float32, Float64)
        cspace = CommonSpaces.ColumnSpace(
            FT;
            z_elem = 8,
            z_min = 500,
            z_max = 8000,
            staggering = CommonSpaces.CellCenter(),
        )
        fspace = CA.Spaces.face_space(cspace)
        lspace = CA.Spaces.level(cspace, 1)
        flspace = CA.Spaces.level(fspace, CA.Fields.half)

        ᶜuforcing = fill(FT(1e-4), cspace)
        ᶜvforcing = fill(FT(-2e-4), cspace)
        τ_x = fill(FT(-0.5), lspace)
        τ_y = fill(FT(0.25), lspace)
        τ_l = fill(FT(1.0), lspace)
        τ_np = fill(FT(0.5), lspace)
        ᶠVτ = fill(FT(10.0), fspace)
        # Put the blocked-layer start above the model top so no z_ref is found and
        # ᶠp_ref stays NaN.
        ᶠz_start = fill(FT(9000.0), flspace)
        ᶠz_ref = zeros(FT, flspace)
        ᶠp_ref = zeros(FT, flspace)
        ᶜmask = fill(false, cspace)
        ᶜweights = zeros(FT, cspace)
        ᶜdiff = zeros(FT, cspace)
        ᶜwtsum = zeros(FT, lspace)
        ᶠz = CA.Fields.coordinate_field(fspace).z
        ᶠdz = CA.Fields.local_geometry_field(fspace).∂x∂ξ.components.data.:1
        ᶠp = @. FT(1e5) * exp(-(ᶠz) / FT(8000))
        ᶜz = CA.Fields.coordinate_field(cspace).z
        ᶜp = @. FT(1e5) * exp(-(ᶜz) / FT(8000))
        ᶠN = fill(FT(0.01), fspace)

        CA.calc_nonpropagating_forcing!(
            ᶜuforcing,
            ᶜvforcing,
            τ_x,
            τ_y,
            τ_l,
            τ_np,
            ᶠVτ,
            ᶠz_start,
            ᶠz_ref,
            ᶠp_ref,
            ᶜmask,
            ᶜweights,
            ᶜdiff,
            ᶜwtsum,
            ᶠp,
            ᶜp,
            ᶠN,
            ᶠz,
            ᶠdz,
            FT(9.81),
        )
        @test all(isnan, parent(ᶠp_ref))
        @test all(iszero, parent(ᶜweights))
        @test all(==(FT(1e-4)), parent(ᶜuforcing))
        @test all(==(FT(-2e-4)), parent(ᶜvforcing))
    end
end

# ======================================================================================
# 2. Blocked (non-propagating) drag
# ======================================================================================

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
       weight:              w_k = p_k - p(z_ref), p_k the cell-center pressure
       pressure thickness:  Delta p_k = p_bot - p_top
       tendency:            du/dt = g * tau_x * tau_np / tau_l * w_k / Sigma (Delta p * w)
     where Sigma sums over the blocked layer, and the same with tau_y for dv/dt. So
     the column total Sigma (du/dt) * Delta p / g equals tau_x * tau_np / tau_l
     exactly.

It also tests the three helpers that prepare the inputs, on a stretched grid (cells of
different thickness):
  - `ogw_face_pressure!`: the pressure at the bottom and top faces, from hydrostatic
    balance.
  - `blocked_layer_start_height!`: the top face of the cell above the PBL-top cell.
  - `ogw_temperature_gradient!`: dT/dz, one-sided in the bottom and top cells.
=#

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
    c_z = Fields.coordinate_field(center_space).z
    c_p = @. FT(p0) * exp(-c_z / FT(H))

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
        c_p,
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
        p_centers = vec(parent(c_p)),
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
    w = in_layer .* (col.p_centers .- pf[i_ref])
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

# ogw_temperature_gradient! on uniform and stretched columns with T = 300 - 0.0065 z +
# 1e-7 z²: interior cells keep the difference across their two faces (face
# temperatures are the mean of the neighbouring centers); the bottom and top cells take
# the one-sided difference to their only neighbour, not half of it.
@testset "OGW buoyancy frequency input: one-sided dT/dz in the end cells" begin
    FT = Float64
    for stretch in
        (Meshes.Uniform(), Meshes.GeneralizedExponentialStretching(FT(30), FT(2_000)))
        center_space = CommonSpaces.ColumnSpace(
            FT;
            z_min = FT(0),
            z_max = FT(30_000),
            z_elem = 40,
            stretch,
            staggering = Grids.CellCenter(),
        )
        face_space = Spaces.face_space(center_space)
        c_z = Fields.coordinate_field(center_space).z
        c_T = @. FT(300) - FT(0.0065) * c_z + FT(1e-7) * c_z^2
        c_dTdz = Fields.zeros(center_space)
        f_dTdz = Fields.zeros(face_space)
        CA.ogw_temperature_gradient!(c_dTdz, f_dTdz, c_T)
        zc = vec(parent(c_z))
        zf = vec(parent(Fields.coordinate_field(face_space).z))
        T = vec(parent(c_T))
        dTdz = vec(parent(c_dTdz))
        n = length(zc)
        T_face = (T[1:(end - 1)] .+ T[2:end]) ./ 2
        centred = (T_face[2:end] .- T_face[1:(end - 1)]) ./ diff(zf)[2:(end - 1)]
        @test dTdz[2:(end - 1)] ≈ centred rtol = 1e-10
        @test dTdz[1] ≈ (T[2] - T[1]) / (zc[2] - zc[1]) rtol = 1e-10
        @test dTdz[n] ≈ (T[n] - T[n - 1]) / (zc[n] - zc[n - 1]) rtol = 1e-10
    end
end

# ogw_source_level_values! on a stretched column with (ρ, u, v, N) that vary with height:
# the base flux reads the first cell above the PBL-top cell (GFDL kpbl), for a PBL-top
# cell low, in the middle and high in the column, and the top cell when it is the
# PBL-top cell.
@testset "OGW base flux reads the first cell above the PBL-top cell" begin
    FT = Float64
    center_space = CommonSpaces.ColumnSpace(
        FT;
        z_min = FT(0),
        z_max = FT(30_000),
        z_elem = 40,
        stretch = Meshes.GeneralizedExponentialStretching(FT(30), FT(2_000)),
        staggering = Grids.CellCenter(),
    )
    c_z = Fields.coordinate_field(center_space).z
    c_ρ = @. exp(-c_z / FT(8_000))
    c_u = @. FT(5) + c_z / FT(1_000)
    c_v = @. -FT(2) + c_z / FT(3_000)
    c_N = @. FT(0.01) + c_z * FT(1e-7)
    zc = vec(parent(c_z))
    n = length(zc)
    surf = Spaces.level(center_space, 1)
    for k in (1, 5, 17, n - 1, n)
        values = Fields.Field(Tuple{FT, FT, FT, FT}, surf)
        CA.ogw_source_level_values!(
            values,
            c_ρ,
            c_u,
            c_v,
            c_N,
            c_z,
            Fields.ones(surf) .* zc[k],
        )
        got = [only(parent(getproperty(values, i))) for i in 1:4]
        k_source = min(k + 1, n)
        @test got ≈ [
            vec(parent(c_ρ))[k_source],
            vec(parent(c_u))[k_source],
            vec(parent(c_v))[k_source],
            vec(parent(c_N))[k_source],
        ]
    end
end

# ======================================================================================
# 3. Single columns against GFDL topo_drag
# ======================================================================================

#=
Single-column comparison of the ClimaAtmos orographic gravity wave (OGW) drag with GFDL
`topo_drag` (atmos_phys, topo_drag.F90), using stored GFDL numbers.

How the reference numbers were made (the Fortran is NOT in this repository, for licence
reasons; only generated numbers are stored, as Julia literals in
test_ogw_gfdl_reference_data.jl):

  1. A Python script built idealized single columns: face heights (uniform 500 m cells or
     cells stretched from 50 m to 1500 m, top near 40 km), a piecewise-linear temperature,
     exact hydrostatic pressure, dry air, analytic winds, a non-symmetric drag tensor, and
     hmax, hmin. Constants are the ClimaAtmos values (g = 9.81, R_d = 287.0,
     cp_d = 1004.5). The face pressure is what `ogw_face_pressure!` makes from the cell
     centers, so both codes take every pressure thickness from the same numbers.
     The data file stores each column as this definition, and the column builder below
     rebuilds it with the same arithmetic: face heights and temperatures come out bitwise
     equal, pressure and winds (exp, log) can differ in the last bit, far below every
     tolerance and every yes/no threshold (the closest phase-sum margin is 1.5e-3).
  2. GFDL topo_drag (lines 1-710 of topo_drag.F90; no physics line changed: the FMS modules
     are replaced by stubs, and diagnostic copies, a setter and an optional `kref` override
     are added) ran each column in double precision
     with bounds checking, with the AM4 production values of `&topo_drag_nml` (run
     "matched"; NOAA-GFDL/AM4 run/input.nml, last changed in commit 21e877102ea2):
     frcrit 0.7, alin 0.9, anonlin 3.0, use_mg_scaling true, beta 0.5, gamma 0.4, epsi 0,
     h_frac 0.1, zref_fac 1, no_drag_frac 0.1, tboost 1.5, pcut 0, samp 2 (h_frac and
     no_drag_frac have no effect here: the columns give hmin directly). AM4 also sets
     use_pbl_from_lock = true, so AM4 takes its PBL level from the Lock boundary-layer
     scheme instead of the temperature test in `get_pbl`, and use_uref_4stable = true.
     ClimaAtmos has neither option (it uses the `get_pbl` temperature test), so both are
     off in these runs.
     Run "matched_kref" forces GFDL's blocked-layer top `kref` to the face the ClimaAtmos
     phase rule picks (replayed exactly on GFDL's own per-face step sizes), so the blocked
     drag can be compared cell by cell.
  3. GFDL's face density, face N and face wind along the drag are not stored: they are
     recomputed below with GFDL's own formulas (topo_drag.F90 satur_flux and
     topo_drag_tend), which only use the column and the stored base-flux values.
     GFDL's v tendencies equal its u tendencies times tauy/taux, so only u is stored.
  4. `ca_dudt` is this code's own `orographic_gravity_wave_forcing!` output at the time the
     reference was made (regression pin, not GFDL).

What is tested, case by case (cells and faces counted from the bottom; "source cell" is the
first cell above the PBL-top cell, GFDL `kpbl`):

  - PBL: the source cell matches GFDL. In `pbl_second_level_only` ClimaAtmos picks the
    cell one higher, on purpose: ClimaAtmos also tests the second-lowest level as a PBL
    top, GFDL does not.
  - Base flux, with GFDL's density and buoyancy frequency N in the source cell (the two
    codes compute those two values differently on purpose): all drag components to 1e-12.
  - Wind curvature (`ᶜd2dz2`) projected on the drag, in the cells where GFDL computes it.
  - Saturation flux profile on faces, fed GFDL's face density and N (through cell-center
    values chosen so that their face averages are GFDL's): the whole profile, including
    the leftover-flux step, matches GFDL. In `curvature_binding_uniform` the running
    minimum of the saturation velocity binds at faces where the curvature term
    1 - 2·V·d²V/N² is between 0.5 and 2, so the curvature factor and its sign change the
    result.
  - Wind along the drag at faces: interpolate, then floor, as GFDL. The recomputed GFDL
    face values are checked against GFDL's own output around the wind reversals.
  - Blocked layer: the top `z_ref` equals the ClimaAtmos face rule replayed on GFDL's step
    sizes (the phase sum steps on faces on purpose, so the top can be one cell higher than
    GFDL's), and with the same top the blocked drag matches GFDL cell by cell.
  - Propagating drag from the same flux profile, with density Δp/(g Δz), matches GFDL.
  - End to end, ClimaAtmos computing everything itself, without the limiter: the column
    stress agrees with GFDL within 1%. Measured differences are at most 0.41%; they come
    from the deliberate differences (density and N in the source cell, N and density at
    faces interpolated from cell centers, the phase sum on faces), which move drag between
    cells but keep the column total. Not checked for `pbl_second_level_only` (the
    different source cell changes it by 21%).
  - Where the wave breaks right at the source (`low_level_jet`, `weak_surface_wind`), less
    than 1% of the propagating stress lands below the source cell (GFDL: 0.02% and 0.17%).
  - Regression pin of ClimaAtmos's own end-to-end output (after the limiter) at 1e-10.

N in ClimaAtmos is built here with the same lines as
`orographic_gravity_wave_compute_tendency!` (dry air); those lines are checked by reading,
not by running that routine.

Regenerating:
  - The regression pin `ca_dudt` needs no Fortran. Run this file with the environment
    variable OGW_GFDL_PRINT_PIN=true; it prints `ca_dudt = [...]` for every column, to
    paste into test_ogw_gfdl_reference_data.jl in place of the old arrays. Only do this
    when a change of the ClimaAtmos output is intended.
  - The GFDL numbers come from a small harness kept by the author (Ray Chew): a column
    generator, a script that extracts the GFDL column routines, a Fortran driver, the
    comparison script, and the writer of the data file. It needs the GFDL source, which
    is not copied into ClimaAtmos: NOAA-GFDL/atmos_phys
    atmos_param/topo_drag/topo_drag.F90 at commit 3dacaaa38d24 (git blob
    66b0a0ef504fe6047a7294a7e55fa752a9a6fa41; sha256
    1b35752c3be235d6827f018094c9cbf79d35fe8df3cce5b87151612f5892786d).

The full comparison (19 columns, every piece, measured sizes of every difference) was done
outside the repository; this file keeps the parts that must not change.
=#

const FT = Float64

# ------------------------------------------------------------------ reference data
include(joinpath(@__DIR__, "test_ogw_gfdl_reference_data.jl"))

# ------------------------------------------------------------------ parameters
const params = CA.ClimaAtmosParameters(FT)
const grav = CAP.grav(params)
const cp_d = CAP.cp_d(params)
const R_d = CAP.R_d(params)
const cp_m_dry = CA.TD.cp_m(CAP.thermodynamics_params(params), FT(0), FT(0), FT(0))
const ogwp = CA.OrographicGravityWaveParameters(FT)
const ogw_params = (;
    Fr_crit = ogwp.Fr_crit,
    topo_ρscale = ogwp.ρscale,
    topo_L0 = ogwp.L0,
    topo_a0 = ogwp.a0,
    topo_a1 = ogwp.a1,
    topo_γ = ogwp.γ,
    topo_β = ogwp.β,
    topo_ϵ = ogwp.ϵ,
    topo_a_tofd = ogwp.a_tofd,
)

# ------------------------------------------------------------------ column builder
# Same arithmetic as the generator that made the GFDL inputs (see the header).
uniform_faces() = [k * 500.0 for k in 0:80]
function stretched_faces()
    faces = [0.0]
    dz = 50.0
    while faces[end] < 40000.0 - 1e-9
        push!(faces, faces[end] + dz)
        dz = min(dz * 1.12, 1500.0)
    end
    return faces
end

# Piecewise-linear temperature and its exact hydrostatic log-pressure, p(0) = 1e5 Pa
function temperature_and_pressure(T0, segments, zs)
    bounds = []
    za, Ta, lnpa = 0.0, T0, log(1e5)
    for (zt, lap) in segments
        push!(bounds, (za, zt, Ta, lap, lnpa))
        Tt = Ta - lap * (zt - za)
        lnpa =
            lap == 0.0 ? lnpa - grav * (zt - za) / (R_d * Ta) :
            lnpa + grav / (R_d * lap) * log(Tt / Ta)
        za, Ta = zt, Tt
    end
    T = similar(zs)
    p = similar(zs)
    for (i, z) in enumerate(zs)
        (za, _, Ta, lap, lnpa) = bounds[findfirst(b -> z <= b[2] + 1e-9, bounds)]
        T[i] = Ta - lap * (z - za)
        p[i] = exp(
            lap == 0.0 ? lnpa - grav * (z - za) / (R_d * Ta) :
            lnpa + grav / (R_d * lap) * log((Ta - lap * (z - za)) / Ta),
        )
    end
    return T, p
end

# GFDL face quantities at interior faces j = 2..n (between cells j-1 and j), from
# topo_drag.F90 satur_flux (density, N) and topo_drag_tend (wind along the drag, unfloored)
function gfdl_face_values(zc, Tc, pc, u, v, taux, tauy, taub)
    n = length(zc)
    lapse = grav / cp_d
    N_h = fill(NaN, n + 1)
    rho_h = fill(NaN, n + 1)
    vtau_h = fill(NaN, n + 1)
    vtau_c = -(u .* taux .+ v .* tauy) ./ taub
    for j in 2:n
        dz = zc[j] - zc[j - 1]
        rho_h[j] = (pc[j - 1] - pc[j]) / (grav * dz)
        N2 = grav * ((Tc[j] - Tc[j - 1]) / dz + lapse) / (0.5 * (Tc[j] + Tc[j - 1]))
        N_h[j] = sqrt(max(1e-20, N2))
        vtau_h[j] = 0.5 * (vtau_c[j] + vtau_c[j - 1])
    end
    return (; N_h, rho_h, vtau_h)
end

# ------------------------------------------------------------------ column helpers
function column_spaces(zf)
    domain = Domains.IntervalDomain(
        Geometry.ZPoint(zf[1]),
        Geometry.ZPoint(zf[end]);
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain, Geometry.ZPoint.(zf))
    cs = CommonSpaces.ColumnSpace(
        FT;
        z_elem = length(zf) - 1,
        z_min = zf[1],
        z_max = zf[end],
        z_mesh = mesh,
        staggering = Grids.CellCenter(),
    )
    return cs, Spaces.face_space(cs)
end

values_of(f) = vec(copy(parent(f)))
function field_from(space, a)
    f = Fields.zeros(space)
    parent(f) .= reshape(a, :, 1)
    return f
end
level_field(cs) = similar(Fields.level(Fields.zeros(cs), 1))
function level_field(cs, x)
    f = level_field(cs)
    parent(f) .= x
    return f
end
value_of(f) = parent(f)[1]

# N as in orographic_gravity_wave_compute_tendency! (dry air), at centers and faces
function buoyancy_frequency(ᶜT, fs)
    ᶜdTdz = similar(ᶜT)
    CA.ogw_temperature_gradient!(ᶜdTdz, Fields.zeros(fs), ᶜT)
    ᶜN = similar(ᶜT)
    @. ᶜN = (grav / ᶜT) * (ᶜdTdz + grav / cp_m_dry)
    @. ᶜN = ifelse(ᶜN < eps(FT), sqrt(eps(FT)), sqrt(abs(ᶜN)))
    ᶠN = Fields.zeros(fs)
    @. ᶠN = CA.ᶠinterp(ᶜN)
    return ᶜN, ᶠN
end

base_fields(cs) = (;
    τ_x = level_field(cs), τ_y = level_field(cs), τ_l = level_field(cs),
    τ_p = level_field(cs), τ_np = level_field(cs), U_sat = level_field(cs),
    FrU_sat = level_field(cs), FrU_clp = level_field(cs), FrU_max = level_field(cs),
    FrU_min = level_field(cs),
    values = similar(Fields.level(Fields.zeros(cs), 1), Tuple{FT, FT, FT, FT}),
)

function topo_info(cs, c)
    (; t11, t21, t12, t22, hmax, hmin) = c.topo
    return (;
        hmax = level_field(cs, hmax), hmin = level_field(cs, hmin),
        t11 = level_field(cs, t11), t12 = level_field(cs, t12),
        t21 = level_field(cs, t21), t22 = level_field(cs, t22),
    )
end

function base_from_gfdl(cs, g)
    b = base_fields(cs)
    for (f, x) in ((b.τ_x, g.taux_base), (b.τ_y, g.tauy_base), (b.τ_l, g.taul),
        (b.τ_p, g.taup), (b.τ_np, g.taun), (b.U_sat, g.usat_base), (b.FrU_sat, g.frunl),
        (b.FrU_clp, g.fruclp), (b.FrU_max, g.fruhi), (b.FrU_min, g.frulo))
        parent(f) .= x
    end
    return b
end

base_flux!(b, cs, c, z_pbl, ᶜρ, ᶜu, ᶜv, ᶜz, ᶜN) = CA.calc_base_flux!(
    b.τ_x, b.τ_y, b.τ_l, b.τ_p, b.τ_np, b.U_sat, b.FrU_sat, b.FrU_clp, b.FrU_max,
    b.FrU_min, z_pbl, b.values, ogw_params, topo_info(cs, c), ᶜρ, ᶜu, ᶜv, ᶜz, ᶜN,
)

function saturation_profile(cs, fs, b, z_pbl, ᶜρ, ᶜu, ᶜv, ᶠp, ᶜN, ᶜz, ᶠz)
    ᶠτ_sat = Fields.zeros(fs)
    ᶠVτ = Fields.zeros(fs)
    CA.calc_saturation_profile!(
        ᶠτ_sat, ᶠVτ, b.U_sat, b.FrU_sat, b.FrU_clp, b.FrU_max, b.FrU_min,
        b.τ_x, b.τ_y, b.τ_p, z_pbl, ogw_params, ᶜρ, ᶜu, ᶜv, ᶠp, ᶜN, ᶜz, ᶠz,
        Fields.zeros(Spaces.level(fs, half)),
    )
    return ᶠτ_sat, ᶠVτ
end

# Cell-center values whose face averages (`ᶠinterp`) equal given values at the interior
# faces 2..n, so ClimaAtmos's interpolated face density and N can be set to GFDL's face
# values exactly: c[j] = 2 f[j] - c[j-1]. The one free value, the lowest center, is chosen
# so that both end centers are positive (the unused boundary faces copy the end centers,
# and a square root is taken there).
function centers_from_faces(f_interior)
    n = length(f_interior) + 1
    solve(c1) = accumulate((c, f) -> 2 * f - c, f_interior; init = c1)
    A = solve(0.0)[end]   # top center for c1 = 0; it changes by ±c1
    c1 = iseven(n) ? A / 2 : max(f_interior[1], 1e-3 - A)
    return vcat(c1, solve(c1))
end

function blocked_drag(cs, fs, b, ᶠVτ, ᶠz_start, ᶠp, ᶜp, ᶠN, ᶠz)
    ᶜuf = Fields.zeros(cs)
    ᶜvf = Fields.zeros(cs)
    fl = Spaces.level(fs, half)
    ᶠz_ref = Fields.zeros(fl)
    CA.calc_nonpropagating_forcing!(
        ᶜuf, ᶜvf, b.τ_x, b.τ_y, b.τ_l, b.τ_np, ᶠVτ, ᶠz_start,
        ᶠz_ref, Fields.zeros(fl), Fields.Field(Bool, cs), Fields.zeros(cs),
        Fields.zeros(cs), Fields.zeros(Spaces.level(cs, 1)),
        ᶠp, ᶜp, ᶠN, ᶠz, Fields.Δz_field(fs), grav,
    )
    return values_of(ᶜuf), values_of(ᶜvf), value_of(ᶠz_ref)
end

function propagating_drag(cs, b, ᶠτ_sat, ᶜρ)
    ᶜuf = Fields.zeros(cs)
    ᶜvf = Fields.zeros(cs)
    CA.calc_propagate_forcing!(ᶜuf, ᶜvf, b.τ_x, b.τ_y, b.τ_l, ᶠτ_sat, Fields.zeros(cs), ᶜρ)
    return values_of(ᶜuf), values_of(ᶜvf)
end

function start_height(fs, z_pbl)
    ᶠz_start = Fields.zeros(Spaces.level(fs, half))
    CA.blocked_layer_start_height!(ᶠz_start, z_pbl, Fields.coordinate_field(fs).z)
    return ᶠz_start
end

# all entries close: relative to each value, with an absolute floor tied to the largest
close_to(a, b; rtol, atol_frac = rtol) =
    all(isapprox.(a, b; rtol, atol = atol_frac * maximum(abs, b; init = 0.0)))

# ------------------------------------------------------------------ tests
# Checks for cases with GFDL drag data (all but the PBL-only cases)
function full_case_checks(col)
    (;
        c,
        cs,
        fs,
        ᶜz,
        ᶠz,
        zf,
        zc,
        n,
        Tc,
        pc,
        ρc,
        ᶜp,
        ᶜT,
        ᶜρ,
        ᶠp,
        pf,
        dp,
        src,
        z_pbl_gfdl,
        ᶠz_start,
    ) = col
    g = c.gfdl
    u = c.u.(zc)
    v = c.v.(zc)
    ᶜu = field_from(cs, u)
    ᶜv = field_from(cs, v)
    ᶜN, ᶠN = buoyancy_frequency(ᶜT, fs)
    τx, τy = g.taux_base, g.tauy_base
    ratio = τy / τx   # GFDL's v tendency = its u tendency times tauy/taux
    gf = gfdl_face_values(zc, Tc, pc, u, v, τx, τy, g.taub_base)

    # Base flux with GFDL's density and N in the source cell
    ρ2 = copy(ρc)
    ρ2[src] = g.density_src
    N2 = values_of(ᶜN)
    N2[src] = g.bfreq_src
    b = base_fields(cs)
    base_flux!(b, cs, c, z_pbl_gfdl, field_from(cs, ρ2), ᶜu, ᶜv, ᶜz, field_from(cs, N2))
    for (f, x) in ((b.τ_x, g.taux_base), (b.τ_y, g.tauy_base), (b.τ_l, g.taul),
        (b.τ_p, g.taup), (b.τ_np, g.taun), (b.U_sat, g.usat_base),
        (b.FrU_sat, g.frunl), (b.FrU_max, g.fruhi), (b.FrU_min, g.frulo),
        (b.FrU_clp, g.fruclp))
        @test isapprox(value_of(f), x; rtol = 1e-12, atol = 1e-15)
    end

    # Curvature in the cells where GFDL computes it (k = 2..min(kdim-1, kpbl))
    if hasproperty(g, :curv_c)
        d2u = similar(ᶜu)
        d2u .= CA.ᶜd2dz2(ᶜu)
        d2v = similar(ᶜv)
        d2v .= CA.ᶜd2dz2(ᶜv)
        curv = -(values_of(d2u) .* τx .+ values_of(d2v) .* τy) ./ hypot(τx, τy)
        cells = g.curv_first:(n - 1)
        curv_scale = maximum(abs, vcat(u, v)) / minimum(diff(zf))^2
        @test maximum(abs.(curv[cells] .- g.curv_c)) <= 1e-12 * curv_scale
    end

    # Saturation profile with GFDL's face inputs: centers chosen so that the interpolated
    # face density and N are GFDL's (winds as they are: both codes average the two
    # centers' wind along the drag and curvature). The whole face profile, including the
    # leftover-flux step on face pressures, must match GFDL. It is judged on the column
    # maximum, since near the top the leftover-flux step subtracts nearly equal numbers.
    ρ_in = centers_from_faces(gf.rho_h[2:n])
    N_in = centers_from_faces(gf.N_h[2:n])
    ᶠρ_check = Fields.zeros(fs)
    ᶠρ_check .= CA.ᶠinterp.(field_from(cs, ρ_in))
    @test close_to(values_of(ᶠρ_check)[2:n], gf.rho_h[2:n]; rtol = 1e-14, atol_frac = 1e-14)
    ᶠN_check = Fields.zeros(fs)
    ᶠN_check .= CA.ᶠinterp.(field_from(cs, N_in))
    @test close_to(values_of(ᶠN_check)[2:n], gf.N_h[2:n]; rtol = 1e-14, atol_frac = 1e-14)
    ᶠτ_in, _ = saturation_profile(cs, fs, base_from_gfdl(cs, g), z_pbl_gfdl,
        field_from(cs, ρ_in), ᶜu, ᶜv, ᶠp, field_from(cs, N_in), ᶜz, ᶠz)
    @test maximum(abs.(values_of(ᶠτ_in) .- g.tausat_res)) <=
          1e-13 * maximum(abs, g.tausat_res)

    # Wind along the drag at faces: interpolate, then floor (as GFDL)
    bg = base_from_gfdl(cs, g)
    _, ᶠVτ = saturation_profile(cs, fs, bg, z_pbl_gfdl, ᶜρ, ᶜu, ᶜv, ᶠp, ᶜN, ᶜz, ᶠz)
    interior = 2:n
    if hasproperty(g, :vtau_faces)   # GFDL's own output around the wind reversal
        @test all(
            isapprox.(gf.vtau_h[g.vtau_faces], g.vtau_unfl;
                atol = 1e-12 * maximum(abs, gf.vtau_h[interior])),
        )
    end
    @test close_to(values_of(ᶠVτ)[interior], max.(1e-20, gf.vtau_h[interior]);
        rtol = 1e-14, atol_frac = 1e-14)

    # Blocked layer with GFDL's face N and wind along the drag
    V_in = copy(gf.vtau_h)
    V_in[[1, end]] .= 1.0     # boundary faces: not used, GFDL has no value there
    N_in = copy(gf.N_h)
    N_in[[1, end]] .= 0.01
    u_np, v_np, z_ref = blocked_drag(cs, fs, base_from_gfdl(cs, g),
        field_from(fs, V_in), ᶠz_start, ᶠp, ᶜp, field_from(fs, N_in), ᶠz)
    @test z_ref == zf[g.z_ref_face]
    expected_np = zeros(n)
    expected_np[g.np_first:(g.np_first + length(g.dudt_np_kref) - 1)] .= g.dudt_np_kref
    @test close_to(u_np, expected_np; rtol = 1e-12)
    @test close_to(v_np, expected_np .* ratio; rtol = 1e-12)
    # ClimaAtmos self-consistency: the blocked column stress is τ_x·τ_np/τ_l
    @test isapprox(sum(u_np .* dp) / grav, τx * g.taun / g.taul; rtol = 1e-12)

    # Propagating drag from GFDL's flux profile, density Δp/(g Δz)
    if hasproperty(g, :dudt_p)
        ρ_dp = dp ./ (grav .* diff(zf))
        u_p, v_p = propagating_drag(cs, base_from_gfdl(cs, g),
            field_from(fs, g.tausat_res), field_from(cs, ρ_dp))
        @test close_to(u_p, g.dudt_p; rtol = 1e-12)
        @test close_to(v_p, g.dudt_p .* ratio; rtol = 1e-12)
    end

    # End to end: ClimaAtmos computes everything itself, no limiter
    b8 = base_fields(cs)
    z8 = level_field(cs)
    CA.get_pbl_z!(z8, ᶜp, ᶠp, ᶜT, ᶜz, grav, cp_d)
    base_flux!(b8, cs, c, z8, ᶜρ, ᶜu, ᶜv, ᶜz, ᶜN)
    ᶠτ8, ᶠVτ8 = saturation_profile(cs, fs, b8, z8, ᶜρ, ᶜu, ᶜv, ᶠp, ᶜN, ᶜz, ᶠz)
    up8, vp8 = propagating_drag(cs, b8, ᶠτ8, ᶜρ)
    unp8, vnp8, _ = blocked_drag(cs, fs, b8, ᶠVτ8, start_height(fs, z8), ᶠp, ᶜp, ᶠN, ᶠz)
    u8 = up8 .+ unp8
    v8 = vp8 .+ vnp8
    stress_ca = hypot(sum(u8 .* dp), sum(v8 .* dp)) / grav
    stress_gfdl = hypot(g.stress_x_noclip, g.stress_y_noclip)
    if c.name != "pbl_second_level_only"   # the source cell differs there, on purpose
        @test isapprox(stress_ca, stress_gfdl; rtol = 0.01)
    end
    # Where the wave breaks right at the source, breaking drag starts in the source cell:
    # the PBL cells get only their share of the leftover-flux step (GFDL: 0.02% and 0.17%)
    if c.name in ("low_level_jet", "weak_surface_wind")
        below = 1:(findfirst(>(value_of(z8)), zc) - 1)
        share = sum(up8[below] .* dp[below]) / sum(up8 .* dp)
        @test share < 0.01
    end

    # The same chain through orographic_gravity_wave_forcing! (with the limiter)
    cl = Spaces.level(cs, 1)
    fl = Spaces.level(fs, half)
    cache = (;
        ogw_params, topo_ᶠτ_sat = Fields.zeros(fs),
        topo_ᶠVτ = Fields.zeros(fs), topo_τ_x = level_field(cs),
        topo_τ_y = level_field(cs), topo_τ_l = level_field(cs),
        topo_τ_p = level_field(cs), topo_τ_np = level_field(cs),
        topo_U_sat = level_field(cs), topo_FrU_sat = level_field(cs),
        topo_FrU_max = level_field(cs), topo_FrU_min = level_field(cs),
        topo_FrU_clp = level_field(cs), topo_ᶜz_pbl = level_field(cs),
        topo_ᶠz_blocked_start = Fields.zeros(fl),
        values_at_z_pbl = similar(Fields.level(Fields.zeros(cs), 1), Tuple{FT, FT, FT, FT}),
        topo_info = topo_info(cs, c), ᶜmask = Fields.Field(Bool, cs),
        ᶠp_ref = Fields.zeros(fl),
    )
    scratch = (; ᶜtemp_scalar = Fields.zeros(cs), ᶜtemp_scalar_2 = Fields.zeros(cs),
        temp_field_level = Fields.zeros(cl), ᶠtemp_field_level = Fields.zeros(fl))
    ᶜuf = Fields.zeros(cs)
    ᶜvf = Fields.zeros(cs)
    CA.orographic_gravity_wave_forcing!(ᶜu, ᶜv, ᶜN, ᶠN, ᶜz, ᶠz, Fields.Δz_field(fs),
        ᶜuf, ᶜvf, ᶜρ, ᶜp, ᶠp, ᶜT, grav, cp_d,
        (; orographic_gravity_wave = cache, scratch))
    unclipped = hypot.(u8, v8) .<= 3e-3 * sqrt(2)
    @test values_of(ᶜuf)[unclipped] == u8[unclipped]
    @test values_of(ᶜvf)[unclipped] == v8[unclipped]
    if get(ENV, "OGW_GFDL_PRINT_PIN", "false") == "true"
        println(c.name, ": ca_dudt = ", repr(values_of(ᶜuf)))
    end
    # Regression pin of this code's own output (v follows from u: the limiter
    # scales both by one factor, and both parts have v/u = τ_y/τ_x)
    @test close_to(values_of(ᶜuf), c.ca_dudt; rtol = 1e-10)
    @test close_to(values_of(ᶜvf), values_of(ᶜuf) .* (value_of(b8.τ_y) / value_of(b8.τ_x));
        rtol = 1e-10)
end

@testset "GFDL topo_drag reference" begin
    # The stored GFDL numbers were made with these values; if they change, regenerate.
    @testset "constants and parameters used for the reference" begin
        @test grav == 9.81
        @test R_d == 287.0
        @test cp_d == 1004.5
        @test cp_m_dry == 1004.5
        @test (ogwp.Fr_crit, ogwp.ρscale, ogwp.L0) == (0.7, 1.2, 80e3)
        @test (ogwp.γ, ogwp.β, ogwp.ϵ) == (0.4, 0.5, 0.0)
        @test (ogwp.a0, ogwp.a1) == (0.9, 3.0)
    end

    for c in GFDL_REFERENCE_COLUMNS
        @testset "$(c.name)" begin
            zf = c.grid == :uniform ? uniform_faces() : stretched_faces()
            n = length(zf) - 1
            cs, fs = column_spaces(zf)
            ᶜz = Fields.coordinate_field(cs).z
            ᶠz = Fields.coordinate_field(fs).z
            zc = values_of(ᶜz)
            @test n == c.kdim
            @test zf[end] ≈ c.z_top rtol = 1e-15
            @test zc ≈ 0.5 .* (zf[1:n] .+ zf[2:(n + 1)]) rtol = 1e-14

            Tc, pc = temperature_and_pressure(c.T0, c.lapse_segments, zc)
            ρc = pc ./ (R_d .* Tc)
            ᶜp = field_from(cs, pc)
            ᶜT = field_from(cs, Tc)
            ᶜρ = field_from(cs, ρc)
            ᶠp = Fields.zeros(fs)
            CA.ogw_face_pressure!(ᶠp, ᶜp, ᶜρ, ᶜz, ᶠz, grav)
            pf = values_of(ᶠp)
            dp = pf[1:n] .- pf[2:(n + 1)]

            # PBL: source cell = first cell above the PBL-top cell
            src = n + 1 - c.kpbl
            z_pbl = level_field(cs)
            CA.get_pbl_z!(z_pbl, ᶜp, ᶠp, ᶜT, ᶜz, grav, cp_d)
            ca_src = findfirst(>(value_of(z_pbl)), zc)
            if c.name == "pbl_second_level_only"
                # on purpose: ClimaAtmos also tests the second-lowest level as a PBL top
                @test ca_src == src + 1
            else
                @test ca_src == src
            end
            z_pbl_gfdl = level_field(cs, zc[src - 1])  # center of GFDL's PBL-top cell
            ᶠz_start = start_height(fs, z_pbl_gfdl)
            @test value_of(ᶠz_start) == zf[src + 1]

            hasproperty(c, :gfdl) && full_case_checks((;
                c,
                cs,
                fs,
                ᶜz,
                ᶠz,
                zf,
                zc,
                n,
                Tc,
                pc,
                ρc,
                ᶜp,
                ᶜT,
                ᶜρ,
                ᶠp,
                pf,
                dp,
                src,
                z_pbl_gfdl,
                ᶠz_start,
            ))
        end
    end
end

# ======================================================================================
# 4. Turbulent orographic form drag (TOFD)
# ======================================================================================

#=
Unit tests for the turbulent orographic form drag (TOFD) of Beljaars et al. (2004), in the
IFS form (IFS Documentation CY47R1, Part IV, Sections 3.4 and 11.3).

- `tofd_drag_coefficient!` on a single column, passing the bottom face level as the
  surface height (as the OGW cache does), against the closed form in the height above
  ground z_agl = z - z_sfc:

    Cd(z) = a_tofd * alpha*beta*Cmd*Ccorr*2.109 * exp(-(z_agl/1500)^1.5) * a2 * z_agl^-1.2,
    a2 = sigma_flt^2 * k1^(n1-n2) / (I_H * kflt^n1),   alpha = 35.

- The IFS smoothing operator and the band-pass-filtered orography variance that defines
  sigma_flt^2, against the exact discrete transfer function for sinusoidal orography
  varying in longitude (at the equator and at 60N, where the longitude spacing carries
  cos(lat)) and in latitude; ocean relief clipped to zero; and the area-weighted block
  average.
- `tofd_implicit_rate`: the backward-Euler rate reduces to the explicit drag for weak
  damping, is 1/(2 dt) at r dt = 1, and can never reverse the wind within one step for
  strong damping; stepped forward many times with forward Euler, the wind decays
  monotonically and equals the exact solution u0 / (1 + Cd u0 t) of du/dt = -Cd*|u|*u
  at every step, for weak and strong damping alike.
=#

# Reference IFS TOFD drag coefficient, independent of the source under test.
function ifs_tofd_Cd(z, σ²_flt, a_tofd)
    α, β, C_md, C_corr = 35.0, 1.0, 0.005, 0.6
    I_H, k_flt, k1 = 0.00102, 0.00035, 0.003
    n1, n2 = -1.9, -2.8
    a2 = σ²_flt * k1^(n1 - n2) / (I_H * k_flt^n1)
    zc = max(1.0, z)
    return a_tofd * α * β * C_md * C_corr * 2.109 * exp(-(zc / 1500)^1.5) * a2 * zc^(-1.2)
end

function tofd_column(FT, z_min)
    center_space = CommonSpaces.ColumnSpace(
        FT;
        z_min,
        z_max = FT(20_000),
        z_elem = 80,
        staggering = Grids.CellCenter(),
    )
    face_space = Spaces.face_space(center_space)
    c_z = Fields.coordinate_field(center_space).z
    # Surface height from the bottom face level, exactly as the OGW cache passes it.
    z_sfc = Fields.level(Fields.coordinate_field(face_space).z, half)
    return center_space, c_z, z_sfc
end

@testset "OGW TOFD: drag coefficient profile" begin
    FT = Float64
    σ²_flt = FT(200)^2
    a_tofd = FT(1)

    center_space, c_z, z_sfc = tofd_column(FT, 0)  # flat surface at z = 0
    surf_c = Spaces.level(center_space, 1)
    σ²_f = Fields.ones(surf_c) .* σ²_flt

    Cd = Fields.zeros(center_space)
    CA.tofd_drag_coefficient!(Cd, c_z, z_sfc, σ²_f, a_tofd)
    cd = vec(parent(Cd))
    zc = vec(parent(c_z))

    @test all(isapprox.(cd, ifs_tofd_Cd.(zc, σ²_flt, a_tofd); rtol = 1e-10))
    @test all(cd .> 0)
    @test all(diff(cd) .< 0)
    @test sum(cd[zc .< 1500]) > 0.8 * sum(cd)

    # The coefficient is linear in a_tofd and vanishes when TOFD is disabled.
    Cd_half = Fields.zeros(center_space)
    CA.tofd_drag_coefficient!(Cd_half, c_z, z_sfc, σ²_f, FT(0.5))
    @test vec(parent(Cd_half)) ≈ 0.5 .* cd rtol = 1e-12
    Cd0 = Fields.ones(center_space)
    CA.tofd_drag_coefficient!(Cd0, c_z, z_sfc, σ²_f, FT(0))
    @test iszero(maximum(abs.(parent(Cd0))))

    # On elevated terrain the profile follows the height above ground, not altitude.
    z_ground = FT(1_000)
    center_hi, c_z_hi, z_sfc_hi = tofd_column(FT, z_ground)
    σ²_hi = Fields.ones(Spaces.level(center_hi, 1)) .* σ²_flt
    Cd_hi = Fields.zeros(center_hi)
    CA.tofd_drag_coefficient!(Cd_hi, c_z_hi, z_sfc_hi, σ²_hi, a_tofd)
    cd_hi = vec(parent(Cd_hi))
    zc_hi = vec(parent(c_z_hi))
    @test all(
        isapprox.(cd_hi, ifs_tofd_Cd.(zc_hi .- z_ground, σ²_flt, a_tofd); rtol = 1e-10),
    )
end

@testset "OGW TOFD: IFS smoothing operator" begin
    FT = Float64
    # Top hat of width Δ with cosine edges of width δ, sampled at Δx (IFS Eq. 11.4).
    w = CA.ifs_smoothing_weights(FT(100), FT(20_000), FT(1_000), 10_000)
    J = (length(w) - 1) ÷ 2
    r = abs.(((-J):J) .* 100)
    @test sum(w) ≈ 1
    @test w ≈ reverse(w)
    @test J == 110                                   # ⌈(Δ/2 + δ)/Δx⌉
    @test all(w[r .< 9_000] .≈ maximum(w))           # flat top
    @test w[r .== 10_000][1] ≈ maximum(w) / 2        # half height at r = Δ/2
    @test all(w[r .>= 11_000] .== 0)
    # Δ = 2 km with δ = 1 km has no flat part: a raised cosine of half-width 2 km.
    w2 = CA.ifs_smoothing_weights(FT(500), FT(2_000), FT(1_000), 10_000)
    @test w2 ≈
          [0, 0.5 - 0.5 * cos(π / 4), 0.5, 0.5 + 0.5 * cos(π / 4), 1,
        0.5 + 0.5 * cos(π / 4), 0.5, 0.5 - 0.5 * cos(π / 4), 0] ./ 4 atol = 1e-14
    # Grid spacing wider than the operator: no smoothing.
    @test CA.ifs_smoothing_weights(FT(5_000), FT(2_000), FT(1_000), 10_000) == [0, 1, 0]
    # The half-width is capped.
    @test length(CA.ifs_smoothing_weights(FT(1), FT(20_000), FT(1_000), 50)) == 101
end

@testset "OGW TOFD: filtered-orography variance" begin
    FT = Float64
    R = FT(6.371e6)
    A = FT(300)
    # Strips at 30 arc seconds, periodic in longitude over 9° and 1° tall.
    dlon = FT(1 / 120)
    nlon, nlat = 1080, 120
    lon = collect(range(dlon / 2; step = dlon, length = nlon))
    strip(lat_center) =
        collect(range(lat_center - FT(0.5) + dlon / 2; step = dlon, length = nlat))

    # Exact discrete gain of the IFS operator of width Δ at spacing Δx for a periodic
    # wave with n_waves wavelengths over n_points samples.
    function gain(Δx, Δ, n_waves, n_points, J_max)
        w = CA.ifs_smoothing_weights(FT(Δx), FT(Δ), FT(1_000), J_max)
        J = (length(w) - 1) ÷ 2
        return sum(w[m + J + 1] * cos(2π * n_waves * m / n_points) for m in (-J):J)
    end
    # Gain of the band pass (S_2km − S_20km) for a longitude wave, by row: the longitude
    # pass at the spacing R·Δλ·cos φ of each row, then the latitude pass across rows.
    function band_gain_lon(lat, n_waves, j)
        Δy = R * deg2rad(dlon)
        g = map((2_000, 20_000)) do Δ
            w = CA.ifs_smoothing_weights(Δy, FT(Δ), FT(1_000), nlat)
            J = (length(w) - 1) ÷ 2
            sum(
                w[m + J + 1] * gain(
                    R * deg2rad(dlon) * cosd(lat[clamp(j + m, 1, nlat)]),
                    Δ,
                    n_waves,
                    nlon,
                    nlon ÷ 8,
                ) for m in (-J):J
            )
        end
        return g[1] - g[2]
    end

    # Rows away from the strip edges, so the latitude pass repeats no edge value.
    interior = 20:100
    # At 60°N the row spacing is R·Δλ·cos φ ≈ half the equatorial spacing.
    # One check per latitude over all wavelengths and interior rows.
    for lat_center in (FT(0), FT(60))
        lat = strip(lat_center)
        # n_waves = 20, 100, 400: wavelengths of 50, 10, 2.5 km at the equator.
        wave_ok = map((20, 100, 400)) do n_waves
            # Positive orography: no clipping.
            elev = [A * (1 + sin(2π * n_waves * (i - 1) / nlon)) for i in 1:nlon, _ in lat]
            variance = CA.calc_filtered_orography_variance_latlon(elev, lon, lat, R)
            expected = [A^2 / 2 * band_gain_lon(lat, n_waves, j)^2 for j in interior]
            all(
                isapprox.(
                    vec(sum(variance[:, interior]; dims = 1)) ./ nlon,
                    expected;
                    rtol = 1e-8,
                ),
            )
        end
        @test all(wave_ok)
    end
    # The 10 km wave is inside the band and keeps most of its variance.
    @test band_gain_lon(strip(FT(0)), 100, 60)^2 > 0.5
    # At 60°N the 100-wave pattern is 5 km long, so a filter that ignored cos φ (spacing
    # R·Δλ) would see a 10 km wave, and the per-row check above would fail.
    Δx_eq = R * deg2rad(dlon)
    Δx_60 = Δx_eq * cosd(FT(60))
    band100(Δx) =
        gain(Δx, 2_000, 100, nlon, nlon ÷ 8) - gain(Δx, 20_000, 100, nlon, nlon ÷ 8)
    @test abs(band100(Δx_60)^2 - band100(Δx_eq)^2) > 0.1

    # Latitude-only wave, an integer number of waves across the strip: the longitude
    # pass leaves it unchanged and the latitude pass at spacing R·Δφ sets its gain.
    # Interior rows are far enough from the strip edges that no edge value repeats.
    lat = strip(FT(0))
    Δy = R * deg2rad(dlon)
    lat_ok = map((6, 12, 30)) do n_waves   # wavelengths of 18.5, 9.3, 3.7 km
        elev = [A * (1 + sin(2π * n_waves * (j - 1) / nlat)) for _ in lon, j in 1:nlat]
        variance = CA.calc_filtered_orography_variance_latlon(elev, lon, lat, R)
        band =
            gain(Δy, 2_000, n_waves, nlat, nlat) - gain(Δy, 20_000, n_waves, nlat, nlat)
        expected = [
            (A * band * sin(2π * n_waves * (j - 1) / nlat))^2 for _ in lon, j in interior
        ]
        abs(band) > 0.05 && all(abs.(variance[:, interior] .- expected) .<= 1e-8 * A^2)
    end
    @test all(lat_ok)

    # Uniform orography and ocean have no filtered variance. The ocean has relief, but it
    # is clipped to zero before filtering.
    @test maximum(
        CA.calc_filtered_orography_variance_latlon(fill(FT(1500), nlon, nlat), lon, lat, R),
    ) < 1e-12
    ocean = [FT(-3000) + A * sin(2π * 100 * (i - 1) / nlon) for i in 1:nlon, _ in lat]
    @test iszero(CA.calc_filtered_orography_variance_latlon(ocean, lon, lat, R))

    # Area-weighted block average of a field varying in longitude and latitude,
    # f = (1 + λ/360)·cos φ: block (ib, jb) averages to
    # mean(1 + λ/360) · Σ cos²φ / Σ cos φ over its points.
    lat_g = collect(range(-87.5; step = 5, length = 36))
    lon_g = collect(range(2.5; step = 5, length = 72))
    f = [(1 + λ / 360) * cosd(φ) for λ in lon_g, φ in lat_g]
    lon_c, lat_c, f_c = CA.block_average_latlon(f, lon_g, lat_g, 6)
    @test size(f_c) == (12, 6)
    @test lon_c ≈ collect(range(15.0; step = 30, length = 12))
    @test lat_c ≈ collect(range(-75.0; step = 30, length = 6))
    block_ok = map(((1, 1), (7, 4))) do (ib, jb)
        cols = ((ib - 1) * 6 + 1):(ib * 6)
        rows = ((jb - 1) * 6 + 1):(jb * 6)
        expected =
            sum(1 .+ lon_g[cols] ./ 360) / 6 * sum(cosd.(lat_g[rows]) .^ 2) /
            sum(cosd.(lat_g[rows]))
        isapprox(f_c[ib, jb], expected; rtol = 1e-12)
    end
    @test all(block_ok)
end

@testset "OGW TOFD: backward-Euler rate" begin
    FT = Float64
    dt = FT(450)
    speed = FT(10)

    # Weak damping (r·dt ≪ 1): the explicit drag Cd·|U| is recovered.
    Cd_weak = FT(1e-9)
    @test CA.tofd_implicit_rate(Cd_weak, speed, dt) ≈ Cd_weak * speed rtol = 1e-5

    # Strong damping (r·dt ≫ 1): the rate saturates at 1/dt, so one step brings the
    # wind to rest at most and never reverses it.
    Cd_strong = FT(10)
    rate_strong = CA.tofd_implicit_rate(Cd_strong, speed, dt)
    @test rate_strong * dt < 1
    @test rate_strong ≈ 1 / dt rtol = 1e-3

    # Middle (r·dt = 1): the rate is r/2 = 1/(2·dt).
    Cd_mid = 1 / (speed * dt)
    @test CA.tofd_implicit_rate(Cd_mid, speed, dt) ≈ 1 / (2 * dt) rtol = 1e-12

    grid = Iterators.product(
        (FT(0), FT(1e-7), FT(1e-4), FT(1e-2), FT(1)),
        (FT(0), FT(1), FT(30)),
    )
    rates = [(Cd, U, CA.tofd_implicit_rate(Cd, U, dt)) for (Cd, U) in grid]
    @test all(0 <= rate * dt < 1 && rate <= Cd * U for (Cd, U, rate) in rates)
end

@testset "OGW TOFD: wind over many steps" begin
    # Step a single wind u forward with the tendency the model adds,
    # du/dt = -tofd_implicit_rate(Cd, |u|, dt) * u, as forward-Euler steps of length dt.
    # Each step gives u_{n+1} = u_n / (1 + Cd·u_n·dt), i.e. 1/u_{n+1} = 1/u_n + Cd·dt,
    # so the steps reproduce the exact solution of du/dt = -Cd·|u|·u,
    # u(t) = u0 / (1 + Cd·u0·t), at every step, for any Cd and dt.
    FT = Float64
    dt = FT(450)
    nsteps = 200
    u0 = FT(10)
    t = dt .* (0:nsteps)
    function step_wind(Cd, u0, dt, nsteps)
        u = zeros(FT, nsteps + 1)
        u[1] = u0
        for n in 1:nsteps
            u[n + 1] = u[n] - dt * CA.tofd_implicit_rate(Cd, abs(u[n]), dt) * u[n]
        end
        return u
    end

    # Weak damping (Cd·u0·dt = 4.5e-3) and strong damping (Cd·u0·dt = 2250 on the first
    # step): the wind decays monotonically, never changes sign, and matches the exact
    # solution.
    winds = [(Cd, step_wind(Cd, u0, dt, nsteps)) for Cd in (FT(1e-6), FT(0.5))]
    @test all(all(u .> 0) && all(diff(u) .< 0) for (_, u) in winds)
    @test all(
        all(isapprox.(u, u0 ./ (1 .+ Cd .* u0 .* t); rtol = 1e-10)) for (Cd, u) in winds
    )
end

# ======================================================================================
# 5. Preprocessing: hmax and the drag tensor from elevation
# ======================================================================================

#=
Tests of the offline preprocessing that turns elevation into `hmax` and the drag tensor
(`calc_∇A`, `calc_orographic_tensor`, `calc_velocity_potential`, `calc_hpoz_latlon`), on
small latitude-longitude grids with analytic mountains. Expected values come from the
analytic shapes (exact slopes, closed-form moments) or from exact properties of the
sphere (rotating or shifting a mountain), never from the code under test. These catch:

  - wrong east-west grid spacing near the poles (a cos(latitude) floor made east-west
    slopes too small poleward of 60°): slopes, tensor and isotropy at 60°-85°;
  - a tensor that is not the rotated copy for a ridge turned by 90°, or whose off-diagonal
    entries are swapped (ridge at 45° to the grid);
  - a zonal finite-difference step that changes abruptly with latitude;
  - rows on or next to the poles: finite slopes, and a tensor that is zero poleward of 88°;
  - elevation sampling that coarsens with the smoothing scale;
  - a tensor that is not symmetric and definite for a mountain;
  - longitude handled as an edge instead of a circle (date-line seam): a mountain moved
    across the seam must give the same hmax and tensor, moved;
  - a subgrid height that does not follow the smoothing scale α·Δx.
=#

const R_e = 6.371e6

# Cell-centered lat-lon grid: full longitude circle, a latitude band.
function latlon_grid(dlon, dlat, lat_lo, lat_hi)
    lon = collect(range(-180 + dlon / 2, 180 - dlon / 2; step = dlon))
    lat = collect(range(lat_lo + dlat / 2, lat_hi - dlat / 2; step = dlat))
    return lon, lat
end

# Local angular coordinates (east `a`, north `b`, radians) of the point (λ, φ) about the
# center (λ0, φ0), measured along the tangent plane at the center (gnomonic angles).
function local_angles(λ, φ, λ0, φ0)
    p = (cosd(φ) * cosd(λ), cosd(φ) * sind(λ), sind(φ))
    c = (cosd(φ0) * cosd(λ0), cosd(φ0) * sind(λ0), sind(φ0))
    e = (-sind(λ0), cosd(λ0), 0.0)
    n = (-sind(φ0) * cosd(λ0), -sind(φ0) * sind(λ0), cosd(φ0))
    dot3(x, y) = x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
    pc = dot3(p, c)
    return atan(dot3(p, e), pc), atan(dot3(p, n), pc)
end

# Gaussian mountain with half-widths `L_east`, `L_north` [m] along axes turned by `θ`
# (degrees, counterclockwise) from the local east and north directions at its center:
# rotating the sphere moves it without changing its shape.
gaussian_mountain(H, λ0, φ0, L_east, L_north, θ = 0.0) =
    (λ, φ) -> begin
        a0, b0 = local_angles(λ, φ, λ0, φ0)
        a = cosd(θ) * a0 + sind(θ) * b0
        b = -sind(θ) * a0 + cosd(θ) * b0
        H * exp(-(a * R_e / L_east)^2 - (b * R_e / L_north)^2)
    end

on_grid(f, lon, lat) = [f(λ, φ) for λ in lon, φ in lat]

# Exact east and north derivatives [m/m] of an analytic field, by a centered difference
# with a 1e-5° step (truncation error ~1e-10 relative, far below the tolerances).
function exact_gradient(f, lon, lat)
    δ = 1e-5
    dfdx = [
        (f(λ + δ, φ) - f(λ - δ, φ)) / (2 * deg2rad(δ) * R_e * cosd(φ)) for
        λ in lon, φ in lat
    ]
    dfdy = [
        (f(λ, φ + δ) - f(λ, φ - δ)) / (2 * deg2rad(δ) * R_e) for λ in lon, φ in lat
    ]
    return dfdx, dfdy
end

relerr(x, x_exact) = sqrt(sum(abs2, x .- x_exact) / sum(abs2, x_exact))

# Area-weighted sum of a field over the grid.
area_sum(field, lat) = sum(field .* cosd.(lat)')

# Area-integrated tensor (t11, t21, t12, t22) as a 2x2 matrix [[t11 t12]; [t21 t22]],
# rows = derivative of h, columns = derivative of χ (stored tᵢⱼ = −∂χ/∂xⱼ·∂h/∂xᵢ).
tensor_matrix(t11, t21, t12, t22, lat) = [
    area_sum(t11, lat) area_sum(t12, lat)
    area_sum(t21, lat) area_sum(t22, lat)
]

@testset "OGW preprocessing: east-west slopes at all latitudes" begin
    # A mountain 150 km wide east-west and 300 km north-south, on a 0.1° grid. The
    # finite-difference slope must match the exact slope at every latitude, including
    # poleward of 60°, where the zonal grid spacing R·Δλ·cos(lat) keeps shrinking.
    lon, lat0 = latlon_grid(0.1, 0.1, -90, 90)
    for φ0 in (0, 45, 60, 70, 80, 85, -80)
        lat = filter(φ -> abs(φ - φ0) < 6, lat0)
        f = gaussian_mountain(2000.0, 30.0, φ0, 150e3, 300e3)
        h = on_grid(f, lon, lat)
        dhdx, dhdy = CA.calc_∇A(h, lon, lat, R_e)
        dhdx_exact, dhdy_exact = exact_gradient(f, lon, lat)
        @test relerr(dhdx, dhdx_exact) < 0.01
        @test relerr(dhdy, dhdy_exact) < 0.02
    end
end

@testset "OGW preprocessing: east-west slope balance is continuous across latitude" begin
    # Round bumps 40 km wide on a 0.25° grid, just below and just above 48.2°, where
    # 1/cos(lat) passes 1.5. The east-west/north-south balance of the squared slopes must
    # change smoothly with latitude: a zonal step that jumps from one to two columns there
    # would change it by about a third.
    lon, lat0 = latlon_grid(0.25, 0.25, -90, 90)
    balance(φ0) = begin
        lat = filter(φ -> abs(φ - φ0) < 4, lat0)
        h = zeros(length(lon), length(lat))
        for a in -2:2, b in -2:2
            # bumps on a 120 km lattice around (30°E, φ0)
            λc = 30.0 + rad2deg(a * 120e3 / (R_e * cosd(φ0)))
            φc = φ0 + rad2deg(b * 120e3 / R_e)
            h .+= on_grid(gaussian_mountain(1000.0, λc, φc, 40e3, 40e3), lon, lat)
        end
        dhdx, dhdy = CA.calc_∇A(h, lon, lat, R_e)
        area_sum(dhdx .^ 2, lat) / area_sum(dhdy .^ 2, lat)
    end
    @test balance(47.5) ≈ balance(49.0) rtol = 0.05
    @test balance(45.0) ≈ balance(52.0) rtol = 0.1
end

@testset "OGW preprocessing: rows on and next to the poles" begin
    # Node grid with rows exactly on the poles (cos(lat) = 0) and a cell grid like the
    # subsampled ETOPO one, whose first and last rows sit 0.008° from the poles: slopes
    # stay finite, and the tensor is zero poleward of 88° (as in Garner's transform.f90)
    # and nonzero for a mountain just equatorward of it.
    for (lon, lat) in (
        (collect(0.0:1.0:359.0), collect(-90.0:0.5:90.0)),
        (collect(-179.95:0.1:179.95), collect(range(-89.99; stop = 89.99, length = 1500))),
    )
        h =
            on_grid(gaussian_mountain(2000.0, 30.0, 86.5, 100e3, 100e3), lon, lat) .+
            on_grid(gaussian_mountain(2000.0, 30.0, -86.5, 100e3, 100e3), lon, lat) .+
            on_grid(gaussian_mountain(1000.0, 0.0, 89.5, 50e3, 50e3), lon, lat)
        χ =
            on_grid(gaussian_mountain(1.0e8, 30.0, 86.5, 150e3, 150e3), lon, lat) .+
            on_grid(gaussian_mountain(1.0e8, 30.0, -86.5, 150e3, 150e3), lon, lat) .+
            on_grid(gaussian_mountain(1.0e8, 0.0, 89.5, 80e3, 80e3), lon, lat)
        dhdx, dhdy = CA.calc_∇A(h, lon, lat, R_e)
        @test all(isfinite, dhdx) && all(isfinite, dhdy)
        tensor = CA.calc_orographic_tensor(copy(h), χ, lon, lat, R_e)
        polar = abs.(lat) .> 88
        for t in tensor
            @test all(isfinite, t)
            @test all(iszero, t[:, polar])
            @test maximum(abs, t[:, .!polar]) > 0
        end
    end
end

@testset "OGW preprocessing: elevation sampling does not follow the smoothing scale" begin
    # The raw 1-arcminute elevation (21600 longitudes, 1.85 km) is sampled every 4th
    # point (7.4 km) for every smoothing scale wider than 30 km: a wider window must not
    # coarsen the slopes. A narrower window samples finer, so it still holds a few points
    # across. The stride always divides 21600, so the longitudes close the circle evenly.
    for L in (28.8e3, 51e3, 192e3, 768e3, 3000e3)
        @test CA.ogw_topography_stride(L, R_e) == 4
    end
    @test CA.ogw_topography_stride(8e3, R_e) == 1
    for L in range(1e3, 1000e3; length = 50)
        stride = CA.ogw_topography_stride(L, R_e)
        @test 21600 % stride == 0
        @test stride * 2π * R_e / 21600 <= min(7.5e3, L / 4) * 1.5 + 2π * R_e / 21600
    end
end

@testset "OGW preprocessing: tensor from given slopes, rotated over latitude" begin
    # With χ given analytically, the tensor −∇χ (∇h)ᵀ has an exact value at every point,
    # so the area-integrated tensor must match it at every latitude. A ridge long in one
    # direction and the same ridge turned by 90° swap their large tensor entries.
    lon, lat0 = latlon_grid(0.05, 0.05, -90, 90)
    long, short = 200e3, 70e3
    results = Dict{Tuple{Float64, Symbol}, Matrix{Float64}}()
    for φ0 in (0.0, 60.0, 70.0, 80.0), orientation in (:east_west, :north_south)
        lat = filter(φ -> abs(φ - φ0) < 5, lat0)
        L_east, L_north = orientation == :east_west ? (long, short) : (short, long)
        f = gaussian_mountain(2000.0, 30.0, φ0, L_east, L_north)
        g = gaussian_mountain(1.0e8, 30.0, φ0, 1.5 * L_east, 1.5 * L_north)
        h = on_grid(f, lon, lat)
        χ = on_grid(g, lon, lat)
        t11, t21, t12, t22 = CA.calc_orographic_tensor(h, χ, lon, lat, R_e)
        dhdx, dhdy = exact_gradient(f, lon, lat)
        dχdx, dχdy = exact_gradient(g, lon, lat)
        exact = tensor_matrix(
            -dχdx .* dhdx, -dχdx .* dhdy, -dχdy .* dhdx, -dχdy .* dhdy, lat,
        )
        T = tensor_matrix(t11, t21, t12, t22, lat)
        @test T ≈ exact rtol = 0.02
        results[(φ0, orientation)] = T
    end
    T_ref = results[(0.0, :east_west)]
    for φ0 in (0.0, 60.0, 70.0, 80.0)
        T_ew = results[(φ0, :east_west)]
        T_ns = results[(φ0, :north_south)]
        # An east-west ridge has its slopes north-south, so t22 dominates; turned by 90°,
        # t11 dominates with the same value. The same ridge gives the same large entry at
        # every latitude (the small entries also feel how the local east direction turns
        # across a mountain near the pole, so they are not compared).
        @test abs(T_ew[2, 2]) > 5 * abs(T_ew[1, 1])
        @test abs(T_ns[1, 1]) > 5 * abs(T_ns[2, 2])
        @test T_ns[1, 1] ≈ T_ew[2, 2] rtol = 0.03
        @test T_ew[2, 2] ≈ T_ref[2, 2] rtol = 0.03
    end
end

@testset "OGW preprocessing: off-diagonal entries for a ridge turned by 45°" begin
    # A ridge at 45° to the grid lines with an analytic χ elongated along another axis.
    # Integrated over the mountain the tensor is symmetric (integrate by parts), so the
    # check is point by point: the fields t12 = −∂χ/∂y·∂h/∂x and t21 = −∂χ/∂x·∂h/∂y differ
    # there, and each must match its own exact field. Swapping t12 and t21, or the
    # derivative order, fails here.
    lon, lat0 = latlon_grid(0.05, 0.05, -90, 90)
    for φ0 in (0.0, 45.0, 70.0)
        lat = filter(φ -> abs(φ - φ0) < 5, lat0)
        f = gaussian_mountain(2000.0, 30.0, φ0, 200e3, 70e3, 45.0)
        g = gaussian_mountain(1.0e8, 30.0, φ0, 250e3, 120e3, 10.0)
        h = on_grid(f, lon, lat)
        χ = on_grid(g, lon, lat)
        t11, t21, t12, t22 = CA.calc_orographic_tensor(h, χ, lon, lat, R_e)
        dhdx, dhdy = exact_gradient(f, lon, lat)
        dχdx, dχdy = exact_gradient(g, lon, lat)
        t12_exact = -dχdy .* dhdx
        t21_exact = -dχdx .* dhdy
        # The two exact fields are far apart ...
        @test relerr(t21_exact, t12_exact) > 0.3
        # ... and each computed field matches its own.
        @test relerr(t12, t12_exact) < 0.02
        @test relerr(t21, t21_exact) < 0.02
        @test relerr(t11, -dχdx .* dhdx) < 0.02
        @test relerr(t22, -dχdy .* dhdy) < 0.02
    end
end

@testset "OGW preprocessing: full pipeline tensor is symmetric, definite, isotropic" begin
    # Round mountain: velocity potential from the windowed Hilbert transform, then the
    # tensor. Integrated over the mountain, −∫∇χ(∇h)ᵀ is symmetric and negative
    # semidefinite (the kernel 1/r has a positive Fourier transform), and for a round
    # mountain it is a multiple of the identity: t11 = t22, t12 = t21 = 0.
    lon, lat0 = latlon_grid(0.2, 0.2, -90, 90)
    for φ0 in (0.0, 60.0, 70.0, 80.0)
        lat = filter(φ -> abs(φ - φ0) < 8, lat0)
        f = gaussian_mountain(2000.0, 30.0, φ0, 150e3, 150e3)
        h = on_grid(f, lon, lat)
        χ = CA.calc_velocity_potential(
            copy(h), lon, lat, R_e; smoothing_length_scale = 600e3,
        )
        t11, t21, t12, t22 = CA.calc_orographic_tensor(h, χ, lon, lat, R_e)
        T = tensor_matrix(t11, t21, t12, t22, lat)
        scale = abs(T[1, 1]) + abs(T[2, 2])
        @test abs(T[1, 2] - T[2, 1]) < 0.02 * scale
        # A symmetric 2x2 matrix is negative definite when its trace is negative and its
        # determinant positive.
        S = (T + T') / 2
        @test S[1, 1] + S[2, 2] < 0
        @test S[1, 1] * S[2, 2] - S[1, 2]^2 > 0
        @test T[1, 1] ≈ T[2, 2] rtol = 0.05
        @test abs(T[1, 2]) < 0.02 * scale
    end
end

@testset "OGW preprocessing: longitude shift, including across the date line" begin
    # Moving the terrain by a whole number of grid columns must move hmax and the tensor
    # with it, exactly: nothing about the sphere depends on longitude. The second
    # mountain sits on the ±180° seam of the grid.
    lon, lat0 = latlon_grid(0.25, 0.25, -90, 90)
    lat = filter(φ -> abs(φ - 45) < 10, lat0)
    nlon = length(lon)
    shift = nlon ÷ 4
    @testset "mountain at $λ0" for λ0 in (30.0, 179.9)
        f = gaussian_mountain(2000.0, λ0, 45.0, 200e3, 100e3)
        bump = gaussian_mountain(500.0, λ0 + 1, 46.0, 40e3, 40e3)
        h = on_grid(f, lon, lat) .+ on_grid(bump, lon, lat)
        h_shifted = circshift(h, (shift, 0))
        stats(e) = begin
            L = 300e3
            hpoz =
                CA.calc_hpoz_latlon(copy(e), lon, lat, R_e; smoothing_length_scale = L)
            χ = CA.calc_velocity_potential(
                copy(e),
                lon,
                lat,
                R_e;
                smoothing_length_scale = L,
            )
            (hpoz, CA.calc_orographic_tensor(copy(e), χ, lon, lat, R_e)...)
        end
        a = stats(h)
        b = stats(h_shifted)
        for (x, y) in zip(a, b)
            @test circshift(x, (shift, 0)) ≈ y rtol = 1e-10 atol = 1e-12 * maximum(abs, x)
        end
    end
end

@testset "OGW preprocessing: subgrid height and the smoothing scale" begin
    # Sinusoidal terrain h = h₀ + A·sin(nλ) with n whole waves around the latitude circle,
    # near 40°N (where the latitude factor sin 40°/sin(max(20°, |lat|)) of the smoothing
    # scale is exactly 1). The raw subgrid height is the fourth root of the fourth moment
    # about the local mean:
    #   - window much wider than the wavelength: local mean = h₀, fourth moment of a sine =
    #     3A⁴/8, so h = (3/8)^(1/4)·A;
    #   - window of half-width s much narrower than the wavelength: the terrain is a
    #     straight line across the window, so h ≤ max|∂h/∂x|·s = A·k·s.
    # A larger smoothing fraction α (scale α·Δ) therefore gives a larger subgrid height.
    lon, lat0 = latlon_grid(0.1, 0.1, -90, 90)
    lat = filter(φ -> abs(φ - 40) < 2.5, lat0)
    j40 = argmin(abs.(lat .- 40))
    A = 300.0
    Δ = 200e3
    circle = 2π * R_e * cosd(40.0)
    subgrid_height(α, n_waves) = begin
        h = [1000 + A * sin(n_waves * deg2rad(λ)) for λ in lon, φ in lat]
        hpoz = CA.calc_hpoz_latlon(h, lon, lat, R_e; smoothing_length_scale = α * Δ)
        hpoz[:, j40]
    end
    # 60 km waves under a 200 km window
    h_wide = subgrid_height(1.0, round(Int, circle / 60e3))
    @test sum(h_wide) / length(h_wide) ≈ (3 / 8)^(1 / 4) * A rtol = 0.05
    # 1000 km waves under a 30 km window
    n_long = round(Int, circle / 1000e3)
    k = 2π * n_long / circle
    @test maximum(subgrid_height(0.15, n_long)) <= A * k * 0.15 * Δ
    # 300 km waves: the default α = 1 sees more of each wave than α = 0.15
    n_mid = round(Int, circle / 300e3)
    @test sum(subgrid_height(1.0, n_mid)) > 2 * sum(subgrid_height(0.15, n_mid))
end

@testset "OGW preprocessing: block size follows the model node spacing" begin
    # The 7.4 km grid compute_OGW_info builds from ETOPO (every 4th of 21600 × 10800 points).
    lat = collect(range(-90 + 1 / 30; step = 1 / 15, length = 2700))
    Δy = R_e * deg2rad(1 / 15)
    for (Δh, n_expected) in ((192e3, 25), (96e3, 12), (48e3, 6), (512e3, 60), (5e3, 1))
        n = CA.latlon_block_size(Δh, lat, 5400, 2700, R_e)
        @test n == n_expected
        @test 5400 % n == 0 && 2700 % n == 0
        @test n == 1 || n * Δy <= Δh
    end
end

@testset "OGW preprocessing: tensor and hpos are averaged over model cells" begin
    # Two mountains in a 12° latitude band on a 0.1° grid, averaged over 3° blocks (30
    # points, 333 km north-south, 236 km east-west at 45°):
    #   - an inland mountain in the middle of one block;
    #   - a coastal mountain off-center in another block, so the block center and the
    #     points around it are sea: point sampling at the center gives a zero tensor
    #     there, and no drag for any wind.
    # Block edges: longitude −180 + 3k, latitude 39 + 3k; block centers at −178.5 + 3k
    # and 40.5 + 3k.
    lon, lat0 = latlon_grid(0.1, 0.1, -90, 90)
    lat = filter(φ -> abs(φ - 45) < 6, lat0)
    n = 30
    γ = 0.4
    inland = gaussian_mountain(2000.0, 31.5, 46.5, 50e3, 50e3)
    coastal = gaussian_mountain(1500.0, 61.5 + 0.6, 43.5 + 0.6, 30e3, 30e3)
    # Sea floor at −100 m away from the mountains, clipped to 0 by the preprocessing.
    elev = on_grid((λ, φ) -> inland(λ, φ) + coastal(λ, φ) - 100.0, lon, lat)
    latlon_fields(block_size) = CA.compute_OGW_latlon(
        copy(elev),
        lon,
        lat,
        R_e,
        γ;
        smoothing_length_scale = 300e3,
        block_size,
    )
    point = latlon_fields(1)
    cell = latlon_fields(n)
    @test size(cell.t11) == (length(lon) ÷ n, length(lat) ÷ n)
    @test cell.lon ≈ collect(range(-178.5; step = 3, length = 120))
    @test cell.lat ≈ [40.5, 43.5, 46.5, 49.5]

    # Symmetric part S of a tensor; negative definite when trace < 0 and det > 0.
    sym(t11, t21, t12, t22) = (t11, (t12 + t21) / 2, t22)
    negdef(t11, t21, t12, t22) = begin
        (a, b, d) = sym(t11, t21, t12, t22)
        a + d < 0 && a * d - b^2 > 0
    end
    # A point tensor −∇χ (∇h)ᵀ has rank one: det S = −(∂χ/∂x ∂h/∂y − ∂χ/∂y ∂h/∂x)²/4 ≤ 0,
    # so no point is negative definite.
    scale = maximum(abs, point.t11) + maximum(abs, point.t22)
    det_point = map(point.t11, point.t21, point.t12, point.t22) do t11, t21, t12, t22
        (a, b, d) = sym(t11, t21, t12, t22)
        a * d - b^2
    end
    @test maximum(det_point) <= 1e-12 * scale^2
    # Over the whole mountain the cell mean is negative definite.
    i_inland, j_inland = findfirst(≈(31.5), cell.lon), findfirst(≈(46.5), cell.lat)
    @test negdef(
        cell.t11[i_inland, j_inland],
        cell.t21[i_inland, j_inland],
        cell.t12[i_inland, j_inland],
        cell.t22[i_inland, j_inland],
    )

    # Coastal block: the point tensor at the four grid points around the block center is
    # exactly zero, so interpolating point values gives no drag there. The cell mean is
    # nonzero and negative definite.
    i_coast, j_coast = findfirst(≈(61.5), cell.lon), findfirst(≈(43.5), cell.lat)
    ic = findall(λ -> abs(λ - 61.5) < 0.1, lon)
    jc = findall(φ -> abs(φ - 43.5) < 0.1, lat)
    @test length(ic) == 2 && length(jc) == 2
    for t in (point.t11, point.t21, point.t12, point.t22)
        @test all(iszero, t[ic, jc])
    end
    @test negdef(
        cell.t11[i_coast, j_coast],
        cell.t21[i_coast, j_coast],
        cell.t12[i_coast, j_coast],
        cell.t22[i_coast, j_coast],
    )

    # The block means conserve the area integral: Σ_blocks mean·(Σ cos φ over the block)
    # equals Σ_points value·cos φ, to rounding. The off-diagonal integrals nearly cancel
    # (round and coastal mountains), so the tolerance is relative to Σ |value|·cos φ.
    # hpos = max(0, h₀)^(2−γ) is averaged the same way (as GFDL remaps hpos).
    w_point = cosd.(lat)'
    w_cell =
        [n * sum(cosd.(lat[((jb - 1) * n + 1):(jb * n)])) for jb in 1:length(cell.lat)]'
    for name in (:hpos, :t11, :t12, :t21, :t22)
        x_point = getproperty(point, name)
        x_cell = getproperty(cell, name)
        @test sum(x_cell .* w_cell) ≈ sum(x_point .* w_point) atol =
            1e-10 * sum(abs.(x_point) .* w_point)
    end
    @test maximum(point.hpos) > 0
end

@testset "OGW preprocessing: computes in Float64 whatever the input float type" begin
    # The same Float32 inputs, passed once as Float32 and once converted to Float64, must
    # give the same Float64 fields, bit for bit: the preprocessing precision does not
    # depend on the model float type. The values are Float32 numbers, so both calls see
    # exactly the same input.
    lon64, lat0 = latlon_grid(0.2, 0.2, -90, 90)
    lat64 = filter(φ -> abs(φ - 45) < 6, lat0)
    mountain = gaussian_mountain(2000.0, 31.5, 46.5, 80e3, 50e3, 30.0)
    elev32 = Float32.(on_grid((λ, φ) -> mountain(λ, φ) - 100.0, lon64, lat64))
    lon32, lat32 = Float32.(lon64), Float32.(lat64)
    elev32_before = copy(elev32)
    fields(elev, lon, lat, R, γ, L) =
        CA.compute_OGW_latlon(
            elev, lon, lat, R, γ; smoothing_length_scale = L, block_size = 15,
        )
    from32 = fields(elev32, lon32, lat32, Float32(R_e), 0.4f0, 300.0f3)
    from64 = fields(
        Float64.(elev32), Float64.(lon32), Float64.(lat32), Float64(Float32(R_e)),
        Float64(0.4f0), 300.0e3,
    )
    @test elev32 == elev32_before
    for name in (:lon, :lat, :hpos, :t11, :t12, :t21, :t22)
        @test eltype(getproperty(from32, name)) == Float64
        @test getproperty(from32, name) == getproperty(from64, name)
    end
    @test maximum(from64.hpos) > 0

    # The TOFD band-pass variance on the lat-lon grid, likewise.
    lon_t = collect(range(1 / 240; step = 1 / 120, length = 1080))
    lat_t = collect(range(44.5 + 1 / 240; step = 1 / 120, length = 120))
    elev_t = Float32[
        300 * (1 + sin(2π * 100 * (i - 1) / 1080)) *
        (1 + 0.5 * cos(2π * 3 * (j - 1) / 120))
        for i in 1:1080, j in 1:120
    ]
    var32 = CA.calc_filtered_orography_variance_latlon(
        elev_t, Float32.(lon_t), Float32.(lat_t), Float32(R_e),
    )
    var64 = CA.calc_filtered_orography_variance_latlon(
        Float64.(elev_t), Float64.(Float32.(lon_t)), Float64.(Float32.(lat_t)),
        Float64(Float32(R_e)),
    )
    @test eltype(var32) == Float64
    @test var32 == var64
    @test maximum(var64) > 0
end

@testset "OGW preprocessing: hpos is converted to hmax after regridding" begin
    # Lat-lon input on a 4° grid (block centers at ..., 0, 4, 8, ...): hpos is H at and
    # north of the center a + 4 and 0 at and south of a, so model nodes between a and
    # a + 4 lie between a mountain block and a flat block (a is the band holding the most
    # nodes). The tensor is uniform. Linear interpolation of hpos, flat beyond the
    # outermost rows, is exactly H·clamp((φ − a)/4, 0, 1) at a node of latitude φ.
    #   - hmax must be the conversion of the interpolated hpos, as GFDL `topo_drag` does.
    #     The conversion hmax ∝ hpos^(1/(2−γ)) is concave, so interpolating hmax instead
    #     gives a lower value between the two blocks: (φ − a)/4 · hmax(H).
    #   - The tensor is zero at nodes poleward of 88° (the cubed sphere with an even
    #     h_elem has nodes at the poles) and unchanged elsewhere.
    FT = Float64
    γ, h_frac = 0.4, 0.1
    H = 1.0e4
    hspace = CommonSpaces.CubedSphereSpace(FT; radius = R_e, h_elem = 4, n_quad_points = 4)
    lon, lat = latlon_grid(4.0, 4.0, -90, 90)
    node_lat = vec(Array(parent(Fields.coordinate_field(hspace).lat)))
    in_band(a) = findall(φ -> a + 0.5 < φ < a + 3, node_lat)
    a = argmax(a -> length(in_band(a)), 0.0:4.0:40.0)
    hpos_ll = [φ >= a + 3.5 ? H : 0.0 for _ in lon, φ in lat]
    c = (γ + 2) / (2γ) * (1 - h_frac^(2γ)) / (1 - h_frac^(γ + 2))
    hmax_of(hpos) = (hpos * c)^(1 / (2 - γ))
    latlon_info = (;
        lon,
        lat,
        hpos = hpos_ll,
        t11 = fill(-1.0, length(lon), length(lat)),
        t12 = fill(0.25, length(lon), length(lat)),
        t21 = fill(0.25, length(lon), length(lat)),
        t22 = fill(-2.0, length(lon), length(lat)),
    )
    topo = CA.regrid_OGW_latlon(hspace, latlon_info, γ, h_frac)
    node_values(name) = vec(Array(parent(getproperty(topo, name))))
    hmax, hmin = node_values(:hmax), node_values(:hmin)

    hpos_expected = H .* clamp.((node_lat .- a) ./ 4, 0, 1)
    @test hmax ≈ hmax_of.(hpos_expected) rtol = 1e-8 atol = 1e-8 * hmax_of(H)
    @test hmin ≈ h_frac .* hmax
    between = in_band(a)
    @test length(between) > 10
    hmax_interpolated = (node_lat[between] .- a) ./ 4 .* hmax_of(H)
    @test all(hmax[between] .> 1.05 .* hmax_interpolated)

    polar = abs.(node_lat) .> 88
    @test count(polar) > 0
    for (name, value) in ((:t11, -1.0), (:t12, 0.25), (:t21, 0.25), (:t22, -2.0))
        t = node_values(name)
        @test all(iszero, t[polar])
        @test t[.!polar] ≈ fill(value, count(.!polar)) rtol = 1e-12
    end
end

@testset "OGW preprocessing: drag files carry a build record" begin
    # A small synthetic drag field (h_elem = 2, Float32, as a Float32 run stores it) is
    # written with write_computed_drag! and read back. Copies of the file with the build
    # record changed must be refused with a message naming the file and how to fix it.
    FT = Float32
    h_elem = 2
    hspace = CommonSpaces.CubedSphereSpace(FT; radius = R_e, h_elem, n_quad_points = 4)
    ctx = ClimaComms.context(hspace)
    ᶜlat = Fields.coordinate_field(hspace).lat
    drag = fill(
        (; t11 = FT(0), t12 = FT(0), t21 = FT(0), t22 = FT(0), hmin = FT(0), hmax = FT(0)),
        hspace,
    )
    @. drag.t11 = -1 - cosd(ᶜlat)
    @. drag.t12 = FT(0.25) * sind(ᶜlat)
    @. drag.t21 = FT(0.25) * sind(ᶜlat)
    @. drag.t22 = -2 - cosd(ᶜlat)^2
    @. drag.hmax = 100 * cosd(ᶜlat)^2
    @. drag.hmin = FT(0.1) * drag.hmax
    shape = (; γ = 0.4, h_frac = 0.1, α_smoothing = 1.0)

    dir = mktempdir()
    filename = cd(dir) do
        CA.write_computed_drag!(
            drag,
            ctx;
            topography = "Earth",
            topo_smoothing = false,
            topography_damping_factor = 1,
            h_elem,
            shape...,
            elevation_source = "synthetic",
        )
    end
    path = joinpath(dir, "$(filename).hdf5")
    HDF5 = InputOutput.HDF5
    stamp = HDF5.h5open(path) do file
        attributes = HDF5.attrs(file)
        Dict(key => attributes[key] for key in keys(attributes))
    end
    @test stamp["ogw_preprocessing_version"] == CA.OGW_PREPROCESSING_VERSION
    @test stamp["h_elem"] == h_elem
    @test stamp["gamma"] == 0.4
    @test stamp["h_frac"] == 0.1
    @test stamp["alpha_smoothing"] == 1.0
    @test stamp["field_float_type"] == "Float32"
    @test stamp["compute_float_type"] == "Float64"
    @test stamp["elevation_source"] == "synthetic"

    # The matching run loads the file without a warning, with the float types logged at
    # the debug level only.
    loaded = @test_logs min_level = Logging.Warn CA.read_ogw_drag_file(
        path, ctx; h_elem, shape...,
    )
    for name in (:hmax, :hmin, :t11, :t12, :t21, :t22)
        @test parent(getproperty(loaded, name)) == parent(getproperty(drag, name))
    end
    @test_logs (:debug, "Orographic drag file float types") min_level = Logging.Debug match_mode =
        :any CA.read_ogw_drag_file(path, ctx; h_elem, shape...)
    # Float32 run parameters against the Float64 stamp: still no warning.
    @test_logs min_level = Logging.Warn CA.read_ogw_drag_file(
        path, ctx; h_elem, γ = 0.4f0, h_frac = 0.1f0, α_smoothing = 1.0f0,
    )

    # A run with another γ loads the file but warns.
    @test_logs (:warn, r"γ = 0.5 \(file 0.4\)") CA.read_ogw_drag_file(
        path, ctx; h_elem, shape..., γ = 0.5,
    )
    # A run at another resolution is refused.
    @test_throws "built for h_elem = 2, but the run has h_elem = 3" CA.read_ogw_drag_file(
        path, ctx; h_elem = 3, shape...,
    )

    # Copies of the file with another preprocessing version, and with no stamp.
    function copy_with(change!)
        copy_path = joinpath(dir, "copy_$(rand(UInt32)).hdf5")
        cp(path, copy_path)
        HDF5.h5open(change!, copy_path, "r+")
        return copy_path
    end
    newer = copy_with() do file
        HDF5.delete_attribute(file, "ogw_preprocessing_version")
        HDF5.write_attribute(
            file,
            "ogw_preprocessing_version",
            CA.OGW_PREPROCESSING_VERSION + 1,
        )
    end
    expected = "built with preprocessing version $(CA.OGW_PREPROCESSING_VERSION + 1), \
                but this code expects version $(CA.OGW_PREPROCESSING_VERSION)"
    @test_throws expected CA.read_ogw_drag_file(newer, ctx; h_elem, shape...)
    @test_throws "raw_topo_online" CA.read_ogw_drag_file(newer, ctx; h_elem, shape...)
    @test_throws newer CA.read_ogw_drag_file(newer, ctx; h_elem, shape...)
    unstamped = copy_with() do file
        HDF5.delete_attribute(file, "ogw_preprocessing_version")
    end
    @test_throws "has no build record" CA.read_ogw_drag_file(
        unstamped, ctx; h_elem, shape...,
    )
end

# ======================================================================================
# 6. One step on an Earth-topography sphere
# ======================================================================================

#=
One orographic gravity-wave step on a coarse sphere with Earth topography.

Builds a small simulation (`h_elem = 4`, 10 levels) with `topography: "Earth"` and
`orographic_gravity_wave: "raw_topo"`, which loads the shipped `ogw_computed_drag_h4` drag
file (checking its build record), and builds the TOFD coefficient from the Earth elevation data, sets a uniform 10 m/s westerly and
2 m/s southerly wind (the model starts at rest, which gives no drag), and calls the OGW
compute step once. It checks the wiring end to end: the drag is finite, nonzero, within
the limiter, and on balance opposes the wind. The single-column physics is checked
against GFDL `topo_drag` in section 3 of this file.

Takes about 3 minutes on one CPU, almost all of it the simulation build.
=#

@testset "OGW step on an Earth-topography sphere" begin
    output_dir = mktempdir()
    config = CA.AtmosConfig(
        Dict(
            "h_elem" => 4,
            "z_elem" => 10,
            "z_max" => 30000.0,
            "z_stretch" => false,
            "dt" => "400secs",
            "t_end" => "400secs",
            "topography" => "Earth",
            "topo_smoothing" => true,
            "orographic_gravity_wave" => "raw_topo",
            "output_default_diagnostics" => false,
            "output_dir" => output_dir,
        );
        job_id = "ogw_earth_sphere",
    )
    simulation = CA.get_simulation(config)
    Y = simulation.integrator.u
    p = simulation.integrator.p
    FT = eltype(Y.c.ρ)

    u_wind, v_wind = FT(10), FT(2)
    ᶜlg = Fields.local_geometry_field(Y.c)
    Y.c.uₕ .= Geometry.Covariant12Vector.(Geometry.UVVector.(u_wind, v_wind), ᶜlg)
    CA.orographic_gravity_wave_compute_tendency!(
        Y,
        p,
        p.atmos.orographic_gravity_wave,
    )

    uf = Array(parent(p.orographic_gravity_wave.ᶜuforcing))
    vf = Array(parent(p.orographic_gravity_wave.ᶜvforcing))
    @test all(isfinite, uf) && all(isfinite, vf)
    @test count(!iszero, uf) > 0
    # The limiter caps the drag vector at 3e-3·√2 m/s².
    @test maximum(sqrt.(uf .^ 2 .+ vf .^ 2)) <= FT(3e-3) * sqrt(FT(2)) * (1 + sqrt(eps(FT)))
    # Summed over the globe, the drag decelerates the imposed wind.
    @test sum(uf .* u_wind .+ vf .* v_wind) < 0
    # TOFD uses the Earth elevation data, so its coefficient is nonzero somewhere.
    @test maximum(parent(p.orographic_gravity_wave.ᶜtofd_Cd)) > 0
end
