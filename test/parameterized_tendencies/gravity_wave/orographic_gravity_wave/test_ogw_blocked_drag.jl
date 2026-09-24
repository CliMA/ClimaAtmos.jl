#=
Unit tests for the Garner/GFDL blocked (non-propagating) orographic drag.

Calls `calc_nonpropagating_forcing!` on a single column with constant N and V_tau
so the WKB phase and the column-stress integral have closed-form targets:

  - z_ref is the first face where Sigma Delta z * N/V_tau exceeds pi (local
    layer thickness, not the full height above the PBL at every face).
  - Sigma (du/dt) * Delta p / g = tau_x * tau_np / tau_l
    (wtsum = Sigma Delta p * weight).
=#

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: CommonSpaces, Fields, Grids, Spaces, Utilities

const half = Utilities.half

@testset "OGW blocked drag: phase depth and column stress" begin
    FT = Float64
    z_max = FT(20_000)
    z_elem = 80
    z_pbl = FT(1_000)
    N = FT(0.01)          # inside the [0.7e-2, 1.7e-2] clamp
    Vtau = FT(10)
    grav = FT(9.81)
    p0 = FT(1e5)
    H = FT(8_000)

    tau_x = FT(1)
    tau_y = FT(0)
    tau_l = FT(1)
    tau_np = FT(0.5)

    center_space = CommonSpaces.ColumnSpace(
        FT;
        z_min = 0,
        z_max = z_max,
        z_elem,
        staggering = Grids.CellCenter(),
    )
    face_space = Spaces.face_space(center_space)

    c_z = Fields.coordinate_field(center_space).z
    f_z = Fields.coordinate_field(face_space).z
    f_dz = Fields.Δz_field(face_space)
    uforcing = Fields.zeros(center_space)
    vforcing = Fields.zeros(center_space)
    mask = Fields.Field(Bool, center_space)
    weights = Fields.zeros(center_space)
    diffs = Fields.zeros(center_space)

    f_N = Fields.ones(face_space) .* N
    f_Vtau = Fields.ones(face_space) .* Vtau
    f_p = @. p0 * exp(-f_z / H)

    surf_c = Spaces.level(center_space, 1)
    surf_f = Spaces.level(face_space, half)
    tau_x_f = Fields.ones(surf_c) .* tau_x
    tau_y_f = Fields.ones(surf_c) .* tau_y
    tau_l_f = Fields.ones(surf_c) .* tau_l
    tau_np_f = Fields.ones(surf_c) .* tau_np
    f_z_pbl = Fields.ones(surf_f) .* z_pbl
    f_z_ref = Fields.zeros(surf_f)
    f_p_ref = Fields.zeros(surf_f)
    wtsum = Fields.zeros(surf_c)

    p_bottom = Fields.level(f_p, half)
    dz_bottom = Fields.level(f_dz, half)
    p_extrap = @. p_bottom * exp(dz_bottom / H)
    f_p_m1 = similar(f_p)
    CA.field_shiftface_down!(f_p, f_p_m1, p_extrap)

    CA.calc_nonpropagating_forcing!(
        uforcing,
        vforcing,
        tau_x_f,
        tau_y_f,
        tau_l_f,
        tau_np_f,
        f_Vtau,
        f_z_pbl,
        f_z_ref,
        f_p_ref,
        mask,
        weights,
        diffs,
        wtsum,
        f_p,
        f_p_m1,
        f_N,
        f_z,
        f_dz,
        grav,
    )

    # First face above z_pbl + pi * V / N, i.e. the first face where
    # Sigma Delta z * N/V > pi. The inverted (z - z_pbl) accumulation
    # would reach pi much lower.
    z_ref = Fields.maximum(f_z_ref)
    expected_depth = FT(pi) * Vtau / N
    dz = z_max / z_elem
    @test z_ref ≈ z_pbl + ceil(expected_depth / dz) * dz
    @test z_ref > z_pbl + expected_depth
    # Old bug: phase += (z - z_pbl)*N/V at every face reaches pi far lower, so
    # z_ref would sit well below z_pbl + pi V/N (~3141 m). The fixed depth here is
    # ceil(pi V/N / dz)*dz = 3250 m, comfortably above this discriminating bound.
    @test z_ref > z_pbl + FT(3_000)

    # Column-integrated blocked stress: Sigma (du/dt) Delta p / g = tau_x tau_np / tau_l.
    column_stress = sum(parent(uforcing) .* parent(diffs)) / grav
    @test column_stress ≈ tau_x * tau_np / tau_l rtol = 1e-6
    @test iszero(maximum(abs.(vforcing)))
    @test maximum(wtsum) > 0
    # Forcing is confined to [z_pbl, z_ref) and is not clamp-saturated.
    @test maximum(abs.(uforcing)) < FT(1e-3)
    u = parent(uforcing)
    zc = parent(c_z)
    for i in eachindex(u)
        if zc[i] < z_pbl - dz || zc[i] > z_ref
            @test abs(u[i]) < 10 * eps(FT)
        end
    end
end
