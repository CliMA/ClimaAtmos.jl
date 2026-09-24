#=
Unit tests for the Beljaars (2004) turbulent orographic form drag (TOFD).

Calls `calc_tofd_forcing!` on a single column with constant wind and a flat
surface (z_sfc = 0) so the drag profile has a closed-form target:

  du/dt = -Cd(z) * |U| * u,
  Cd(z) = a_tofd * K * (0.5 * hmax)^2 * exp(-(z/1500)^1.5) * z^-1.2,
  K = alpha*beta*Ccorr*Cmd*2.109*Cavar,
  Cavar = k1^(n1-n2) / (Cih * kflt^n1),

matching the IFS / SURFEX sso_beljaars04 implementation. Also checks that the
term opposes the wind, decays with height, and is bit-for-bit zero when disabled.
=#

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: CommonSpaces, Fields, Grids, Spaces, Utilities

const half = Utilities.half

# Reference Beljaars (2004) drag coefficient, independent of the source under test.
function beljaars_Cd(z, hmax, a_tofd)
    c_alpha, c_beta, c_cmd, c_cor = 12.0, 1.0, 0.005, 0.6
    c_ih, c_kflt, c_k1 = 0.00102, 0.00035, 0.003
    c_n1, c_n2 = -1.9, -2.8
    c_avar = c_k1^(c_n1 - c_n2) / (c_ih * c_kflt^c_n1)
    K = c_alpha * c_beta * c_cor * c_cmd * 2.109 * c_avar
    σ = 0.5 * hmax
    zc = max(1.0, z)
    return a_tofd * K * σ^2 * exp(-(zc / 1500)^1.5) * zc^(-1.2)
end

@testset "OGW TOFD: Beljaars profile, direction, decay, disable" begin
    FT = Float64
    z_max = FT(20_000)
    z_elem = 80
    u0 = FT(8)
    v0 = FT(6)          # speed = 10
    speed = sqrt(u0^2 + v0^2)
    hmax = FT(500)
    a_tofd = FT(1)

    center_space = CommonSpaces.ColumnSpace(
        FT;
        z_min = 0,
        z_max = z_max,
        z_elem,
        staggering = Grids.CellCenter(),
    )
    face_space = Spaces.face_space(center_space)

    c_z = Fields.coordinate_field(center_space).z
    u = Fields.ones(center_space) .* u0
    v = Fields.ones(center_space) .* v0
    uforcing = Fields.zeros(center_space)
    vforcing = Fields.zeros(center_space)

    surf_c = Spaces.level(center_space, 1)
    z_sfc = Fields.zeros(surf_c)          # flat surface at z = 0
    hmax_f = Fields.ones(surf_c) .* hmax

    CA.calc_tofd_forcing!(uforcing, vforcing, u, v, c_z, z_sfc, hmax_f, a_tofd)

    uf = vec(parent(uforcing))
    vf = vec(parent(vforcing))
    zc = vec(parent(c_z))

    # Matches the analytic Beljaars profile at every level.
    for i in eachindex(uf)
        Cd = beljaars_Cd(zc[i], hmax, a_tofd)
        @test uf[i] ≈ -Cd * speed * u0 rtol = 1e-10
        @test vf[i] ≈ -Cd * speed * v0 rtol = 1e-10
    end

    # Drag opposes the wind (u, v > 0 here) and keeps the wind direction.
    @test all(uf .< 0)
    @test all(vf .< 0)
    @test all(abs.(uf .* v0 .- vf .* u0) .< 1e-12 .* abs.(uf))

    # Magnitude decays monotonically with height.
    @test all(diff(abs.(uf)) .< 0)

    # Concentrated near the surface: most of the column-integrated drag is low.
    total = sum(abs.(uf))
    low = sum(abs.(uf[zc .< 1500]))
    @test low > 0.8 * total

    # Disabling (a_tofd = 0) leaves the forcing bit-for-bit zero.
    uforcing0 = Fields.zeros(center_space)
    vforcing0 = Fields.zeros(center_space)
    CA.calc_tofd_forcing!(uforcing0, vforcing0, u, v, c_z, z_sfc, hmax_f, FT(0))
    @test iszero(maximum(abs.(uforcing0)))
    @test iszero(maximum(abs.(vforcing0)))
end
