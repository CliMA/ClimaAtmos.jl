#=
Unit tests for the Beljaars (2004) turbulent orographic form drag (TOFD).

Calls `tofd_drag_coefficient!` on a single column with a flat surface (z_sfc = 0)
so the coefficient profile has a closed-form target:

  Cd(z) = a_tofd * K * (0.5 * hmax)^2 * exp(-(z/1500)^1.5) * z^-1.2,
  K = alpha*beta*Ccorr*Cmd*2.109*Cavar,
  Cavar = k1^(n1-n2) / (Cih * kflt^n1),

matching the IFS / SURFEX sso_beljaars04 implementation. Also checks that the
profile decays with height, is zero when disabled, and that the backward-Euler
rate `tofd_implicit_rate` reduces to the explicit drag for weak damping and can
never reverse the wind within one step for strong damping.
=#

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: CommonSpaces, Fields, Grids, Spaces

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

@testset "OGW TOFD: Beljaars coefficient profile" begin
    FT = Float64
    hmax = FT(500)
    a_tofd = FT(1)

    center_space = CommonSpaces.ColumnSpace(
        FT;
        z_min = 0,
        z_max = FT(20_000),
        z_elem = 80,
        staggering = Grids.CellCenter(),
    )
    c_z = Fields.coordinate_field(center_space).z
    surf_c = Spaces.level(center_space, 1)
    z_sfc = Fields.zeros(surf_c)          # flat surface at z = 0
    hmax_f = Fields.ones(surf_c) .* hmax

    Cd = Fields.zeros(center_space)
    CA.tofd_drag_coefficient!(Cd, c_z, z_sfc, hmax_f, a_tofd)
    cd = vec(parent(Cd))
    zc = vec(parent(c_z))

    for i in eachindex(cd)
        @test cd[i] ≈ beljaars_Cd(zc[i], hmax, a_tofd) rtol = 1e-10
    end
    @test all(cd .> 0)
    @test all(diff(cd) .< 0)
    @test sum(cd[zc .< 1500]) > 0.8 * sum(cd)

    # The coefficient is linear in a_tofd and vanishes when TOFD is disabled.
    Cd_half = Fields.zeros(center_space)
    CA.tofd_drag_coefficient!(Cd_half, c_z, z_sfc, hmax_f, FT(0.5))
    @test vec(parent(Cd_half)) ≈ 0.5 .* cd rtol = 1e-12
    Cd0 = Fields.ones(center_space)
    CA.tofd_drag_coefficient!(Cd0, c_z, z_sfc, hmax_f, FT(0))
    @test iszero(maximum(abs.(parent(Cd0))))
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

    for Cd in (FT(0), FT(1e-7), FT(1e-4), FT(1e-2), FT(1)), U in (FT(0), FT(1), FT(30))
        rate = CA.tofd_implicit_rate(Cd, U, dt)
        @test 0 <= rate * dt < 1
        @test rate <= Cd * U
    end
end
