#=
Unit tests for the turbulent orographic form drag (TOFD) of Beljaars et al. (2004), in the
IFS form (IFS Documentation CY47R1, Part IV, Sections 3.4 and 11.3).

- `tofd_drag_coefficient!` on a single column, passing the bottom face level as the
  surface height (as the OGW cache does), against the closed form in the height above
  ground z_agl = z - z_sfc:

    Cd(z) = a_tofd * alpha*beta*Cmd*Ccorr*2.109 * exp(-(z_agl/1500)^1.5) * a2 * z_agl^-1.2,
    a2 = sigma_flt^2 * k1^(n1-n2) / (I_H * kflt^n1),   alpha = 35.

- The IFS smoothing operator and the band-pass-filtered orography variance that defines
  sigma_flt^2, against the exact discrete transfer function for a sinusoidal orography.
- `tofd_implicit_rate`: the backward-Euler rate reduces to the explicit drag for weak
  damping and can never reverse the wind within one step for strong damping.
=#

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: CommonSpaces, Fields, Grids, Spaces, Utilities

const half = Utilities.half

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

    for i in eachindex(cd)
        @test cd[i] ≈ ifs_tofd_Cd(zc[i], σ²_flt, a_tofd) rtol = 1e-10
    end
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
    for i in eachindex(cd_hi)
        @test cd_hi[i] ≈ ifs_tofd_Cd(zc_hi[i] - z_ground, σ²_flt, a_tofd) rtol = 1e-10
    end
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
    # Equatorial strip at 30 arc seconds, periodic in longitude over 9° (~1000 km).
    dlon = FT(1 / 120)
    lon = collect(range(dlon / 2; step = dlon, length = 1080))
    lat = collect(range(-0.5 + dlon / 2; step = dlon, length = 120))
    x = R .* deg2rad.(lon)
    L = R * deg2rad(dlon) * length(lon)
    A = FT(300)

    # Exact discrete transfer function of the longitude pass at the equator.
    function transfer(Δ, k)
        w = CA.ifs_smoothing_weights(R * deg2rad(dlon), FT(Δ), FT(1_000), length(lon) ÷ 8)
        J = (length(w) - 1) ÷ 2
        return sum(w[m + J + 1] * cos(k * m * R * deg2rad(dlon)) for m in (-J):J)
    end

    interior = 20:100
    for n_waves in (20, 100, 400)   # wavelengths of 50, 10 and 2.5 km
        k = 2π * n_waves / L
        elev = [A * (1 + sin(k * xi)) for xi in x, _ in lat]   # positive: no clipping
        variance = CA.calc_filtered_orography_variance_latlon(elev, lon, lat, R)
        expected = A^2 / 2 * (transfer(2_000, k) - transfer(20_000, k))^2
        @test sum(variance[:, interior]) / length(variance[:, interior]) ≈ expected rtol =
            1e-3
    end
    # The 10 km wave is inside the band and keeps most of its variance.
    k10 = 2π * 100 / L
    @test (transfer(2_000, k10) - transfer(20_000, k10))^2 > 0.5

    # Uniform orography and ocean (clipped to zero) have no filtered variance.
    @test maximum(
        CA.calc_filtered_orography_variance_latlon(fill(FT(1500), 1080, 120), lon, lat, R),
    ) <
          1e-12
    @test iszero(
        CA.calc_filtered_orography_variance_latlon(fill(FT(-3000), 1080, 120), lon, lat, R),
    )

    # Area-weighted block average: a field equal to cos φ averages to Σcos²φ / Σcos φ.
    lat_g = collect(range(-87.5; step = 5, length = 36))
    lon_g = collect(range(2.5; step = 5, length = 72))
    f = [cosd(φ) for _ in lon_g, φ in lat_g]
    lon_c, lat_c, f_c = CA.block_average_latlon(f, lon_g, lat_g, 6)
    @test size(f_c) == (12, 6)
    @test lon_c[1] ≈ sum(lon_g[1:6]) / 6
    rows = 1:6
    @test f_c[1, 1] ≈ sum(cosd.(lat_g[rows]) .^ 2) / sum(cosd.(lat_g[rows]))
    @test all(f_c[:, 3] .≈ f_c[1, 3])
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
