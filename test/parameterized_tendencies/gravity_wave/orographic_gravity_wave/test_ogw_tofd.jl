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
