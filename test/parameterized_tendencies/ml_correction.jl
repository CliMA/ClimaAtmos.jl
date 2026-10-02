using Test
import Random
import Dates
import ClimaAtmos as CA

Random.seed!(1234)

@testset "approx_erf and gelu_erf" begin
    # erf(x) from tables, to 10 digits
    for (x, erf_x) in (
        (0.0, 0.0),
        (0.1, 0.1124629160),
        (0.5, 0.5204998778),
        (1.0, 0.8427007929),
        (2.0, 0.9953222650),
        (3.5, 0.9999992569),
    )
        @test CA.approx_erf(x) ≈ erf_x atol = 2e-7
        @test CA.approx_erf(-x) ≈ -erf_x atol = 2e-7
    end
    @test CA.approx_erf(1.0f0) isa Float32
    @test CA.gelu_erf(0.0) == 0
    @test CA.gelu_erf(1.0) ≈ 0.5 * (1 + 0.6826894921) atol = 1e-7 # erf(1/√2)
    @test CA.gelu_erf(-10.0) ≈ 0 atol = 1e-12
end

@testset "horizontal neighbor operations" begin
    nlon, nlat, L = 7, 5, 3
    a = rand(nlon, nlat, L)
    for s in (-9, -2, 0, 1, 3, 7)
        @test CA.roll_lon(a, s) == circshift(a, (s, 0, 0))
    end
    for s in (-6, -2, 0, 1, 4, 8)
        expected = [a[i, clamp(j + s, 1, nlat), k] for i in 1:nlon, j in 1:nlat, k in 1:L]
        @test CA.shift_lat_nearest(a, s) == expected
    end

    r = 1
    expected_box = [
        sum(
            a[mod1(i + di, nlon), clamp(j + dj, 1, nlat), k] for di in (-r):r,
            dj in (-r):r
        ) for i in 1:nlon, j in 1:nlat, k in 1:L
    ]
    @test CA.box_sum(a, r) ≈ expected_box

    m = rand(nlon, nlat, L) .> 0.3
    expected_mask = [
        m[i, j, k] && m[mod1(i - 1, nlon), j, k] && m[mod1(i + 1, nlon), j, k] &&
            (j == 1 || m[i, j - 1, k]) && (j == nlat || m[i, j + 1, k]) for
        i in 1:nlon, j in 1:nlat, k in 1:L
    ]
    @test CA.erode_mask(m, 1) == expected_mask
    @test CA.erode_mask(m, 2) == CA.erode_mask(expected_mask, 1)

    x = rand(nlon, nlat, L, 2)
    xnb = CA.neighbor_anomaly(x, m, r)
    mf = Float64.(m)
    den = CA.box_sum(mf, r)
    for v in 1:2
        mean_v = CA.box_sum(x[:, :, :, v] .* mf, r) ./ max.(den, 1e-6)
        @test xnb[:, :, :, v] ≈ (x[:, :, :, v] .- mean_v) .* mf
    end
    @test all(iszero, xnb[.!m, :])

    # A constant field is unchanged by smoothing where valid, and zero elsewhere
    c = fill(2.5, nlon, nlat, L)
    smoothed = CA.gaussian_smooth(c, m, 1.0)
    @test smoothed[m] ≈ c[m]
    @test all(iszero, smoothed[.!m])
    @test CA.gaussian_smooth(x, m, 0.0) === x
end

# Column-by-column reference of `column_net_forward!`
function naive_forward(member, f, mm, dilations)
    N, L, _ = size(f)
    T = size(member.out_w, 2)
    y = zeros(N, L, T)
    gelu = CA.gelu_erf
    for n in 1:N
        mask = mm[n, :]
        h = (f[n, :, :] * member.inp_w .+ member.inp_b[1, 1, :]') .* mask
        for (b, d) in enumerate(dilations)
            pool = vec(sum(h .* mask; dims = 1)) ./ max(sum(mask), 1)
            ctx = member.ctx_w[:, :, b]' * pool .+ member.ctx_b[1, :, b]
            z = zeros(L, size(h, 2))
            for l in 1:L
                z[l, :] .= member.conv_b[1, 1, :, b] .+ ctx
                for k in 1:3
                    src = l + (k - 2) * d
                    1 <= src <= L || continue
                    z[l, :] .+= member.conv_w[:, :, k, b]' * h[src, :]
                end
            end
            h = (h .+ gelu.(z) * member.mix_w[:, :, b] .+ member.mix_b[1, 1, :, b]') .* mask
        end
        y[n, :, :] = (h * member.out_w .+ member.out_b[1, 1, :]') .* mask
    end
    return y
end

@testset "column_net_forward! matches a column-by-column reference" begin
    N, L, fin, W, T = 4, 9, 5, 6, 2
    dilations = [1, 2, 4]
    nb = length(dilations)
    rnd(dims...) = randn(dims...) ./ sqrt(W)
    member = CA.ColumnNetMember(
        rnd(fin, W), rnd(1, 1, W),
        rnd(W, W, 3, nb), rnd(1, 1, W, nb),
        rnd(W, W, nb), rnd(1, 1, W, nb),
        rnd(W, W, nb), rnd(1, W, nb),
        rnd(W, T), rnd(1, 1, T),
        zeros(1, 1, fin - 1), ones(1, 1, fin - 1), ones(1, 1, T), 1e-3,
    )
    f = randn(N, L, fin)
    mm = Float64.(rand(N, L) .> 0.25)
    mm[1, :] .= 0  # a fully masked column
    buffers = (;
        h = zeros(N, L, W), z = zeros(N, L, W), s = zeros(N, L, W),
        pool = zeros(N, 1, W),
    )
    y = zeros(N, L, T)
    CA.column_net_forward!(y, member, f, mm, dilations, buffers)
    @test y ≈ naive_forward(member, f, mm, dilations)
    @test all(iszero, y[1, :, :])
end

@testset "cos_zenith_noaa" begin
    lat = [-89.5, 0.5, 45.5]
    lon = [-179.5, 0.5, 90.5]
    cz = CA.cos_zenith_noaa(Dates.DateTime(2020, 6, 21, 12), lat, lon)
    @test size(cz) == (length(lon), length(lat))
    @test all(c -> -1 <= c <= 1, cz)
    # Near local noon at the June solstice the sun is ~23.4° north of the equator
    @test cz[2, 3] ≈ cosd(45.5 - 23.44) atol = 0.02
    @test cz[2, 1] < 0  # polar night
end

@testset "ml_correction_ramp" begin
    FT = Float32
    mlc(start, ramp) =
        CA.MLTendencyCorrection{FT, Nothing}(
            nothing,
            0.5,
            3600,
            true,
            true,
            1e-4,
            1e-7,
            0,
            1e4,
            5e3,
            start,
            ramp,
        )
    @test CA.ml_correction_ramp(0.0, mlc(0, 0)) == 1
    @test CA.ml_correction_ramp(100.0, mlc(600, 0)) == 0
    @test CA.ml_correction_ramp(700.0, mlc(600, 0)) == 1
    @test CA.ml_correction_ramp(900.0, mlc(600, 600)) ≈ 0.5
    @test CA.ml_correction_ramp(5000.0, mlc(600, 600)) == 1
    @test CA.ml_correction_ramp(900.0, mlc(600, 600)) isa FT
end

@testset "ml_correction_pressure_taper" begin
    taper(p) = CA.ml_correction_pressure_taper(p, 1.0e4, 5.0e3)
    @test taper(1.0e5) == 1
    @test taper(1.0e4) == 1
    @test taper(7.5e3) ≈ 0.5
    @test taper(5.0e3) == 0
    @test taper(1.0e2) == 0
    @test issorted(taper.(range(4.0e3, 1.1e4; length = 50)))
    @test CA.ml_correction_pressure_taper(7.5f3, 1.0f4, 5.0f3) isa Float32
end

@testset "get_ml_correction_model" begin
    FT = Float32
    parsed_args = CA.AtmosConfig(Dict(); job_id = "ml_correction_test").parsed_args
    @test isnothing(CA.get_ml_correction_model(parsed_args, FT))

    path, io = mktemp()
    close(io)
    args = merge(
        parsed_args,
        Dict(
            "ml_correction" => path,
            "ml_correction_variables" => "tq",
            "ml_correction_q_cap" => 3.6e-4,
            "ml_correction_dt" => "30mins",
            "ml_correction_ramp" => "1hours",
        ),
    )
    mlc = CA.get_ml_correction_model(args, FT)
    @test mlc isa CA.MLTendencyCorrection{FT, String}
    @test mlc.correct_temperature && mlc.correct_humidity
    @test mlc.q_cap ≈ FT(1e-7)
    @test mlc.t_cap ≈ FT(0.5 / 3600)
    @test (mlc.p_full, mlc.p_zero) == (FT(1e4), FT(5e3))
    @test_throws ErrorException CA.get_ml_correction_model(
        merge(args, Dict("ml_correction_p_zero" => 2.0e4)), FT,
    )
    @test mlc.refresh_period == 1800
    @test mlc.ramp_time == 3600
    @test !CA.get_ml_correction_model(
        merge(args, Dict("ml_correction_variables" => "q")), FT,
    ).correct_temperature
    @test_throws ErrorException CA.get_ml_correction_model(
        merge(args, Dict("ml_correction_variables" => "u")), FT,
    )
    @test_throws ErrorException CA.get_ml_correction_model(
        merge(args, Dict("ml_correction" => path * ".missing")), FT,
    )
end

# Parity with the training code on its exported reference case. The network file
# is large and lives outside the repository, so this runs only when
# `CLIMAATMOS_ML_CORRECTION_NET` points to one.
if haskey(ENV, "CLIMAATMOS_ML_CORRECTION_NET")
    import NCDatasets
    @testset "column_net_predict matches the exported reference" begin
        FT = Float32
        path = ENV["CLIMAATMOS_ML_CORRECTION_NET"]
        net = CA.load_column_net(path, FT)
        ref = NCDatasets.NCDataset(path) do ds
            (;
                x = FT.(Array(ds["ref_x"])),
                ps = FT.(Array(ds["ref_ps"])),
                cz = FT.(Array(ds["ref_cz"])),
                mask = Array(ds["ref_mask"]) .!= 0,
                pred = FT.(Array(ds["ref_pred"])),
            )
        end
        @test CA.column_net_mask(net, ref.ps) == ref.mask
        pred, _ = CA.column_net_predict(net, ref.x, ref.ps, ref.cz; members = 1:1)
        for i in eachindex(net.targets)
            expected = ref.pred[:, :, :, i, 1]
            @test maximum(abs, pred[:, :, :, i] .- expected) <=
                  1e-3 * maximum(abs, expected)
        end
    end
end
