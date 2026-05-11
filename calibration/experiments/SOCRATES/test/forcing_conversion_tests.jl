using Test
import NCDatasets as NC
import ClimaUtilities: TimeVaryingInputs

include(joinpath(@__DIR__, "..", "model_interface.jl"))

function write_synthetic_raw_forcing(path::String)
    lev = Float64[97500.0, 90000.0, 80000.0]
    tsec = Float64[0.0, 3600.0]

    temperature = [285.0 284.0; 280.0 279.0; 275.0 274.0]
    u = fill(5.0, 3, 2)
    v = fill(-1.0, 3, 2)
    q = fill(0.003, 3, 2)
    omega = fill(-0.02, 3, 2)
    divt = zeros(3, 2)
    divq = zeros(3, 2)
    ps = fill(98420.0, 2)
    tg = fill(280.0, 2)

    NC.NCDataset(path, "c") do ds
        NC.defDim(ds, "lev", length(lev))
        NC.defDim(ds, "time", length(tsec))

        NC.defVar(ds, "lev", lev, ("lev",))
        NC.defVar(ds, "tsec", tsec, ("time",))
        NC.defVar(ds, "T", temperature, ("lev", "time"))
        NC.defVar(ds, "u", u, ("lev", "time"))
        NC.defVar(ds, "v", v, ("lev", "time"))
        NC.defVar(ds, "q", q, ("lev", "time"))
        NC.defVar(ds, "omega", omega, ("lev", "time"))
        NC.defVar(ds, "divT", divt, ("lev", "time"))
        NC.defVar(ds, "divq", divq, ("lev", "time"))
        NC.defVar(ds, "Ps", ps, ("time",))
        NC.defVar(ds, "Tg", tg, ("time",))
    end
    return nothing
end

function write_synthetic_raw_forcing(path::String, tsec::Vector{Float64})
    lev = Float64[97500.0, 90000.0, 80000.0]
    nt = length(tsec)

    # Profiles vary smoothly in time so interpolation behavior can be checked explicitly.
    temperature = zeros(Float64, length(lev), nt)
    for (j, t) in enumerate(tsec)
        hours = t / 3600.0
        temperature[:, j] .= [285.0, 280.0, 275.0] .- 0.25 * hours
    end
    u = fill(5.0, length(lev), nt)
    v = fill(-1.0, length(lev), nt)
    q = fill(0.003, length(lev), nt)
    omega = fill(-0.02, length(lev), nt)
    divt = zeros(length(lev), nt)
    divq = zeros(length(lev), nt)
    ps = fill(98420.0, nt)
    tg = fill(280.0, nt)

    NC.NCDataset(path, "c") do ds
        NC.defDim(ds, "lev", length(lev))
        NC.defDim(ds, "time", nt)

        NC.defVar(ds, "lev", lev, ("lev",))
        NC.defVar(ds, "tsec", tsec, ("time",))
        NC.defVar(ds, "T", temperature, ("lev", "time"))
        NC.defVar(ds, "u", u, ("lev", "time"))
        NC.defVar(ds, "v", v, ("lev", "time"))
        NC.defVar(ds, "q", q, ("lev", "time"))
        NC.defVar(ds, "omega", omega, ("lev", "time"))
        NC.defVar(ds, "divT", divt, ("lev", "time"))
        NC.defVar(ds, "divq", divq, ("lev", "time"))
        NC.defVar(ds, "Ps", ps, ("time",))
        NC.defVar(ds, "Tg", tg, ("time",))
    end
    return nothing
end

@testset "SOCRATES forcing conversion guards" begin
    @testset "pressure_to_height produces monotonic levels" begin
        pressure_levels = [97500.0, 90000.0, 80000.0]
        temperature_profile = [285.0, 280.0, 275.0]
        z = pressure_to_height(pressure_levels, temperature_profile, 98420.0)
        @test length(z) == 3
        @test all(isfinite, z)
        @test z[2] > z[1]
        @test z[3] > z[2]
        @test z[1] < 200.0
    end

    @testset "pressure_to_height is order-robust" begin
        p_desc = [97500.0, 90000.0, 80000.0, 70000.0]
        T_desc = [285.0, 280.0, 275.0, 270.0]
        z_desc = pressure_to_height(p_desc, T_desc, 98420.0)

        p_asc = reverse(p_desc)
        T_asc = reverse(T_desc)
        z_asc = pressure_to_height(p_asc, T_asc, 98420.0)

        @test all(isfinite, z_desc)
        @test all(isfinite, z_asc)
        @test z_desc ≈ reverse(z_asc) atol = 1e-6
    end

    @testset "is_valid_converted_forcing_file catches collapsed z" begin
        mktempdir() do tmp
            bad_path = joinpath(tmp, "bad.nc")
            NC.NCDataset(bad_path, "c") do ds
                NC.defDim(ds, "z", 3)
                NC.defVar(ds, "z", Float32, ("z",))
                ds["z"][:] = Float32[0, 0, 0]
            end
            @test !is_valid_converted_forcing_file(bad_path)
        end
    end

    @testset "convert_socrates_forcing_file writes valid z" begin
        mktempdir() do tmp
            raw_path = joinpath(tmp, "raw.nc")
            out_path = joinpath(tmp, "converted.nc")
            write_synthetic_raw_forcing(raw_path)

            convert_socrates_forcing_file(raw_path, out_path, "20100101")
            @test is_valid_converted_forcing_file(out_path)

            NC.NCDataset(out_path, "r") do ds
                z = vec(ds["z"][:])
                @test length(unique(z)) > 1
                @test maximum(z) > minimum(z)
            end
        end
    end

    @testset "reader helpers preserve expected dimensions" begin
        mktempdir() do tmp
            raw_path = joinpath(tmp, "raw_dims.nc")
            write_synthetic_raw_forcing(raw_path)

            NC.NCDataset(raw_path, "a") do ds
                NC.defDim(ds, "x", 1)
                NC.defDim(ds, "y", 1)
                t2d = Array(ds["T"][:])
                ps1d = vec(ds["Ps"][:])

                NC.defVar(ds, "T4", Float64, ("x", "y", "lev", "time"))
                ds["T4"][1, 1, :, :] = t2d

                NC.defVar(ds, "Ps3", Float64, ("x", "y", "time"))
                ds["Ps3"][1, 1, :] = ps1d
            end

            NC.NCDataset(raw_path, "r") do ds
                prof = read_input_profile_zt(ds, "T4")
                ts = read_input_timeseries_t(ds, "Ps3")
                @test size(prof) == (3, 2)
                @test length(ts) == 2
                @test all(isfinite, prof)
                @test all(isfinite, ts)
            end
        end
    end
end

# ---------------------------------------------------------------------------
# parse_duration_seconds
# ---------------------------------------------------------------------------
@testset "parse_duration_seconds" begin
    @test parse_duration_seconds(0.0) == 0.0
    @test parse_duration_seconds(10) == 10.0
    @test parse_duration_seconds("0secs") == 0.0
    @test parse_duration_seconds("10secs") == 10.0
    @test parse_duration_seconds("10s") == 10.0
    @test parse_duration_seconds("10seconds") == 10.0
    @test parse_duration_seconds("60mins") == 3600.0
    @test parse_duration_seconds("2minutes") == 120.0
    @test parse_duration_seconds("14hours") ≈ 50400.0
    @test parse_duration_seconds("1hr") == 3600.0
    @test parse_duration_seconds("1day") == 86400.0
    @test parse_duration_seconds("2days") == 172800.0
    @test_throws Exception parse_duration_seconds("bad_unit")
    @test_throws Exception parse_duration_seconds("nonnumber secs")
end

# ---------------------------------------------------------------------------
# parse_datetime_flexible
# ---------------------------------------------------------------------------
@testset "parse_datetime_flexible" begin
    expected = Dates.DateTime(2010, 1, 1, 0, 0, 0)
    @test parse_datetime_flexible("2010-01-01T00:00:00") == expected
    @test parse_datetime_flexible("2010-01-01 00:00:00") == expected
    @test parse_datetime_flexible("2010-01-01T00:00") == expected
    @test parse_datetime_flexible("2010-01-01 00:00") == expected
    @test parse_datetime_flexible("2010-01-01") == expected

    expected2 = Dates.DateTime(2020, 6, 15, 12, 30, 0)
    @test parse_datetime_flexible("2020-06-15T12:30:00") == expected2
    @test parse_datetime_flexible("2020-06-15 12:30") == expected2

    @test_throws Exception parse_datetime_flexible("not-a-date")
    @test_throws Exception parse_datetime_flexible("Jan 1 2010")
end

# ---------------------------------------------------------------------------
# forcing_time_unit_to_seconds
# ---------------------------------------------------------------------------
@testset "forcing_time_unit_to_seconds" begin
    @test forcing_time_unit_to_seconds("days") == 86400.0
    @test forcing_time_unit_to_seconds("day") == 86400.0
    @test forcing_time_unit_to_seconds("hours") == 3600.0
    @test forcing_time_unit_to_seconds("hour") == 3600.0
    @test forcing_time_unit_to_seconds("hr") == 3600.0
    @test forcing_time_unit_to_seconds("minutes") == 60.0
    @test forcing_time_unit_to_seconds("mins") == 60.0
    @test forcing_time_unit_to_seconds("seconds") == 1.0
    @test forcing_time_unit_to_seconds("s") == 1.0
    @test_throws Exception forcing_time_unit_to_seconds("fortnights")
end

# ---------------------------------------------------------------------------
# linear_interp_profile: verify clamping at boundaries and in-bounds linearity
# ---------------------------------------------------------------------------
@testset "linear_interp_profile boundary behavior" begin
    z_src = Float64[100.0, 500.0, 1000.0, 5000.0]
    y_src = Float64[0.0, 1.0, 2.0, 3.0]

    # Exact grid points
    @test linear_interp_profile(z_src, y_src, 100.0)  ≈ 0.0 atol = 1e-10
    @test linear_interp_profile(z_src, y_src, 500.0)  ≈ 1.0 atol = 1e-10
    @test linear_interp_profile(z_src, y_src, 5000.0) ≈ 3.0 atol = 1e-10

    # Interior midpoints
    @test linear_interp_profile(z_src, y_src, 300.0)  ≈ 0.5 atol = 1e-10
    @test linear_interp_profile(z_src, y_src, 750.0)  ≈ 1.5 atol = 1e-10

    # Below minimum: must clamp, not extrapolate (no NaN, no blow-up)
    @test linear_interp_profile(z_src, y_src, 0.0)   == y_src[1]
    @test linear_interp_profile(z_src, y_src, -500.0) == y_src[1]

    # Above maximum: must clamp (no NaN, no blow-up)
    @test linear_interp_profile(z_src, y_src, 10000.0) == y_src[end]
    @test linear_interp_profile(z_src, y_src, 1e6)     == y_src[end]

    # All results finite regardless of query
    for z_query in [-1000.0, 0.0, 50.0, 300.0, 750.0, 5000.0, 1e5]
        @test isfinite(linear_interp_profile(z_src, y_src, z_query))
    end
end

# ---------------------------------------------------------------------------
# pressure_to_height sensitivity to R_d and grav parameters
# ---------------------------------------------------------------------------
@testset "pressure_to_height respects R_d and grav from ClimaAtmosParameters" begin
    p = [97500.0, 90000.0, 80000.0]
    T = [285.0, 280.0, 275.0]
    ps = 98420.0

    # Default keyword args should use CA.ClimaAtmosParameters internally.
    z_default = pressure_to_height(p, T, ps)
    @test all(isfinite, z_default)
    @test z_default[1] < z_default[2] < z_default[3]

    # Explicitly passing the same values should give identical results.
    R_d_ca = CA.Parameters.R_d(CA.ClimaAtmosParameters(Float64))
    grav_ca = CA.Parameters.grav(CA.ClimaAtmosParameters(Float64))
    z_explicit = pressure_to_height(p, T, ps; R_d = R_d_ca, grav = grav_ca)
    @test z_default ≈ z_explicit atol = 1e-10

    # Higher R_d → larger scale height → higher z everywhere (above surface)
    z_high_R = pressure_to_height(p, T, ps; R_d = R_d_ca * 1.1, grav = grav_ca)
    @test all(z_high_R[2:end] .> z_default[2:end])

    # Higher grav → smaller scale height → lower z everywhere (above surface)
    z_high_g = pressure_to_height(p, T, ps; R_d = R_d_ca, grav = grav_ca * 1.1)
    @test all(z_high_g[2:end] .< z_default[2:end])
end

# ---------------------------------------------------------------------------
# Converted file: time axis is DateTime-decodable by NCDatasets (as ClimaUtilities does)
# ---------------------------------------------------------------------------
@testset "converted forcing time axis decodes to DateTime via NCDatasets" begin
    mktempdir() do tmp
        raw_path = joinpath(tmp, "raw.nc")
        out_path = joinpath(tmp, "converted.nc")
        write_synthetic_raw_forcing(raw_path)
        convert_socrates_forcing_file(raw_path, out_path, "20100101")

        NC.NCDataset(out_path, "r") do ds
            # NCDatasets auto-decodes CF time ("days/hours since ...") as DateTimeStandard
            times = ds["time"][:]
            @test length(times) == 2  # write_synthetic_raw_forcing has tsec=[0,3600]
            # First time point: 2010-01-01 00:00 (tsec=0)
            @test Dates.DateTime(times[1]) == Dates.DateTime(2010, 1, 1, 0, 0, 0)
            # Second time point: 2010-01-01 01:00 (tsec=3600)
            @test Dates.DateTime(times[2]) == Dates.DateTime(2010, 1, 1, 1, 0, 0)
            # Sequence is strictly increasing
            @test Dates.DateTime(times[2]) > Dates.DateTime(times[1])
        end
    end
end

# ---------------------------------------------------------------------------
# validate_forcing_time_support: happy path and out-of-range rejection
# ---------------------------------------------------------------------------
@testset "validate_forcing_time_support" begin
    mktempdir() do tmp
        forcing_path = joinpath(tmp, "forcing_tvi.nc")
        # 29 hourly time points: 0 to 28h, starting 2010-01-01
        tsec_full = collect(0.0:3600.0:100800.0)  # 29 points
        time_axis = build_time_axis("20100101", tsec_full)
        NC.NCDataset(forcing_path, "c") do ds
            NC.defDim(ds, "time", length(time_axis))
            NC.defVar(ds, "time", time_axis, ("time",))
        end

        # Model runs 0 to 14h (50400s): fully inside 0–28h forcing window
        cfg_ok = Dict(
            "external_forcing_file" => forcing_path,
            "start_date" => "20100101",
            "t_end" => "50400secs",
        )
        @test_nowarn validate_forcing_time_support(cfg_ok)

        # t_end pushes beyond forcing support (29h > 28h)
        cfg_long = Dict(
            "external_forcing_file" => forcing_path,
            "start_date" => "20100101",
            "t_end" => "104400secs",   # 29h
        )
        @test_throws Exception validate_forcing_time_support(cfg_long)

        # start_date two days later: model window starts entirely outside forcing file
        cfg_late = Dict(
            "external_forcing_file" => forcing_path,
            "start_date" => "20100103",
            "t_end" => "3600secs",
        )
        @test_throws Exception validate_forcing_time_support(cfg_late)
    end
end

# ---------------------------------------------------------------------------
# Forcing z-interpolation to model levels produces finite, physical values
# This exercises the same logic ClimaUtilities uses at model runtime.
# ---------------------------------------------------------------------------
@testset "forcing z-interpolation to model z levels is finite and physical" begin
    mktempdir() do tmp
        raw_path = joinpath(tmp, "raw.nc")
        out_path = joinpath(tmp, "converted.nc")
        write_synthetic_raw_forcing(raw_path)
        convert_socrates_forcing_file(raw_path, out_path, "20100101")

        NC.NCDataset(out_path, "r") do forcing_ds
            z_src = Float64.(vec(forcing_ds["z"][:]))
            ta_profile = read_forcing_profile_series(forcing_ds, "ta")  # (z, time)

            # Model z levels: below forcing grid, spanning forcing grid, above forcing grid
            z_below = range(0.0, z_src[1] * 0.5; length = 4)
            z_within = range(z_src[1], z_src[end]; length = 10)
            z_above  = range(z_src[end], z_src[end] * 1.5; length = 4)
            z_model  = sort(unique(vcat(collect(z_below), collect(z_within), collect(z_above))))

            for t_idx in axes(ta_profile, 2)
                profile = view(ta_profile, :, t_idx)
                interped = [linear_interp_profile(z_src, profile, Float64(z)) for z in z_model]
                @test all(isfinite, interped)
                # Clamping: below-grid queries must not extrapolate below sfc value
                @test all(v -> v >= minimum(profile), interped)
                # Clamping: above-grid queries must not extrapolate above top value
                @test all(v -> v <= maximum(profile), interped)
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Time-interpolation coverage: model simulation times map inside forcing window
# This is the in-code equivalent of what TimeVaryingInput checks at runtime.
# ---------------------------------------------------------------------------
@testset "model simulation times lie within forcing file DateTime support" begin
    mktempdir() do tmp
        raw_path = joinpath(tmp, "raw.nc")
        out_path = joinpath(tmp, "converted.nc")
        write_synthetic_raw_forcing(raw_path)
        convert_socrates_forcing_file(raw_path, out_path, "20100101")

        start_dt = Dates.DateTime(2010, 1, 1, 0, 0, 0)
        t_end_sec = 3600.0   # 1h — write_synthetic_raw_forcing spans exactly 1h

        NC.NCDataset(out_path, "r") do ds
            times = ds["time"][:]
            forcing_start = Dates.DateTime(times[1])
            forcing_end   = Dates.DateTime(times[end])

            # Model time window expressed as absolute datetimes
            model_start = start_dt
            model_end   = start_dt + Dates.Second(round(Int, t_end_sec))

            @test forcing_start <= model_start
            @test forcing_end   >= model_end

            # Evaluating at N evenly-spaced model times: all within forcing support
            model_times = [start_dt + Dates.Second(round(Int, s))
                           for s in range(0.0, t_end_sec; length = 20)]
            for mt in model_times
                @test forcing_start <= mt <= forcing_end
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Runtime-style time interpolation object evaluation over full simulation times
# This directly evaluates the same TimeVaryingInput interpolation type used by
# ClimaAtmos forcing callbacks.
# ---------------------------------------------------------------------------
@testset "TimeVaryingInput evaluates across simulation domain" begin
    mktempdir() do tmp
        raw_path = joinpath(tmp, "raw_long.nc")
        out_path = joinpath(tmp, "converted_long.nc")

        # 29 hourly points, 0h..28h, so a 14h model run sits fully inside support.
        tsec_full = collect(0.0:3600.0:100800.0)
        write_synthetic_raw_forcing(raw_path, tsec_full)
        convert_socrates_forcing_file(raw_path, out_path, "20100101")

        NC.NCDataset(out_path, "r") do ds
            times_dt = Dates.DateTime.(ds["time"][:])
            t0 = times_dt[1]
            times = Float64.(Dates.value.(times_dt .- t0) ./ 1000)
            ta_zt = read_forcing_profile_series(ds, "ta")
            vals = vec(ta_zt[1, :])

            input = TimeVaryingInputs.TimeVaryingInput(
                times,
                vals;
                method = TimeVaryingInputs.LinearInterpolation(),
            )

            sim_times = Float64.(600 .* collect(0:84))  # 14h at 10 min cadence

            dest = zeros(Float64, 1)
            samples = Float64[]
            for t in sim_times
                TimeVaryingInputs.evaluate!(dest, input, t)
                @test isfinite(dest[1])
                push!(samples, dest[1])
            end

            # Synthetic temperature decreases linearly in time; interpolated values should be non-increasing.
            @test all(diff(samples) .<= 1e-10)

            # Out-of-domain evaluation should throw with default Throw() extrapolation.
            @test_throws ErrorException TimeVaryingInputs.evaluate!(
                dest,
                input,
                35.0 * 3600.0,
            )
        end
    end
end

# ---------------------------------------------------------------------------
# Integration test: real SSCF forcing file used with TimeVaryingInput at
# ClimaAtmos simulation times.
# ---------------------------------------------------------------------------
@testset "real SSCF forcing on ClimaAtmos times" begin
    case = Dict("flight_number" => 1, "forcing_type" => "obs_data")
    forcing_file = get_socrates_forcing_file(case, "20100101")
    @test isfile(forcing_file)

    cfg = Dict(
        "external_forcing_file" => forcing_file,
        "start_date" => "20100101",
        "t_start" => "0secs",
        "t_end" => "14hours",
    )
    @test_nowarn validate_forcing_time_support(cfg)

    NC.NCDataset(forcing_file, "r") do ds
        times_dt = Dates.DateTime.(ds["time"][:])
        times_s = Float64.(Dates.value.(times_dt .- times_dt[1]) ./ 1000)

        ta_zt = read_forcing_profile_series(ds, "ta")
        vals = vec(ta_zt[1, :])

        input = TimeVaryingInputs.TimeVaryingInput(
            times_s,
            vals;
            method = TimeVaryingInputs.LinearInterpolation(),
        )

        # ClimaAtmos default SOCRATES timing: 14h with diagnostics every 10 minutes.
        sim_times = Float64.(600 .* collect(0:84))
        dest = zeros(Float64, 1)
        for t in sim_times
            TimeVaryingInputs.evaluate!(dest, input, t)
            @test isfinite(dest[1])
        end
    end
end
