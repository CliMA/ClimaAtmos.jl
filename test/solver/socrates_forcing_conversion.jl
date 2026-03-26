using Test
import NCDatasets as NC

include(joinpath(
    @__DIR__,
    "..",
    "..",
    "calibration",
    "experiments",
    "SOCRATES",
    "model_interface.jl",
))

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

@testset "SOCRATES forcing conversion guards" begin
    @testset "pressure_to_height produces monotonic levels" begin
        pressure_levels = [97500.0, 90000.0, 80000.0]
        temperature_profile = [285.0, 280.0, 275.0]
        z = pressure_to_height(pressure_levels, temperature_profile, 98420.0)
        @test length(z) == 3
        @test all(isfinite, z)
        @test z[2] > z[1]
        @test z[3] > z[2]
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
end