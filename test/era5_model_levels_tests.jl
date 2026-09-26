#=
Unit tests for the ERA5 model-level to altitude-level conversion.

These build a small synthetic file in the shape CDS delivers rather than
downloading anything, so they are fast and need no credentials.
=#

using Test
import NCDatasets
import ClimaAtmos as CA

const NLON = 8
const NLAT = 5
const NLEV = 137
const SP = 101325.0f0
const T0 = 288.0f0

"""
Write a file shaped like an `era5_raw_*.nc`: model levels running 1 at the
top to `NLEV` at the surface, an isothermal dry column, and flat terrain.
"""
function write_fake_raw(path)
    NCDatasets.NCDataset(path, "c") do ds
        NCDatasets.defDim(ds, "longitude", NLON)
        NCDatasets.defDim(ds, "latitude", NLAT)
        NCDatasets.defDim(ds, "model_level", NLEV)
        NCDatasets.defDim(ds, "valid_time", 1)

        NCDatasets.defVar(ds, "longitude", collect(range(-180.0, 157.5, length = NLON)), ("longitude",))
        # Decreasing, the way ERA5 stores it
        NCDatasets.defVar(ds, "latitude", collect(range(90.0, -90.0, length = NLAT)), ("latitude",))
        NCDatasets.defVar(ds, "model_level", collect(1:NLEV), ("model_level",))
        NCDatasets.defVar(ds, "valid_time", [0], ("valid_time",))

        dims4 = ("longitude", "latitude", "model_level", "valid_time")
        for (name, value) in
            (("t", T0), ("q", 0.0f0), ("u", 10.0f0), ("v", -5.0f0), ("w", 0.5f0))
            NCDatasets.defVar(ds, name, fill(value, NLON, NLAT, NLEV, 1), dims4)
        end

        dims3 = ("longitude", "latitude", "valid_time")
        NCDatasets.defVar(ds, "sp", fill(SP, NLON, NLAT, 1), dims3)
        NCDatasets.defVar(ds, "skt", fill(290.0f0, NLON, NLAT, 1), dims3)
        NCDatasets.defVar(ds, "surface_geopotential", zeros(Float32, NLON, NLAT, 1), dims3)
    end
    return path
end

@testset "ERA5 model levels to altitude levels" begin
    mktempdir() do dir
        source = write_fake_raw(joinpath(dir, "era5_raw_fake.nc"))
        target = joinpath(dir, "out.nc")
        levels = 0.0:500.0:20000.0

        CA.to_z_levels_3d_model(source, target, levels, Float32)

        NCDatasets.NCDataset(target) do ds
            @test size(ds["t"]) == (NLON, NLAT, length(levels))
            @test Array(ds["z"]) ≈ collect(levels)
            # Latitude has to come out increasing for SpaceVaryingInput
            @test issorted(Array(ds["lat"]))
            @test Array(ds["lon"]) ≈ Array(NCDatasets.NCDataset(source)["longitude"])

            p = Array(ds["p_3d"])
            # Pressure falls with altitude in every column
            @test all(issorted(p[i, j, :]; rev = true) for i in 1:NLON, j in 1:NLAT)
            # The lowest level sits at the surface, so it is near sp
            @test all(isapprox.(p[:, :, 1], SP; rtol = 0.02))
            # An isothermal dry column has a known scale height
            scale_height = 287.06f0 * T0 / 9.81f0
            expected = SP .* exp.(-Float32.(levels) ./ scale_height)
            @test all(isapprox.(p[1, 1, :], expected; rtol = 0.05))

            @test all(Array(ds["t"]) .≈ T0)
            @test all(Array(ds["q"]) .>= 0)
            @test all(Array(ds["u"]) .≈ 10.0f0)
            @test all(Array(ds["v"]) .≈ -5.0f0)

            # w is discarded unless asked for, even though the source has it
            @test all(iszero, Array(ds["w"]))

            # Surface fields are broadcast over z until the reader takes 2D
            @test all(Array(ds["skt"]) .≈ 290.0f0)
            @test all(iszero, Array(ds["z_sfc"]))
            @test all(isapprox.(Array(ds["p"]), SP))
        end

        target_w = joinpath(dir, "out_w.nc")
        CA.to_z_levels_3d_model(source, target_w, levels, Float32; interp_w = true)
        NCDatasets.NCDataset(target_w) do ds
            w = Array(ds["w"])
            # Positive omega is downward, so a geometric w must be negative
            @test all(w[:, :, 1] .< 0)
            @test NCDatasets.attribs(ds["w"])["units"] == "m s-1"
        end
    end

    @testset "rejects pressure-level input" begin
        mktempdir() do dir
            path = joinpath(dir, "plev.nc")
            NCDatasets.NCDataset(path, "c") do ds
                NCDatasets.defDim(ds, "pressure_level", 3)
                NCDatasets.defVar(ds, "pressure_level", [1000.0, 500.0, 100.0], ("pressure_level",))
            end
            @test_throws ErrorException CA.to_z_levels_3d_model(
                path, joinpath(dir, "o.nc"), 0.0:100.0:1000.0, Float32,
            )
        end
    end
end
