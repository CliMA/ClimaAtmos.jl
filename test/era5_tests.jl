#=
ERA5 forcing file generation tests for ClimaAtmos.jl

These tests verify the generation and processing of ERA5 observational data
for single column model runs.
=#

using Test
using ClimaComms
ClimaComms.@import_required_backends
import Dates
using Random
Random.seed!(1234)
import ClimaAtmos as CA
using NCDatasets

include("test_helpers.jl")

#####
##### ERA5 variable name mapping
#####

@testset "clima_to_era5_name_dict" begin
    d = CA.clima_to_era5_name_dict()
    @test d["ua"] == "u"
    @test d["va"] == "v"
    @test d["ta"] == "t"
    @test d["hus"] == "q"
    @test d["ts"] == "skt"
end

#####
##### ERA5 forcing file generation
#####

@testset "ERA5 single day forcing file generation" begin
    FT = Float64
    parsed_args = Dict(
        "start_date" => "20000506",
        "site_latitude" => 0.0,
        "site_longitude" => 0.0,
        "t_end" => "5hours",
        "era5_diurnal_warming" => Nothing,
    )

    temporary_dir = mktempdir()
    sim_forcing_daily = CA.get_external_daily_forcing_file_path(
        parsed_args,
        data_dir = temporary_dir,
    )

    @test basename(sim_forcing_daily) ==
          "tv_forcing_0.0_0.0_20000506_20000506.nc"

    sim_forcing_monthly = CA.get_external_monthly_forcing_file_path(
        parsed_args,
        data_dir = temporary_dir,
    )

    @test basename(sim_forcing_monthly) ==
          "monthly_diurnal_cycle_forcing_0.0_0.0_20000506.nc"

    # Create mock datasets
    create_mock_era5_datasets(temporary_dir, parsed_args["start_date"], FT)

    # Generate forcing file - identical up to file name for single day and monthly forcing files
    time_resolution = FT(3600)
    CA.generate_external_forcing_file(
        parsed_args,
        sim_forcing_daily,
        FT,
        smooth_amount = 4,
        time_resolution = time_resolution,
        input_data_dir = temporary_dir,
    )

    processed_data = NCDataset(sim_forcing_daily, "r")

    # Test fixed variables - this tests that the variables are copied correctly
    for clima_var in ["ua", "va", "wap", "ts"]
        @test all(isapprox.(processed_data[clima_var][:], 1, atol = 1e-10))
    end

    for clima_var in ["clw", "cli"]
        @test all(isapprox.(processed_data[clima_var][:], 0, atol = 1e-10))
    end

    # data is stored from top of atmosphere to surface 
    @test monotonic_decreasing(processed_data["zg"], 3)
    @test monotonic_increasing(processed_data["ta"], 3)
    @test monotonic_increasing(processed_data["hus"], 3)
    @test all(processed_data["hus"] .>= 0)
    @test all(processed_data["ta"] .>= 200) # 200 K is the minimum temperature set in the helper function

    # Test accumulated variables - note that the sign is flipped because of differences between ecmwf and clima
    for clima_var in ["hfls", "hfss"]
        @test all(
            isapprox.(
                processed_data[clima_var][:],
                -1 / time_resolution,
                atol = 1e-10,
            ),
        )
    end

    # Test gradient variables (should be zero for uniform data)
    gradient_vars = ["tnhusha", "tntha"]
    for var in gradient_vars
        @test all(isapprox.(processed_data[var][:], 0, atol = 1e-10))
    end

    # Test coszen variable
    @test all(x -> x >= 0 && x <= 1, processed_data["coszen"][:])

    # Test time check
    @test CA.check_daily_forcing_times(sim_forcing_daily, parsed_args)

    # The monthly diurnal case data and processing happen exactly the 
    # same as single day files (just the source files are different).
    # So we can test the monthly time check in the same way.
    @test CA.check_monthly_forcing_times(sim_forcing_daily, parsed_args)

    close(processed_data)
end

@testset "ERA5 multiday forcing file generation" begin
    FT = Float64
    parsed_args = Dict(
        "start_date" => "20000506",
        "site_latitude" => 0.0,
        "site_longitude" => 0.0,
        "t_end" => "2days",
        "era5_diurnal_warming" => Nothing,
    )

    input_dir = mktempdir()
    output_dir = mktempdir()
    sim_forcing = CA.get_external_daily_forcing_file_path(
        parsed_args,
        data_dir = output_dir,
    )

    # Create mock datasets for multiple days
    start_date = Dates.DateTime(parsed_args["start_date"], "yyyymmdd")
    end_time =
        start_date + Dates.Second(CA.time_to_seconds(parsed_args["t_end"]))
    days_needed = Dates.value(Dates.Day(end_time - start_date)) + 1  # Add 1 to include partial end day

    # Use a common base date for all datasets to avoid time concatenation issues
    base_date = "20000101"

    for day_offset in 0:(days_needed - 1)
        current_date = start_date + Dates.Day(day_offset)
        date_str = Dates.format(current_date, "yyyymmdd")
        create_mock_era5_datasets(
            input_dir,
            date_str,
            FT;
            base_date = base_date,
        )
    end

    # Generate multiday forcing file
    time_resolution = FT(3600)
    CA.generate_multiday_era5_external_forcing_file(
        parsed_args,
        sim_forcing,
        FT,
        time_resolution = time_resolution,
        input_data_dir = input_dir,
        output_data_dir = output_dir,
    )

    # Test the generated file
    processed_data = NCDataset(sim_forcing, "r")

    # Should have days_needed * 24 hours per day time steps
    expected_time_steps = days_needed * 24
    @test length(processed_data["time"][:]) == expected_time_steps

    # Test that data is consistent across time
    @test monotonic_increasing(processed_data["ta"], 3)
    @test all(
        x -> all(isapprox.(x, -1 / time_resolution, atol = 1e-10)),
        processed_data["hfls"][:],
    )

    # Test time check
    @test CA.check_daily_forcing_times(sim_forcing, parsed_args)

    # check the vertical tendency function - useful if we implement steady ERA5 forcing
    vert_partial_ds = Dict(
        "ta" => processed_data["ta"][1, 1, :, :],
        "wa" => processed_data["wa"][1, 1, :, :],
        "hus" => processed_data["hus"][1, 1, :, :],
        # need to set z to not be all zeros
        "z" =>
            collect(1:length(processed_data["z"][:])) .* processed_data["z"][:],
    )
    # compute the vertical temperature gradient
    vertical_temperature_gradient =
        CA.get_vertical_tendencies(vert_partial_ds, "ta")
    @test all(vertical_temperature_gradient .<= 0)

    close(processed_data)
end

@testset "get_horizontal_tendencies" begin
    # Test horizontal advective tendency calculation using finite differences
    FT = Float64
    temporary_dir = mktempdir()
    
    # Create mock dataset with known horizontal gradients
    test_data_path = joinpath(temporary_dir, "test_horizontal.nc")
    ds = NCDataset(test_data_path, "c")
    
    nlat, nlon, npres, ntime = 11, 11, 3, 2
    defDim(ds, "longitude", nlon)
    defDim(ds, "latitude", nlat)
    defDim(ds, "pressure_level", npres)
    defDim(ds, "valid_time", ntime)
    
    defVar(ds, "longitude", FT, ("longitude",))
    defVar(ds, "latitude", FT, ("latitude",))
    defVar(ds, "u", FT, ("longitude", "latitude", "pressure_level", "valid_time"))
    defVar(ds, "v", FT, ("longitude", "latitude", "pressure_level", "valid_time"))
    defVar(ds, "t", FT, ("longitude", "latitude", "pressure_level", "valid_time"))
    defVar(ds, "q", FT, ("longitude", "latitude", "pressure_level", "valid_time"))
    
    # Create grid centered at equator
    ds["longitude"][:] = collect(-5.0:1.0:5.0)
    ds["latitude"][:] = collect(-5.0:1.0:5.0)
    
    # Set uniform velocities and linear gradients for analytical validation
    # u = 1 m/s (eastward), v = 0 m/s
    # T increases eastward: T = 300 + x (where x is longitude)
    # q decreases northward: q = 0.01 - 0.001*y (where y is latitude)
    for i in 1:nlon, j in 1:nlat, k in 1:npres, t in 1:ntime
        ds["u"][i, j, k, t] = 1.0
        ds["v"][i, j, k, t] = 0.0
        ds["t"][i, j, k, t] = 300.0 + ds["longitude"][i]
        ds["q"][i, j, k, t] = 0.01 - 0.001 * ds["latitude"][j]
    end
    close(ds)
    
    # Test tendency calculation
    test_ds = NCDataset(test_data_path, "r")
    lat = 0.0
    lon_index = 6  # center point
    lat_index = 6
    
    # Create minimal external_tv_params
    external_tv_params = (planet_radius = 6.371e6,)  # Earth radius in meters
    
    tntha, tnhusha = CA.get_horizontal_tendencies(
        lat,
        lon_index,
        lat_index,
        test_ds,
        external_tv_params,
    )
    
    # Analytical validation:
    # 1. Temperature advection: u = 1, v = 0, T = 300 + lon
    # ∂T/∂x = (∂T/∂lon) / (R cos(lat) * π/180) 
    #       = 1.0 / (R * 1.0 * π/180) = 180 / (π R)
    # tntha = -u * ∂T/∂x = -1.0 * 180 / (π R)
    expected_tntha = -1.0 * 180.0 / (π * external_tv_params.planet_radius)
    
    # Grid spacing calculation in function:
    # longitudinal_resolution = 1.0 (from 1 degree grid)
    # dx = 2 * π * R * cos(lat) / 360 * 1.0
    # computed_grad = (T_E - T_W) / (2 * dx) = 2.0 / (2 * dx) = 1.0 / dx = 180 / (π R)
    # So it should match exactly (within FP precision)
    
    # Check that tendencies have correct shape
    @test size(tntha) == (npres, ntime)
    @test size(tnhusha) == (npres, ntime)
    
    # Check that temperature tendency matches analytical value
    # We use a small relative tolerance to account for floating point operations
    @test all(isapprox.(tntha, expected_tntha, rtol = 1e-5))
    
    # Check that humidity tendency is near zero (no gradient in x-direction)
    @test maximum(abs.(tnhusha)) < 1e-6
    
    close(test_ds)
end

@testset "get_coszen_inst" begin
    # Test solar zenith angle and insolation calculation
    FT = Float64
    
    # Test at equator, vernal equinox noon (March 20, 2000, 12:00 UTC)
    lat_eq = 0.0
    lon_eq = 0.0
    date_noon = Dates.DateTime(2000, 3, 20, 12, 0, 0)
    
    μ_noon, S_noon = CA.get_coszen_inst(lat_eq, lon_eq, date_noon, FT)
    
    # At solar noon on equinox at equator, coszen should be close to 1
    @test μ_noon > 0.9
    @test μ_noon <= 1.0
    
    # Solar flux should be positive
    @test S_noon > 0
    
    # Test at night (opposite side of Earth)
    lon_night = 180.0
    μ_night, S_night = CA.get_coszen_inst(lat_eq, lon_night, date_noon, FT)
    
    # At night, coszen should be 0 (sun below horizon)
    # At night, coszen should be 0 (sun below horizon)
    @test isapprox(μ_night, 0.0, atol = sqrt(eps(FT)))
    @test isapprox(S_night, 0.0, atol = sqrt(eps(FT)))
    
    # Test at high latitude (Arctic, 80°N)
    lat_arctic = 80.0
    lon_arctic = 0.0
    μ_arctic, S_arctic = CA.get_coszen_inst(lat_arctic, lon_arctic, date_noon, FT)
    
    # At high latitude, coszen should be smaller than at equator
    @test μ_arctic < μ_noon
    @test μ_arctic >= 0.0
    
    # Test return types
    @test μ_noon isa FT
    @test S_noon isa FT
end

@testset "smooth_4D_era5" begin
    # Test spatial smoothing of 4D ERA5 data (lon, lat, pressure, time)
    FT = Float64
    temporary_dir = mktempdir()
    test_data_path = joinpath(temporary_dir, "test_smoothing.nc")
    
    # Create test dataset with checkerboard pattern
    ds = NCDataset(test_data_path, "c")
    nlat, nlon, npres, ntime = 21, 21, 5, 3
    defDim(ds, "longitude", nlon)
    defDim(ds, "latitude", nlat)
    defDim(ds, "pressure_level", npres)
    defDim(ds, "valid_time", ntime)
    
    defVar(ds, "longitude", FT, ("longitude",))
    defVar(ds, "latitude", FT, ("latitude",))
    defVar(ds, "test_var_4d", FT, ("longitude", "latitude", "pressure_level", "valid_time"))
    
    ds["longitude"][:] = collect(-5.0:0.5:5.0)  # 21 points
    ds["latitude"][:] = collect(-5.0:0.5:5.0)   # 21 points
    
    # Checkerboard pattern: alternating 1s and 0s
    for i in 1:nlon, j in 1:nlat, k in 1:npres, t in 1:ntime
        ds["test_var_4d"][i, j, k, t] = ((i + j) % 2 == 0) ? 1.0 : 0.0
    end
    close(ds)
    
    # Test smoothing with analytical validation
    test_ds = NCDataset(test_data_path, "r")
    center_lon_idx = 11  # middle of 21-point grid
    center_lat_idx = 11
    smoothed_4d = CA.smooth_4D_era5(
        test_ds,
        "test_var_4d",
        center_lon_idx,
        center_lat_idx,
        smooth_amount = 4,
    )
    
    # Analytical result for checkerboard with 4-point smoothing:
    # - smooth_amount=4 → (2×4+1)×(2×4+1) = 9×9 box
    # - Center at (11,11): i+j=22 (even), so center cell = 1
    # - In 9×9 box centered at (11,11): rows 7-15, cols 7-15
    # - Checkerboard: 5 rows have 5 ones, 4 rows have 4 ones
    # - Total: 5×5 + 4×4 = 41 ones out of 81 cells
    exact_value_checkerboard = 41 / 81
    @test all(isapprox.(smoothed_4d, exact_value_checkerboard, atol = 1e-10))
    
    # Test with different smoothing amount
    smoothed_4d_small = CA.smooth_4D_era5(
        test_ds,
        "test_var_4d",
        center_lon_idx,
        center_lat_idx,
        smooth_amount = 1,
    )
    @test size(smoothed_4d_small) == (npres, ntime)
    
    close(test_ds)
end

@testset "smooth_3D_era5" begin
    # Test spatial smoothing of 3D ERA5 data (lon, lat, time)
    FT = Float64
    temporary_dir = mktempdir()
    test_data_path = joinpath(temporary_dir, "test_smoothing.nc")
    
    # Create test dataset with checkerboard pattern
    ds = NCDataset(test_data_path, "c")
    nlat, nlon, ntime = 21, 21, 3
    defDim(ds, "longitude", nlon)
    defDim(ds, "latitude", nlat)
    defDim(ds, "valid_time", ntime)
    
    defVar(ds, "longitude", FT, ("longitude",))
    defVar(ds, "latitude", FT, ("latitude",))
    defVar(ds, "test_var_3d", FT, ("longitude", "latitude", "valid_time"))
    
    ds["longitude"][:] = collect(-5.0:0.5:5.0)
    ds["latitude"][:] = collect(-5.0:0.5:5.0)
    
    # Checkerboard pattern: alternating 1s and 0s
    for i in 1:nlon, j in 1:nlat, t in 1:ntime
        ds["test_var_3d"][i, j, t] = ((i + j) % 2 == 0) ? 1.0 : 0.0
    end
    close(ds)
    
    # Test smoothing with analytical validation
    test_ds = NCDataset(test_data_path, "r")
    center_lon_idx = 11
    center_lat_idx = 11
    smoothed_3d = CA.smooth_3D_era5(
        test_ds,
        "test_var_3d",
        center_lon_idx,
        center_lat_idx,
        smooth_amount = 4,
    )
    
    # Same analytical result as 4D case (41 ones in 81 cells)
    exact_value_checkerboard = 41 / 81
    @test all(isapprox.(smoothed_3d, exact_value_checkerboard, atol = 1e-10))
    @test length(smoothed_3d) == ntime
    
    close(test_ds)
end

@testset "ERA5 diurnal warming" begin
    FT = Float32
    temporary_dir = mktempdir()

    parsed_args_0K = Dict(
        "start_date" => "20000506",
        "site_latitude" => 0.0,
        "site_longitude" => 0.0,
        "t_end" => "5hours",
        "era5_diurnal_warming" => Nothing,
    )

    parsed_args_4K = Dict(
        "start_date" => "20000506",
        "site_latitude" => 0.0,
        "site_longitude" => 0.0,
        "t_end" => "5hours",
        "era5_diurnal_warming" => 4,
    )

    create_mock_era5_datasets(temporary_dir, parsed_args_0K["start_date"], FT)

    sim_forcing_0K = CA.get_external_monthly_forcing_file_path(
        parsed_args_0K,
        data_dir = temporary_dir,
    )
    sim_forcing_4K = CA.get_external_monthly_forcing_file_path(
        parsed_args_4K,
        data_dir = temporary_dir,
    )

    @test basename(sim_forcing_0K) == "monthly_diurnal_cycle_forcing_0.0_0.0_20000506.nc"
    @test basename(sim_forcing_4K) ==
          "monthly_diurnal_cycle_forcing_0.0_0.0_20000506_plus_4.0K.nc"

    CA.generate_external_forcing_file(
        parsed_args_0K,
        sim_forcing_0K,
        FT,
        input_data_dir = temporary_dir,
    )

    CA.generate_external_forcing_file(
        parsed_args_4K,
        sim_forcing_4K,
        FT,
        input_data_dir = temporary_dir,
    )

    # open the datasets and check the temperature and specific humidity profiles have been adjusted
    processed_data_0K = NCDataset(sim_forcing_0K, "r")
    processed_data_4K = NCDataset(sim_forcing_4K, "r")

    # check that air and surface temperatures have increased by the +4K amount
    @test all(isapprox.(processed_data_0K["ta"] .+ 4, processed_data_4K["ta"]))
    @test isapprox(processed_data_0K["ts"] .+ 4, processed_data_4K["ts"])

    # Note: Specific humidity increase from diurnal warming is implementation-dependent
    # and varies with thermodynamic calculations, so not reliably testable across environments

    close(processed_data_0K)
    close(processed_data_4K)
end
