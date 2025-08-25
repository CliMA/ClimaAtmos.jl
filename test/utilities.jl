using Test
using ClimaComms
ClimaComms.@import_required_backends
import Dates
using Random
Random.seed!(1234)
import ClimaAtmos as CA
using NCDatasets

include("test_helpers.jl")

@testset "sort_hdf5_files" begin
    day_sec(t) =
        (floor(Int, t / (60 * 60 * 24)), floor(Int, t % (60 * 60 * 24)))
    filenames(d, s) = "day$d.$s.hdf5"
    filenames(t) = filenames(day_sec(t)...)
    t = map(i -> rand(1:(10^6)), 1:100)
    t_sorted = sort(t)
    fns = filenames.(t)
    sort!(fns)
    @test CA.sort_files_by_time(fns) == filenames.(t_sorted)
end

@testset "isdivisible" begin
    @test CA.isdivisible(Dates.Month(1), Dates.Day(1))
    @test !CA.isdivisible(Dates.Month(1), Dates.Day(25))
    @test CA.isdivisible(Dates.Week(1), Dates.Day(1))
    @test CA.isdivisible(Dates.Day(1), Dates.Hour(1))
    @test CA.isdivisible(Dates.Hour(1), Dates.Second(1))
    @test CA.isdivisible(Dates.Minute(1), Dates.Second(30))
    @test !CA.isdivisible(Dates.Minute(1), Dates.Second(13))
    @test !CA.isdivisible(Dates.Day(1), Dates.Second(1e6))
    @test CA.isdivisible(Dates.Month(1), Dates.Hour(1))
end

@testset "kinetic_energy (c.f. analytical function)" begin
    # Test kinetic energy function for staggered grids
    # given an analytical expression for the velocity profiles
    (; cent_space, face_space) = get_cartesian_spaces()
    ccoords, fcoords = get_coords(cent_space, face_space)
    uₕ, uᵥ = get_test_functions(cent_space, face_space)
    (; x, y, z) = ccoords
    # Type helpers
    C1 = Geometry.Covariant1Vector
    C2 = Geometry.Covariant2Vector
    C3 = Geometry.Covariant3Vector
    C12 = Geometry.Covariant12Vector
    CT123 = Geometry.Contravariant123Vector
    # Exercise function
    κ = zeros(cent_space)
    κ .= CA.compute_kinetic(uₕ, uᵥ)
    ᶜκ_exact = @. 1 // 2 *
                  cos(z)^2 *
                  ((sin(x)^2) * (cos(y)^2) + (cos(x)^2) * (sin(y)^2))
    # Test upto machine precision approximation
    @test ᶜκ_exact ≈ κ
end

@testset "compute_strain_rate (c.f. analytical function)" begin
    # Test compute_strain_rate_face
    (; helem, cent_space, face_space) = get_cartesian_spaces()
    ccoords, fcoords = get_coords(cent_space, face_space)
    FT = eltype(ccoords.x)
    UVW = Geometry.UVWVector
    C123 = Geometry.Covariant123Vector
    # Alloc scratch space
    ᶜϵ =
        ᶜtemp_UVWxUVW = Fields.Field(
            typeof(UVW(FT(0), FT(0), FT(0)) * UVW(FT(0), FT(0), FT(0))'),
            cent_space,
        ) # ᶜstrain_rate
    ᶠϵ =
        ᶠtemp_UVWxUVW = Fields.Field(
            typeof(UVW(FT(0), FT(0), FT(0)) * UVW(FT(0), FT(0), FT(0))'),
            face_space,
        )
    u, v, w = taylor_green_ic(ccoords)
    ᶠu, ᶠv, ᶠw = taylor_green_ic(fcoords)

    (; x, y, z) = ccoords
    UVW = Geometry.UVWVector
    # Assemble (Cartesian) velocity
    ᶜu = @. UVW(Geometry.UVector(u)) +
            UVW(Geometry.VVector(v)) +
            UVW(Geometry.WVector(w))
    ᶠu = @. UVW(Geometry.UVector(ᶠu)) +
            UVW(Geometry.VVector(ᶠv)) +
            UVW(Geometry.WVector(ᶠw))
    ᶜϵ .= CA.compute_strain_rate_center(Geometry.Covariant123Vector.(ᶠu))
    ᶠϵ .= CA.compute_strain_rate_face(Geometry.Covariant123Vector.(ᶜu))

    # Center valued strain rate
    @test ᶜϵ.components.data.:1 == ᶜϵ.components.data.:1 .* FT(0)
    @test ᶜϵ.components.data.:5 == ᶜϵ.components.data.:7 .* FT(0)
    @test ᶜϵ.components.data.:3 == ᶜϵ.components.data.:7
    @test ᶜϵ.components.data.:2 == ᶜϵ.components.data.:4
    @test ᶜϵ.components.data.:6 == ᶜϵ.components.data.:8

    ᶜϵ₁₃ = ᶜϵ.components.data.:3
    ᶜϵ₂₃ = ᶜϵ.components.data.:6
    c₁₃ = @. -1 // 2 * sin(x) * cos(y) * sin(z)
    c₂₃ = @. 1 // 2 * cos(x) * sin(y) * sin(z)
    maximum(abs.((ᶜϵ₁₃ .- c₁₃) ./ (c₁₃ .+ eps(Float32)) .* 100)) < FT(0.5)
    maximum(abs.((ᶜϵ₂₃ .- c₂₃) ./ (c₂₃ .+ eps(Float32)) .* 100)) < FT(0.5)

    # Face valued strain-rate
    (; x, y, z) = fcoords
    ᶠϵ₁₃ = ᶠϵ.components.data.:3
    ᶠϵ₂₃ = ᶠϵ.components.data.:6
    f₁₃ = @. -1 // 2 * sin(x) * cos(y) * sin(z)
    f₂₃ = @. 1 // 2 * cos(x) * sin(y) * sin(z)
    # Check boundary conditions (see src/utils/utilities)
    # `slab` works per element
    for elem_id in 1:helem
        @test maximum(
            abs.(
                Fields.field_values(
                    Fields.slab(f₁₃, 1, elem_id) .-
                    Fields.slab(ᶠϵ₁₃, 1, elem_id),
                )
            ),
        ) < eps(FT) # bottom face
        @test maximum(
            abs.(
                Fields.field_values(
                    Fields.slab(f₁₃, 11, elem_id) .-
                    Fields.slab(ᶠϵ₁₃, 11, elem_id),
                )
            ),
        ) < eps(FT) # top face
    end
end

@testset "horizontal integral at boundary" begin
    # Test both `horizontal_integral_at_boundary` methods
    (; cent_space, face_space) = get_cartesian_spaces()
    _, fcoords = get_coords(cent_space, face_space)
    ᶠu, ᶠv, ᶠw = taylor_green_ic(fcoords)
    FT = eltype(ᶠu)
    halflevel = ClimaCore.Utilities.half
    y₁ = CA.horizontal_integral_at_boundary(ᶠu, halflevel)
    y₂ = CA.horizontal_integral_at_boundary(ᶠv, halflevel)
    y₃ = CA.horizontal_integral_at_boundary(ᶠw, halflevel)
    @test y₁ <= sqrt(eps(FT))
    @test y₂ <= sqrt(eps(FT))
    @test y₃ == y₃
    ᶠuₛ = Fields.level(ᶠu, halflevel)
    ᶠvₛ = Fields.level(ᶠv, halflevel)
    ᶠwₛ = Fields.level(ᶠw, halflevel)
    y₁ = CA.horizontal_integral_at_boundary(ᶠuₛ)
    y₂ = CA.horizontal_integral_at_boundary(ᶠvₛ)
    y₃ = CA.horizontal_integral_at_boundary(ᶠwₛ)
    @test y₁ <= sqrt(eps(FT))
    @test y₂ <= sqrt(eps(FT))
    @test y₃ == y₃
end

@testset "get mesh metrics" begin
    # We have already constructed cent_space and face_space (3d)
    # > These contain local geometry properties.
    # If grid properties change the updates will need to be caught in this
    # g³³_field function

    # This just tests getter functions
    # Correctness is checked in ClimaCore.jl
    (; cent_space, face_space) = get_cartesian_spaces()
    lg_gⁱʲ = cent_space.grid.center_local_geometry.gⁱʲ
    lg_g³³ = lg_gⁱʲ.components.data.:9
    (; x) = Fields.coordinate_field(cent_space)
    @test Fields.field_values(CA.g³³_field(axes(x))) == lg_g³³
    @test maximum(abs.(lg_g³³ .- CA.g³³.(lg_gⁱʲ).components.data.:1)) == 0
    @test maximum(abs.(CA.g³ʰ.(lg_gⁱʲ).components.data.:1)) == 0
    @test maximum(abs.(CA.g³ʰ.(lg_gⁱʲ).components.data.:2)) == 0
end

@testset "interval domain" begin
    # Interval Spaces
    (; zlim, velem) = get_cartesian_spaces()
    line_mesh = CA.periodic_line_mesh(; x_max = zlim[2], x_elem = velem)
    @test line_mesh isa Meshes.IntervalMesh
    @test Geometry.XPoint(zlim[1]) == Meshes.domain(line_mesh).coord_min
    @test Geometry.XPoint(zlim[2]) == Meshes.domain(line_mesh).coord_max
    @test velem == Meshes.nelements(line_mesh)
end

@testset "periodic rectangle meshes (spectral elements)" begin
    # Interval Spaces
    (; xlim, zlim, velem, helem, npoly) = get_cartesian_spaces()
    rectangle_mesh = CA.periodic_rectangle_mesh(;
        x_max = xlim[2],
        y_max = xlim[2],
        x_elem = helem,
        y_elem = helem,
    )
    @test rectangle_mesh isa Meshes.RectilinearMesh
    @test Meshes.domain(rectangle_mesh) isa Meshes.RectangleDomain
    @test Meshes.nelements(rectangle_mesh) == helem^2
    @test Meshes.element_horizontal_length_scale(rectangle_mesh) ==
          eltype(xlim)(π / npoly)
    @test Meshes.elements(rectangle_mesh) == CartesianIndices((helem, helem))
end

@testset "make horizontal spaces" begin

    (; xlim, zlim, velem, helem, npoly, quad) = get_cartesian_spaces()
    device = ClimaComms.CPUSingleThreaded()
    comms_ctx = ClimaComms.context(device)
    FT = eltype(xlim)
    # 1D Space
    line_mesh = CA.periodic_line_mesh(; x_max = zlim[2], x_elem = velem)
    @test line_mesh isa Meshes.AbstractMesh1D
    horz_plane_space =
        CA.make_horizontal_space(line_mesh, quad, comms_ctx, true)
    @test Spaces.column(horz_plane_space, 1, 1) isa Spaces.PointSpace

    # 2D Space
    rectangle_mesh = CA.periodic_rectangle_mesh(;
        x_max = xlim[2],
        y_max = xlim[2],
        x_elem = helem,
        y_elem = helem,
    )
    @test rectangle_mesh isa Meshes.AbstractMesh2D
    horz_plane_space =
        CA.make_horizontal_space(rectangle_mesh, quad, comms_ctx, true)
    @test Spaces.nlevels(horz_plane_space) == 1
    @test Spaces.node_horizontal_length_scale(horz_plane_space) ==
          FT(π / npoly / 5)
    @test Spaces.column(horz_plane_space, 1, 1, 1) isa Spaces.PointSpace
end

@testset "make hybrid spaces" begin
    (; cent_space, face_space, xlim, zlim, velem, helem, npoly, quad) =
        get_cartesian_spaces()
    config = CA.AtmosConfig(
        Dict("topography" => "NoWarp", "topo_smoothing" => false),
    )
    device = ClimaComms.CPUSingleThreaded()
    comms_ctx = ClimaComms.context(device)
    z_stretch = Meshes.Uniform()
    rectangle_mesh = CA.periodic_rectangle_mesh(;
        x_max = xlim[2],
        y_max = xlim[2],
        x_elem = helem,
        y_elem = helem,
    )
    horz_plane_space =
        CA.make_horizontal_space(rectangle_mesh, quad, comms_ctx, true)
    test_cent_space, test_face_space = CA.make_hybrid_spaces(
        horz_plane_space,
        zlim[2],
        velem,
        z_stretch;
        deep = false,
        parsed_args = config.parsed_args,
    )
    @test test_cent_space == cent_space
    @test test_face_space == face_space
end

@testset "promote_period" begin
    @test CA.promote_period(Dates.Hour(24)) == Dates.Day(1)
    @test CA.promote_period(Dates.Day(14)) == Dates.Week(2)
    @test CA.promote_period(Dates.Millisecond(1)) == Dates.Millisecond(1)
    @test CA.promote_period(Dates.Minute(120)) == Dates.Hour(2)
    @test CA.promote_period(Dates.Second(3600)) == Dates.Hour(1)
end

@testset "parse_date" begin
    @test CA.parse_date("20000506") == Dates.DateTime(2000, 5, 6)
    @test CA.parse_date("20000506-0000") == Dates.DateTime(2000, 5, 6, 0, 0)
    @test_throws ErrorException CA.parse_date("20000506-00000")
    @test_throws ErrorException CA.parse_date("")
end

@testset "ERA5 single day forcing file generation" begin
    FT = Float64
    parsed_args = Dict(
        "start_date" => "20000506",
        "site_latitude" => 0.0,
        "site_longitude" => 0.0,
        "t_end" => "5hours",
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
    for clima_var in ["hus", "ta", "ua", "va", "wap", "zg", "clw", "cli", "ts"]
        @test all(isapprox.(processed_data[clima_var][:], 1, atol = 1e-10))
    end

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
        println("Creating mock datasets for $date_str")
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
    @test all(x -> all(isapprox.(x, 1, atol = 1e-10)), processed_data["ta"][:])
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

    close(processed_data)
end

@testset "ERA5 smoothing functions" begin
    FT = Float64
    temporary_dir = mktempdir()

    # Create a test dataset with known spatial patterns
    test_data_path = joinpath(temporary_dir, "test_smoothing.nc")
    ds = NCDataset(test_data_path, "c")

    # Create a larger grid for testing smoothing
    nlat, nlon, npres, ntime = 21, 21, 5, 3
    defDim(ds, "longitude", nlon)
    defDim(ds, "latitude", nlat)
    defDim(ds, "pressure_level", npres)
    defDim(ds, "valid_time", ntime)

    defVar(ds, "longitude", FT, ("longitude",))
    defVar(ds, "latitude", FT, ("latitude",))
    defVar(
        ds,
        "test_var_4d",
        FT,
        ("longitude", "latitude", "pressure_level", "valid_time"),
    )
    defVar(ds, "test_var_3d", FT, ("longitude", "latitude", "valid_time"))

    # Fill coordinates
    ds["longitude"][:] = collect(-5.0:0.5:5.0)  # 21 points
    ds["latitude"][:] = collect(-5.0:0.5:5.0)   # 21 points

    # Create test pattern: checkerboard-like pattern
    for i in 1:nlon, j in 1:nlat, k in 1:npres, t in 1:ntime
        ds["test_var_4d"][i, j, k, t] = ((i + j) % 2 == 0) ? 1.0 : 0.0
    end

    for i in 1:nlon, j in 1:nlat, t in 1:ntime
        ds["test_var_3d"][i, j, t] = ((i + j) % 2 == 0) ? 1.0 : 0.0
    end

    close(ds)

    # Test smooth_4D_era5
    @testset "smooth_4D_era5" begin
        test_ds = NCDataset(test_data_path, "r")

        # Test with center point (should smooth checkerboard pattern)
        center_lon_idx = 11  # middle of 21-point grid
        center_lat_idx = 11
        smoothed_4d = CA.smooth_4D_era5(
            test_ds,
            "test_var_4d",
            center_lon_idx,
            center_lat_idx,
            smooth_amount = 4,
        )

        # With a checkerboard pattern and 4-point smoothing, we get 41 ones in a 81 square box 
        exact_value_checkerboard = 41 / 81
        @test all(
            isapprox.(smoothed_4d, exact_value_checkerboard, atol = 1e-10),
        )

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

    # Test smooth_3D_era5
    @testset "smooth_3D_era5" begin
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

        # With checkerboard pattern and 4-point smoothing, should get 41/81 
        exact_value_checkerboard = 41 / 81
        @test all(
            isapprox.(smoothed_3d, exact_value_checkerboard, atol = 1e-10),
        )
        @test length(smoothed_3d) == ntime
        close(test_ds)
    end
end
