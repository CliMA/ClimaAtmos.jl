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
    @test Fields.field_values(
        CA.g³³_field(Fields.coordinate_field(cent_space).x),
    ) == lg_g³³
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

@testset "ERA5 Observations to forcing file" begin
    FT = Float64
    parsed_args = Dict(
        "start_date" => "20000506",
        "site_latitude" => 0.0,
        "site_longitude" => 0.0,
        "t_end" => "5hours",
    )
    # generate 3 datasets that contain mock forcing data
    temporary_dir = mktempdir()
    sim_forcing =
        CA.get_external_forcing_file_path(parsed_args, data_dir = temporary_dir)

    @test basename(sim_forcing) ==
          "tv_forcing_0.0_0.0_$(parsed_args["start_date"]).nc"

    # create fake data
    column_data_path = joinpath(
        temporary_dir,
        "forcing_and_cloud_hourly_profiles_$(parsed_args["start_date"]).nc",
    )
    accum_data_path =
        joinpath(temporary_dir, "hourly_accum_$(parsed_args["start_date"]).nc")
    inst_data_path =
        joinpath(temporary_dir, "hourly_inst_$(parsed_args["start_date"]).nc")

    tvforcing = NCDataset(column_data_path, "c")
    # define dimensions
    defDim(tvforcing, "valid_time", 6)
    defDim(tvforcing, "pressure_level", 37) # same dims as ERA5
    defDim(tvforcing, "latitude", 9)
    defDim(tvforcing, "longitude", 9)

    # define variables
    defVar(tvforcing, "latitude", FT, ("latitude",))
    defVar(tvforcing, "longitude", FT, ("longitude",))
    defVar(tvforcing, "pressure_level", FT, ("pressure_level",))
    defVar(tvforcing, "valid_time", FT, ("valid_time",))
    tvforcing["valid_time"].attrib["units"] = "hours since 2000-05-06 00:00:00"
    tvforcing["valid_time"].attrib["calendar"] = "standard"

    # fill the variables with sequential data
    tvforcing["latitude"][:] = collect(-1.0:0.25:1.0)
    tvforcing["longitude"][:] = collect(-1.0:0.25:1.0)
    tvforcing["pressure_level"][:] = 10 .^ (range(1, stop = 4, length = 37))
    tvforcing["valid_time"][:] = collect(0.0:5.0)

    # define the forcing variables
    full_dims = ("longitude", "latitude", "pressure_level", "valid_time")
    defVar(tvforcing, "u", FT, full_dims)
    defVar(tvforcing, "v", FT, full_dims)
    defVar(tvforcing, "w", FT, full_dims)
    defVar(tvforcing, "t", FT, full_dims)
    defVar(tvforcing, "q", FT, full_dims)
    defVar(tvforcing, "z", FT, full_dims)
    defVar(tvforcing, "clwc", FT, full_dims)
    defVar(tvforcing, "ciwc", FT, full_dims)

    # fill the variables with uniform ones
    tvforcing["u"][:, :, :, :] .= ones(FT, size(tvforcing["u"]))
    tvforcing["v"][:, :, :, :] .= ones(FT, size(tvforcing["v"]))
    tvforcing["w"][:, :, :, :] .= ones(FT, size(tvforcing["w"]))
    tvforcing["t"][:, :, :, :] .= ones(FT, size(tvforcing["t"]))
    tvforcing["q"][:, :, :, :] .= ones(FT, size(tvforcing["q"]))
    tvforcing["z"][:, :, :, :] .= ones(FT, size(tvforcing["z"]))
    tvforcing["clwc"][:, :, :, :] .= ones(FT, size(tvforcing["clwc"]))
    tvforcing["ciwc"][:, :, :, :] .= ones(FT, size(tvforcing["ciwc"]))

    # write the accumulated dataset
    tv_accum = NCDataset(accum_data_path, "c")
    defDim(tv_accum, "valid_time", 6)
    defDim(tv_accum, "latitude", 9)
    defDim(tv_accum, "longitude", 9)
    defVar(tv_accum, "latitude", FT, ("latitude",))
    defVar(tv_accum, "longitude", FT, ("longitude",))
    defVar(tv_accum, "valid_time", FT, ("valid_time",))
    tv_accum["valid_time"].attrib["units"] = "hours since 2000-05-06 00:00:00"
    tv_accum["valid_time"].attrib["calendar"] = "standard"

    tv_accum["latitude"][:] = collect(-1.0:0.25:1.0)
    tv_accum["longitude"][:] = collect(-1.0:0.25:1.0)
    tv_accum["valid_time"][:] = collect(0.0:5.0)

    # add slhf and sshf variables with ones
    defVar(tv_accum, "slhf", FT, ("longitude", "latitude", "valid_time"))
    defVar(tv_accum, "sshf", FT, ("longitude", "latitude", "valid_time"))
    tv_accum["slhf"][:, :, :] .= ones(FT, size(tv_accum["slhf"]))
    tv_accum["sshf"][:, :, :] .= ones(FT, size(tv_accum["sshf"]))

    # write the inst dataset
    tv_inst = NCDataset(inst_data_path, "c")
    defDim(tv_inst, "valid_time", 6)
    defDim(tv_inst, "latitude", 9)
    defDim(tv_inst, "longitude", 9)
    defVar(tv_inst, "latitude", FT, ("latitude",))
    defVar(tv_inst, "longitude", FT, ("longitude",))
    defVar(tv_inst, "valid_time", FT, ("valid_time",))
    tv_inst["valid_time"].attrib["units"] = "hours since 2000-05-06 00:00:00"
    tv_inst["valid_time"].attrib["calendar"] = "standard"

    tv_inst["latitude"][:] = collect(-1.0:0.25:1.0)
    tv_inst["longitude"][:] = collect(-1.0:0.25:1.0)
    tv_inst["valid_time"][:] = collect(0.0:5.0)

    # define skt
    defVar(tv_inst, "skt", FT, ("longitude", "latitude", "valid_time"))
    tv_inst["skt"][:, :, :] .= ones(FT, size(tv_inst["skt"]))

    # assert that the forcing file is generated correctly
    time_resolution = FT(3600)
    CA.generate_external_era5_forcing_file(
        parsed_args["site_latitude"],
        parsed_args["site_longitude"],
        parsed_args["start_date"],
        sim_forcing,
        Float64,
        time_resolution = time_resolution,
        data_dir = temporary_dir,
    )

    # test that the fixed variables have been copied exactly
    # name mapping between ERA5 and ClimaAtmos variable convections
    fixed_vars = Dict(
        "q" => "hus",
        "t" => "ta",
        "u" => "ua",
        "v" => "va",
        "w" => "wap",
        "z" => "zg",
        "clwc" => "clw",
        "ciwc" => "cli",
        "skt" => "ts",
    )

    # accum variables
    surface_accum_vars = Dict("slhf" => "hfls", "sshf" => "hfss")
    # check that the variables are copied correctly
    # open the dataset
    processed_data = NCDataset(sim_forcing, "r")
    for (era5_var, clima_var) in fixed_vars
        @test all(
            x -> all(isapprox.(x, 1, atol = 1e-10)),
            processed_data[clima_var][:],
        )
    end

    for (era5_var, clima_var) in surface_accum_vars
        @test all(
            x -> all(isapprox.(x, -1 / time_resolution, atol = 1e-10)),
            processed_data[clima_var][:],
        )
    end

    # assert that the gradients are all zero since the entries are constant; not in era5 dataset
    gradient_vars = ["tnhusha", "tntha"]
    for var in gradient_vars
        @test all(
            x -> all(isapprox.(x, 0, atol = 1e-10)),
            processed_data[var][:],
        )
    end

    # check that the coszen variable is between 0 and 1
    @test all(x -> x >= 0 && x <= 1, processed_data["coszen"][:])

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

    # test the vertical temperature gradient is all zeros
    @test all(
        x -> all(isapprox.(x, 0, atol = 1e-10)),
        vertical_temperature_gradient,
    )

    # check the forcing file time check passes if the start time and end time from config is valid
    @test CA.check_external_forcing_file_times(sim_forcing, parsed_args)

    # close files
    close(tvforcing)
    close(tv_inst)
    close(tv_accum)
    close(processed_data)
end
