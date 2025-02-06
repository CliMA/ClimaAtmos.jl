using Test
using ClimaComms
ClimaComms.@import_required_backends
import Dates
using Random
Random.seed!(1234)
import ClimaAtmos as CA

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
    bc_kinetic = CA.compute_kinetic(uₕ, uᵥ)
    @. κ = bc_kinetic
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
    bc_strain_rate =
        CA.compute_strain_rate_center(Geometry.Covariant123Vector.(ᶠu))
    @. ᶜϵ = bc_strain_rate
    bc_strain_rate =
        CA.compute_strain_rate_face(Geometry.Covariant123Vector.(ᶜu))
    @. ᶠϵ = bc_strain_rate

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
