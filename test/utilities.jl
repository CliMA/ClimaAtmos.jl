using Test
using Random
Random.seed!(1234)
import ClimaAtmos as CA

### BoilerPlate Code
using ClimaComms
using IntervalSets

import ClimaCore:
    ClimaCore,
    Domains,
    Geometry,
    Grids,
    Fields,
    Operators,
    Meshes,
    Spaces,
    Quadratures,
    Topologies,
    Hypsography


FT = Float32

function get_cartesian_spaces(; FT = Float32)
    xlim = (FT(0), FT(π))
    zlim = (FT(0), FT(π))
    helem = 5
    velem = 10
    npoly = 5
    ndims = 3
    stretch = Meshes.Uniform()
    device = ClimaComms.CPUSingleThreaded()
    comms_context = ClimaComms.SingletonCommsContext(device)
    # Horizontal Grid Construction
    quad = Quadratures.GLL{npoly + 1}()
    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(xlim[1]) .. Geometry.XPoint{FT}(xlim[2]),
        Geometry.YPoint{FT}(xlim[1]) .. Geometry.YPoint{FT}(xlim[2]),
        x1periodic = true,
        x2periodic = true,
    )
    # Assume same number of elems (helem) in (x,y) directions
    horzmesh = Meshes.RectilinearMesh(horzdomain, helem, helem)
    horz_topology = Topologies.Topology2D(
        comms_context,
        horzmesh,
        Topologies.spacefillingcurve(horzmesh),
    )
    h_space =
        Spaces.SpectralElementSpace2D(horz_topology, quad, enable_bubble = true)

    horz_grid = Spaces.grid(h_space)

    # Vertical Grid Construction
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, stretch, nelems = velem)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)
    vert_topology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        vertmesh,
    )
    vert_grid = Grids.FiniteDifferenceGrid(vert_topology)
    ArrayType = ClimaComms.array_type(device)
    grid = Grids.ExtrudedFiniteDifferenceGrid(horz_grid, vert_grid)
    cent_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)
    return (helem = helem, cent_space = cent_space, face_space = face_space)
end

function get_coords(cent_space, face_space)
    ccoords = Fields.coordinate_field(cent_space)
    fcoords = Fields.coordinate_field(face_space)
    return ccoords, fcoords
end

function taylor_green_ic(coords)
    u = @. sin(coords.x) * cos(coords.y) * cos(coords.z)
    v = @. -cos(coords.x) * sin(coords.y) * cos(coords.z)
    #TODO: If a w field is introduced include it here. 
    return u, v, u .* 0
end

function get_test_functions(cent_space, face_space)
    ccoords, fcoords = get_coords(cent_space, face_space)
    FT = eltype(ccoords)
    Q = zero.(ccoords.x)
    # Exact velocity profiles
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
    # Get covariant components
    uₕ = @. Geometry.Covariant12Vector(ᶜu)
    uᵥ = @. Geometry.Covariant3Vector(ᶠu)
    return uₕ, uᵥ
end


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

@testset "gaussian_smooth" begin
    # No smooth on constant
    @test CA.gaussian_smooth(3.0 * ones(132, 157)) ≈ 3.0 * ones(132, 157)
    randy = rand(123, 145)
    smoothed = CA.gaussian_smooth(randy)
    # min
    @test extrema(randy)[1] <= smoothed[1]
    # max
    @test extrema(randy)[2] >= smoothed[2]
end

@testset "kinetic_energy (c.f. analytical function)" begin
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
    CA.compute_kinetic!(κ, uₕ, uᵥ)
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
    CA.compute_strain_rate_center!(ᶜϵ, Geometry.Covariant123Vector.(ᶠu))
    CA.compute_strain_rate_face!(ᶠϵ, Geometry.Covariant123Vector.(ᶜu))
    # Check by component, verify symmetry.
    # Strain rate functions only compute vertical derivatives right now.
    # Thus, terms in horizontal derivatives must be zero. 
    # FIXME: (This needs to be updated if a 3d operator is introduced.

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
    @test maximum(abs.(lg_g³³ .- CA.g³³.(lg_gⁱʲ).components.data.:1)) == FT(0)
    @test maximum(abs.(CA.g³ʰ.(lg_gⁱʲ).components.data.:1)) == FT(0)
    @test maximum(abs.(CA.g³ʰ.(lg_gⁱʲ).components.data.:2)) == FT(0)
end
