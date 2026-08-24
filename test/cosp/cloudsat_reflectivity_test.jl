using Test
import ClimaAtmos.COSP.COSPCloudSatReflectivity as CCR
using ClimaCore: Domains, Meshes, Spaces, Fields, Geometry

function make_center_field(
    FT;
    value,
    nelems = 3,
    z_top = 3000,
    stretch = Meshes.Uniform(),
)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(z_top);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, stretch; nelems)
    face_space = Spaces.FaceFiniteDifferenceSpace(z_mesh)
    center_space = Spaces.CenterFiniteDifferenceSpace(face_space)

    field = Fields.Field(FT, center_space)
    @. field = FT(value)
    return field
end

function make_center_profile_field(
    FT,
    profile;
    z_top = 3000,
    stretch = Meshes.Uniform(),
)
    field = make_center_field(
        FT;
        value = 0,
        nelems = length(profile),
        z_top,
        stretch,
    )
    for (ilev, value) in enumerate(profile)
        Fields.level(field, ilev) .= FT(value)
    end
    return field
end

function level_values(field)
    return [
        Fields.level(field, ilev)[] for
        ilev in 1:Spaces.nlevels(axes(field))
    ]
end

function model_top_height_km(field)
    nlevels = Spaces.nlevels(axes(field))
    face_height =
        Fields.coordinate_field(Spaces.face_space(axes(field))).z
    return Fields.level(face_height, nlevels + Fields.half) ./
           eltype(field)(1000)
end

@testset "CloudSat reflectivity" begin
    FT = Float64
    nelems = 3

    z_vol = make_center_profile_field(FT, [100, 10, 1])
    kr_vol = make_center_profile_field(FT, [0.1, 0.2, 0.3])
    g_vol = make_center_profile_field(FT, [0.01, 0.02, 0.03])
    DBZe = make_center_field(FT; value = 999, nelems)
    gas_path = make_center_field(FT; value = 999, nelems)
    height_km = Fields.coordinate_field(axes(g_vol)).z ./ FT(1000)
    top_height_km = model_top_height_km(g_vol)

    @test isnothing(
        CCR.cloudsat_gas_path_attenuation!(
            gas_path,
            g_vol,
            height_km,
            top_height_km,
        ),
    )
    result = CCR.cloudsat_reflectivity_subcolumn!(
        DBZe,
        z_vol,
        kr_vol,
        gas_path,
        height_km,
        top_height_km,
    )

    @test level_values(Fields.coordinate_field(axes(z_vol)).z) ==
          FT[500, 1500, 2500]
    @test isnothing(result)
    @test isapprox(
        parent(DBZe),
        FT[18.79, 9.12, -0.33];
        rtol = 1e-12,
        atol = 1e-12,
    )

    z_vol = make_center_profile_field(FT, [0, -1, 1])
    CCR.cloudsat_reflectivity_subcolumn!(
        DBZe,
        z_vol,
        kr_vol,
        gas_path,
        height_km,
        top_height_km,
    )

    @test parent(DBZe)[1:2] == FT[-1e30, -1e30]
    @test isfinite(parent(DBZe)[3])

end

@testset "CloudSat reflectivity in a single layer" begin
    FT = Float64
    z_vol = make_center_profile_field(FT, [10])
    kr_vol = make_center_profile_field(FT, [0.1])
    g_vol = make_center_profile_field(FT, [0.01])
    DBZe = similar(z_vol)
    gas_path = similar(z_vol)
    height_km = Fields.coordinate_field(axes(z_vol)).z ./ FT(1000)
    top_height_km = model_top_height_km(z_vol)

    CCR.cloudsat_gas_path_attenuation!(
        gas_path,
        g_vol,
        height_km,
        top_height_km,
    )
    CCR.cloudsat_reflectivity_subcolumn!(
        DBZe,
        z_vol,
        kr_vol,
        gas_path,
        height_km,
        top_height_km,
    )

    distance_from_top = top_height_km[] - level_values(height_km)[1]
    expected_DBZe = FT(10) - FT(2) * (FT(0.1) + FT(0.01)) * distance_from_top
    @test parent(DBZe) ≈ FT[expected_DBZe] rtol = 1e-12 atol = 1e-12
end

@testset "CloudSat reflectivity on a stretched grid" begin
    FT = Float64
    nelems = 4
    stretch = Meshes.HyperbolicTangentStretching{FT}(FT(250))
    z_vol = make_center_profile_field(
        FT,
        ones(FT, nelems);
        z_top = 4000,
        stretch,
    )
    kr_vol = make_center_profile_field(
        FT,
        fill(FT(0.1), nelems);
        z_top = 4000,
        stretch,
    )
    g_vol = make_center_profile_field(
        FT,
        fill(FT(0.01), nelems);
        z_top = 4000,
        stretch,
    )
    DBZe = similar(z_vol)
    gas_path = similar(z_vol)
    height_km = Fields.coordinate_field(axes(z_vol)).z ./ FT(1000)
    top_height_km = model_top_height_km(z_vol)

    CCR.cloudsat_gas_path_attenuation!(
        gas_path,
        g_vol,
        height_km,
        top_height_km,
    )
    CCR.cloudsat_reflectivity_subcolumn!(
        DBZe,
        z_vol,
        kr_vol,
        gas_path,
        height_km,
        top_height_km,
    )

    center_heights_km = level_values(height_km)
    @test !isapprox(
        center_heights_km[2] - center_heights_km[1],
        center_heights_km[end] - center_heights_km[end - 1],
    )
    expected_DBZe =
        -FT(2) * (FT(0.1) + FT(0.01)) *
        (top_height_km[] .- center_heights_km)
    @test isapprox(
        parent(DBZe),
        expected_DBZe;
        rtol = 1e-12,
        atol = 1e-12,
    )
end
