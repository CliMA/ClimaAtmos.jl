using Test
import ClimaAtmos.COSP.COSPCloudSatReflectivity as CCR
using ClimaCore: Domains, Meshes, Spaces, Fields, Geometry

function make_center_field(FT; value, nelems = 3, z_top = 3000)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(z_top);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = nelems)
    face_space = Spaces.FaceFiniteDifferenceSpace(z_mesh)
    center_space = Spaces.CenterFiniteDifferenceSpace(face_space)

    field = Fields.Field(FT, center_space)
    @. field = FT(value)
    return field
end

function make_center_profile_field(FT, profile; z_top = 3000)
    field = make_center_field(FT; value = 0, nelems = length(profile), z_top)
    for (ilev, value) in enumerate(profile)
        Fields.level(field, ilev) .= FT(value)
    end
    return field
end

make_subcolumn_fields(FT, nsubcolumns, nelems; value = 0) =
    ntuple(_ -> make_center_field(FT; value, nelems), nsubcolumns)

function level_values(field)
    return [
        Fields.level(field, ilev)[] for
        ilev in 1:Spaces.nlevels(axes(field))
    ]
end

@testset "COSP CloudSat reflectivity" begin
    FT = Float64
    nsubcolumns = 1
    nelems = 3

    z_vol = (make_center_profile_field(FT, [100, 10, 1]),)
    kr_vol = (make_center_profile_field(FT, [0.1, 0.2, 0.3]),)
    g_vol = make_center_profile_field(FT, [0.01, 0.02, 0.03])
    Ze_non = make_subcolumn_fields(FT, nsubcolumns, nelems; value = 999)
    DBZe = make_subcolumn_fields(FT, nsubcolumns, nelems; value = 999)

    result = CCR.cloudsat_reflectivity!(Ze_non, DBZe, z_vol, kr_vol, g_vol)

    @test level_values(Fields.coordinate_field(axes(z_vol[1])).z) ==
          FT[500, 1500, 2500]
    @test isnothing(result)
    @test isapprox(parent(Ze_non[1]), FT[20, 10, 0]; atol = 1e-12)
    @test isapprox(
        parent(DBZe[1]),
        FT[18.845, 9.16, -0.315];
        rtol = 1e-12,
        atol = 1e-12,
    )

    z_vol = (make_center_profile_field(FT, [0, -1, 1]),)
    Ze_non = make_subcolumn_fields(FT, nsubcolumns, nelems; value = 999)
    DBZe = make_subcolumn_fields(FT, nsubcolumns, nelems; value = 999)

    CCR.cloudsat_reflectivity!(Ze_non, DBZe, z_vol, kr_vol, g_vol)

    @test parent(Ze_non[1])[1:2] == FT[-1e30, -1e30]
    @test parent(DBZe[1])[1:2] == FT[-1e30, -1e30]
    @test isfinite(parent(Ze_non[1])[3])
    @test isfinite(parent(DBZe[1])[3])

    z_vol = (
        make_center_profile_field(FT, [100, 10, 1]),
        make_center_profile_field(FT, [1000, 100, 10]),
    )
    kr_vol = (
        make_center_profile_field(FT, [0.1, 0.2, 0.3]),
        make_center_profile_field(FT, [0.05, 0.1, 0.15]),
    )
    g_vol = make_center_profile_field(FT, [0.01, 0.02, 0.03])
    Ze_non = make_subcolumn_fields(FT, 2, nelems; value = 999)
    DBZe = make_subcolumn_fields(FT, 2, nelems; value = 999)

    CCR.cloudsat_reflectivity!(Ze_non, DBZe, z_vol, kr_vol, g_vol)

    @test isapprox(parent(Ze_non[1]), FT[20, 10, 0]; atol = 1e-12)
    @test isapprox(parent(Ze_non[2]), FT[30, 20, 10]; atol = 1e-12)
    @test isapprox(
        parent(DBZe[1]),
        FT[18.845, 9.16, -0.315];
        rtol = 1e-12,
        atol = 1e-12,
    )
    @test isapprox(
        parent(DBZe[2]),
        FT[29.395, 19.56, 9.835];
        rtol = 1e-12,
        atol = 1e-12,
    )
end
