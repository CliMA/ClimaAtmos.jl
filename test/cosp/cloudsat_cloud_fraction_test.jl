using Test
import ClimaAtmos.COSP.COSPCloudSatCloudFraction as CCF
using ClimaCore: Domains, Meshes, Spaces, Fields, Geometry

function make_center_profile_field(FT, profile)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(1000);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = length(profile))
    face_space = Spaces.FaceFiniteDifferenceSpace(z_mesh)
    center_space = Spaces.CenterFiniteDifferenceSpace(face_space)
    field = Fields.Field(FT, center_space)
    for (ilev, value) in enumerate(profile)
        Fields.level(field, ilev) .= FT(value)
    end
    return field
end

@testset "COSP CloudSat total cloud cover" begin
    for FT in (Float32, Float64)
        DBZe = (
            make_center_profile_field(FT, [-31, -40, -1e30]),
            make_center_profile_field(FT, [-30, -40, -1e30]),
            make_center_profile_field(FT, [-29, -20, -1e30]),
            make_center_profile_field(FT, [-1e30, -1e30, -1e30]),
        )
        cloudsat_tcc = similar(Fields.level(DBZe[1], 1), FT)
        detected_column = similar(cloudsat_tcc, Bool)

        @test isnothing(
            CCF.cloudsat_cloud_fraction!(
                cloudsat_tcc,
                detected_column,
                DBZe,
            ),
        )
        # The exact -30 dBZ boundary is detectable, and a subcolumn with
        # several detectable levels contributes only once.
        @test all(==(FT(50)), parent(cloudsat_tcc))

        all_clear = ntuple(
            _ -> make_center_profile_field(FT, [-31, -40, -1e30]),
            4,
        )
        CCF.cloudsat_cloud_fraction!(
            cloudsat_tcc,
            detected_column,
            all_clear,
        )
        @test all(iszero, parent(cloudsat_tcc))
        @test all(!, parent(detected_column))
    end
end
