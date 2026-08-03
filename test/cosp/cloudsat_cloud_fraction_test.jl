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

@testset "streamed CloudSat total cloud cover above 1 km" begin
    for FT in (Float32, Float64)
        DBZe = (
            make_center_profile_field(FT, [-20, -20, -40]),
            make_center_profile_field(FT, [-40, -40, -20]),
        )
        height_km = make_center_profile_field(FT, [0.5, 1, 1.5])
        cloudsat_tcc = similar(Fields.level(DBZe[1], 1), FT)
        cloudsat_tcc2 = similar(cloudsat_tcc)
        surface_height_km = similar(cloudsat_tcc)
        detected_column = similar(cloudsat_tcc, Bool)
        surface_height_km .= zero(FT)

        CCF.initialize_cloudsat_cloud_fraction!(cloudsat_tcc)
        CCF.initialize_cloudsat_cloud_fraction!(cloudsat_tcc2)
        contribution = FT(100) / FT(length(DBZe))
        for DBZe_subcolumn in DBZe
            CCF.accumulate_cloudsat_cloud_fraction!(
                cloudsat_tcc,
                cloudsat_tcc2,
                detected_column,
                DBZe_subcolumn,
                height_km,
                surface_height_km,
                contribution,
            )
        end

        # Both subcolumns contain detectable reflectivity, but only the second
        # contains it strictly more than 1 km above the local surface.
        @test all(==(FT(100)), parent(cloudsat_tcc))
        @test all(==(FT(50)), parent(cloudsat_tcc2))
    end
end

@testset "COSPv2 CloudSat total cloud cover semantics" begin
    for FT in (Float32, Float64)
        DBZe = (
            make_center_profile_field(FT, [-31, -40, -1e30]),
            make_center_profile_field(FT, [-30, -40, -1e30]),
            make_center_profile_field(FT, [-29, -20, -1e30]),
            make_center_profile_field(FT, [10, -40, -1e30]),
            make_center_profile_field(FT, [11, 20, -1e30]),
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
        # The exact -30 and 10 dBZ boundaries are detectable, reflectivities
        # above 10 dBZ are excluded, and a subcolumn with several detectable
        # levels contributes only once.
        @test all(==(FT(50)), parent(cloudsat_tcc))

        all_clear = ntuple(
            i -> make_center_profile_field(
                FT,
                isodd(i) ? [-31, -40, -1e30] : [11, 20, -1e30],
            ),
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
