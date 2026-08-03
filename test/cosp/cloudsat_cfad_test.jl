using Test
import ClimaAtmos.COSP.COSPCloudSatCFAD as CCFAD
using ClimaCore: Domains, Meshes, Spaces, Fields, Geometry

function make_cfad_center_field(FT, profile)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(1000);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = length(profile))
    face_space = Spaces.FaceFiniteDifferenceSpace(z_mesh)
    center_space = Spaces.CenterFiniteDifferenceSpace(face_space)
    field = Fields.Field(FT, center_space)
    for (level, value) in enumerate(profile)
        Fields.level(field, level) .= FT(value)
    end
    return field
end

@testset "COSPv2 CloudSat reflectivity CFAD semantics" begin
    for FT in (Float32, Float64)
        edges = CCFAD.cloudsat_cfad_bin_edges(FT)
        centers = CCFAD.cloudsat_cfad_bin_centers(FT)
        @test collect(edges) == FT[
            -100,
            -45,
            -40,
            -35,
            -30,
            -25,
            -20,
            -15,
            -10,
            -5,
            0,
            5,
            10,
            15,
            20,
            80,
        ]
        @test length(centers) == 15
        @test centers[1] == FT(-72.5)
        @test centers[end] == FT(50)

        first_subcolumn = make_cfad_center_field(
            FT,
            [-100, -45, -30, 10, 20, 79, 80, -1e30],
        )
        second_subcolumn = make_cfad_center_field(
            FT,
            [-99, -44, -29, 11, 21, -1e30, 80, -1e30],
        )
        cfad = ntuple(_ -> similar(first_subcolumn), length(centers))
        CCFAD.initialize_cloudsat_cfad!(cfad)

        contribution = FT(0.5)
        CCFAD.accumulate_cloudsat_cfad!(
            cfad,
            first_subcolumn,
            edges,
            contribution,
        )
        CCFAD.accumulate_cloudsat_cfad!(
            cfad,
            second_subcolumn,
            edges,
            contribution,
        )

        # Lower edges are inclusive and upper edges are exclusive. Values at
        # 80 dBZ and the missing-value sentinel are outside every bin.
        @test vec(parent(cfad[1])) == FT[1, 0, 0, 0, 0, 0, 0, 0]
        @test vec(parent(cfad[2])) == FT[0, 1, 0, 0, 0, 0, 0, 0]
        @test vec(parent(cfad[5])) == FT[0, 0, 1, 0, 0, 0, 0, 0]
        @test vec(parent(cfad[13])) == FT[0, 0, 0, 1, 0, 0, 0, 0]
        @test vec(parent(cfad[15])) == FT[0, 0, 0, 0, 1, 0.5, 0, 0]
        for level in (7, 8)
            @test all(iszero, map(bin -> parent(bin)[level], cfad))
        end
    end
end
