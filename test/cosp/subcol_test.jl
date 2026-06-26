using Test
using ClimaAtmos
using ClimaCore: Domains, Meshes, Spaces, Fields, Geometry

const CA = ClimaAtmos

function make_subcol_simulation(device; job_id)
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "DYCOMS_RF02",
            "microphysics_model" => "0M",
            "config" => "column",
            "output_default_diagnostics" => false,
            "device" => device,
        );
        job_id,
    )
    return CA.get_simulation(config)
end

function make_center_field(FT; value, nelems = 10)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(1000);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = nelems)
    face_space = Spaces.FaceFiniteDifferenceSpace(z_mesh)
    center_space = Spaces.CenterFiniteDifferenceSpace(face_space)

    field = Fields.Field(FT, center_space)
    @. field = FT(value)
    return field
end

function make_center_profile_field(FT, profile)
    field = make_center_field(FT; value = 0, nelems = length(profile))
    for (ilev, value) in enumerate(profile)
        Fields.level(field, ilev) .= FT(value)
    end
    return field
end

@testset "COSP subcolumns" begin
    FT = Float64
    seed = UInt64(1)

    @testset "p.precomputed.ᶜcloud_fraction input" begin
        simulation = make_subcol_simulation(
            "CPUSingleThreaded";
            job_id = "cosp_subcol_precomputed",
        )
        p = simulation.integrator.p

        cloud_fraction = p.precomputed.ᶜcloud_fraction
        @. cloud_fraction = FT(0.4)

        frac_out = p.precomputed.ᶜsubcolumn_cloud
        nsubcolumns = length(frac_out)
        threshold = p.precomputed.ᶜsubcolumn_threshold

        result = CA.COSPSubcolumns.scops!(
            frac_out,
            threshold,
            cloud_fraction,
            seed;
            overlap = :maximum,
        )

        @test isnothing(result)

        for isubcolumn in 1:nsubcolumns
            expected =
                FT(0.4) > (FT(isubcolumn) - FT(0.5)) / FT(nsubcolumns) ? FT(1) : FT(0)
            @test all(==(expected), parent(frac_out[isubcolumn]))
        end
    end

    @testset "subcolumn callback updates cached masks" begin
        simulation = make_subcol_simulation(
            "CPUSingleThreaded";
            job_id = "cosp_subcol_callback",
        )
        p = simulation.integrator.p

        @. p.precomputed.ᶜcloud_fraction = FT(1)
        CA.subcol_model_callback!(simulation.integrator)

        for frac in p.precomputed.ᶜsubcolumn_cloud
            @test all(==(FT(1)), parent(frac))
        end
    end

end
