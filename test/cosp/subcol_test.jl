using Test
import ClimaComms
ClimaComms.@import_required_backends
using ClimaAtmos
using ClimaCore: Domains, Meshes, Spaces, Fields, Geometry, to_cpu, to_device

const CA = ClimaAtmos
const HAS_CUDA = let
    try
        @eval import CUDA
        CUDA.functional()
    catch
        false
    end
end

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

    @testset "GPU subcolumn path matches CPU" begin
        if !HAS_CUDA
            @test true
        else
            CUDA.allowscalar(false)

            cpu_cloud_fraction = make_center_field(FT; value = 0.4)
            cpu_frac_out = ntuple(_ -> similar(cpu_cloud_fraction, FT), 4)
            cpu_threshold = ntuple(_ -> similar(cpu_cloud_fraction, FT), 4)

            gpu_cloud_fraction =
                to_device(ClimaComms.CUDADevice(), cpu_cloud_fraction)
            gpu_frac_out = ntuple(_ -> similar(gpu_cloud_fraction, FT), 4)
            gpu_threshold = ntuple(_ -> similar(gpu_cloud_fraction, FT), 4)

            for overlap in (:maximum, :random, :maximum_random)
                CA.COSPSubcolumns.scops!(
                    cpu_frac_out,
                    cpu_threshold,
                    cpu_cloud_fraction,
                    seed;
                    overlap,
                )
                CA.COSPSubcolumns.scops!(
                    gpu_frac_out,
                    gpu_threshold,
                    gpu_cloud_fraction,
                    seed;
                    overlap,
                )

                for isubcolumn in eachindex(cpu_frac_out)
                    @test parent(to_cpu(gpu_frac_out[isubcolumn])) ==
                          parent(cpu_frac_out[isubcolumn])
                    @test parent(to_cpu(gpu_threshold[isubcolumn])) ≈
                          parent(cpu_threshold[isubcolumn])
                end
            end

            @. gpu_cloud_fraction = FT(1)
            gpu_precomputed = (;
                ᶜcloud_fraction = gpu_cloud_fraction,
                ᶜsubcolumn_cloud = gpu_frac_out,
                ᶜsubcolumn_threshold = gpu_threshold,
            )
            gpu_integrator = (; p = (; precomputed = gpu_precomputed), t = FT(0))
            CA.subcol_model_callback!(gpu_integrator)
            for frac in gpu_frac_out
                @test all(==(FT(1)), parent(to_cpu(frac)))
            end
        end
    end
end
