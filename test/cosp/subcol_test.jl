using Test
import ClimaAtmos as CA
using ClimaCore: Domains, Meshes, Spaces, Fields, Geometry

function make_subcol_simulation(device; job_id)
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "DYCOMS_RF02",
            "microphysics_model" => "0M",
            "config" => "column",
            "output_default_diagnostics" => false,
            "dt_subcol" => "10mins",
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

function center_profile(field)
    return [Fields.level(field, ilev)[] for ilev in 1:Spaces.nlevels(axes(field))]
end

function mean_profile(fields)
    nsubcolumns = length(fields)
    profile = center_profile(fields[1])
    for isubcolumn in 2:nsubcolumns
        profile .+= center_profile(fields[isubcolumn])
    end
    return profile ./ nsubcolumns
end

make_subcolumn_fields(FT, nsubcolumns, nelems; value = -1) =
    ntuple(_ -> make_center_field(FT; value, nelems), nsubcolumns)

function make_hydrometeor_subcolumns(grid_mean, nsubcolumns)
    subcolumn_values =
        map(
            field -> ntuple(_ -> similar(field), nsubcolumns),
            Base.values(grid_mean),
        )
    return NamedTuple{keys(grid_mean)}(subcolumn_values)
end

function make_reff_subcolumns(reference, nsubcolumns)
    return (;
        Reff_lcl = ntuple(_ -> similar(reference), nsubcolumns),
        Reff_icl = ntuple(_ -> similar(reference), nsubcolumns),
        Reff_rai = ntuple(_ -> similar(reference), nsubcolumns),
        Reff_sno = ntuple(_ -> similar(reference), nsubcolumns),
    )
end

function make_number_subcolumns(reference, nsubcolumns)
    return (;
        Np_lcl = ntuple(_ -> similar(reference), nsubcolumns),
        Np_icl = ntuple(_ -> similar(reference), nsubcolumns),
        Np_rai = ntuple(_ -> similar(reference), nsubcolumns),
        Np_sno = ntuple(_ -> similar(reference), nsubcolumns),
    )
end

expected_maximum_mask(FT, isubcolumn, nsubcolumns, cloud_fraction) =
    FT(cloud_fraction) > (FT(isubcolumn) - FT(0.5)) / FT(nsubcolumns) ? FT(1) : FT(0)

function reference_scops_profiles(FT, cloud_profile, nsubcolumns, seed, overlap)
    nlev = length(cloud_profile)
    coords = center_profile(Fields.coordinate_field(axes(make_center_profile_field(FT, cloud_profile))))
    thresholds = [Vector{FT}(undef, nlev) for _ in 1:nsubcolumns]
    masks = [Vector{FT}(undef, nlev) for _ in 1:nsubcolumns]

    for ilev in nlev:-1:1
        total_cloud = clamp(FT(cloud_profile[ilev]), zero(FT), one(FT))
        previous_total_cloud =
            ilev == nlev ? zero(FT) :
            clamp(FT(cloud_profile[ilev + 1]), zero(FT), one(FT))
        convective_cloud = zero(FT)

        for isubcolumn in 1:nsubcolumns
            box_position = (FT(isubcolumn) - FT(0.5)) / FT(nsubcolumns)
            old_threshold =
                ilev == nlev ? box_position : thresholds[isubcolumn][ilev + 1]

            thresholds[isubcolumn][ilev] = reference_new_threshold(
                box_position,
                total_cloud,
                previous_total_cloud,
                convective_cloud,
                old_threshold,
                seed,
                coords[ilev],
                isubcolumn,
                overlap,
            )
            masks[isubcolumn][ilev] =
                total_cloud > thresholds[isubcolumn][ilev] ? one(FT) : zero(FT)
        end
    end

    return thresholds, masks
end

function reference_new_threshold(
    box_position,
    total_cloud,
    previous_total_cloud,
    convective_cloud,
    old_threshold,
    seed,
    coords,
    isubcolumn,
    overlap,
)
    in_convective_region = box_position <= convective_cloud

    if overlap === :maximum
        return box_position
    elseif overlap === :random
        threshold_min = convective_cloud
        random_number = CA.COSP.COSPSubcolumns._rand_for_point(
            seed,
            coords,
            isubcolumn,
        )

        return in_convective_region ? box_position :
               threshold_min + (one(threshold_min) - threshold_min) * random_number
    else
        common_cloud = min(previous_total_cloud, total_cloud)
        threshold_min = max(convective_cloud, common_cloud)
        random_number = CA.COSP.COSPSubcolumns._rand_for_point(
            seed,
            coords,
            isubcolumn,
        )
        maximally_overlap_stratiform =
            old_threshold < common_cloud && old_threshold > convective_cloud

        return in_convective_region ? box_position :
               maximally_overlap_stratiform ? old_threshold :
               threshold_min + (one(threshold_min) - threshold_min) * random_number
    end
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
        prec_frac = p.precomputed.ᶜsubcolumn_precip
        subcolumn_reff = p.precomputed.ᶜsubcolumn_reff
        subcolumn_Np = p.precomputed.ᶜsubcolumn_Np
        large_scale_precipitation_flux =
            p.precomputed.ᶜlarge_scale_precipitation_flux

        result = CA.COSP.COSPSubcolumns.scops!(
            frac_out,
            threshold,
            cloud_fraction,
            seed;
            overlap = :maximum,
        )

        @test isnothing(result)
        @test length(prec_frac) == nsubcolumns
        @test all(axes(prec) == axes(cloud_fraction) for prec in prec_frac)
        @test length(subcolumn_reff.Reff_lcl) == nsubcolumns
        @test length(subcolumn_Np.Np_lcl) == nsubcolumns
        @test axes(subcolumn_reff.Reff_lcl[1]) == axes(cloud_fraction)
        @test axes(subcolumn_Np.Np_lcl[1]) == axes(cloud_fraction)
        @test axes(large_scale_precipitation_flux) == axes(cloud_fraction)

        for isubcolumn in 1:nsubcolumns
            expected = expected_maximum_mask(FT, isubcolumn, nsubcolumns, 0.4)
            @test all(==(expected), parent(frac_out[isubcolumn]))
        end

        @. large_scale_precipitation_flux = FT(1)
        for precip in prec_frac
            @. precip = FT(-1)
        end

        precip_result = CA.COSP.COSPPrecipSubcolumns.prec_scops!(
            prec_frac,
            large_scale_precipitation_flux,
            frac_out,
        )

        @test isnothing(precip_result)
        for isubcolumn in 1:nsubcolumns
            expected = expected_maximum_mask(FT, isubcolumn, nsubcolumns, 0.4)
            @test all(==(expected), parent(prec_frac[isubcolumn]))
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
        for field_group in values(p.precomputed.ᶜsubcolumn_reff), field in field_group
            @test all(iszero, parent(field))
        end
        for field_group in values(p.precomputed.ᶜsubcolumn_Np), field in field_group
            @test all(iszero, parent(field))
        end
    end

    @testset "maximum-random overlap carries previous threshold" begin
        cloud_fraction = make_center_profile_field(FT, [1, 1, 1])
        frac_out = make_subcolumn_fields(FT, 4, 3)
        threshold = make_subcolumn_fields(FT, 4, 3)

        result = CA.COSP.COSPSubcolumns.scops!(
            frac_out,
            threshold,
            cloud_fraction,
            seed;
            overlap = :maximum_random,
        )

        @test isnothing(result)
        for isubcolumn in 1:length(threshold)
            threshold_profile = center_profile(threshold[isubcolumn])
            @test 0 <= threshold_profile[1] < 1
            @test threshold_profile[2:end] ==
                  fill(threshold_profile[1], length(threshold_profile) - 1)
            @test center_profile(frac_out[isubcolumn]) == FT[1, 1, 1]
        end
    end

    @testset "thresholds match reference recurrence for overlap modes" begin
        cloud_profile = FT[0.2, 0.8, 0.4, 1.0]
        nsubcolumns = 4

        for overlap in (:maximum, :random, :maximum_random)
            cloud_fraction = make_center_profile_field(FT, cloud_profile)
            frac_out = make_subcolumn_fields(FT, nsubcolumns, 4)
            threshold = make_subcolumn_fields(FT, nsubcolumns, 4)

            CA.COSP.COSPSubcolumns.scops!(
                frac_out,
                threshold,
                cloud_fraction,
                seed;
                overlap,
            )

            expected_thresholds, expected_masks =
                reference_scops_profiles(FT, cloud_profile, nsubcolumns, seed, overlap)

            for isubcolumn in 1:nsubcolumns
                @test isapprox(
                    center_profile(threshold[isubcolumn]),
                    expected_thresholds[isubcolumn],
                )
                @test center_profile(frac_out[isubcolumn]) ==
                      expected_masks[isubcolumn]
                @test all(
                    mask in (zero(FT), one(FT)) for
                    mask in center_profile(frac_out[isubcolumn])
                )
            end
        end
    end

    @testset "reverse accumulation matches SCOPS reference with random overlap" begin
        nsubcolumns = 4

        for cloud_profile in (
            FT[0.15, 0.85, 0.30, 0.65, 0.45],
            FT[-0.20, 1.20, 0.60, 0.00, 0.35],
        )
            for overlap in (:random, :maximum_random)
                cloud_fraction = make_center_profile_field(FT, cloud_profile)
                frac_out = make_subcolumn_fields(
                    FT,
                    nsubcolumns,
                    length(cloud_profile),
                )
                threshold = make_subcolumn_fields(
                    FT,
                    nsubcolumns,
                    length(cloud_profile),
                )

                result = CA.COSP.COSPSubcolumns.scops!(
                    frac_out,
                    threshold,
                    cloud_fraction,
                    seed;
                    overlap,
                )

                expected_thresholds, expected_masks = reference_scops_profiles(
                    FT,
                    cloud_profile,
                    nsubcolumns,
                    seed,
                    overlap,
                )

                @test isnothing(result)
                for isubcolumn in 1:nsubcolumns
                    @test isapprox(
                        center_profile(threshold[isubcolumn]),
                        expected_thresholds[isubcolumn],
                    )
                    @test center_profile(frac_out[isubcolumn]) ==
                          expected_masks[isubcolumn]
                end
            end
        end
    end

    @testset "maximum overlap samples each level cloud fraction" begin
        cloud_profile = FT[0.00, 0.25, 0.50, 0.75, 1.00]
        nsubcolumns = 4
        cloud_fraction = make_center_profile_field(FT, cloud_profile)
        frac_out = make_subcolumn_fields(FT, nsubcolumns, length(cloud_profile))
        threshold = make_subcolumn_fields(FT, nsubcolumns, length(cloud_profile))

        result = CA.COSP.COSPSubcolumns.scops!(
            frac_out,
            threshold,
            cloud_fraction,
            seed;
            overlap = :maximum,
        )

        @test isnothing(result)
        expected_masks = [
            FT[0, 1, 1, 1, 1],
            FT[0, 0, 1, 1, 1],
            FT[0, 0, 0, 1, 1],
            FT[0, 0, 0, 0, 1],
        ]
        for isubcolumn in 1:nsubcolumns
            @test center_profile(frac_out[isubcolumn]) ==
                  expected_masks[isubcolumn]
        end
    end

    @testset "precipitation subcolumns assume zero convective precipitation" begin
        large_scale_precipitation_flux =
            make_center_profile_field(FT, [1, 1, 0])

        frac_out = (
            make_center_profile_field(FT, [0, 0, 0]),
            make_center_profile_field(FT, [0, 1, 0]),
            make_center_profile_field(FT, [0, 0, 0]),
            make_center_profile_field(FT, [0, 0, 1]),
        )
        prec_frac = make_subcolumn_fields(FT, 4, 3)

        result = CA.COSP.COSPPrecipSubcolumns.prec_scops!(
            prec_frac,
            large_scale_precipitation_flux,
            frac_out,
        )

        @test isnothing(result)
        @test center_profile(prec_frac[1]) == FT[0, 0, 0]
        @test center_profile(prec_frac[2]) == FT[1, 1, 0]
        @test center_profile(prec_frac[3]) == FT[0, 0, 0]
        @test center_profile(prec_frac[4]) == FT[0, 0, 0]
    end

    @testset "precipitation carries through non-cloudy lower levels" begin
        large_scale_precipitation_flux =
            make_center_profile_field(FT, [1, 1, 1])

        frac_out = (
            make_center_profile_field(FT, [0, 0, 0]),
            make_center_profile_field(FT, [1, 0, 0]),
            make_center_profile_field(FT, [0, 0, 0]),
            make_center_profile_field(FT, [0, 0, 0]),
        )
        prec_frac = make_subcolumn_fields(FT, 4, 3)

        CA.COSP.COSPPrecipSubcolumns.prec_scops!(
            prec_frac,
            large_scale_precipitation_flux,
            frac_out,
        )

        @test center_profile(prec_frac[1]) == FT[0, 0, 0]
        @test center_profile(prec_frac[2]) == FT[1, 1, 1]
        @test center_profile(prec_frac[3]) == FT[0, 0, 0]
        @test center_profile(prec_frac[4]) == FT[0, 0, 0]
    end

    @testset "precipitation falls toward lower cloudy subcolumns" begin
        large_scale_precipitation_flux =
            make_center_profile_field(FT, [0, 1, 0])

        frac_out = (
            make_center_profile_field(FT, [0, 0, 0]),
            make_center_profile_field(FT, [1, 0, 0]),
            make_center_profile_field(FT, [0, 0, 0]),
            make_center_profile_field(FT, [0, 0, 1]),
        )
        prec_frac = make_subcolumn_fields(FT, 4, 3)

        CA.COSP.COSPPrecipSubcolumns.prec_scops!(
            prec_frac,
            large_scale_precipitation_flux,
            frac_out,
        )

        @test center_profile(prec_frac[1]) == FT[0, 0, 0]
        @test center_profile(prec_frac[2]) == FT[0, 1, 0]
        @test center_profile(prec_frac[3]) == FT[0, 0, 0]
        @test center_profile(prec_frac[4]) == FT[0, 0, 0]
    end

    @testset "hydrometeor subcolumns conserve sampled grid mean" begin
        cloud_mask = (
            make_center_profile_field(FT, [1, 0, 2]),
            make_center_profile_field(FT, [0, 1, 0]),
            make_center_profile_field(FT, [0, 0, 0]),
            make_center_profile_field(FT, [1, 1, 0]),
        )
        precip_mask = (
            make_center_profile_field(FT, [1, 0, 0]),
            make_center_profile_field(FT, [0, 3, 0]),
            make_center_profile_field(FT, [0, 0, 0]),
            make_center_profile_field(FT, [1, 2, 0]),
        )
        grid_mean = (;
            q_lcl = make_center_profile_field(FT, [1, 2, 4]),
            q_icl = make_center_profile_field(FT, [0.5, 1, 2]),
            q_rai = make_center_profile_field(FT, [10, 20, 0]),
            q_sno = make_center_profile_field(FT, [5, 15, 0]),
        )

        subcolumns = make_hydrometeor_subcolumns(grid_mean, length(cloud_mask))
        sampled_cloud_fraction = make_center_field(FT; value = -1, nelems = 3)
        sampled_precip_fraction = make_center_field(FT; value = -1, nelems = 3)

        result = CA.COSP.COSPHydrometeorSubcolumns.slice_hydrometeor_subcolumns!(
            subcolumns,
            cloud_mask,
            precip_mask,
            grid_mean,
            sampled_cloud_fraction,
            sampled_precip_fraction,
        )

        @test isnothing(result)
        @test length(subcolumns.q_lcl) == length(cloud_mask)
        @test axes(subcolumns.q_lcl[1]) == axes(grid_mean.q_lcl)

        @test center_profile(subcolumns.q_lcl[2]) == FT[0, 4, 0]
        @test center_profile(subcolumns.q_icl[3]) == FT[0, 0, 0]
        @test center_profile(subcolumns.q_rai[1]) == FT[20, 0, 0]
        @test center_profile(subcolumns.q_sno[2]) == FT[0, 30, 0]
        @test center_profile(sampled_cloud_fraction) == FT[0.5, 0.5, 0.25]
        @test center_profile(sampled_precip_fraction) == FT[0.5, 0.5, 0]

        @test isapprox(
            mean_profile(subcolumns.q_lcl),
            center_profile(grid_mean.q_lcl),
        )
        @test isapprox(
            mean_profile(subcolumns.q_icl),
            center_profile(grid_mean.q_icl),
        )
        @test isapprox(
            mean_profile(subcolumns.q_rai),
            center_profile(grid_mean.q_rai),
        )
        @test isapprox(
            mean_profile(subcolumns.q_sno),
            center_profile(grid_mean.q_sno),
        )
    end

    @testset "hydrometeor slicing handles zero sampled fractions" begin
        cloud_mask = ntuple(_ -> make_center_profile_field(FT, [0, 0]), 4)
        precip_mask = ntuple(_ -> make_center_profile_field(FT, [0, 0]), 4)
        grid_mean = (;
            q_lcl = make_center_profile_field(FT, [1, 0]),
            q_rai = make_center_profile_field(FT, [2, 0]),
        )
        subcolumns = make_hydrometeor_subcolumns(grid_mean, length(cloud_mask))
        sampled_cloud_fraction = make_center_field(FT; value = -1, nelems = 2)
        sampled_precip_fraction = make_center_field(FT; value = -1, nelems = 2)

        CA.COSP.COSPHydrometeorSubcolumns.slice_hydrometeor_subcolumns!(
            subcolumns,
            cloud_mask,
            precip_mask,
            grid_mean,
            sampled_cloud_fraction,
            sampled_precip_fraction,
        )

        @test center_profile(sampled_cloud_fraction) == FT[0, 0]
        @test center_profile(sampled_precip_fraction) == FT[0, 0]

        for field in subcolumns.q_lcl
            @test center_profile(field) == FT[1, 0]
        end
        for field in subcolumns.q_rai
            @test center_profile(field) == FT[2, 0]
        end

        @test isapprox(
            mean_profile(subcolumns.q_lcl),
            center_profile(grid_mean.q_lcl),
        )
        @test isapprox(
            mean_profile(subcolumns.q_rai),
            center_profile(grid_mean.q_rai),
        )

        for field_group in values(subcolumns), field in field_group
            @test all(isfinite, parent(field))
        end
    end

    @testset "hydrometeor callback helper uses grid-mean state" begin
        ρ = make_center_profile_field(FT, [2, 2])
        Y = (;
            c = (;
                ρ,
                ρq_lcl = make_center_profile_field(FT, [2, 4]),
                ρq_icl = make_center_profile_field(FT, [1, 2]),
                ρq_rai = make_center_profile_field(FT, [20, 40]),
                ρq_sno = make_center_profile_field(FT, [10, 30]),
            ),
        )
        cloud_mask = (
            make_center_profile_field(FT, [1, 0]),
            make_center_profile_field(FT, [0, 1]),
        )
        precip_mask = (
            make_center_profile_field(FT, [1, 0]),
            make_center_profile_field(FT, [0, 3]),
        )
        subcolumns = (;
            q_lcl = make_subcolumn_fields(FT, 2, 2),
            q_icl = make_subcolumn_fields(FT, 2, 2),
            q_rai = make_subcolumn_fields(FT, 2, 2),
            q_sno = make_subcolumn_fields(FT, 2, 2),
        )
        sampled_cloud_fraction = make_center_field(FT; value = -1, nelems = 2)
        sampled_precip_fraction = make_center_field(FT; value = -1, nelems = 2)
        p = (;
            precomputed = (;
                ᶜsubcolumn_cloud = cloud_mask,
                ᶜsubcolumn_precip = precip_mask,
                ᶜsubcolumn_hydrometeors = subcolumns,
                ᶜsampled_cloud_fraction = sampled_cloud_fraction,
                ᶜsampled_precip_fraction = sampled_precip_fraction,
            ),
        )

        result = CA.set_cosp_hydrometeor_subcolumns!(
            Y,
            p,
            CA.NonEquilibriumMicrophysics1M(),
        )

        @test isnothing(result)
        @test center_profile(sampled_cloud_fraction) == FT[0.5, 0.5]
        @test center_profile(sampled_precip_fraction) == FT[0.5, 0.5]
        @test center_profile(subcolumns.q_lcl[1]) == FT[2, 0]
        @test center_profile(subcolumns.q_icl[2]) == FT[0, 2]
        @test center_profile(subcolumns.q_rai[1]) == FT[20, 0]
        @test center_profile(subcolumns.q_sno[2]) == FT[0, 30]
    end

    @testset "reff/np callback helper uses subcolumn hydrometeors" begin
        ρ = make_center_profile_field(FT, [2, 2])
        Y = (; c = (; ρ))
        subcolumns = (;
            q_lcl = (
                make_center_profile_field(FT, [1, 0]),
                make_center_profile_field(FT, [0, 2]),
            ),
            q_icl = (
                make_center_profile_field(FT, [0, 1]),
                make_center_profile_field(FT, [2, 0]),
            ),
            q_rai = (
                make_center_profile_field(FT, [3, 0]),
                make_center_profile_field(FT, [0, 4]),
            ),
            q_sno = (
                make_center_profile_field(FT, [0, 5]),
                make_center_profile_field(FT, [6, 0]),
            ),
        )
        reff = make_reff_subcolumns(ρ, length(subcolumns.q_lcl))
        np = make_number_subcolumns(ρ, length(subcolumns.q_lcl))
        p = (;
            precomputed = (;
                ᶜsubcolumn_hydrometeors = subcolumns,
                ᶜsubcolumn_reff = reff,
                ᶜsubcolumn_Np = np,
            ),
        )

        result = CA.set_cosp_reff_np_subcolumns!(
            Y,
            p,
            CA.NonEquilibriumMicrophysics1M(),
        )

        @test isnothing(result)
        N_lcl =
            CA.COSP.COSP1MReffNpDiagnostics.DEFAULT_REFF_NP_1M_PARAMETERS.n_lcl
        @test center_profile(np.Np_lcl[1]) == FT[N_lcl, 0]
        @test center_profile(reff.Reff_lcl[1])[1] > FT(0)
        @test center_profile(reff.Reff_lcl[1])[2] == FT(0)
        @test center_profile(reff.Reff_rai[1])[1] > FT(0)
        @test center_profile(np.Np_rai[1])[1] > FT(0)
        @test center_profile(reff.Reff_sno[2])[1] > FT(0)
        @test center_profile(np.Np_sno[2])[1] > FT(0)
    end

end
