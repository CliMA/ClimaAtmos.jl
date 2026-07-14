using Test
import ClimaAtmos as CA
using ClimaCore: Domains, Meshes, Spaces, Fields, Geometry

# Generated with D:/SURF/cosp_subcolumn_ref_driver.f90 from COSPv2 commit
# 5eb05e51187dd2d0e448b78c4e6b28e1d6f65493. COSPv2 profiles are ordered
# from model top to surface; reverse their level dimension for ClimaAtmos.
const COSPV2_SUBCOLUMN_REFERENCES = (;
    scops_maximum = (;
        cloud_fraction = Float64[1, 0.75, 0.5, 0.25, 0],
        cloud_mask = Float64[
            1 1 1 1 0
            1 1 1 0 0
            1 1 0 0 0
            1 0 0 0 0
        ],
    ),
    precip_primary_and_inheritance = (;
        large_scale_flux = Float64[1, 1, 1],
        cloud_mask = Float64[1 0 0; 0 0 0],
        precip_mask = Float64[1 1 1; 0 0 0],
    ),
    precip_cloud_below = (;
        large_scale_flux = Float64[1, 1, 1],
        cloud_mask = Float64[0 1 0; 0 0 0],
        precip_mask = Float64[1 1 1; 0 0 0],
    ),
    precip_cloud_elsewhere = (;
        large_scale_flux = Float64[1, 1, 1],
        cloud_mask = Float64[0 0 1; 0 0 0],
        precip_mask = Float64[1 1 1; 0 0 0],
    ),
    precip_all_clear_fallback = (;
        large_scale_flux = Float64[1, 1, 1],
        cloud_mask = Float64[0 0 0; 0 0 0],
        precip_mask = Float64[1 1 1; 1 1 1],
    ),
    precip_zero_flux_interrupts = (;
        large_scale_flux = Float64[1, 0, 1],
        cloud_mask = Float64[1 0 1; 0 0 0],
        precip_mask = Float64[1 0 1; 0 0 0],
    ),
    hydrometeor_large_scale = (;
        cloud_mask = Float64[
            1 0 1
            0 1 0
            0 0 0
            1 1 0
        ],
        precip_mask = Float64[
            1 0 0
            0 3 0
            0 0 0
            1 3 0
        ],
        grid_mean_cloud = Float64[1, 2, 4],
        grid_mean_precip = Float64[10, 20, 0],
        sampled_cloud_fraction = Float64[0.5, 0.5, 0.25],
        sampled_precip_fraction = Float64[0.5, 0.5, 0],
        cloud_subcolumns = Float64[
            2 0 16
            0 4 0
            0 0 0
            2 4 0
        ],
        precip_subcolumns = Float64[
            20 0 0
            0 40 0
            0 0 0
            20 40 0
        ],
    ),
)

function make_subcol_simulation(
    device;
    job_id,
    microphysics_model = "0M",
    z_elem = 10,
)
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "DYCOMS_RF02",
            "microphysics_model" => microphysics_model,
            "config" => "column",
            "output_default_diagnostics" => false,
            "dt_subcol" => "10mins",
            "device" => device,
            "z_elem" => z_elem,
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

function set_center_profile!(field, profile)
    @assert Spaces.nlevels(axes(field)) == length(profile)
    FT = eltype(field)
    for (ilev, value) in enumerate(profile)
        Fields.level(field, ilev) .= FT(value)
    end
    return nothing
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

cosp_bottom_to_top(profile) = reverse(collect(profile))

function make_cosp_subcolumn_fields(FT, matrix)
    return ntuple(size(matrix, 1)) do isubcolumn
        make_center_profile_field(
            FT,
            cosp_bottom_to_top(matrix[isubcolumn, :]),
        )
    end
end

function materialize_scops!(
    frac_out,
    threshold,
    cloud_fraction,
    random_seed;
    overlap,
)
    @assert length(frac_out) == length(threshold)
    nsubcolumns = length(frac_out)
    for isubcolumn in 1:nsubcolumns
        CA.COSP.COSPSubcolumns.scops_subcolumn!(
            frac_out[isubcolumn],
            threshold[isubcolumn],
            cloud_fraction,
            isubcolumn,
            nsubcolumns,
            random_seed;
            overlap,
        )
    end
    return nothing
end

function make_hydrometeor_subcolumns(grid_mean, nsubcolumns)
    subcolumn_values =
        map(
            field -> ntuple(_ -> similar(field), nsubcolumns),
            Base.values(grid_mean),
        )
    return NamedTuple{keys(grid_mean)}(subcolumn_values)
end

function slice_materialized_hydrometeors!(
    subcolumns,
    cloud_masks,
    precip_masks,
    grid_mean,
    sampled_cloud_fraction,
    sampled_precip_fraction,
)
    @assert length(cloud_masks) == length(precip_masks)
    nsubcolumns = length(cloud_masks)
    FT = eltype(sampled_cloud_fraction)
    @. sampled_cloud_fraction = zero(FT)
    @. sampled_precip_fraction = zero(FT)
    for isubcolumn in 1:nsubcolumns
        CA.COSP.COSPHydrometeorSubcolumns.accumulate_sampled_cloud_fraction!(
            sampled_cloud_fraction,
            cloud_masks[isubcolumn],
            nsubcolumns,
        )
        CA.COSP.COSPHydrometeorSubcolumns.accumulate_sampled_precip_fraction!(
            sampled_precip_fraction,
            precip_masks[isubcolumn],
            nsubcolumns,
        )
    end
    for isubcolumn in 1:nsubcolumns
        output = map(fields -> fields[isubcolumn], subcolumns)
        CA.COSP.COSPHydrometeorSubcolumns.slice_hydrometeor_subcolumn!(
            output,
            cloud_masks[isubcolumn],
            precip_masks[isubcolumn],
            grid_mean,
            sampled_cloud_fraction,
            sampled_precip_fraction,
        )
    end
    return nothing
end

expected_maximum_mask(FT, isubcolumn, nsubcolumns, cloud_fraction) =
    FT(cloud_fraction) > (FT(isubcolumn) - FT(0.5)) / FT(nsubcolumns) ? FT(1) : FT(0)

function reference_scops_profiles(FT, cloud_profile, nsubcolumns, seed, overlap)
    nlev = length(cloud_profile)
    coords = center_profile(
        Fields.coordinate_field(axes(make_center_profile_field(FT, cloud_profile))),
    )
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

function make_precip_scratch(FT, nlev)
    return (;
        cloud = make_center_field(FT; value = 0, nelems = nlev),
        cloud_below = make_center_field(FT; value = 0, nelems = nlev),
        any_cloud = make_center_field(FT; value = 0, nelems = nlev),
        column_any = make_center_field(FT; value = 0, nelems = nlev),
    )
end

function selectors_from_cloud_masks(FT, cloud_masks)
    nlev = Spaces.nlevels(axes(first(cloud_masks)))
    has_cloud = make_center_field(FT; value = 0, nelems = nlev)
    for cloud_mask in cloud_masks
        @. has_cloud = max(has_cloud, ifelse(cloud_mask > 0, FT(1), FT(0)))
    end
    has_cloud_below = similar(has_cloud)
    has_cloud_anywhere = similar(has_cloud)
    scratch = similar(has_cloud)
    CA.COSP.COSPSubcolumns.shift_up!(has_cloud_below, has_cloud)
    CA.COSP.COSPSubcolumns.column_any!(
        has_cloud_anywhere,
        has_cloud,
        scratch,
    )
    return (; has_cloud, has_cloud_below, has_cloud_anywhere)
end

function streamed_precipitation(cloud_masks, flux)
    FT = eltype(flux)
    nlev = Spaces.nlevels(axes(flux))
    selectors = selectors_from_cloud_masks(FT, cloud_masks)
    scratch = make_precip_scratch(FT, nlev)
    outputs = ntuple(_ -> similar(flux), length(cloud_masks))
    for isubcolumn in eachindex(cloud_masks)
        CA.COSP.COSPPrecipSubcolumns.scops_subcolumn_precip!(
            outputs[isubcolumn],
            cloud_masks[isubcolumn],
            flux,
            selectors,
            scratch,
        )
    end
    return outputs
end

function reference_precipitation(cloud_profiles, flux_profile)
    nsubcolumns = length(cloud_profiles)
    nlev = length(flux_profile)
    precip = ntuple(_ -> zeros(eltype(flux_profile), nlev), nsubcolumns)
    has_cloud_anywhere = map(profile -> any(>(0), profile), cloud_profiles)
    for ilev in nlev:-1:1
        flux_profile[ilev] > 0 || continue
        selected = false
        for isubcolumn in 1:nsubcolumns
            cloud = cloud_profiles[isubcolumn][ilev] > 0
            inherited = ilev < nlev && precip[isubcolumn][ilev + 1] > 0
            if cloud || inherited
                precip[isubcolumn][ilev] = one(eltype(flux_profile))
                selected = true
            end
        end
        if !selected && ilev > 1
            for isubcolumn in 1:nsubcolumns
                if cloud_profiles[isubcolumn][ilev - 1] > 0
                    precip[isubcolumn][ilev] = one(eltype(flux_profile))
                    selected = true
                end
            end
        end
        if !selected
            for isubcolumn in 1:nsubcolumns
                if has_cloud_anywhere[isubcolumn]
                    precip[isubcolumn][ilev] = one(eltype(flux_profile))
                    selected = true
                end
            end
        end
        if !selected
            for isubcolumn in 1:nsubcolumns
                precip[isubcolumn][ilev] = one(eltype(flux_profile))
            end
        end
    end
    return precip
end

@testset "COSP subcolumns" begin
    FT = Float64
    seed = UInt64(1)

    @testset "point RNG supports Float32 and Float64" begin
        for RNGFT in (Float32, Float64)
            coords = (; x = RNGFT(1.25), y = RNGFT(-2.5), z = RNGFT(300))
            random_number =
                CA.COSP.COSPSubcolumns._rand_for_point(seed, coords, 1)

            @test random_number isa RNGFT
            @test zero(RNGFT) <= random_number < one(RNGFT)
            @test random_number ==
                  CA.COSP.COSPSubcolumns._rand_for_point(seed, coords, 1)
            @test random_number !=
                  CA.COSP.COSPSubcolumns._rand_for_point(
                seed + one(seed),
                coords,
                1,
            )
            @test random_number !=
                  CA.COSP.COSPSubcolumns._rand_for_point(seed, coords, 2)
            @test random_number != CA.COSP.COSPSubcolumns._rand_for_point(
                seed,
                merge(coords, (; z = RNGFT(301))),
                1,
            )

            swapped_xy = (; x = coords.y, y = coords.x, z = coords.z)
            @test random_number !=
                  CA.COSP.COSPSubcolumns._rand_for_point(seed, swapped_xy, 1)

            lat_long = (; lat = RNGFT(10), long = RNGFT(20), z = coords.z)
            swapped_lat_long =
                (; lat = lat_long.long, long = lat_long.lat, z = coords.z)
            @test CA.COSP.COSPSubcolumns._rand_for_point(seed, lat_long, 1) !=
                  CA.COSP.COSPSubcolumns._rand_for_point(
                seed,
                swapped_lat_long,
                1,
            )

            endpoint =
                CA.COSP.COSPSubcolumns._uint64_to_unit_interval(
                    RNGFT,
                    typemax(UInt64),
                )
            @test zero(RNGFT) <= endpoint < one(RNGFT)
            @test endpoint == prevfloat(one(RNGFT))

            cloud_fraction =
                make_center_profile_field(RNGFT, fill(one(RNGFT), 3))
            cloud_s = similar(cloud_fraction)
            threshold_s = similar(cloud_fraction)
            for isubcolumn in 1:4
                CA.COSP.COSPSubcolumns.scops_subcolumn!(
                    cloud_s,
                    threshold_s,
                    cloud_fraction,
                    isubcolumn,
                    4,
                    seed;
                    overlap = :maximum_random,
                )
                @test all(==(one(RNGFT)), parent(cloud_s))
                @test all(x -> zero(RNGFT) <= x < one(RNGFT), parent(threshold_s))
            end
        end
    end

    @testset "COSPv2 golden references" begin
        @testset "SCOPS maximum overlap" begin
            reference = COSPV2_SUBCOLUMN_REFERENCES.scops_maximum
            cloud_fraction = make_center_profile_field(
                FT,
                cosp_bottom_to_top(reference.cloud_fraction),
            )
            nsubcolumns = size(reference.cloud_mask, 1)
            frac_out = make_subcolumn_fields(
                FT,
                nsubcolumns,
                length(reference.cloud_fraction),
            )
            threshold = make_subcolumn_fields(
                FT,
                nsubcolumns,
                length(reference.cloud_fraction),
            )

            result = materialize_scops!(
                frac_out,
                threshold,
                cloud_fraction,
                seed;
                overlap = :maximum,
            )

            @test isnothing(result)
            for isubcolumn in 1:nsubcolumns
                expected = cosp_bottom_to_top(
                    reference.cloud_mask[isubcolumn, :],
                )
                @test center_profile(frac_out[isubcolumn]) == expected
            end
        end

        @testset "PREC_SCOPS large-scale placement" begin
            case_names = (
                :precip_primary_and_inheritance,
                :precip_cloud_below,
                :precip_cloud_elsewhere,
                :precip_all_clear_fallback,
                :precip_zero_flux_interrupts,
            )
            for case_name in case_names
                reference = getproperty(COSPV2_SUBCOLUMN_REFERENCES, case_name)
                cloud_masks =
                    make_cosp_subcolumn_fields(FT, reference.cloud_mask)
                flux = make_center_profile_field(
                    FT,
                    cosp_bottom_to_top(reference.large_scale_flux),
                )

                actual = streamed_precipitation(cloud_masks, flux)

                for isubcolumn in eachindex(actual)
                    expected = cosp_bottom_to_top(
                        reference.precip_mask[isubcolumn, :],
                    )
                    @test center_profile(actual[isubcolumn]) == expected
                end
            end
        end

        @testset "COSP large-scale hydrometeor distribution" begin
            reference = COSPV2_SUBCOLUMN_REFERENCES.hydrometeor_large_scale
            cloud_masks = make_cosp_subcolumn_fields(FT, reference.cloud_mask)
            precip_masks = make_cosp_subcolumn_fields(FT, reference.precip_mask)
            grid_mean = (;
                q_lcl = make_center_profile_field(
                    FT,
                    cosp_bottom_to_top(reference.grid_mean_cloud),
                ),
                q_rai = make_center_profile_field(
                    FT,
                    cosp_bottom_to_top(reference.grid_mean_precip),
                ),
            )
            subcolumns =
                make_hydrometeor_subcolumns(grid_mean, length(cloud_masks))
            sampled_cloud_fraction = similar(grid_mean.q_lcl)
            sampled_precip_fraction = similar(grid_mean.q_rai)

            result = slice_materialized_hydrometeors!(
                subcolumns,
                cloud_masks,
                precip_masks,
                grid_mean,
                sampled_cloud_fraction,
                sampled_precip_fraction,
            )

            @test isnothing(result)
            @test center_profile(sampled_cloud_fraction) == cosp_bottom_to_top(
                reference.sampled_cloud_fraction,
            )
            @test center_profile(sampled_precip_fraction) == cosp_bottom_to_top(
                reference.sampled_precip_fraction,
            )
            for isubcolumn in eachindex(cloud_masks)
                expected_cloud = cosp_bottom_to_top(
                    reference.cloud_subcolumns[isubcolumn, :],
                )
                expected_precip = cosp_bottom_to_top(
                    reference.precip_subcolumns[isubcolumn, :],
                )
                @test center_profile(subcolumns.q_lcl[isubcolumn]) ==
                      expected_cloud
                @test center_profile(subcolumns.q_rai[isubcolumn]) ==
                      expected_precip
            end
            @test mean_profile(subcolumns.q_lcl) == center_profile(grid_mean.q_lcl)
            @test mean_profile(subcolumns.q_rai) == center_profile(grid_mean.q_rai)
        end
    end

    @testset "p.precomputed.ᶜcloud_fraction input" begin
        simulation = make_subcol_simulation(
            "CPUSingleThreaded";
            job_id = "cosp_subcol_precomputed",
        )
        p = simulation.integrator.p

        cloud_fraction = p.precomputed.ᶜcloud_fraction
        @. cloud_fraction = FT(0.4)

        cloud_s = p.precomputed.ᶜsubcolumn_cloud
        nsubcolumns = length(p.precomputed.ᶜsubcolumn_hydrometeors.q_lcl)
        threshold = p.precomputed.ᶜsubcolumn_threshold
        precip_s = p.precomputed.ᶜsubcolumn_precip
        selectors = p.precomputed.ᶜscops_selectors
        scratch = p.precomputed.ᶜprecip_subcolumn_scratch
        large_scale_precipitation_flux =
            p.precomputed.ᶜlarge_scale_precipitation_flux

        result = CA.COSP.COSPSubcolumns.set_scops_selectors!(
            selectors,
            cloud_s,
            threshold,
            cloud_fraction,
            nsubcolumns,
            seed,
            :maximum,
            scratch.column_any,
        )

        @test isnothing(result)
        @test !(cloud_s isa Tuple)
        @test !(threshold isa Tuple)
        @test !(precip_s isa Tuple)
        @test all(
            axes(field) == axes(cloud_fraction) for
            field in values(selectors)
        )
        @test axes(large_scale_precipitation_flux) == axes(cloud_fraction)

        @. large_scale_precipitation_flux = FT(1)
        for isubcolumn in 1:nsubcolumns
            CA.COSP.COSPSubcolumns.scops_subcolumn!(
                cloud_s,
                threshold,
                cloud_fraction,
                isubcolumn,
                nsubcolumns,
                seed;
                overlap = :maximum,
            )
            precip_result =
                CA.COSP.COSPPrecipSubcolumns.scops_subcolumn_precip!(
                    precip_s,
                    cloud_s,
                    large_scale_precipitation_flux,
                    selectors,
                    scratch,
                )
            expected = expected_maximum_mask(FT, isubcolumn, nsubcolumns, 0.4)
            @test isnothing(precip_result)
            @test all(==(expected), parent(cloud_s))
            @test all(==(expected), parent(precip_s))
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

        @test all(==(FT(1)), parent(p.precomputed.ᶜsubcolumn_cloud))
        @test all(==(FT(1)), parent(p.precomputed.ᶜsampled_cloud_fraction))
    end

    @testset "1M callback streams masks and conserves hydrometeors" begin
        simulation = make_subcol_simulation(
            "CPUSingleThreaded";
            job_id = "cosp_subcol_1m_callback",
            microphysics_model = "1M",
            z_elem = 4,
        )
        integrator = simulation.integrator
        Y = integrator.u
        p = integrator.p

        density_profile = FT[1, 2, 1, 2]
        cloud_fraction_profile = FT[0.25, 0.5, 0.75, 0.1]
        ρq_lcl_profile = FT[0.2, 0.4, 0, 0.2]
        ρq_icl_profile = FT[0.1, 0, 0.2, 0.4]
        ρq_rai_profile = FT[0.1, 0.2, 0.3, 0.4]
        ρq_sno_profile = FT[0.2, 0.1, 0.1, 0.2]
        rain_velocity_profile = FT[1, 2, -1, 1]
        snow_velocity_profile = FT[0.5, 1, -1, -3]

        set_center_profile!(Y.c.ρ, density_profile)
        set_center_profile!(Y.c.ρq_lcl, ρq_lcl_profile)
        set_center_profile!(Y.c.ρq_icl, ρq_icl_profile)
        set_center_profile!(Y.c.ρq_rai, ρq_rai_profile)
        set_center_profile!(Y.c.ρq_sno, ρq_sno_profile)
        set_center_profile!(
            p.precomputed.ᶜcloud_fraction,
            cloud_fraction_profile,
        )
        set_center_profile!(p.precomputed.ᶜwᵣ, rain_velocity_profile)
        set_center_profile!(p.precomputed.ᶜwₛ, snow_velocity_profile)

        CA.subcol_model_callback!(integrator)

        expected_flux = similar(p.precomputed.ᶜlarge_scale_precipitation_flux)
        @. expected_flux = max(
            FT(0),
            Y.c.ρq_rai * p.precomputed.ᶜwᵣ +
            Y.c.ρq_sno * p.precomputed.ᶜwₛ,
        )
        @test center_profile(p.precomputed.ᶜlarge_scale_precipitation_flux) ≈
              center_profile(expected_flux)

        sampled_cloud_fraction = p.precomputed.ᶜsampled_cloud_fraction
        sampled_precip_fraction = p.precomputed.ᶜsampled_precip_fraction
        for fraction in (sampled_cloud_fraction, sampled_precip_fraction)
            @test all(isfinite, parent(fraction))
            @test all(x -> zero(FT) <= x <= one(FT), parent(fraction))
        end

        nsubcolumns = length(p.precomputed.ᶜsubcolumn_hydrometeors.q_lcl)
        explicit_cloud_fraction = similar(sampled_cloud_fraction)
        explicit_precip_fraction = similar(sampled_precip_fraction)
        @. explicit_cloud_fraction = zero(FT)
        @. explicit_precip_fraction = zero(FT)
        cloud_s = similar(p.precomputed.ᶜsubcolumn_cloud)
        threshold_s = similar(p.precomputed.ᶜsubcolumn_threshold)
        precip_s = similar(p.precomputed.ᶜsubcolumn_precip)
        selectors = (;
            has_cloud = similar(p.precomputed.ᶜcloud_fraction),
            has_cloud_below = similar(p.precomputed.ᶜcloud_fraction),
            has_cloud_anywhere = similar(p.precomputed.ᶜcloud_fraction),
        )
        scratch = (;
            cloud = similar(p.precomputed.ᶜcloud_fraction),
            cloud_below = similar(p.precomputed.ᶜcloud_fraction),
            any_cloud = similar(p.precomputed.ᶜcloud_fraction),
            column_any = similar(p.precomputed.ᶜcloud_fraction),
        )
        cosp = p.atmos.cosp
        CA.COSP.COSPSubcolumns.set_scops_selectors!(
            selectors,
            cloud_s,
            threshold_s,
            p.precomputed.ᶜcloud_fraction,
            nsubcolumns,
            cosp.random_seed,
            cosp.overlap,
            scratch.column_any,
        )
        for isubcolumn in 1:nsubcolumns
            CA.COSP.COSPSubcolumns.scops_subcolumn!(
                cloud_s,
                threshold_s,
                p.precomputed.ᶜcloud_fraction,
                isubcolumn,
                nsubcolumns,
                cosp.random_seed;
                overlap = cosp.overlap,
            )
            CA.COSP.COSPPrecipSubcolumns.scops_subcolumn_precip!(
                precip_s,
                cloud_s,
                p.precomputed.ᶜlarge_scale_precipitation_flux,
                selectors,
                scratch,
            )
            CA.COSP.COSPHydrometeorSubcolumns.accumulate_sampled_cloud_fraction!(
                explicit_cloud_fraction,
                cloud_s,
                nsubcolumns,
            )
            CA.COSP.COSPHydrometeorSubcolumns.accumulate_sampled_precip_fraction!(
                explicit_precip_fraction,
                precip_s,
                nsubcolumns,
            )
        end
        @test center_profile(sampled_cloud_fraction) ≈
              center_profile(explicit_cloud_fraction)
        @test center_profile(sampled_precip_fraction) ≈
              center_profile(explicit_precip_fraction)

        expected_grid_mean = (;
            q_lcl = ρq_lcl_profile ./ density_profile,
            q_icl = ρq_icl_profile ./ density_profile,
            q_rai = ρq_rai_profile ./ density_profile,
            q_sno = ρq_sno_profile ./ density_profile,
        )
        for name in keys(expected_grid_mean)
            stored_subcolumns =
                getproperty(p.precomputed.ᶜsubcolumn_hydrometeors, name)
            @test mean_profile(stored_subcolumns) ≈
                  getproperty(expected_grid_mean, name)
        end
    end

    @testset "maximum-random overlap carries previous threshold" begin
        cloud_fraction = make_center_profile_field(FT, [1, 1, 1])
        frac_out = make_subcolumn_fields(FT, 4, 3)
        threshold = make_subcolumn_fields(FT, 4, 3)

        result = materialize_scops!(
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

            materialize_scops!(
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

                result = materialize_scops!(
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

        result = materialize_scops!(
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

    @testset "one-subcolumn generation is exact and reproducible" begin
        cloud_fraction = make_center_profile_field(FT, [0.2, 0.8, 0.4, 1.0])
        nsubcolumns = 4
        all_cloud = make_subcolumn_fields(FT, nsubcolumns, 4)
        all_threshold = make_subcolumn_fields(FT, nsubcolumns, 4)
        materialize_scops!(
            all_cloud,
            all_threshold,
            cloud_fraction,
            seed;
            overlap = :maximum_random,
        )

        cloud_s = similar(cloud_fraction)
        threshold_s = similar(cloud_fraction)
        for isubcolumn in 1:nsubcolumns
            CA.COSP.COSPSubcolumns.scops_subcolumn!(
                cloud_s,
                threshold_s,
                cloud_fraction,
                isubcolumn,
                nsubcolumns,
                seed;
                overlap = :maximum_random,
            )
            @test parent(cloud_s) == parent(all_cloud[isubcolumn])
            @test parent(threshold_s) == parent(all_threshold[isubcolumn])

            cloud_bits = copy(parent(cloud_s))
            threshold_bits = copy(parent(threshold_s))
            @. cloud_s = FT(-1)
            @. threshold_s = FT(-1)
            CA.COSP.COSPSubcolumns.scops_subcolumn!(
                cloud_s,
                threshold_s,
                cloud_fraction,
                isubcolumn,
                nsubcolumns,
                seed;
                overlap = :maximum_random,
            )
            @test isequal(parent(cloud_s), cloud_bits)
            @test isequal(parent(threshold_s), threshold_bits)
        end

        maximum_cloud = make_center_profile_field(FT, [0.5])
        cloud_s = similar(maximum_cloud)
        threshold_s = similar(maximum_cloud)
        CA.COSP.COSPSubcolumns.scops_subcolumn!(
            cloud_s,
            threshold_s,
            maximum_cloud,
            1,
            nsubcolumns,
            seed;
            overlap = :maximum,
        )
        first_mask = copy(parent(cloud_s))
        CA.COSP.COSPSubcolumns.scops_subcolumn!(
            cloud_s,
            threshold_s,
            maximum_cloud,
            4,
            nsubcolumns,
            seed;
            overlap = :maximum,
        )
        @test first_mask != parent(cloud_s)
    end

    @testset "backend-safe vertical selector helpers" begin
        input = make_center_profile_field(FT, [1, 2, 3, 4])
        output = similar(input)
        scratch = similar(input)
        CA.COSP.COSPSubcolumns.shift_up!(output, input)
        @test center_profile(output) == FT[0, 1, 2, 3]

        for profile in (FT[0, 0, 0], FT[0, 0, 1], FT[0, 1, 0], FT[1, 0, 0])
            input = make_center_profile_field(FT, profile)
            output = similar(input)
            scratch = similar(input)
            CA.COSP.COSPSubcolumns.column_any!(output, input, scratch)
            expected = any(x -> !iszero(x), profile) ? ones(FT, 3) : zeros(FT, 3)
            @test center_profile(output) == expected
        end
    end

    @testset "selectors reduce actual finite sampled masks" begin
        nsubcolumns = 4
        cloud_fraction = make_center_profile_field(FT, [0.1, 0.6, 0.1])
        cloud_s = similar(cloud_fraction)
        threshold_s = similar(cloud_fraction)
        selectors = (;
            has_cloud = similar(cloud_fraction),
            has_cloud_below = similar(cloud_fraction),
            has_cloud_anywhere = similar(cloud_fraction),
        )
        scratch = similar(cloud_fraction)
        CA.COSP.COSPSubcolumns.set_scops_selectors!(
            selectors,
            cloud_s,
            threshold_s,
            cloud_fraction,
            nsubcolumns,
            seed,
            :maximum,
            scratch,
        )

        explicit_masks = make_subcolumn_fields(FT, nsubcolumns, 3)
        explicit_thresholds = make_subcolumn_fields(FT, nsubcolumns, 3)
        materialize_scops!(
            explicit_masks,
            explicit_thresholds,
            cloud_fraction,
            seed;
            overlap = :maximum,
        )
        explicit = selectors_from_cloud_masks(FT, explicit_masks)
        for name in keys(selectors)
            @test parent(getproperty(selectors, name)) ==
                  parent(getproperty(explicit, name))
        end

        @test center_profile(cloud_fraction)[1] > 0
        @test center_profile(selectors.has_cloud)[[1, 3]] == FT[0, 0]

        finite_clear_fraction = make_center_profile_field(FT, [0.1])
        finite_clear_masks = make_subcolumn_fields(FT, nsubcolumns, 1)
        finite_clear_thresholds = make_subcolumn_fields(FT, nsubcolumns, 1)
        materialize_scops!(
            finite_clear_masks,
            finite_clear_thresholds,
            finite_clear_fraction,
            seed;
            overlap = :maximum,
        )
        finite_clear_selectors = selectors_from_cloud_masks(FT, finite_clear_masks)
        @test center_profile(finite_clear_selectors.has_cloud) == FT[0]
        precip = streamed_precipitation(
            finite_clear_masks,
            make_center_profile_field(FT, [1]),
        )
        @test all(center_profile(mask) == FT[1] for mask in precip)
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
        prec_frac = streamed_precipitation(frac_out, large_scale_precipitation_flux)

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
        prec_frac = streamed_precipitation(frac_out, large_scale_precipitation_flux)

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
        prec_frac = streamed_precipitation(frac_out, large_scale_precipitation_flux)

        @test center_profile(prec_frac[1]) == FT[0, 0, 0]
        @test center_profile(prec_frac[2]) == FT[0, 1, 0]
        @test center_profile(prec_frac[3]) == FT[0, 0, 0]
        @test center_profile(prec_frac[4]) == FT[0, 0, 0]
    end

    @testset "all four precipitation placement rules" begin
        rule1_cloud = (
            make_center_profile_field(FT, [0, 0, 1]),
            make_center_profile_field(FT, [0, 0, 0]),
        )
        rule1 = streamed_precipitation(
            rule1_cloud,
            make_center_profile_field(FT, [1, 1, 1]),
        )
        @test center_profile(rule1[1]) == FT[1, 1, 1]
        @test center_profile(rule1[2]) == FT[0, 0, 0]

        rule2_cloud = (
            make_center_profile_field(FT, [1, 0, 0]),
            make_center_profile_field(FT, [0, 0, 1]),
        )
        rule2 = streamed_precipitation(
            rule2_cloud,
            make_center_profile_field(FT, [0, 1, 0]),
        )
        @test center_profile(rule2[1]) == FT[0, 1, 0]
        @test center_profile(rule2[2]) == FT[0, 0, 0]

        rule3_cloud = (
            make_center_profile_field(FT, [1, 0, 0]),
            make_center_profile_field(FT, [0, 0, 0]),
        )
        rule3 = streamed_precipitation(
            rule3_cloud,
            make_center_profile_field(FT, [0, 0, 1]),
        )
        @test center_profile(rule3[1]) == FT[0, 0, 1]
        @test center_profile(rule3[2]) == FT[0, 0, 0]

        clear = ntuple(_ -> make_center_profile_field(FT, [0, 0, 0]), 2)
        rule4 = streamed_precipitation(
            clear,
            make_center_profile_field(FT, [0, 0, 1]),
        )
        @test all(center_profile(mask) == FT[0, 0, 1] for mask in rule4)
    end

    @testset "zero and NaN flux interrupt precipitation inheritance" begin
        cloud_masks = (
            make_center_profile_field(FT, [0, 0, 1]),
            make_center_profile_field(FT, [1, 0, 0]),
        )
        for middle_flux in (FT(0), FT(NaN))
            precip = streamed_precipitation(
                cloud_masks,
                make_center_profile_field(FT, [1, middle_flux, 1]),
            )
            @test center_profile(precip[1]) == FT[0, 0, 1]
            @test center_profile(precip[2]) == FT[1, 0, 0]
        end
    end

    @testset "streamed precipitation matches independent COSP-style reference" begin
        cases = (
            (
                (FT[0, 0, 0], FT[0, 1, 0], FT[0, 0, 1]),
                FT[1, 1, 0],
            ),
            (
                (FT[1, 0, 0, 0], FT[0, 0, 0, 1], FT[0, 0, 0, 0]),
                FT[1, 1, 1, 1],
            ),
            (
                (FT[0, 0, 0], FT[0, 0, 0]),
                FT[1, 0, 1],
            ),
        )
        for (cloud_profiles, flux_profile) in cases
            masks = map(profile -> make_center_profile_field(FT, profile), cloud_profiles)
            flux = make_center_profile_field(FT, flux_profile)
            actual = streamed_precipitation(masks, flux)
            expected = reference_precipitation(cloud_profiles, flux_profile)
            for isubcolumn in eachindex(actual)
                @test center_profile(actual[isubcolumn]) == expected[isubcolumn]
            end
        end
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

        result = slice_materialized_hydrometeors!(
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

        slice_materialized_hydrometeors!(
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

    @testset "one streamed hydrometeor subcolumn uses final fractions" begin
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
        @. sampled_cloud_fraction = FT(0.5)
        @. sampled_precip_fraction = FT(0.5)
        grid_mean = (;
            q_lcl = make_center_profile_field(FT, [1, 2]),
            q_icl = make_center_profile_field(FT, [0.5, 1]),
            q_rai = make_center_profile_field(FT, [10, 20]),
            q_sno = make_center_profile_field(FT, [5, 15]),
        )

        output = map(fields -> fields[1], subcolumns)
        result = CA.COSP.COSPHydrometeorSubcolumns.slice_hydrometeor_subcolumn!(
            output,
            cloud_mask[1],
            precip_mask[1],
            grid_mean,
            sampled_cloud_fraction,
            sampled_precip_fraction,
        )

        @test isnothing(result)
        @test center_profile(subcolumns.q_lcl[1]) == FT[2, 0]
        @test center_profile(subcolumns.q_rai[1]) == FT[20, 0]
    end

end
