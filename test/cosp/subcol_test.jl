using Test
import ClimaAtmos as CA
using ClimaCore: Domains, Meshes, Spaces, Fields, Geometry

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
)

# Generated with COSPv2 commit 5eb05e51187dd2d0e448b78c4e6b28e1d6f65493.
# Profiles are ordered from model top to surface and use random seed 1.
const COSPV2_1M_STREAMED_E2E_REFERENCE = (;
    nsubcolumns = 4,
    overlap = :maximum,
    density = Float32[2, 1, 2, 1],
    cloud_fraction = Float32[0.2, 0.4, 0.6, 0.8],
    rho_q_lcl = Float32[2e-4, 2e-4, 8e-4, 8e-4],
    rho_q_icl = Float32[1e-4, 1e-4, 4e-4, 4e-4],
    rho_q_rai = Float32[4e-5, 5e-5, 2e-4, 2e-4],
    rho_q_sno = Float32[2e-5, 2e-5, 1e-4, 1e-4],
    w_rain = Float32[1.5, 0.75, 2, 1],
    w_snow = Float32[0.75, 0.25, 1, 0.5],
    large_scale_precipitation_flux = Float32[7.5e-5, 4.25e-5, 5e-4, 2.5e-4],
    cloud_masks = Float32[
        1 1 1 1
        0 1 1 1
        0 0 0 1
        0 0 0 0
    ],
    precip_masks = Float32[
        1 1 1 1
        0 1 1 1
        0 0 0 1
        0 0 0 0
    ],
    sampled_cloud_fraction = Float32[0.25, 0.5, 0.5, 0.75],
    sampled_precip_fraction = Float32[0.25, 0.5, 0.5, 0.75],
    q_lcl_subcolumns = Float32[
        4e-4 4e-4 8e-4 1.0666667e-3
        0 4e-4 8e-4 1.0666667e-3
        0 0 0 1.0666667e-3
        0 0 0 0
    ],
    q_icl_subcolumns = Float32[
        2e-4 2e-4 4e-4 5.3333334e-4
        0 2e-4 4e-4 5.3333334e-4
        0 0 0 5.3333334e-4
        0 0 0 0
    ],
    q_rai_subcolumns = Float32[
        8e-5 1e-4 2e-4 2.6666667e-4
        0 1e-4 2e-4 2.6666667e-4
        0 0 0 2.6666667e-4
        0 0 0 0
    ],
    q_sno_subcolumns = Float32[
        4e-5 4e-5 1e-4 1.3333333e-4
        0 4e-5 1e-4 1.3333333e-4
        0 0 0 1.3333333e-4
        0 0 0 0
    ],
)

function make_1m_subcol_simulation(;
    z_elem = 10,
    cosp_n_subcolumns = 256,
    cosp_overlap = "maximum_random",
)
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "DYCOMS_RF02",
            "microphysics_model" => "1M",
            "config" => "column",
            "output_default_diagnostics" => false,
            "dt_subcol" => "10mins",
            "cosp_n_subcolumns" => cosp_n_subcolumns,
            "cosp_overlap" => cosp_overlap,
            "device" => "CPUSingleThreaded",
            "z_elem" => z_elem,
        );
        job_id = "cosp_subcol_1m_cospv2_golden",
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
            variants = (
                (seed + one(seed), coords, 1),
                (seed, coords, 2),
                (seed, merge(coords, (; z = RNGFT(301))), 1),
                (seed, (; x = coords.y, y = coords.x, z = coords.z), 1),
            )
            @test all(
                random_number !=
                CA.COSP.COSPSubcolumns._rand_for_point(args...) for
                args in variants
            )

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

            materialize_scops!(
                frac_out,
                threshold,
                cloud_fraction,
                seed;
                overlap = :maximum,
            )

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

    end

    @testset "1M callback matches COSPv2-derived streamed reference" begin
        reference = COSPV2_1M_STREAMED_E2E_REFERENCE
        reference_tolerance = 5eps(Float32)

        simulation = make_1m_subcol_simulation(;
            z_elem = length(reference.density),
            cosp_n_subcolumns = reference.nsubcolumns,
            cosp_overlap = String(reference.overlap),
        )
        Y = simulation.integrator.u
        p = simulation.integrator.p
        @test CA._cosp_nsubcolumns(p.atmos.cosp.n_subcolumns) ==
              reference.nsubcolumns
        @test p.atmos.cosp.overlap === reference.overlap
        @test all(
            field -> !(field isa Tuple),
            (
                p.precomputed.ᶜsubcolumn_cloud,
                p.precomputed.ᶜsubcolumn_threshold,
                p.precomputed.ᶜsubcolumn_precip,
            ),
        )
        @test !hasproperty(p.precomputed, :ᶜsubcolumn_hydrometeors)
        @test length(p.precomputed.DBZe_cloudsat) == reference.nsubcolumns
        for removed_cache in (
            :z_vol_cloudsat,
            :kr_vol_cloudsat,
            :Ze_non_cloudsat,
        )
            @test !hasproperty(p.precomputed, removed_cache)
        end
        @test all(
            field -> !(field isa Tuple),
            (
                p.precomputed.z_vol_cloudsat_work,
                p.precomputed.kr_vol_cloudsat_work,
                p.precomputed.Ze_non_cloudsat_work,
                p.precomputed.hydro_path_attenuation_cloudsat_work,
                p.precomputed.gas_path_attenuation_cloudsat,
            ),
        )

        # COSPv2 writes levels from model top to surface. ClimaAtmos center
        # fields use level 1 at the surface, so reverse every input profile.
        set_center_profile!(Y.c.ρ, cosp_bottom_to_top(reference.density))
        set_center_profile!(Y.c.ρq_lcl, cosp_bottom_to_top(reference.rho_q_lcl))
        set_center_profile!(Y.c.ρq_icl, cosp_bottom_to_top(reference.rho_q_icl))
        set_center_profile!(Y.c.ρq_rai, cosp_bottom_to_top(reference.rho_q_rai))
        set_center_profile!(Y.c.ρq_sno, cosp_bottom_to_top(reference.rho_q_sno))
        set_center_profile!(
            p.precomputed.ᶜcloud_fraction,
            cosp_bottom_to_top(reference.cloud_fraction),
        )
        set_center_profile!(
            p.precomputed.ᶜwᵣ,
            cosp_bottom_to_top(reference.w_rain),
        )
        set_center_profile!(
            p.precomputed.ᶜwₛ,
            cosp_bottom_to_top(reference.w_snow),
        )
        @test isnothing(CA.subcol_model_callback!(simulation.integrator))
        @test any(
            DBZe -> any(>(eltype(Y)(-1e30)), parent(DBZe)),
            p.precomputed.DBZe_cloudsat,
        )
        @test any(>(zero(eltype(Y))), parent(p.precomputed.cloudsat_tcc))

        nsubcolumns = reference.nsubcolumns
        grid_mean_template = (;
            q_lcl = Y.c.ρ,
            q_icl = Y.c.ρ,
            q_rai = Y.c.ρ,
            q_sno = Y.c.ρ,
        )
        actual_hydrometeors =
            make_hydrometeor_subcolumns(grid_mean_template, nsubcolumns)
        actual_cloud_masks = ntuple(
            _ -> similar(p.precomputed.ᶜsubcolumn_cloud),
            nsubcolumns,
        )
        actual_precip_masks = ntuple(
            _ -> similar(p.precomputed.ᶜsubcolumn_precip),
            nsubcolumns,
        )
        consumed_subcolumns = Int[]

        CA.foreach_cosp_subcolumn(Y, p) do isubcolumn, hydrometeors
            push!(consumed_subcolumns, isubcolumn)

            # The masks and lazy hydrometeor broadcasts borrow streamed
            # working storage, so materialize all of them before returning.
            @. actual_cloud_masks[isubcolumn] =
                p.precomputed.ᶜsubcolumn_cloud
            @. actual_precip_masks[isubcolumn] =
                p.precomputed.ᶜsubcolumn_precip
            for name in keys(actual_hydrometeors)
                output = getproperty(actual_hydrometeors, name)[isubcolumn]
                hydrometeor = getproperty(hydrometeors, name)
                @. output = hydrometeor
            end
        end

        @test consumed_subcolumns == collect(1:nsubcolumns)
        @test isapprox(
            center_profile(p.precomputed.ᶜlarge_scale_precipitation_flux),
            cosp_bottom_to_top(reference.large_scale_precipitation_flux);
            rtol = reference_tolerance,
            atol = 0,
        )
        @test center_profile(p.precomputed.ᶜsampled_cloud_fraction) ==
              cosp_bottom_to_top(reference.sampled_cloud_fraction)
        @test center_profile(p.precomputed.ᶜsampled_precip_fraction) ==
              cosp_bottom_to_top(reference.sampled_precip_fraction)

        hydrometeor_reference_names = (;
            q_lcl = :q_lcl_subcolumns,
            q_icl = :q_icl_subcolumns,
            q_rai = :q_rai_subcolumns,
            q_sno = :q_sno_subcolumns,
        )
        for isubcolumn in 1:nsubcolumns
            # Each matrix row is one subcolumn; reverse only its level axis.
            @test center_profile(actual_cloud_masks[isubcolumn]) ==
                  cosp_bottom_to_top(reference.cloud_masks[isubcolumn, :])
            @test center_profile(actual_precip_masks[isubcolumn]) ==
                  cosp_bottom_to_top(reference.precip_masks[isubcolumn, :])

            for name in keys(actual_hydrometeors)
                reference_name = getproperty(hydrometeor_reference_names, name)
                expected = cosp_bottom_to_top(
                    getproperty(reference, reference_name)[isubcolumn, :],
                )
                actual = center_profile(
                    getproperty(actual_hydrometeors, name)[isubcolumn],
                )
                @test isapprox(
                    actual,
                    expected;
                    rtol = reference_tolerance,
                    atol = 0,
                )
            end
        end

        cached_objects = (;
            z_vol = p.precomputed.z_vol_cloudsat_work,
            kr_vol = p.precomputed.kr_vol_cloudsat_work,
            g_vol = p.precomputed.g_vol_cloudsat,
            Ze_non = p.precomputed.Ze_non_cloudsat_work,
            hydro_path =
                p.precomputed.hydro_path_attenuation_cloudsat_work,
            gas_path = p.precomputed.gas_path_attenuation_cloudsat,
            height_km = p.precomputed.height_km_cloudsat,
            DBZe = p.precomputed.DBZe_cloudsat,
            detected = p.precomputed.detected_column_cloudsat,
            tcc = p.precomputed.cloudsat_tcc,
        )

        gas_before_refresh = copy(parent(p.precomputed.g_vol_cloudsat))
        state_FT = eltype(Y)
        energy_increment = state_FT(1000)
        zero_state = zero(state_FT)
        missing_reflectivity = state_FT(-1e30)
        @. Y.c.ρe_tot += Y.c.ρ * energy_increment
        CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)
        CA.subcol_model_callback!(simulation.integrator)
        @test parent(p.precomputed.g_vol_cloudsat) != gas_before_refresh

        @. Y.c.ρq_lcl = zero_state
        @. Y.c.ρq_icl = zero_state
        @. Y.c.ρq_rai = zero_state
        @. Y.c.ρq_sno = zero_state
        @. p.precomputed.ᶜcloud_fraction = zero_state
        CA.subcol_model_callback!(simulation.integrator)

        @test all(
            DBZe -> all(==(missing_reflectivity), parent(DBZe)),
            p.precomputed.DBZe_cloudsat,
        )
        @test all(iszero, parent(p.precomputed.cloudsat_tcc))
        @test all(iszero, parent(p.precomputed.z_vol_cloudsat_work))
        @test all(iszero, parent(p.precomputed.kr_vol_cloudsat_work))
        @test all(
            ==(missing_reflectivity),
            parent(p.precomputed.Ze_non_cloudsat_work),
        )
        @test all(
            iszero,
            parent(p.precomputed.hydro_path_attenuation_cloudsat_work),
        )
        @test all(!, parent(p.precomputed.detected_column_cloudsat))
        @test p.precomputed.z_vol_cloudsat_work === cached_objects.z_vol
        @test p.precomputed.kr_vol_cloudsat_work === cached_objects.kr_vol
        @test p.precomputed.g_vol_cloudsat === cached_objects.g_vol
        @test p.precomputed.Ze_non_cloudsat_work === cached_objects.Ze_non
        @test p.precomputed.hydro_path_attenuation_cloudsat_work ===
              cached_objects.hydro_path
        @test p.precomputed.gas_path_attenuation_cloudsat ===
              cached_objects.gas_path
        @test p.precomputed.height_km_cloudsat === cached_objects.height_km
        @test p.precomputed.DBZe_cloudsat === cached_objects.DBZe
        @test p.precomputed.detected_column_cloudsat === cached_objects.detected
        @test p.precomputed.cloudsat_tcc === cached_objects.tcc
    end

    @testset "unsupported CloudSat outputs" begin
        DBZe_cloudsat = ntuple(
            _ -> make_center_field(FT; value = 42, nelems = 2),
            3,
        )
        cloudsat_tcc = similar(Fields.level(DBZe_cloudsat[1], 1), FT)
        cloudsat_tcc .= FT(100)
        untouched_gas = make_center_field(FT; value = 7, nelems = 2)
        precomputed = (; DBZe_cloudsat, cloudsat_tcc, untouched_gas)

        CA.fill_unsupported_cloudsat_outputs!(precomputed, FT)

        @test all(
            DBZe -> all(==(FT(-1e30)), parent(DBZe)),
            DBZe_cloudsat,
        )
        @test all(iszero, parent(cloudsat_tcc))
        @test all(==(FT(7)), parent(untouched_gas))
    end

    @testset "COSP microphysics support" begin
        density = make_center_profile_field(FT, [1, 1])
        similar_center_field = () -> similar(density)
        profile_field = values -> begin
            field = similar_center_field()
            set_center_profile!(field, values)
            return field
        end
        Y = (;
            c = (;
                ρ = density,
                ρq_lcl = profile_field(FT[0, 0]),
                ρq_icl = profile_field(FT[0, 0]),
                ρq_rai = profile_field(FT[0.1, 0.2]),
                ρq_sno = profile_field(FT[0.2, 0.1]),
            ),
        )
        cloud_fraction = profile_field(FT[1, 1])
        p = (;
            atmos = (;
                cosp = CA.COSPModel(;
                    n_subcolumns = Val(4),
                    overlap = :maximum,
                    random_seed = UInt64(1),
                ),
                microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            ),
            precomputed = (;
                ᶜcloud_fraction = cloud_fraction,
                ᶜsubcolumn_cloud = similar_center_field(),
                ᶜsubcolumn_threshold = similar_center_field(),
                ᶜsubcolumn_precip = similar_center_field(),
                ᶜscops_selectors = (;
                    has_cloud = similar_center_field(),
                    has_cloud_below = similar_center_field(),
                    has_cloud_anywhere = similar_center_field(),
                ),
                ᶜprecip_subcolumn_scratch = (;
                    cloud = similar_center_field(),
                    cloud_below = similar_center_field(),
                    any_cloud = similar_center_field(),
                    column_any = similar_center_field(),
                ),
                ᶜsampled_cloud_fraction = similar_center_field(),
                ᶜsampled_precip_fraction = similar_center_field(),
                ᶜlarge_scale_precipitation_flux = similar_center_field(),
                ᶜwᵣ = profile_field(FT[1, -1]),
                ᶜwₛ = profile_field(FT[0.5, -2]),
            ),
            scratch = (;
                ᶜtemp_scalar = similar_center_field(),
                ᶜtemp_scalar_2 = similar_center_field(),
                ᶜtemp_scalar_3 = similar_center_field(),
                ᶜtemp_scalar_4 = similar_center_field(),
            ),
        )

        cases = (
            (CA.NonEquilibriumMicrophysics2M(), true),
            (CA.DryModel(), false),
            (CA.EquilibriumMicrophysics0M(), false),
            (CA.NonEquilibriumMicrophysics2MP3(), false),
        )
        for (microphysics_model, supported) in cases
            p_case = merge(
                p,
                (; atmos = merge(p.atmos, (; microphysics_model))),
            )
            if supported
                CA.foreach_cosp_subcolumn((_, _) -> nothing, Y, p_case)
                # ClimaAtmos center profiles are ordered from bottom to top.
                @test center_profile(
                    p.precomputed.ᶜlarge_scale_precipitation_flux,
                ) ≈ FT[0.2, 0]
            else
                @test_throws ArgumentError CA.foreach_cosp_subcolumn(
                    (_, _) -> nothing,
                    Y,
                    p_case,
                )
            end
        end
    end

    @testset "random overlap thresholds match recurrence" begin
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

                materialize_scops!(
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

    @testset "backend-safe vertical selector helpers" begin
        input = make_center_profile_field(FT, [1, 2, 3, 4])
        output = similar(input)
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
        @test center_profile(finite_clear_fraction) == FT[0.1]
        @test all(center_profile(mask) == FT[0] for mask in finite_clear_masks)

        finite_clear_selectors = selectors_from_cloud_masks(FT, finite_clear_masks)
        @test center_profile(finite_clear_selectors.has_cloud) == FT[0]
        precip = streamed_precipitation(
            finite_clear_masks,
            make_center_profile_field(FT, [1]),
        )
        @test all(center_profile(mask) == FT[1] for mask in precip)
    end

    @testset "NaN flux interrupts precipitation inheritance" begin
        cloud_masks = (
            make_center_profile_field(FT, [0, 0, 1]),
            make_center_profile_field(FT, [1, 0, 0]),
        )
        precip = streamed_precipitation(
            cloud_masks,
            make_center_profile_field(FT, [1, NaN, 1]),
        )
        @test center_profile(precip[1]) == FT[0, 0, 1]
        @test center_profile(precip[2]) == FT[1, 0, 0]
    end

    @testset "hydrometeor slicing zeroes zero sampled fractions" begin
        cloud_mask = make_center_profile_field(FT, [0, 0])
        precip_mask = make_center_profile_field(FT, [0, 0])
        grid_mean = (;
            q_lcl = make_center_profile_field(FT, [1, 0]),
            q_icl = make_center_profile_field(FT, [3, 0]),
            q_rai = make_center_profile_field(FT, [2, 0]),
            q_sno = make_center_profile_field(FT, [4, 0]),
        )
        zero_fraction = make_center_profile_field(FT, [0, 0])
        hydrometeors =
            CA.COSP.COSPHydrometeorSubcolumns.lazy_hydrometeor_subcolumn(
                grid_mean,
                cloud_mask,
                precip_mask,
                zero_fraction,
                zero_fraction,
            )

        for hydrometeor in values(hydrometeors)
            output = similar(cloud_mask)
            @. output = hydrometeor
            @test center_profile(output) == FT[0, 0]
        end
    end

end
