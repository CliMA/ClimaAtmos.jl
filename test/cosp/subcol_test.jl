using Test
import ClimaAtmos as CA
using ClimaCore: Domains, Meshes, Spaces, Fields, Geometry

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
        job_id = "cosp_subcol_1m_callback",
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

    @testset "1M streamed subcolumns and CloudSat callback" begin
        nlevels = 4
        nsubcolumns = 4
        overlap = :maximum

        simulation = make_1m_subcol_simulation(;
            z_elem = nlevels,
            cosp_n_subcolumns = nsubcolumns,
            cosp_overlap = String(overlap),
        )
        Y = simulation.integrator.u
        p = simulation.integrator.p
        state_FT = eltype(Y)
        @test CA._cosp_nsubcolumns(p.atmos.cosp.n_subcolumns) == nsubcolumns
        @test CA._cosp_overlap(p.atmos.cosp.overlap) === overlap
        @test isbitstype(typeof(p.atmos.cosp))
        temporary_cosp_quantities = (
            :ᶜsubcolumn_cloud,
            :ᶜsubcolumn_threshold,
            :ᶜsubcolumn_precip,
            :ᶜscops_selectors,
            :ᶜprecip_subcolumn_scratch,
            :ᶜsampled_cloud_fraction,
            :ᶜsampled_precip_fraction,
            :ᶜlarge_scale_precipitation_flux,
        )
        @test all(
            name -> !hasproperty(p.precomputed, name),
            temporary_cosp_quantities,
        )
        # ClimaAtmos center profiles are ordered from surface to model top.
        set_center_profile!(Y.c.ρ, state_FT[1, 2, 1, 2])
        set_center_profile!(Y.c.ρq_lcl, state_FT[8e-4, 8e-4, 2e-4, 2e-4])
        set_center_profile!(Y.c.ρq_icl, state_FT[4e-4, 4e-4, 1e-4, 1e-4])
        set_center_profile!(Y.c.ρq_rai, state_FT[2e-4, 2e-4, 5e-5, 4e-5])
        set_center_profile!(Y.c.ρq_sno, state_FT[1e-4, 1e-4, 2e-5, 2e-5])
        set_center_profile!(
            p.precomputed.ᶜcloud_fraction,
            state_FT[0.8, 0.6, 0.4, 0.2],
        )
        set_center_profile!(
            p.precomputed.ᶜwᵣ,
            state_FT[1, 2, 0.75, 1.5],
        )
        set_center_profile!(
            p.precomputed.ᶜwₛ,
            state_FT[0.5, 1, 0.25, 0.75],
        )
        CA.subcol_model_callback!(simulation.integrator)
        @test any(
            index -> any(
                >(zero(eltype(Y))),
                parent(getproperty(p.precomputed.cfadDbze94, index)),
            ),
            eachindex(p.precomputed.cloudsat_dbze_bin_centers),
        )
        @test any(>(zero(eltype(Y))), parent(p.precomputed.cloudsat_tcc))
        @test all(
            parent(p.precomputed.cloudsat_tcc2) .<=
            parent(p.precomputed.cloudsat_tcc),
        )

        @testset "CloudSat callback refreshes and clears outputs" begin
            gas_before_refresh = copy(parent(p.scratch.g_vol_cloudsat))
            energy_increment = state_FT(1000)
            zero_state = zero(state_FT)
            @. Y.c.ρe_tot += Y.c.ρ * energy_increment
            CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)
            CA.subcol_model_callback!(simulation.integrator)
            @test parent(p.scratch.g_vol_cloudsat) != gas_before_refresh

            @. Y.c.ρq_lcl = zero_state
            @. Y.c.ρq_icl = zero_state
            @. Y.c.ρq_rai = zero_state
            @. Y.c.ρq_sno = zero_state
            @. p.precomputed.ᶜcloud_fraction = zero_state
            CA.subcol_model_callback!(simulation.integrator)

            @test all(
                index -> all(
                    iszero,
                    parent(getproperty(p.precomputed.cfadDbze94, index)),
                ),
                eachindex(p.precomputed.cloudsat_dbze_bin_centers),
            )
            @test all(iszero, parent(p.precomputed.cloudsat_tcc))
            @test all(iszero, parent(p.precomputed.cloudsat_tcc2))
        end
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
                    overlap = Val(:maximum),
                    random_seed = UInt64(1),
                ),
                microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            ),
            precomputed = (;
                ᶜcloud_fraction = cloud_fraction,
                ᶜwᵣ = profile_field(FT[1, -1]),
                ᶜwₛ = profile_field(FT[0.5, -2]),
            ),
            scratch = (;
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
                    p.scratch.ᶜlarge_scale_precipitation_flux,
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

    @testset "vertical selector helpers" begin
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

        for hydrometeor in (hydrometeors.q_lcl, hydrometeors.q_rai)
            output = similar(cloud_mask)
            @. output = hydrometeor
            @test center_profile(output) == FT[0, 0]
        end
    end

end
