import ClimaCore.Fields: @fused_direct

struct NoOpCOSPSubcolumnConsumer end
(::NoOpCOSPSubcolumnConsumer)(_, _) = nothing

struct CloudSatSubcolumnConsumer{W, A, O, S}
    work::W
    atmosphere::A
    optics::O
    statistics::S
end

function (consumer::CloudSatSubcolumnConsumer)(_, hydrometeors)
    (; work, atmosphere, optics, statistics) = consumer
    hydrometeor_optics = work.hydrometeor_optics
    COSP.COSPCloudSatOptics.cloudsat_optics_subcolumn!(
        hydrometeor_optics,
        hydrometeors,
        optics.grid_mean_sizes,
        atmosphere.temperature,
        atmosphere.rho_air,
        optics.microphysics_params,
        optics.radar_config,
    )
    COSP.COSPCloudSatReflectivity.cloudsat_reflectivity_subcolumn!(
        work.DBZe,
        hydrometeor_optics.z_vol,
        hydrometeor_optics.kr_vol,
        atmosphere.gas_path_attenuation,
        atmosphere.height_km,
        atmosphere.top_height_km,
    )
    COSP.COSPCloudSatCloudFraction.accumulate_cloudsat_cloud_fraction!(
        statistics.cloudsat_tcc,
        statistics.cloudsat_tcc2,
        statistics.detected_column,
        work.DBZe,
        atmosphere.height_km,
        statistics.surface_height_km,
        statistics.percent_contribution,
    )
    COSP.COSPCloudSatCFAD.accumulate_cloudsat_cfad!(
        statistics.cfadDbze94,
        work.DBZe,
        statistics.dbze_bin_edges,
        statistics.fraction_contribution,
    )
    return nothing
end

function run_cosp_cloudsat!(Y, p, ::NonEquilibriumMicrophysics1M)
    (;
        height_km_cloudsat,
        surface_height_km_cloudsat,
        top_height_km_cloudsat,
        cloudsat_dbze_bin_edges,
        cfadDbze94,
        cloudsat_tcc,
        cloudsat_tcc2,
        ᶜp,
        ᶜT,
        ᶜq_tot_nonneg,
        ᶜq_liq,
        ᶜq_ice,
    ) = p.precomputed
    (;
        cloudsat_hydrometeor_optics_work,
        g_vol_cloudsat,
        DBZe_cloudsat_work,
        gas_path_attenuation_cloudsat,
        cloudsat_grid_mean_sizes,
        detected_column_cloudsat,
    ) = p.scratch

    ᶜq_vap = @. lazy(TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
    radar_config =
        COSP.COSPCloudSatOptics.CloudSatRadarConfig(eltype(Y))
    COSP.COSPCloudSatOptics.cloudsat_gas_attenuation!(
        g_vol_cloudsat,
        ᶜT,
        ᶜp,
        ᶜq_vap,
        radar_config,
    )
    COSP.COSPCloudSatReflectivity.cloudsat_gas_path_attenuation!(
        gas_path_attenuation_cloudsat,
        g_vol_cloudsat,
        height_km_cloudsat,
        top_height_km_cloudsat,
    )

    FT = eltype(Y)
    nsubcolumns = _cosp_nsubcolumns(p.atmos.cosp.n_subcolumns)
    fraction_contribution = FT(1) / FT(nsubcolumns)
    percent_contribution = FT(100) * fraction_contribution
    reset_cloudsat_statistics!(cfadDbze94, cloudsat_tcc, cloudsat_tcc2)
    statistics = (;
        cloudsat_tcc,
        cloudsat_tcc2,
        detected_column = detected_column_cloudsat,
        surface_height_km = surface_height_km_cloudsat,
        cfadDbze94,
        dbze_bin_edges = cloudsat_dbze_bin_edges,
        fraction_contribution,
        percent_contribution,
    )

    work = (;
        hydrometeor_optics = cloudsat_hydrometeor_optics_work,
        DBZe = DBZe_cloudsat_work,
    )
    atmosphere = (;
        gas_path_attenuation = gas_path_attenuation_cloudsat,
        height_km = height_km_cloudsat,
        top_height_km = top_height_km_cloudsat,
        temperature = ᶜT,
        rho_air = Y.c.ρ,
    )
    optics = (;
        grid_mean_sizes = cloudsat_grid_mean_sizes,
        microphysics_params = CAP.microphysics_1m_params(p.params),
        radar_config,
    )
    consumer = CloudSatSubcolumnConsumer(work, atmosphere, optics, statistics)
    foreach_cosp_subcolumn(consumer, Y, p)
    return nothing
end

function run_cosp_cloudsat!(
    Y,
    p,
    ::NonEquilibriumMicrophysics2M,
)
    # The subcolumn generator supports 2M, but CloudSat hydrometeor optics do
    # not. Preserve the sampled-fraction workflow while returning explicit
    # unsupported simulator outputs.
    foreach_cosp_subcolumn(NoOpCOSPSubcolumnConsumer(), Y, p)
    reset_cloudsat_statistics!(p.precomputed)
    return nothing
end

function run_cosp_cloudsat!(Y, p, _)
    reset_cloudsat_statistics!(p.precomputed)
    return nothing
end

function reset_cloudsat_statistics!(precomputed)
    (; cfadDbze94, cloudsat_tcc, cloudsat_tcc2) = precomputed
    return reset_cloudsat_statistics!(cfadDbze94, cloudsat_tcc, cloudsat_tcc2)
end

function reset_cloudsat_statistics!(cfadDbze94, cloudsat_tcc, cloudsat_tcc2)
    COSP.COSPCloudSatCFAD.initialize_cloudsat_cfad!(cfadDbze94)
    COSP.COSPCloudSatCloudFraction.initialize_cloudsat_cloud_fraction!(
        cloudsat_tcc,
    )
    COSP.COSPCloudSatCloudFraction.initialize_cloudsat_cloud_fraction!(
        cloudsat_tcc2,
    )
    return nothing
end

function prepare_cosp_subcolumns!(Y, p)
    (; ᶜcloud_fraction) = p.precomputed
    (;
        ᶜsubcolumn_cloud,
        ᶜsubcolumn_threshold,
        ᶜsubcolumn_precip,
        ᶜscops_selectors,
        ᶜprecip_subcolumn_scratch,
        ᶜsampled_cloud_fraction,
        ᶜsampled_precip_fraction,
        ᶜlarge_scale_precipitation_flux,
    ) = p.scratch
    cosp = p.atmos.cosp
    nsubcolumns = _cosp_nsubcolumns(cosp.n_subcolumns)
    overlap = _cosp_overlap(cosp.overlap)

    COSP.COSPSubcolumns.set_scops_selectors!(
        ᶜscops_selectors,
        ᶜsubcolumn_cloud,
        ᶜsubcolumn_threshold,
        ᶜcloud_fraction,
        nsubcolumns,
        cosp.random_seed,
        overlap,
        ᶜprecip_subcolumn_scratch.column_any,
    )

    set_cosp_large_scale_precipitation_flux!(Y, p, p.atmos.microphysics_model)

    FT = eltype(ᶜcloud_fraction)
    @fused_direct begin
        @. ᶜsampled_cloud_fraction = zero(FT)
        @. ᶜsampled_precip_fraction = zero(FT)
    end
    for isubcolumn in 1:nsubcolumns
        COSP.COSPSubcolumns.scops_subcolumn!(
            ᶜsubcolumn_cloud,
            ᶜsubcolumn_threshold,
            ᶜcloud_fraction,
            isubcolumn,
            nsubcolumns,
            cosp.random_seed;
            overlap,
        )
        COSP.COSPPrecipSubcolumns.scops_subcolumn_precip!(
            ᶜsubcolumn_precip,
            ᶜsubcolumn_cloud,
            ᶜlarge_scale_precipitation_flux,
            ᶜscops_selectors,
            ᶜprecip_subcolumn_scratch,
        )
        COSP.COSPHydrometeorSubcolumns.accumulate_sampled_fractions!(
            ᶜsampled_cloud_fraction,
            ᶜsampled_precip_fraction,
            ᶜsubcolumn_cloud,
            ᶜsubcolumn_precip,
            nsubcolumns,
        )
    end

    return nothing
end

function set_cosp_large_scale_precipitation_flux!(
    Y,
    p,
    ::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
)
    (; ᶜwᵣ, ᶜwₛ) = p.precomputed
    (; ᶜlarge_scale_precipitation_flux) = p.scratch
    FT = eltype(ᶜlarge_scale_precipitation_flux)

    @. ᶜlarge_scale_precipitation_flux =
        max(FT(0), Y.c.ρq_rai * ᶜwᵣ + Y.c.ρq_sno * ᶜwₛ)

    return nothing
end

set_cosp_large_scale_precipitation_flux!(_, _, microphysics_model) =
    _check_cosp_microphysics(microphysics_model)

"""
    foreach_cosp_subcolumn(consume!, Y, p)

For 1M microphysics, diagnose grid-mean hydrometeor sizes before preparing the
sampled cloud and precipitation fractions. Then regenerate and stream one
deterministic hydrometeor subcolumn at a time. `consume!` must use the lazy
hydrometeor broadcasts immediately; they borrow working mask and scratch fields
that are overwritten during subsequent iterations.
"""
function foreach_cosp_subcolumn(consume!::F, Y, p) where {F}
    microphysics_model = p.atmos.microphysics_model
    _check_cosp_microphysics(microphysics_model)
    return foreach_cosp_subcolumn(consume!, Y, p, microphysics_model)
end

function foreach_cosp_subcolumn(
    consume!::F,
    Y,
    p,
    microphysics_model::Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
    },
) where {F}
    ᶜq_lcl = p.scratch.ᶜtemp_scalar
    ᶜq_icl = p.scratch.ᶜtemp_scalar_2
    ᶜq_rai = p.scratch.ᶜtemp_scalar_3
    ᶜq_sno = p.scratch.ᶜtemp_scalar_4

    @fused_direct begin
        @. ᶜq_lcl = specific(Y.c.ρq_lcl, Y.c.ρ)
        @. ᶜq_icl = specific(Y.c.ρq_icl, Y.c.ρ)
        @. ᶜq_rai = specific(Y.c.ρq_rai, Y.c.ρ)
        @. ᶜq_sno = specific(Y.c.ρq_sno, Y.c.ρ)
    end

    grid_mean_hydrometeors =
        (; q_lcl = ᶜq_lcl, q_icl = ᶜq_icl, q_rai = ᶜq_rai, q_sno = ᶜq_sno)

    if microphysics_model isa NonEquilibriumMicrophysics1M
        COSP.COSPCloudSatOptics.cloudsat_grid_mean_sizes!(
            p.scratch.cloudsat_grid_mean_sizes,
            grid_mean_hydrometeors,
            Y.c.ρ,
            CAP.microphysics_1m_params(p.params),
        )
    end
    prepare_cosp_subcolumns!(Y, p)
    return foreach_prepared_cosp_subcolumn!(consume!, grid_mean_hydrometeors, p)
end

foreach_cosp_subcolumn(::F, _, _, microphysics_model) where {F} =
    _check_cosp_microphysics(microphysics_model)

_check_cosp_microphysics(
    ::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
) = nothing

function _check_cosp_microphysics(microphysics_model)
    throw(
        ArgumentError(
            "COSP supports only NonEquilibriumMicrophysics1M and " *
            "NonEquilibriumMicrophysics2M; got $(nameof(typeof(microphysics_model)))",
        ),
    )
end

function foreach_prepared_cosp_subcolumn!(
    consume!::F,
    grid_mean_hydrometeors,
    p,
) where {F}
    (; ᶜcloud_fraction) = p.precomputed
    (;
        ᶜsubcolumn_cloud,
        ᶜsubcolumn_threshold,
        ᶜsubcolumn_precip,
        ᶜscops_selectors,
        ᶜprecip_subcolumn_scratch,
        ᶜlarge_scale_precipitation_flux,
        ᶜsampled_cloud_fraction,
        ᶜsampled_precip_fraction,
    ) = p.scratch

    cosp = p.atmos.cosp
    nsubcolumns = _cosp_nsubcolumns(cosp.n_subcolumns)
    overlap = _cosp_overlap(cosp.overlap)
    for isubcolumn in 1:nsubcolumns
        COSP.COSPSubcolumns.scops_subcolumn!(
            ᶜsubcolumn_cloud,
            ᶜsubcolumn_threshold,
            ᶜcloud_fraction,
            isubcolumn,
            nsubcolumns,
            cosp.random_seed;
            overlap,
        )
        COSP.COSPPrecipSubcolumns.scops_subcolumn_precip!(
            ᶜsubcolumn_precip,
            ᶜsubcolumn_cloud,
            ᶜlarge_scale_precipitation_flux,
            ᶜscops_selectors,
            ᶜprecip_subcolumn_scratch,
        )
        hydrometeors =
            COSP.COSPHydrometeorSubcolumns.lazy_hydrometeor_subcolumn(
                grid_mean_hydrometeors,
                ᶜsubcolumn_cloud,
                ᶜsubcolumn_precip,
                ᶜsampled_cloud_fraction,
                ᶜsampled_precip_fraction,
            )
        consume!(isubcolumn, hydrometeors)
    end

    return nothing
end
