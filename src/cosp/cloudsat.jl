struct NoOpCOSPSubcolumnConsumer end
(::NoOpCOSPSubcolumnConsumer)(_, _) = nothing

struct CloudSatSubcolumnConsumer{Z, K, ZE, D, H, G, HT, HTT, T, R, S, MP, C}
    z_vol_work::Z
    kr_vol_work::K
    Ze_non_work::ZE
    DBZe::D
    hydro_path_attenuation_work::H
    gas_path_attenuation::G
    height_km::HT
    top_height_km::HTT
    temperature::T
    rho_air::R
    grid_mean_sizes::S
    microphysics_params::MP
    radar_config::C
end

function (consumer::CloudSatSubcolumnConsumer)(isubcolumn, hydrometeors)
    return consume_cosp_subcolumn!(consumer, isubcolumn, hydrometeors)
end

function consume_cosp_subcolumn!(
    consumer::CloudSatSubcolumnConsumer,
    isubcolumn,
    hydrometeors,
)
    COSP.COSPCloudSatOptics.cloudsat_optics_subcolumn!(
        consumer.z_vol_work,
        consumer.kr_vol_work,
        hydrometeors,
        consumer.grid_mean_sizes,
        consumer.temperature,
        consumer.rho_air,
        consumer.microphysics_params,
        consumer.radar_config,
    )
    COSP.COSPCloudSatReflectivity.cloudsat_reflectivity_subcolumn!(
        consumer.Ze_non_work,
        consumer.DBZe[isubcolumn],
        consumer.z_vol_work,
        consumer.kr_vol_work,
        consumer.hydro_path_attenuation_work,
        consumer.gas_path_attenuation,
        consumer.height_km,
        consumer.top_height_km,
    )
    return nothing
end

function run_cosp_cloudsat!(Y, p, ::NonEquilibriumMicrophysics1M)
    (;
        height_km_cloudsat,
        top_height_km_cloudsat,
        DBZe_cloudsat,
        cloudsat_tcc,
        ᶜp,
        ᶜT,
        ᶜq_tot_nonneg,
        ᶜq_liq,
        ᶜq_ice,
    ) = p.precomputed
    (;
        z_vol_cloudsat_work,
        kr_vol_cloudsat_work,
        g_vol_cloudsat,
        Ze_non_cloudsat_work,
        hydro_path_attenuation_cloudsat_work,
        gas_path_attenuation_cloudsat,
        cloudsat_grid_mean_sizes,
        detected_column_cloudsat,
    ) = p.scratch

    ᶜq_vap = @. lazy(ᶜq_tot_nonneg - ᶜq_liq - ᶜq_ice)
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

    consumer = CloudSatSubcolumnConsumer(
        z_vol_cloudsat_work,
        kr_vol_cloudsat_work,
        Ze_non_cloudsat_work,
        DBZe_cloudsat,
        hydro_path_attenuation_cloudsat_work,
        gas_path_attenuation_cloudsat,
        height_km_cloudsat,
        top_height_km_cloudsat,
        ᶜT,
        Y.c.ρ,
        cloudsat_grid_mean_sizes,
        CAP.microphysics_1m_params(p.params),
        radar_config,
    )
    foreach_cosp_subcolumn(consumer, Y, p)
    COSP.COSPCloudSatCloudFraction.cloudsat_cloud_fraction!(
        cloudsat_tcc,
        detected_column_cloudsat,
        DBZe_cloudsat,
    )
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
    fill_unsupported_cloudsat_outputs!(p.precomputed, eltype(Y))
    return nothing
end

function run_cosp_cloudsat!(Y, p, _)
    fill_unsupported_cloudsat_outputs!(p.precomputed, eltype(Y))
    return nothing
end

function fill_unsupported_cloudsat_outputs!(precomputed, ::Type{FT}) where {FT}
    (; DBZe_cloudsat, cloudsat_tcc) = precomputed
    for DBZe_subcolumn in DBZe_cloudsat
        @. DBZe_subcolumn = FT(-1e30)
    end
    cloudsat_tcc .= zero(FT)
    return nothing
end

function prepare_cosp_subcolumns!(Y, p)
    (;
        ᶜcloud_fraction,
        ᶜsubcolumn_cloud,
        ᶜsubcolumn_threshold,
        ᶜsubcolumn_precip,
        ᶜscops_selectors,
        ᶜprecip_subcolumn_scratch,
        ᶜsampled_cloud_fraction,
        ᶜsampled_precip_fraction,
        ᶜlarge_scale_precipitation_flux,
    ) = p.precomputed
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
    @. ᶜsampled_cloud_fraction = zero(FT)
    @. ᶜsampled_precip_fraction = zero(FT)
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
        COSP.COSPHydrometeorSubcolumns.accumulate_sampled_cloud_fraction!(
            ᶜsampled_cloud_fraction,
            ᶜsubcolumn_cloud,
            nsubcolumns,
        )
        COSP.COSPHydrometeorSubcolumns.accumulate_sampled_precip_fraction!(
            ᶜsampled_precip_fraction,
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
    (; ᶜlarge_scale_precipitation_flux, ᶜwᵣ, ᶜwₛ) = p.precomputed
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

    @. ᶜq_lcl = specific(Y.c.ρq_lcl, Y.c.ρ)
    @. ᶜq_icl = specific(Y.c.ρq_icl, Y.c.ρ)
    @. ᶜq_rai = specific(Y.c.ρq_rai, Y.c.ρ)
    @. ᶜq_sno = specific(Y.c.ρq_sno, Y.c.ρ)

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
    (;
        ᶜcloud_fraction,
        ᶜsubcolumn_cloud,
        ᶜsubcolumn_threshold,
        ᶜsubcolumn_precip,
        ᶜscops_selectors,
        ᶜprecip_subcolumn_scratch,
        ᶜlarge_scale_precipitation_flux,
        ᶜsampled_cloud_fraction,
        ᶜsampled_precip_fraction,
    ) = p.precomputed

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
