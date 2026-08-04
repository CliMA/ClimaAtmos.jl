import ClimaCore: Fields, Spaces

cosp_precomputed_quantities(_, ::Nothing) = (;)

function cosp_precomputed_quantities(Y, ::COSPModel)
    FT = eltype(Y)
    height_km_cloudsat = Fields.coordinate_field(axes(Y.c)).z ./ FT(1000)
    surface_height_km_cloudsat =
        Fields.level(
            Fields.coordinate_field(Spaces.face_space(axes(Y.c))).z,
            Fields.half,
        ) ./ FT(1000)
    top_height_km_cloudsat =
        Fields.level(
            Fields.coordinate_field(Spaces.face_space(axes(Y.c))).z,
            Spaces.nlevels(axes(Y.c)) + Fields.half,
        ) ./ FT(1000)
    cloudsat_dbze_bin_edges =
        COSP.COSPCloudSatCFAD.cloudsat_cfad_bin_edges(FT)
    cloudsat_dbze_bin_centers =
        COSP.COSPCloudSatCFAD.cloudsat_cfad_bin_centers(FT)
    cfadDbze94 = similar(Y.c, typeof(cloudsat_dbze_bin_centers))
    cloudsat_tcc = similar(Fields.level(Y.c.ρ, 1), FT)
    cloudsat_tcc2 = similar(cloudsat_tcc)

    COSP.COSPCloudSatCFAD.initialize_cloudsat_cfad!(cfadDbze94)
    cloudsat_tcc .= zero(FT)
    cloudsat_tcc2 .= zero(FT)
    return (;
        height_km_cloudsat,
        surface_height_km_cloudsat,
        top_height_km_cloudsat,
        cloudsat_dbze_bin_edges,
        cloudsat_dbze_bin_centers,
        cfadDbze94,
        cloudsat_tcc,
        cloudsat_tcc2,
    )
end

cosp_temporary_quantities(_, ::Nothing) = (;)

function cosp_temporary_quantities(Y, ::COSPModel)
    FT = Spaces.undertype(axes(Y.c))
    ᶜsampled_cloud_fraction = similar(Y.c, FT)
    ᶜsampled_precip_fraction = similar(Y.c, FT)
    @. ᶜsampled_cloud_fraction = zero(FT)
    @. ᶜsampled_precip_fraction = zero(FT)
    return (;
        ᶜsubcolumn_cloud = similar(Y.c, FT),
        ᶜsubcolumn_threshold = similar(Y.c, FT),
        ᶜsubcolumn_precip = similar(Y.c, FT),
        ᶜscops_selectors = (;
            has_cloud = similar(Y.c, FT),
            has_cloud_below = similar(Y.c, FT),
            has_cloud_anywhere = similar(Y.c, FT),
        ),
        ᶜprecip_subcolumn_scratch = (;
            cloud = similar(Y.c, FT),
            cloud_below = similar(Y.c, FT),
            any_cloud = similar(Y.c, FT),
            column_any = similar(Y.c, FT),
        ),
        ᶜsampled_cloud_fraction,
        ᶜsampled_precip_fraction,
        ᶜlarge_scale_precipitation_flux = similar(Y.c, FT),
        cloudsat_hydrometeor_optics_work =
            similar(Y.c, @NamedTuple{z_vol::FT, kr_vol::FT}),
        g_vol_cloudsat = similar(Y.c, FT),
        DBZe_cloudsat_work = similar(Y.c, FT),
        gas_path_attenuation_cloudsat = similar(Y.c, FT),
        cloudsat_grid_mean_sizes = (;
            r_lcl = similar(Y.c, FT),
            lambda_inv_icl = similar(Y.c, FT),
            lambda_inv_rai = similar(Y.c, FT),
            lambda_inv_sno = similar(Y.c, FT),
        ),
        detected_column_cloudsat = similar(Fields.level(Y.c.ρ, 1), Bool),
    )
end
