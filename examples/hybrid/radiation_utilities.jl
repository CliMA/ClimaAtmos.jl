import ClimaAtmos.Parameters as CAP
using Statistics: mean
using Dierckx: Spline1D
using Dates: Second
using Insolation: instantaneous_zenith_angle

function rrtmgp_model_cache(
    Y,
    params,
    radiation_mode::RRTMGPI.AbstractRadiationMode = RRTMGPI.ClearSkyRadiation();
    interpolation = RRTMGPI.BestFit(),
    bottom_extrapolation = RRTMGPI.SameAsInterpolation(),
    idealized_insolation = true,
    idealized_h2o = false,
    idealized_clouds = false,
)
    rrtmgp_params = CAP.rrtmgp_params(params)
    thermo_params = CAP.thermodynamics_params(params)
    if idealized_h2o && radiation_mode isa RRTMGPI.GrayRadiation
        error("idealized_h2o can't be used with $radiation_mode")
    end
    if idealized_clouds && (
        radiation_mode isa RRTMGPI.GrayRadiation ||
        radiation_mode isa RRTMGPI.ClearSkyRadiation
    )
        error("idealized_clouds can't be used with $radiation_mode")
    end

    bottom_coords = Fields.coordinate_field(Spaces.level(Y.c, 1))
    if eltype(bottom_coords) <: Geometry.LatLongZPoint
        latitude = RRTMGPI.field2array(bottom_coords.lat)
    else
        latitude = RRTMGPI.field2array(zero(bottom_coords.z)) # flat space is on Equator
    end
    input_data = RRTMGPI.rrtmgp_artifact("atmos_state", "clearsky_as.nc")
    if radiation_mode isa RRTMGPI.GrayRadiation
        kwargs = (;
            lapse_rate = 3.5,
            optical_thickness_parameter = (@. 7.2 +
                                              (1.8 - 7.2) * sind(latitude)^2),
        )
    else
        # the pressure and ozone concentrations are provided for each of 100
        # sites, which we average across
        n = input_data.dim["layer"]
        input_center_pressure =
            vec(mean(reshape(input_data["pres_layer"][:, :], n, :); dims = 2))
        # the first values along the third dimension of the ozone concentration
        # data are the present-day values
        input_center_volume_mixing_ratio_o3 =
            vec(mean(reshape(input_data["ozone"][:, :, 1], n, :); dims = 2))

        # interpolate the ozone concentrations to our initial pressures (set the
        # kinetic energy to 0 when computing the pressure using total energy)
        pressure2ozone =
            Spline1D(input_center_pressure, input_center_volume_mixing_ratio_o3)
        ???ts = thermo_state(Y, params, ???interp, 0)
        ???p = @. TD.air_pressure(thermo_params, ???ts)
        center_volume_mixing_ratio_o3 =
            RRTMGPI.field2array(@. FT(pressure2ozone(???p)))

        # the first value for each global mean volume mixing ratio is the
        # present-day value
        input_vmr(name) =
            input_data[name][1] * parse(FT, input_data[name].attrib["units"])
        kwargs = (;
            use_global_means_for_well_mixed_gases = true,
            center_volume_mixing_ratio_h2o = NaN, # initialize in tendency
            center_volume_mixing_ratio_o3,
            volume_mixing_ratio_co2 = input_vmr("carbon_dioxide_GM"),
            volume_mixing_ratio_n2o = input_vmr("nitrous_oxide_GM"),
            volume_mixing_ratio_co = input_vmr("carbon_monoxide_GM"),
            volume_mixing_ratio_ch4 = input_vmr("methane_GM"),
            volume_mixing_ratio_o2 = input_vmr("oxygen_GM"),
            volume_mixing_ratio_n2 = input_vmr("nitrogen_GM"),
            volume_mixing_ratio_ccl4 = input_vmr("carbon_tetrachloride_GM"),
            volume_mixing_ratio_cfc11 = input_vmr("cfc11_GM"),
            volume_mixing_ratio_cfc12 = input_vmr("cfc12_GM"),
            volume_mixing_ratio_cfc22 = input_vmr("hcfc22_GM"),
            volume_mixing_ratio_hfc143a = input_vmr("hfc143a_GM"),
            volume_mixing_ratio_hfc125 = input_vmr("hfc125_GM"),
            volume_mixing_ratio_hfc23 = input_vmr("hfc23_GM"),
            volume_mixing_ratio_hfc32 = input_vmr("hfc32_GM"),
            volume_mixing_ratio_hfc134a = input_vmr("hfc134a_GM"),
            volume_mixing_ratio_cf4 = input_vmr("cf4_GM"),
            volume_mixing_ratio_no2 = 1e-8, # not available in input_data
            latitude,
        )
        if !(radiation_mode isa RRTMGPI.ClearSkyRadiation)
            kwargs = (;
                kwargs...,
                center_cloud_liquid_effective_radius = 12,
                center_cloud_ice_effective_radius = 95,
                ice_roughness = 2,
            )
            ???z = Fields.coordinate_field(Y.c).z
            ?????z = Fields.local_geometry_field(Y.c).???x?????.components.data.:9
            if idealized_clouds # icy cloud on top and wet cloud on bottom
                ???is_bottom_cloud = Fields.Field(
                    DataLayouts.replace_basetype(Fields.field_values(???z), Bool),
                    axes(Y.c),
                ) # need to fix several ClimaCore bugs in order to simplify this
                ???is_top_cloud = similar(???is_bottom_cloud)
                @. ???is_bottom_cloud = ???z > 1e3 && ???z < 1.5e3
                @. ???is_top_cloud = ???z > 4e3 && ???z < 5e3
                kwargs = (;
                    kwargs...,
                    center_cloud_liquid_water_path = RRTMGPI.field2array(
                        @. ifelse(???is_bottom_cloud, FT(0.002) * ?????z, FT(0))
                    ),
                    center_cloud_ice_water_path = RRTMGPI.field2array(
                        @. ifelse(???is_top_cloud, FT(0.001) * ?????z, FT(0))
                    ),
                    center_cloud_fraction = RRTMGPI.field2array(
                        @. ifelse(
                            ???is_bottom_cloud || ???is_top_cloud,
                            FT(1),
                            FT(0) * ?????z,
                        )
                    ),
                )
            else
                kwargs = (;
                    kwargs...,
                    center_cloud_liquid_water_path = NaN, # initialized in callback
                    center_cloud_ice_water_path = NaN, # initialized in callback
                    center_cloud_fraction = NaN, # initialized in callback
                )
            end
        end
    end

    if RRTMGPI.requires_z(interpolation) ||
       RRTMGPI.requires_z(bottom_extrapolation)
        kwargs = (;
            kwargs...,
            center_z = RRTMGPI.field2array(Fields.coordinate_field(Y.c).z),
            face_z = RRTMGPI.field2array(Fields.coordinate_field(Y.f).z),
        )
    end

    if idealized_insolation # perpetual equinox with no diurnal cycle
        solar_zenith_angle = FT(??) / 3
        weighted_irradiance =
            @. 1360 * (1 + FT(1.2) / 4 * (1 - 3 * sind(latitude)^2)) /
               (4 * cos(solar_zenith_angle))
    else
        solar_zenith_angle = weighted_irradiance = NaN # initialized in callback
    end

    rrtmgp_model = RRTMGPI.RRTMGPModel(
        rrtmgp_params;
        FT = Float64,
        ncol = length(Spaces.all_nodes(axes(Spaces.level(Y.c, 1)))),
        domain_nlay = Spaces.nlevels(axes(Y.c)),
        radiation_mode,
        interpolation,
        bottom_extrapolation,
        add_isothermal_boundary_layer = true,
        center_pressure = NaN, # initialized in callback
        center_temperature = NaN, # initialized in callback
        surface_temperature = NaN, # initialized in callback
        surface_emissivity = 1,
        direct_sw_surface_albedo = 0.38,
        diffuse_sw_surface_albedo = 0.38,
        solar_zenith_angle,
        weighted_irradiance,
        kwargs...,
    )
    close(input_data)
    return (;
        idealized_insolation,
        idealized_h2o,
        idealized_clouds,
        insolation_tuple = similar(Spaces.level(Y.c, 1), Tuple{FT, FT, FT}),
        ???radiation_flux = similar(Y.f, Geometry.WVector{FT}),
        rrtmgp_model,
    )
end
function rrtmgp_model_tendency!(Y???, Y, p, t)
    (; ???radiation_flux) = p
    ???div??? = Operators.DivergenceF2C()
    if :???? in propertynames(Y.c)
        error("rrtmgp_model_tendency! not implemented for ????")
    elseif :??e_tot in propertynames(Y.c)
        @. Y???.c.??e_tot -= ???div???(???radiation_flux)
    elseif :??e_int in propertynames(Y.c)
        @. Y???.c.??e_int -= ???div???(???radiation_flux)
    end
end
function rrtmgp_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    t = integrator.t

    (; ???K, ?????, ???ts, T_sfc, params) = p
    (; idealized_insolation, idealized_h2o, idealized_clouds) = p
    (; insolation_tuple, ???radiation_flux, rrtmgp_model) = p
    thermo_params = CAP.thermodynamics_params(params)
    insolation_params = CAP.insolation_params(params)

    rrtmgp_model.surface_temperature .= RRTMGPI.field2array(T_sfc)

    ???p = RRTMGPI.array2field(rrtmgp_model.center_pressure, axes(Y.c))
    ???T = RRTMGPI.array2field(rrtmgp_model.center_temperature, axes(Y.c))
    @. ???K = norm_sqr(C123(Y.c.u???) + C123(???interp(Y.f.w))) / 2
    thermo_state!(???ts, Y, params, ???interp, ???K)
    @. ???p = TD.air_pressure(thermo_params, ???ts)
    @. ???T = TD.air_temperature(thermo_params, ???ts)

    if !(rrtmgp_model.radiation_mode isa RRTMGPI.GrayRadiation)
        ???vmr_h2o = RRTMGPI.array2field(
            rrtmgp_model.center_volume_mixing_ratio_h2o,
            axes(Y.c),
        )
        if idealized_h2o
            # slowly increase the relative humidity from 0 to 0.6 to account for
            # the fact that we have a very unrealistic initial condition
            max_relative_humidity = FT(0.6)
            t_increasing_humidity = FT(60 * 60 * 24 * 30)
            if t < t_increasing_humidity
                max_relative_humidity *= t / t_increasing_humidity
            end

            # temporarily store ???q_tot in ???vmr_h2o
            ???q_tot = ???vmr_h2o
            @. ???q_tot =
                max_relative_humidity * TD.q_vap_saturation(thermo_params, ???ts)

            # filter ???q_tot so that it is monotonically decreasing with z
            for i in 2:Spaces.nlevels(axes(???q_tot))
                level = Fields.field_values(Spaces.level(???q_tot, i))
                prev_level = Fields.field_values(Spaces.level(???q_tot, i - 1))
                @. level = min(level, prev_level)
            end

            # assume that ???q_vap = ???q_tot when computing ???vmr_h2o
            @. ???vmr_h2o = TD.shum_to_mixing_ratio(???q_tot, ???q_tot)
        else
            @. ???vmr_h2o = TD.vol_vapor_mixing_ratio(
                thermo_params,
                TD.PhasePartition(thermo_params, ???ts),
            )
        end
    end

    if !idealized_insolation
        current_datetime = p.simulation.start_date + Second(round(Int, t)) # current time
        max_zenith_angle = FT(??) / 2 - eps(FT)
        irradiance = FT(CAP.tot_solar_irrad(params))
        au = FT(CAP.astro_unit(params))

        bottom_coords = Fields.coordinate_field(Spaces.level(Y.c, 1))
        if eltype(bottom_coords) <: Geometry.LatLongZPoint
            solar_zenith_angle = RRTMGPI.array2field(
                rrtmgp_model.solar_zenith_angle,
                axes(bottom_coords),
            )
            weighted_irradiance = RRTMGPI.array2field(
                rrtmgp_model.weighted_irradiance,
                axes(bottom_coords),
            )
            @. insolation_tuple = instantaneous_zenith_angle(
                current_datetime,
                Float64(bottom_coords.long),
                Float64(bottom_coords.lat),
                insolation_params,
            ) # the tuple is (zenith angle, azimuthal angle, earth-sun distance)
            @. solar_zenith_angle =
                min(first(insolation_tuple), max_zenith_angle)
            @. weighted_irradiance =
                irradiance * (au / last(insolation_tuple))^2
        else
            # assume that the latitude and longitude are both 0 for flat space
            insolation_tuple = instantaneous_zenith_angle(
                current_datetime,
                0.0,
                0.0,
                insolation_params,
            )
            rrtmgp_model.solar_zenith_angle .=
                min(first(insolation_tuple), max_zenith_angle)
            rrtmgp_model.weighted_irradiance .=
                irradiance * (au / last(insolation_tuple))^2
        end
    end

    if !idealized_clouds && !(
        rrtmgp_model.radiation_mode isa RRTMGPI.GrayRadiation ||
        rrtmgp_model.radiation_mode isa RRTMGPI.ClearSkyRadiation
    )
        ?????z = Fields.local_geometry_field(Y.c).???x?????.components.data.:9
        ???lwp = RRTMGPI.array2field(
            rrtmgp_model.center_cloud_liquid_water_path,
            axes(Y.c),
        )
        ???iwp = RRTMGPI.array2field(
            rrtmgp_model.center_cloud_ice_water_path,
            axes(Y.c),
        )
        ???frac =
            RRTMGPI.array2field(rrtmgp_model.center_cloud_fraction, axes(Y.c))
        # multiply by 1000 to convert from kg/m^2 to g/m^2
        @. ???lwp =
            1000 * Y.c.?? * TD.liquid_specific_humidity(thermo_params, ???ts) * ?????z
        @. ???iwp =
            1000 * Y.c.?? * TD.ice_specific_humidity(thermo_params, ???ts) * ?????z
        @. ???frac =
            ifelse(TD.has_condensate(thermo_params, ???ts), FT(1), FT(0) * ?????z)
    end

    RRTMGPI.update_fluxes!(rrtmgp_model)
    RRTMGPI.field2array(???radiation_flux) .= rrtmgp_model.face_flux
end
