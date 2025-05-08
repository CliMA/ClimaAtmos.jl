import Thermodynamics as TD
import ClimaUtilities.TimeVaryingInputs: evaluate!
import ..Parameters as CAP
import ..PrescribedOzone, ..MaunaLoaCO2

update_atmospheric_state!(integrator) =
    update_atmospheric_state!(integrator.p.atmos.radiation_mode, integrator)

function update_atmospheric_state!(radiation_mode::GrayRadiation, integrator)
    # update temperature & pressure
    update_temperature_pressure!(integrator)
    return nothing
end

function update_atmospheric_state!(radiation_mode::R, integrator) where {R}
    # update temperature & pressure
    update_temperature_pressure!(integrator)
    # update relative humidity 
    update_relative_humidity!(integrator)
    # update column amounts of dry/moist air
    update_concentrations!(radiation_mode, integrator.p.radiation.rrtmgp_model)
    # update gas concentrations (volume mixing ratios)
    update_volume_mixing_ratios!(integrator)
    return nothing
end

"""
    update_temperature_pressure!((; u, p, t)::I) where {I}

Update temperature and pressure.
"""
function update_temperature_pressure!((; u, p, t)::I) where {I}
    params = p.params
    FT = Spaces.undertype(axes(u.c))
    thermo_params = CAP.thermodynamics_params(params)
    T_min = CAP.optics_lookup_temperature_min(params)
    T_max = CAP.optics_lookup_temperature_max(params)

    (; ᶜts, ᶜp, sfc_conditions) = p.precomputed
    model = p.radiation.rrtmgp_model

    # update surface temperature
    sfc_ts = sfc_conditions.ts
    sfc_T = Fields.array2field(model.surface_temperature, axes(sfc_ts))
    @. sfc_T = TD.air_temperature(thermo_params, sfc_ts)

    # update layer pressure
    model.center_pressure .= Fields.field2array(p.precomputed.ᶜp)
    # compute layer temperature
    ᶜT = Fields.array2field(model.center_temperature, axes(u.c))
    # TODO: move this to RRTMGP
    @. ᶜT =
        min(max(TD.air_temperature(thermo_params, ᶜts), FT(T_min)), FT(T_max))
    # compute level temperatures and pressures using interpolation/extrapolation
    update_implied_values!(model)
    return nothing
end

"""
    update_relative_humidity!(integrator)

Update relative humidity.
"""
function update_relative_humidity!((; u, p, t)::I) where {I}
    (; radiation_mode) = p.atmos
    (; rrtmgp_model) = p.radiation
    thermo_params = CAP.thermodynamics_params(p.params)
    FT = eltype(thermo_params)
    (; ᶜts) = p.precomputed
    ᶜrh = Fields.array2field(rrtmgp_model.center_relative_humidity, axes(u.c))
    ᶜvmr_h2o = Fields.array2field(
        rrtmgp_model.center_volume_mixing_ratio_h2o,
        axes(u.c),
    )
    if radiation_mode.idealized_h2o
        # slowly increase the relative humidity from 0 to 0.6 to account for
        # the fact that we have a very unrealistic initial condition
        max_relative_humidity = FT(0.6)
        t_increasing_humidity = FT(60 * 60 * 24 * 30)
        if float(t) < t_increasing_humidity
            max_relative_humidity *= float(t) / t_increasing_humidity
        end
        @. ᶜrh = max_relative_humidity

        # temporarily store ᶜq_tot in ᶜvmr_h2o
        ᶜq_tot = ᶜvmr_h2o
        @. ᶜq_tot =
            max_relative_humidity * TD.q_vap_saturation(thermo_params, ᶜts)

        # filter ᶜq_tot so that it is monotonically decreasing with z
        for i in 2:Spaces.nlevels(axes(ᶜq_tot))
            level = Fields.field_values(Spaces.level(ᶜq_tot, i))
            prev_level = Fields.field_values(Spaces.level(ᶜq_tot, i - 1))
            @. level = min(level, prev_level)
        end

        # assume that ᶜq_vap = ᶜq_tot when computing ᶜvmr_h2o
        @. ᶜvmr_h2o =
            TD.vol_vapor_mixing_ratio(thermo_params, TD.PhasePartition(ᶜq_tot))
    else
        @. ᶜvmr_h2o = TD.vol_vapor_mixing_ratio(
            thermo_params,
            TD.PhasePartition(thermo_params, ᶜts),
        )
        @. ᶜrh = min(max(TD.relative_humidity(thermo_params, ᶜts), 0), 1)
    end

    return nothing
end

"""
    update_volume_mixing_ratios!((; p, t)::I) where {I}

Update volume mixing ratios.
"""
function update_volume_mixing_ratios!((; p, t)::I) where {I}
    # If we have prescribed ozone or aerosols, we need to update them
    update_o3!(p, t, p.atmos.ozone) 
    update_co2!(p, t, p.atmos.co2)

    return nothing
end

update_o3!(_, _, _) = nothing
function update_o3!(p, t, ::PrescribedOzone)
    evaluate!(p.tracers.o3, p.tracers.prescribed_o3_timevaryinginput, t)
    return nothing
end

update_co2!(_, _, _) = nothing
function update_co2!(p, t, ::MaunaLoaCO2)
    evaluate!(p.tracers.co2, p.tracers.prescribed_co2_timevaryinginput, t)
    return nothing
end

