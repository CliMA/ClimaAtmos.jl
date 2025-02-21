import ClimaCore.DataLayouts as DL
import .RRTMGPInterface as RRTMGPI
import Thermodynamics as TD
import LinearAlgebra
import ClimaCore.Fields
import ClimaComms
import ClimaCore as CC
import ClimaCore.Spaces
import SciMLBase
import .Parameters as CAP
import ClimaCore: InputOutput
using Dates
using Insolation: instantaneous_zenith_angle
import ClimaCore.Fields: ColumnField

import ClimaUtilities.TimeVaryingInputs: evaluate!

include("callback_helpers.jl")

function flux_accumulation!(integrator)
    Y = integrator.u
    p = integrator.p
    Δt = integrator.dt
    if !isnothing(p.atmos.radiation_mode)
        (; ᶠradiation_flux) = p.radiation
        (; net_energy_flux_toa, net_energy_flux_sfc) = p
        nlevels = Spaces.nlevels(axes(Y.c))
        net_energy_flux_toa[] +=
            horizontal_integral_at_boundary(ᶠradiation_flux, nlevels + half) * Δt
        if p.atmos.surface_model isa PrescribedSurfaceTemperature
            net_energy_flux_sfc[] +=
                horizontal_integral_at_boundary(ᶠradiation_flux, half) * Δt
        end
    end
    return nothing
end

function external_driven_single_column!(integrator)
    Y = integrator.u
    p = integrator.p
    t = integrator.t

    # unpack external forcing objects that we can directly set.
    FT = Spaces.undertype(axes(Y.c))
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜspecific, ᶜts, ᶜh_tot) = p.precomputed
    (;
        ᶜdTdt_fluc,
        ᶜdqtdt_fluc,
        ᶜdTdt_hadv,
        ᶜdqtdt_hadv,
        ᶜdTdt_rad, # we skip radiation because we're using RRTMGP, but this can be changed for simpler setups
        ᶜT_nudge,
        ᶜqt_nudge,
        ᶜu_nudge,
        ᶜv_nudge,
        ᶜls_subsidence,
        ᶜinv_τ_wind,
        ᶜinv_τ_scalar,
    ) = p.external_forcing
    # unpack tv inputs
    (; hus, rho, ta, tnhusha, tnhusva, tntha, tntva, ua, va, wa, wap) =
        p.external_forcing.column_timevaryinginputs

    # set the external forcing variables; external tendency is updated in a remaining_tendency! call
    evaluate!(ᶜdTdt_fluc, tntva, t)
    evaluate!(ᶜdqtdt_fluc, tnhusva, t)
    evaluate!(ᶜdTdt_hadv, tntha, t)
    evaluate!(ᶜdqtdt_hadv, tnhusha, t)
    evaluate!(ᶜT_nudge, ta, t)
    evaluate!(ᶜqt_nudge, hus, t)
    evaluate!(ᶜu_nudge, ua, t)
    evaluate!(ᶜv_nudge, va, t)

    # subsidence term we need wap / -(rho) / gravity 
    evaluate!(p.external_forcing.column_inputs.rho, rho, t)
    evaluate!(ᶜls_subsidence, wap, t)
    # need space to store rho 
    ᶜls_subsidence .=
        ᶜls_subsidence ./ .-(p.external_forcing.column_inputs.rho) ./ FT(CAP.grav(params))

    eddy_vert_fluctuation!(ᶜdTdt_fluc, ᶜT_nudge, ᶜls_subsidence)
    eddy_vert_fluctuation!(ᶜdqtdt_fluc, ᶜqt_nudge, ᶜls_subsidence)

    @. ᶜinv_τ_wind = 1 / (6 * 3600)
    # set relaxation profile toward reference state - should find better way than field2array 
    Fields.field2array(ᶜinv_τ_scalar) .=
        compute_gcm_driven_scalar_inv_τ.(
            Fields.field2array(Fields.coordinate_field(ᶜinv_τ_scalar).z)[:]
        )

    # also update the surface state (just need the temperature) - the 
    (; ts) = p.external_forcing.surface_timevaryinginputs
    evaluate!(p.external_forcing.surface_inputs.ts, ts, t)
end



NVTX.@annotate function cloud_fraction_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    (; ᶜts, ᶜgradᵥ_θ_virt, ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    if isnothing(p.atmos.turbconv_model)
        @. ᶜgradᵥ_θ_virt = ᶜgradᵥ(ᶠinterp(TD.virtual_pottemp(thermo_params, ᶜts)))
        @. ᶜgradᵥ_q_tot = ᶜgradᵥ(ᶠinterp(TD.total_specific_humidity(thermo_params, ᶜts)))
        @. ᶜgradᵥ_θ_liq_ice = ᶜgradᵥ(ᶠinterp(TD.liquid_ice_pottemp(thermo_params, ᶜts)))
    end
    set_cloud_fraction!(Y, p, p.atmos.moisture_model, p.atmos.cloud_model)
end

# TODO: Move this somewhere else
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


NVTX.@annotate function rrtmgp_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    t = integrator.t

    (; ᶜts, cloud_diagnostics_tuple, sfc_conditions) = p.precomputed
    (; params) = p
    (; ᶠradiation_flux, rrtmgp_model) = p.radiation
    (; radiation_mode) = p.atmos

    # If we have prescribed ozone or aerosols, we need to update them
    update_o3!(p, t, p.atmos.ozone)
    update_co2!(p, t, p.atmos.co2)
    if :prescribed_aerosols_field in propertynames(p.tracers)
        for (key, tv) in pairs(p.tracers.prescribed_aerosol_timevaryinginputs)
            field = getproperty(p.tracers.prescribed_aerosols_field, key)
            evaluate!(field, tv, t)
        end
    end
    if :prescribed_clouds_field in propertynames(p.radiation)
        for (key, tv) in pairs(p.radiation.prescribed_cloud_timevaryinginputs)
            field = getproperty(p.radiation.prescribed_clouds_field, key)
            evaluate!(field, tv, t)
        end
    end

    FT = Spaces.undertype(axes(Y.c))
    thermo_params = CAP.thermodynamics_params(params)
    T_min = CAP.optics_lookup_temperature_min(params)
    T_max = CAP.optics_lookup_temperature_max(params)

    sfc_ts = sfc_conditions.ts
    sfc_T = Fields.array2field(rrtmgp_model.surface_temperature, axes(sfc_ts))
    @. sfc_T = TD.air_temperature(thermo_params, sfc_ts)

    ᶜp = Fields.array2field(rrtmgp_model.center_pressure, axes(Y.c))
    ᶜT = Fields.array2field(rrtmgp_model.center_temperature, axes(Y.c))
    @. ᶜp = TD.air_pressure(thermo_params, ᶜts)
    # TODO: move this to RRTMGP
    @. ᶜT = min(max(TD.air_temperature(thermo_params, ᶜts), FT(T_min)), FT(T_max))

    if !(radiation_mode isa RRTMGPI.GrayRadiation)
        ᶜrh = Fields.array2field(rrtmgp_model.center_relative_humidity, axes(Y.c))
        ᶜvmr_h2o =
            Fields.array2field(rrtmgp_model.center_volume_mixing_ratio_h2o, axes(Y.c))
        if radiation_mode.idealized_h2o
            # slowly increase the relative humidity from 0 to 0.6 to account for
            # the fact that we have a very unrealistic initial condition
            max_relative_humidity = FT(0.6)
            t_increasing_humidity = FT(60 * 60 * 24 * 30)
            if t < t_increasing_humidity
                max_relative_humidity *= t / t_increasing_humidity
            end
            @. ᶜrh = max_relative_humidity

            # temporarily store ᶜq_tot in ᶜvmr_h2o
            ᶜq_tot = ᶜvmr_h2o
            @. ᶜq_tot = max_relative_humidity * TD.q_vap_saturation(thermo_params, ᶜts)

            # filter ᶜq_tot so that it is monotonically decreasing with z
            for i = 2:Spaces.nlevels(axes(ᶜq_tot))
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
    end

    if p.atmos.insolation isa IdealizedInsolation ||
       !(p.atmos.surface_albedo isa CouplerAlbedo)
        set_insolation_variables!(Y, p, t, p.atmos.insolation)
    end

    if radiation_mode isa RRTMGPI.AllSkyRadiation ||
       radiation_mode isa RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics
        if !radiation_mode.idealized_clouds
            ᶜΔz = Fields.Δz_field(Y.c)
            ᶜlwp =
                Fields.array2field(rrtmgp_model.center_cloud_liquid_water_path, axes(Y.c))
            ᶜiwp = Fields.array2field(rrtmgp_model.center_cloud_ice_water_path, axes(Y.c))
            ᶜfrac = Fields.array2field(rrtmgp_model.center_cloud_fraction, axes(Y.c))
            # RRTMGP needs lwp and iwp in g/m^2
            kg_to_g_factor = 1000
            cloud_liquid_water_content =
                radiation_mode.cloud isa PrescribedCloudInRadiation ?
                p.radiation.prescribed_clouds_field.clwc : cloud_diagnostics_tuple.q_liq
            cloud_ice_water_content =
                radiation_mode.cloud isa PrescribedCloudInRadiation ?
                p.radiation.prescribed_clouds_field.ciwc : cloud_diagnostics_tuple.q_ice
            cloud_fraction =
                radiation_mode.cloud isa PrescribedCloudInRadiation ?
                p.radiation.prescribed_clouds_field.cc : cloud_diagnostics_tuple.cf
            @. ᶜlwp =
                kg_to_g_factor * Y.c.ρ * cloud_liquid_water_content * ᶜΔz /
                max(cloud_fraction, eps(FT))
            @. ᶜiwp =
                kg_to_g_factor * Y.c.ρ * cloud_ice_water_content * ᶜΔz /
                max(cloud_fraction, eps(FT))
            @. ᶜfrac = cloud_fraction
        end
    end

    if !(radiation_mode isa RRTMGPI.GrayRadiation)
        if radiation_mode.aerosol_radiation
            ᶜΔz = Fields.Δz_field(Y.c)

            ᶜaero_conc =
                Fields.array2field(rrtmgp_model.center_dust_column_mass_density, axes(Y.c))
            @. ᶜaero_conc = 0
            for prescribed_aerosol_name in [:DST01, :DST02, :DST03, :DST04]
                if prescribed_aerosol_name in
                   propertynames(p.tracers.prescribed_aerosols_field)
                    aerosol_field = getproperty(
                        p.tracers.prescribed_aerosols_field,
                        prescribed_aerosol_name,
                    )
                    @. ᶜaero_conc += aerosol_field * Y.c.ρ * ᶜΔz
                end
            end

            ᶜaero_conc =
                Fields.array2field(rrtmgp_model.center_ss_column_mass_density, axes(Y.c))
            @. ᶜaero_conc = 0
            for prescribed_aerosol_name in [:SSLT01, :SSLT02, :SSLT03, :SSLT04]
                if prescribed_aerosol_name in
                   propertynames(p.tracers.prescribed_aerosols_field)
                    aerosol_field = getproperty(
                        p.tracers.prescribed_aerosols_field,
                        prescribed_aerosol_name,
                    )
                    @. ᶜaero_conc += aerosol_field * Y.c.ρ * ᶜΔz
                end
            end

            aerosol_names_pair = [
                (:center_so4_column_mass_density, :SO4),
                (:center_bcpi_column_mass_density, :CB2),
                (:center_bcpo_column_mass_density, :CB1),
                (:center_ocpi_column_mass_density, :OC2),
                (:center_ocpo_column_mass_density, :OC1),
            ]

            for (rrtmgp_aerosol_name, prescribed_aerosol_name) in aerosol_names_pair
                ᶜaero_conc = Fields.array2field(
                    getproperty(rrtmgp_model, rrtmgp_aerosol_name),
                    axes(Y.c),
                )
                if prescribed_aerosol_name in
                   propertynames(p.tracers.prescribed_aerosols_field)
                    aerosol_field = getproperty(
                        p.tracers.prescribed_aerosols_field,
                        prescribed_aerosol_name,
                    )
                    @. ᶜaero_conc = aerosol_field * Y.c.ρ * ᶜΔz
                else
                    @. ᶜaero_conc = 0
                end
            end

        end
        if :o3 in propertynames(p.tracers)
            ᶜvmr_o3 =
                Fields.array2field(rrtmgp_model.center_volume_mixing_ratio_o3, axes(Y.c))
            @. ᶜvmr_o3 = p.tracers.o3
        end
        if :co2 in propertynames(p.tracers)
            if pkgversion(ClimaUtilities) < v"0.1.21"
                rrtmgp_model.volume_mixing_ratio_co2 .= p.tracers.co2
            else
                rrtmgp_model.volume_mixing_ratio_co2 .= p.tracers.co2[]
            end
        end
    end

    set_surface_albedo!(Y, p, t, p.atmos.surface_albedo)

    RRTMGPI.update_fluxes!(rrtmgp_model, UInt32(t / integrator.p.dt))
    Fields.field2array(ᶠradiation_flux) .= rrtmgp_model.face_flux
    return nothing
end

#Uniform insolation, magnitudes from Wing et al. (2018)
#Note that the TOA downward shortwave fluxes won't be the same as the values in the paper if add_isothermal_boundary_layer is true
function set_insolation_variables!(Y, p, t, ::RCEMIPIIInsolation)
    FT = Spaces.undertype(axes(Y.c))
    (; rrtmgp_model) = p.radiation
    rrtmgp_model.cos_zenith .= cosd(FT(42.05))
    rrtmgp_model.weighted_irradiance .= FT(551.58)
end

function set_insolation_variables!(
    Y,
    p,
    t,
    ::Union{GCMDrivenInsolation,ERA5DrivenInsolation},
)
    (; rrtmgp_model) = p.radiation
    rrtmgp_model.cos_zenith .= Fields.field2array(p.external_forcing.cos_zenith)
    rrtmgp_model.weighted_irradiance .= Fields.field2array(p.external_forcing.insolation)
end

function set_insolation_variables!(Y, p, t, ::ExternalTVInsolation)
    (; rrtmgp_model) = p.radiation
    (; coszen, rsdt) = p.external_forcing.surface_inputs
    coszen_tv = getproperty(p.external_forcing.surface_timevaryinginputs, :coszen)
    rsdt_tv = getproperty(p.external_forcing.surface_timevaryinginputs, :rsdt)

    evaluate!(coszen, coszen_tv, t)
    evaluate!(rsdt, rsdt_tv, t)

    # set variables in rrtmgp 
    # TODO Check arrays have the same 
    rrtmgp_model.cos_zenith .= Fields.field2array(coszen)
    rrtmgp_model.weighted_irradiance .= Fields.field2array(rsdt)
end

function set_insolation_variables!(Y, p, t, ::IdealizedInsolation)
    FT = Spaces.undertype(axes(Y.c))
    bottom_coords = Fields.coordinate_field(Spaces.level(Y.c, 1))
    if eltype(bottom_coords) <: Geometry.LatLongZPoint
        latitude = Fields.field2array(bottom_coords.lat)
    else
        latitude = Fields.field2array(zero(bottom_coords.z)) # flat space is on Equator
    end
    (; rrtmgp_model) = p.radiation
    # perpetual equinox with no diurnal cycle
    rrtmgp_model.cos_zenith .= cos(FT(π) / 3)
    weighted_irradiance =
        @. 1360 * (1 + FT(1.2) / 4 * (1 - 3 * sind(latitude)^2)) / (4 * cos(FT(π) / 3))
    rrtmgp_model.weighted_irradiance .= weighted_irradiance
end

function set_insolation_variables!(Y, p, t, tvi::TimeVaryingInsolation)
    FT = Spaces.undertype(axes(Y.c))
    params = p.params
    insolation_params = CAP.insolation_params(params)
    (; insolation_tuple, rrtmgp_model) = p.radiation

    current_datetime = tvi.start_date + Dates.Second(round(Int, t)) # current time
    max_zenith_angle = FT(π) / 2 - eps(FT)
    irradiance = FT(CAP.tot_solar_irrad(params))
    au = FT(CAP.astro_unit(params))
    # TODO: Where does this date0 come from?
    date0 = DateTime("2000-01-01T11:58:56.816")
    d, δ, η_UTC =
        FT.(
            Insolation.helper_instantaneous_zenith_angle(
                current_datetime,
                date0,
                insolation_params,
            )
        )
    bottom_coords = Fields.coordinate_field(Spaces.level(Y.c, 1))
    cos_zenith = Fields.array2field(rrtmgp_model.cos_zenith, axes(bottom_coords))
    weighted_irradiance =
        Fields.array2field(rrtmgp_model.weighted_irradiance, axes(bottom_coords))
    if eltype(bottom_coords) <: Geometry.LatLongZPoint
        @. insolation_tuple =
            instantaneous_zenith_angle(d, δ, η_UTC, bottom_coords.long, bottom_coords.lat) # the tuple is (zenith angle, azimuthal angle, earth-sun distance)
    else
        # assume that the latitude and longitude are both 0 for flat space,
        # so that insolation_tuple is a constant Field
        insolation_tuple .= Ref(instantaneous_zenith_angle(d, δ, η_UTC, FT(0), FT(0)))
    end
    @. cos_zenith = cos(min(first(insolation_tuple), max_zenith_angle))
    @. weighted_irradiance = irradiance * (au / last(insolation_tuple))^2
end

NVTX.@annotate function save_state_to_disk_func(integrator, output_dir)
    (; t, u, p) = integrator
    Y = u

    day = floor(Int, t / (60 * 60 * 24))
    sec = floor(Int, t % (60 * 60 * 24))
    @info "Saving state to HDF5 file on day $day second $sec"
    output_file = joinpath(output_dir, "day$day.$sec.hdf5")
    comms_ctx = ClimaComms.context(integrator.u.c)
    hdfwriter = InputOutput.HDF5Writer(output_file, comms_ctx)
    # TODO: a better way to write metadata
    InputOutput.HDF5.write_attribute(hdfwriter.file, "time", t)
    InputOutput.HDF5.write_attribute(hdfwriter.file, "atmos_model_hash", hash(p.atmos))
    InputOutput.write!(hdfwriter, Y, "Y")
    Base.close(hdfwriter)
    return nothing
end

function gc_func(integrator)
    num_pre = Base.gc_num()
    alloc_since_last = (num_pre.allocd + num_pre.deferred_alloc) / 2^20
    live_pre = Base.gc_live_bytes() / 2^20
    GC.gc(false)
    live_post = Base.gc_live_bytes() / 2^20
    num_post = Base.gc_num()
    gc_time = (num_post.total_time - num_pre.total_time) / 10^9 # count in ns
    @debug(
        "GC",
        t = integrator.t,
        "alloc since last GC (MB)" = alloc_since_last,
        "live mem pre (MB)" = live_pre,
        "live mem post (MB)" = live_post,
        "GC time (s)" = gc_time,
        "# pause" = num_post.pause,
        "# full_sweep" = num_post.full_sweep,
    )
    return nothing
end

"""
    maybe_graceful_exit(integrator)

This callback is called after every timestep
to allow users to gracefully exit a running
simulation. To do so, users can navigate to
and open `{output_dir}/graceful_exit.dat`, change
the file contents from 0 to 1, and the running
simulation will gracefully exit with the integrator.

!!! note
    This may not be reliable for MPI jobs.
"""
function maybe_graceful_exit(output_dir, integrator)
    file = joinpath(output_dir, "graceful_exit.dat")
    if isfile(file)
        open(file, "r") do io
            while !eof(io)
                try
                    code = parse(Int, read(io, Char))
                    return code != 0
                catch
                    open(io -> print(io, 0), file, "w")
                    return false
                end
            end
            return false
        end
    else
        ispath(output_dir) || mkpath(output_dir)
        open(io -> print(io, 0), file, "w")
    end
end
function reset_graceful_exit(output_dir)
    file = joinpath(output_dir, "graceful_exit.dat")
    ispath(output_dir) || mkpath(output_dir)
    open(io -> print(io, 0), file, "w")
end

function check_nans(integrator)
    any(isnan, parent(integrator.u)) && error("Found NaN")
    return nothing
end
