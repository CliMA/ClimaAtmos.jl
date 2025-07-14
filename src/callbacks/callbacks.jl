import ClimaCore.DataLayouts as DL
import .RRTMGPInterface as RRTMGPI
import Thermodynamics as TD
import CloudMicrophysics as CM
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
            horizontal_integral_at_boundary(ᶠradiation_flux, nlevels + half) *
            float(Δt)
        if p.atmos.surface_model isa PrescribedSST
            net_energy_flux_sfc[] +=
                horizontal_integral_at_boundary(ᶠradiation_flux, half) *
                float(Δt)
        end
    end
    return nothing
end

"""
    external_driven_single_column!(integrator)

Evaluate external time-varying forcing inputs for a single-column atmospheric model onto
objects in the cache.

This callback function evaluates external forcing variables at the current simulation time and
updates the corresponding fields in the model state. It handles various forcing components:

- Temperature and specific humidity tendencies (vertical eddy terms, horizontal advection)
- Nudging fields for temperature, humidity, and horizontal wind components
- Large-scale subsidence computed from vertical velocity

# Arguments
- `integrator`: The ODE integrator containing the current model state (`u`),
  cache (`p`), and time (`t`)

# Notes
The function extracts time-varying inputs from the `column_timevaryinginputs` structure
and evaluates them at the current time using the `evaluate!` function, which updates
the corresponding model fields in place.
"""
function external_driven_single_column!(integrator)
    Y = integrator.u
    p = integrator.p
    t = integrator.t

    @assert p.atmos.sfc_temperature isa ExternalTVColumnSST (
        "SCM reanalysis timevarying setup requires `initial_condition`, " *
        "`external_forcing`, `surface_setup`, and `surface_temperature` " *
        "to be set to `ReanalysisTimeVarying`"
    )

    FT = Spaces.undertype(axes(Y.c))
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    # unpack external forcing objects that we can directly set.
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

    # subsidence
    evaluate!(ᶜls_subsidence, wa, t)
end



NVTX.@annotate function cloud_fraction_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    (; ᶜts, ᶜgradᵥ_θ_virt, ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    if isnothing(p.atmos.turbconv_model)
        @. ᶜgradᵥ_θ_virt =
            ᶜgradᵥ(ᶠinterp(TD.virtual_pottemp(thermo_params, ᶜts)))
        @. ᶜgradᵥ_q_tot =
            ᶜgradᵥ(ᶠinterp(TD.total_specific_humidity(thermo_params, ᶜts)))
        @. ᶜgradᵥ_θ_liq_ice =
            ᶜgradᵥ(ᶠinterp(TD.liquid_ice_pottemp(thermo_params, ᶜts)))
    end
    set_cloud_fraction!(Y, p, p.atmos.moisture_model, p.atmos.cloud_model)
end

NVTX.@annotate function rrtmgp_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    t = integrator.t

    (; params) = p
    (; ᶠradiation_flux, rrtmgp_model) = p.radiation
    (; radiation_mode) = p.atmos

    RRTMGPI.update_atmospheric_state!(integrator)

    set_insolation_variables!(Y, p, t, p.atmos.insolation)
    set_surface_albedo!(Y, p, t, p.atmos.surface_albedo)

    RRTMGPI.update_fluxes!(rrtmgp_model, UInt32(floor(t / integrator.p.dt)))
    Fields.field2array(ᶠradiation_flux) .= rrtmgp_model.face_flux
    return nothing
end

NVTX.@annotate function nogw_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p

    non_orographic_gravity_wave_compute_tendency!(
        Y,
        p,
        p.atmos.non_orographic_gravity_wave,
    )
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

function set_insolation_variables!(Y, p, t, ::GCMDrivenInsolation)
    (; rrtmgp_model) = p.radiation
    rrtmgp_model.cos_zenith .= Fields.field2array(p.external_forcing.cos_zenith)
    rrtmgp_model.weighted_irradiance .=
        Fields.field2array(p.external_forcing.insolation)
end

function set_insolation_variables!(Y, p, t, ::ExternalTVInsolation)
    # unpack objects with time varying data
    (; rrtmgp_model) = p.radiation
    (; coszen, rsdt) = p.external_forcing.surface_inputs
    coszen_tv = p.external_forcing.surface_timevaryinginputs.coszen
    rsdt_tv = p.external_forcing.surface_timevaryinginputs.rsdt
    # evaluate time varying data onto temporary fields
    evaluate!(coszen, coszen_tv, t)
    evaluate!(rsdt, rsdt_tv, t)

    # set insolation variables from the values within the fields
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
        @. 1360 * (1 + FT(1.2) / 4 * (1 - 3 * sind(latitude)^2)) /
           (4 * cos(FT(π) / 3))
    rrtmgp_model.weighted_irradiance .= weighted_irradiance
end

function set_insolation_variables!(Y, p, t, tvi::TimeVaryingInsolation)
    FT = Spaces.undertype(axes(Y.c))
    params = p.params
    insolation_params = CAP.insolation_params(params)
    (; insolation_tuple, rrtmgp_model) = p.radiation

    current_datetime =
        t isa ITime ? ClimaUtilities.TimeManager.date(t) :
        tvi.start_date + Dates.Second(round(Int, t)) # current time
    max_zenith_angle = FT(π) / 2 - eps(FT)
    irradiance = FT(CAP.tot_solar_irrad(params))
    au = FT(CAP.astro_unit(params))
    # date0 references the start of the astronomial epoch J2000
    date0 = DateTime("2000-01-01T11:58:56.816")
    d, δ, η_UTC =
        FT.(
            Insolation.helper_instantaneous_zenith_angle(
                current_datetime,
                date0,
                insolation_params,
            ),
        )
    bottom_coords = Fields.coordinate_field(Spaces.level(Y.c, 1))
    cos_zenith =
        Fields.array2field(rrtmgp_model.cos_zenith, axes(bottom_coords))
    weighted_irradiance = Fields.array2field(
        rrtmgp_model.weighted_irradiance,
        axes(bottom_coords),
    )
    if eltype(bottom_coords) <: Geometry.LatLongZPoint
        @. insolation_tuple = instantaneous_zenith_angle(
            d,
            δ,
            η_UTC,
            bottom_coords.long,
            bottom_coords.lat,
        ) # the tuple is (zenith angle, azimuthal angle, earth-sun distance)
    else
        # assume that the latitude and longitude are both 0 for flat space,
        # so that insolation_tuple is a constant Field
        insolation_tuple .=
            Ref(instantaneous_zenith_angle(d, δ, η_UTC, FT(0), FT(0)))
    end
    @. cos_zenith = cos(min(first(insolation_tuple), max_zenith_angle))
    @. weighted_irradiance = irradiance * (au / last(insolation_tuple))^2
end

NVTX.@annotate function save_state_to_disk_func(integrator, output_dir)
    (; t, u, p) = integrator
    Y = u

    # TODO: Use ITime here
    t = float(t)
    day = floor(Int, t / (60 * 60 * 24))
    sec = floor(Int, t % (60 * 60 * 24))
    @info "Saving state to HDF5 file on day $day second $sec"
    output_file = joinpath(output_dir, "day$day.$sec.hdf5")
    comms_ctx = ClimaComms.context(integrator.u.c)
    hdfwriter = InputOutput.HDF5Writer(output_file, comms_ctx)
    # TODO: a better way to write metadata
    InputOutput.HDF5.write_attribute(hdfwriter.file, "time", t)
    InputOutput.HDF5.write_attribute(
        hdfwriter.file,
        "atmos_model_hash",
        hash(p.atmos),
    )
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
