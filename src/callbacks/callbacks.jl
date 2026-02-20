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

import ClimaCore.Fields: ColumnField

import ClimaUtilities.TimeVaryingInputs: evaluate!


include("callback_helpers.jl")

function flux_accumulation!(integrator)
    Y = integrator.u
    p = integrator.p
    Δt = integrator.dt
    FT = eltype(p.params)
    if !isnothing(p.atmos.radiation_mode)
        (; ᶠradiation_flux) = p.radiation
        (; net_energy_flux_toa, net_energy_flux_sfc) = p
        nlevels = Spaces.nlevels(axes(Y.c))
        net_energy_flux_toa[] +=
            horizontal_integral_at_boundary(ᶠradiation_flux, nlevels + half) *
            FT(Δt)
        if p.atmos.surface_model isa PrescribedSST
            net_energy_flux_sfc[] +=
                horizontal_integral_at_boundary(ᶠradiation_flux, half) *
                FT(Δt)
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
    (; ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno, ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice) =
        p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    if isnothing(p.atmos.turbconv_model)
        @. ᶜgradᵥ_q_tot = ᶜgradᵥ(ᶠinterp(ᶜq_tot_safe))
        @. ᶜgradᵥ_θ_liq_ice = ᶜgradᵥ(
            ᶠinterp(
                TD.liquid_ice_pottemp(
                    thermo_params,
                    ᶜT,
                    Y.c.ρ,
                    ᶜq_tot_safe,
                    ᶜq_liq_rai,
                    ᶜq_ice_sno,
                ),
            ),
        )
    end
    set_cloud_fraction!(Y, p, p.atmos.microphysics_model, p.atmos.cloud_model)
end

NVTX.@annotate function rrtmgp_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    t = integrator.t
    FT = eltype(Y)
    (; params) = p
    (; ᶠradiation_flux, rrtmgp_model) = p.radiation
    (; radiation_mode) = p.atmos

    RRTMGPI.update_atmospheric_state!(integrator)

    set_insolation_variables!(Y, p, t, p.atmos.insolation)
    set_surface_albedo!(Y, p, t, p.atmos.surface_albedo)

    RRTMGPI.update_fluxes!(rrtmgp_model, UInt32(floor(FT(t) / integrator.p.dt)))
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

NVTX.@annotate function edmfx_filter_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    t = integrator.t

    edmfx_filter_tendency!(
        Y,
        p,
        t,
        p.atmos.turbconv_model,
    )
    return nothing
end

#Uniform insolation, magnitudes from Wing et al. (2018)
#Note that the TOA downward shortwave fluxes won't be the same as the values in the paper if add_isothermal_boundary_layer is true
function set_insolation_variables!(Y, p, t, ::RCEMIPIIInsolation)
    FT = Spaces.undertype(axes(Y.c))
    (; rrtmgp_model) = p.radiation
    rrtmgp_model.cos_zenith .= cosd(FT(42.05))
    rrtmgp_model.toa_flux .= FT(551.58)
end

function set_insolation_variables!(Y, p, t, ::GCMDrivenInsolation)
    (; rrtmgp_model) = p.radiation
    rrtmgp_model.cos_zenith .= Fields.field2array(p.external_forcing.cos_zenith)
    rrtmgp_model.toa_flux .=
        Fields.field2array(p.external_forcing.toa_flux)
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
    rrtmgp_model.toa_flux .= Fields.field2array(rsdt ./ coszen)
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
    # Approximate annual mean insolation without diurnal cycle
    # Reference: O'Gorman and Schneider (2008), J. Climate, 21, 3815-3832
    rrtmgp_model.toa_flux .= 680
    @. rrtmgp_model.cos_zenith = (1 + FT(0.3) * (1 - 3 * sind(latitude)^2)) * FT(0.5)
end

function set_insolation_variables!(Y, p, t, tvi::TimeVaryingInsolation)
    FT = Spaces.undertype(axes(Y.c))
    params = p.params
    insolation_params = CAP.insolation_params(params)
    (; insolation_tuple, rrtmgp_model) = p.radiation

    current_datetime =
        t isa ITime ? ClimaUtilities.TimeManager.date(t) :
        tvi.start_date + Dates.Second(round(Int, t)) # current time

    bottom_coords = Fields.coordinate_field(Spaces.level(Y.c, 1))
    cos_zenith =
        Fields.array2field(rrtmgp_model.cos_zenith, axes(bottom_coords))
    toa_flux = Fields.array2field(
        rrtmgp_model.toa_flux,
        axes(bottom_coords),
    )

    # Use Insolation API: insolate_tuple = insolation(datetime, lat, lon, params)
    # Note: μ is already clamped at 0 by Insolation.jl but rrtmgp needs a non-zero μ
    if eltype(bottom_coords) <: Geometry.LatLongZPoint
        # Calculate insolation for each grid point
        @. insolation_tuple = Insolation.insolation(
            current_datetime,
            bottom_coords.lat,
            bottom_coords.long,
            insolation_params,
        )

    else
        # assume that the latitude and longitude are both 0 for flat space
        insolation_tuple .= Ref(Insolation.insolation(
            current_datetime,
            FT(0),
            FT(0),
            insolation_params,
        ))
    end
    @. cos_zenith = max(insolation_tuple.μ, eps(FT))
    @. toa_flux = insolation_tuple.S
end

NVTX.@annotate function save_state_to_disk_func(integrator, output_dir)
    (; t, u, p) = integrator
    Y = u
    FT = eltype(p.params)

    # TODO: Use ITime here
    t = FT(t)
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
