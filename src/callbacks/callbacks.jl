import ClimaCore.DataLayouts as DL
import .RRTMGPInterface as RRTMGPI
import Thermodynamics as TD
import CloudMicrophysics as CM
import LinearAlgebra
import ClimaCore.Fields
import ClimaComms
import ClimaCore as CC
import ClimaCore.Spaces
import .Parameters as CAP
import ClimaCore: InputOutput
using Dates

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
        if !(p.atmos.surface.temperature isa SurfaceConditions.SlabOceanTemperature)
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

    @assert p.atmos.surface.temperature isa SurfaceConditions.ExternalTemperature (
        "SCM reanalysis timevarying setup requires `initial_condition` " *
        "and `external_forcing` to be set to `ReanalysisTimeVarying`"
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

import RRTMGP

NVTX.@annotate function rrtmgp_solver_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    t = integrator.t
    FT = eltype(Y)
    (; params) = p
    (; ᶠradiation_flux, rrtmgp_solver) = p.radiation
    (; radiation_mode) = p.atmos

    RRTMGPI.update_atmospheric_state!(integrator)

    set_insolation_variables!(Y, p, t, p.atmos.insolation)
    set_surface_albedo!(Y, p, t, p.atmos.surface_albedo)

    RRTMGP.update_fluxes!(rrtmgp_solver, UInt32(floor(FT(t) / integrator.p.dt)))
    Fields.field2array(ᶠradiation_flux) .= RRTMGP.net_flux(rrtmgp_solver)
    return nothing
end

NVTX.@annotate function subcol_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    run_cosp_cloudsat!(Y, p, p.atmos.microphysics_model)

    return nothing
end

struct NoOpCOSPSubcolumnConsumer end
(::NoOpCOSPSubcolumnConsumer)(_, _) = nothing

struct CloudSatSubcolumnConsumer{Z, K, ZE, D, H, G, HT, T, R, C}
    z_vol_work::Z
    kr_vol_work::K
    Ze_non_work::ZE
    DBZe::D
    hydro_path_attenuation_work::H
    gas_path_attenuation::G
    height_km::HT
    temperature::T
    rho_air::R
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
        consumer.temperature,
        consumer.rho_air,
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
    )
    return nothing
end

function run_cosp_cloudsat!(Y, p, ::NonEquilibriumMicrophysics1M)
    (;
        z_vol_cloudsat_work,
        kr_vol_cloudsat_work,
        g_vol_cloudsat,
        Ze_non_cloudsat_work,
        hydro_path_attenuation_cloudsat_work,
        gas_path_attenuation_cloudsat,
        height_km_cloudsat,
        DBZe_cloudsat,
        detected_column_cloudsat,
        cloudsat_tcc,
        ᶜp,
        ᶜT,
        ᶜq_tot_nonneg,
        ᶜq_liq,
        ᶜq_ice,
    ) = p.precomputed

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
    )

    consumer = CloudSatSubcolumnConsumer(
        z_vol_cloudsat_work,
        kr_vol_cloudsat_work,
        Ze_non_cloudsat_work,
        DBZe_cloudsat,
        hydro_path_attenuation_cloudsat_work,
        gas_path_attenuation_cloudsat,
        height_km_cloudsat,
        ᶜT,
        Y.c.ρ,
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

    COSP.COSPSubcolumns.set_scops_selectors!(
        ᶜscops_selectors,
        ᶜsubcolumn_cloud,
        ᶜsubcolumn_threshold,
        ᶜcloud_fraction,
        nsubcolumns,
        cosp.random_seed,
        cosp.overlap,
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
            overlap = cosp.overlap,
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

Prepare the sampled cloud and precipitation fractions, then regenerate and
stream one deterministic hydrometeor subcolumn at a time. `consume!` must use
the lazy hydrometeor broadcasts immediately; they borrow working mask and
scratch fields that are overwritten during subsequent iterations.
"""
function foreach_cosp_subcolumn(consume!::F, Y, p) where {F}
    microphysics_model = p.atmos.microphysics_model
    _check_cosp_microphysics(microphysics_model)
    prepare_cosp_subcolumns!(Y, p)
    return foreach_cosp_subcolumn(consume!, Y, p, microphysics_model)
end

function foreach_cosp_subcolumn(
    consume!::F,
    Y,
    p,
    ::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
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
    for isubcolumn in 1:nsubcolumns
        COSP.COSPSubcolumns.scops_subcolumn!(
            ᶜsubcolumn_cloud,
            ᶜsubcolumn_threshold,
            ᶜcloud_fraction,
            isubcolumn,
            nsubcolumns,
            cosp.random_seed;
            overlap = cosp.overlap,
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

NVTX.@annotate function ogw_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p

    orographic_gravity_wave_compute_tendency!(
        Y,
        p,
        p.atmos.orographic_gravity_wave,
    )
    return nothing
end

#Uniform insolation, magnitudes from Wing et al. (2018)
#Note that the TOA downward shortwave fluxes won't be the same as the values in the paper if add_isothermal_boundary_layer is true
function set_insolation_variables!(Y, p, t, ::RCEMIPIIInsolation)
    FT = Spaces.undertype(axes(Y.c))
    (; rrtmgp_solver) = p.radiation
    RRTMGP.cos_zenith(rrtmgp_solver) .= cosd(FT(42.05))
    RRTMGP.toa_flux(rrtmgp_solver) .= FT(551.58)
end

function set_insolation_variables!(Y, p, t, ::GCMDrivenInsolation)
    (; rrtmgp_solver) = p.radiation
    RRTMGP.cos_zenith(rrtmgp_solver) .= Fields.field2array(p.external_forcing.cos_zenith)
    RRTMGP.toa_flux(rrtmgp_solver) .=
        Fields.field2array(p.external_forcing.toa_flux)
end

function set_insolation_variables!(Y, p, t, ::ExternalTVInsolation)
    # unpack objects with time varying data
    (; rrtmgp_solver) = p.radiation
    (; coszen, rsdt) = p.external_forcing.surface_inputs
    coszen_tv = p.external_forcing.surface_timevaryinginputs.coszen
    rsdt_tv = p.external_forcing.surface_timevaryinginputs.rsdt
    # evaluate time varying data onto temporary fields
    evaluate!(coszen, coszen_tv, t)
    evaluate!(rsdt, rsdt_tv, t)

    # set insolation variables from the values within the fields
    RRTMGP.cos_zenith(rrtmgp_solver) .= Fields.field2array(coszen)
    RRTMGP.toa_flux(rrtmgp_solver) .= Fields.field2array(rsdt ./ coszen)
end

function set_insolation_variables!(Y, p, t, ::IdealizedInsolation)
    FT = Spaces.undertype(axes(Y.c))
    bottom_coords = Fields.coordinate_field(Spaces.level(Y.c, 1))
    if eltype(bottom_coords) <: Geometry.LatLongZPoint
        latitude = Fields.field2array(bottom_coords.lat)
    else
        latitude = Fields.field2array(zero(bottom_coords.z)) # flat space is on Equator
    end
    (; rrtmgp_solver) = p.radiation
    # Approximate annual mean insolation without diurnal cycle
    # Reference: O'Gorman and Schneider (2008), J. Climate, 21, 3815-3832
    RRTMGP.toa_flux(rrtmgp_solver) .= 680
    cos_zenith = RRTMGP.cos_zenith(rrtmgp_solver)
    @. cos_zenith =
        (1 + FT(0.3) * (1 - 3 * sind(latitude)^2)) * FT(0.5)
end

function set_insolation_variables!(Y, p, t, ::Larcform1Insolation)
    FT = Spaces.undertype(axes(Y.c))
    (; rrtmgp_solver) = p.radiation
    RRTMGP.cos_zenith(rrtmgp_solver) .= eps(FT) # polar night; keep μ>0 for RRTMGP
    RRTMGP.toa_flux(rrtmgp_solver) .= FT(0)
end

function set_insolation_variables!(Y, p, t, tvi::TimeVaryingInsolation)
    FT = Spaces.undertype(axes(Y.c))
    params = p.params
    insolation_params = CAP.insolation_params(params)
    (; insolation_tuple, rrtmgp_solver) = p.radiation

    current_datetime = if !(t isa ITime) && !isnothing(tvi.start_date)
        tvi.start_date + Dates.Second(round(Int, t))
    else
        ClimaUtilities.TimeManager.date(t)
    end

    bottom_coords = Fields.coordinate_field(Spaces.level(Y.c, 1))
    cos_zenith =
        Fields.array2field(RRTMGP.cos_zenith(rrtmgp_solver), axes(bottom_coords))
    toa_flux = Fields.array2field(
        RRTMGP.toa_flux(rrtmgp_solver),
        axes(bottom_coords),
    )

    # Use Insolation API: insolate_tuple = insolation(datetime, lat, lon, params)
    # Note: μ is already clamped at 0 by Insolation.jl but rrtmgp needs a non-zero μ
    if !isnothing(tvi.latitude) && !isnothing(tvi.longitude)
        # Explicit lat/lon override (e.g. single-column setups whose coordinate
        # system doesn't carry lat/lon).
        insolation_tuple .= Ref(
            Insolation.insolation(
                current_datetime,
                tvi.latitude,
                tvi.longitude,
                insolation_params,
            ),
        )
    elseif eltype(bottom_coords) <: Geometry.LatLongZPoint
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
    if any(isnan, parent(integrator.u))
        # Identify which field(s) have NaN
        Y = integrator.u
        for pn in propertynames(Y)
            sub = getproperty(Y, pn)
            for fn in propertynames(sub)
                field = getproperty(sub, fn)
                if any(isnan, parent(field))
                    n_nan = count(isnan, parent(field))
                    n_tot = length(parent(field))
                    @info "NaN found in Y.$pn.$fn: $n_nan / $n_tot elements"
                end
            end
        end
        error("Found NaN")
    end
    return nothing
end
