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
import UnrolledUtilities: unrolled_foreach


include("callback_helpers.jl")

"""
    flux_accumulation!(integrator)

Accumulate the time-integrated net radiative energy flux through the model boundaries.

Adds `Δt` times the horizontally integrated `ᶠradiation_flux` at the top of the atmosphere
into `p.net_energy_flux_toa`, and likewise at the surface into `p.net_energy_flux_sfc`.
The surface term is skipped for a slab ocean, which carries its own energy budget. Does
nothing when radiation is disabled.

Mutates the `Ref`s in `integrator.p` and returns `nothing`. Installed by
`conservation_checking_callback` and read back by the conservation-check diagnostics.
"""
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

Refresh the external time-varying forcing of a single-column simulation.

Calls `update_forcing_term!` on each term in `p.external_forcing.forcing_terms`, which
evaluates that term's time-varying inputs at the current time and writes them into its
working cache fields. Depending on the configured terms, this covers temperature and
specific-humidity tendencies, nudging targets for temperature, humidity, and horizontal
wind, and large-scale subsidence.

Mutates the term caches in `integrator.p` and returns `nothing`. Nothing is added to the
state here: the refreshed terms are applied later, in `remaining_tendency!`.

# Arguments

  - `integrator`: The ODE integrator, holding the state (`u`), cache (`p`), and time (`t`).
"""
function external_driven_single_column!(integrator)
    p = integrator.p
    t = integrator.t
    (; forcing_terms, term_caches) = p.external_forcing
    unrolled_foreach(forcing_terms, term_caches) do term, cache
        update_forcing_term!(cache, term, t)
    end
    return nothing
end

import RRTMGP

"""
    rrtmgp_solver_callback!(integrator)

Run the RRTMGP radiative transfer solve and store the resulting net flux.

Pushes the current atmospheric state into the solver, refreshes the insolation
(`set_insolation_variables!`) and the surface albedo (`set_surface_albedo!`), solves, and
copies the net flux into `p.radiation.ᶠradiation_flux`. That field is then held fixed and
reused by the tendencies until the next radiation call, which is why the `dt_rad` cadence
is much longer than the timestep.

Mutates the solver and `p.radiation` in `integrator.p`, and returns `nothing`. Installed
by `radiation_callback` for RRTMGP radiation modes only.
"""
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

"""
    subcol_model_callback!(integrator)

Generate the COSP stochastic subcolumns and hand each to the simulator consumers.

Mutates the subcolumn working fields in `integrator.p.precomputed` and returns `nothing`.
Installed by `subcol_callback` on the `dt_subcol` cadence, and only when the model
configures COSP.
"""
NVTX.@annotate function subcol_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    foreach_cosp_subcolumn(consume_cosp_subcolumn!, Y, p)

    return nothing
end

"""
    consume_cosp_subcolumn!(isubcolumn, hydrometeors)

Consume one COSP hydrometeor subcolumn. Placeholder for a future simulator such as
CloudSat; currently discards its arguments and returns `nothing`.
"""
consume_cosp_subcolumn!(_, _) = nothing

"""
    prepare_cosp_subcolumns!(Y, p)

Set up the COSP subcolumn generator and accumulate the sampled cloud and precipitation
fractions.

Fixes the SCOPS overlap selectors once, so that the subcolumns are reproducible across the
two passes, computes the large-scale precipitation flux, then sweeps all subcolumns to
accumulate `ᶜsampled_cloud_fraction` and `ᶜsampled_precip_fraction`. Those sampled
fractions are what the second sweep in `foreach_prepared_cosp_subcolumn!` normalizes
against, which is why this pass must complete first.

Mutates the COSP fields in `p.precomputed` and returns `nothing`.
"""
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

"""
    set_cosp_large_scale_precipitation_flux!(Y, p, microphysics_model)

Compute the grid-mean precipitation mass flux `ρ q_rai w_rai + ρ q_sno w_sno` that COSP
distributes over subcolumns.

Clipped at zero, since the terminal-velocity convention can give a small negative flux.
Defined for the 1M and 2M microphysics models and errors otherwise. Mutates
`p.precomputed.ᶜlarge_scale_precipitation_flux` and returns `nothing`.
"""
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
    foreach_cosp_subcolumn(consume!, Y, p, microphysics_model)

Stream the COSP hydrometeor subcolumns one at a time to `consume!`.

Runs `prepare_cosp_subcolumns!` to fix the overlap selectors and accumulate the sampled
fractions, then regenerates the same subcolumns deterministically and calls
`consume!(isubcolumn, hydrometeors)` on each. Only the 1M and 2M microphysics models are
supported; anything else throws an `ArgumentError`.

Mutates the COSP working fields in `p.precomputed` and the grid-mean hydrometeor scratch
fields, and returns `nothing`.

!!! warning

    `consume!` must use the lazy hydrometeor broadcasts immediately. They borrow working
    mask and scratch fields that the next iteration overwrites.
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

"""
    foreach_prepared_cosp_subcolumn!(consume!, grid_mean_hydrometeors, p)

Regenerate each COSP subcolumn and pass its lazy hydrometeor profiles to `consume!`.

The second pass of `foreach_cosp_subcolumn`, valid only after `prepare_cosp_subcolumns!`
has fixed the SCOPS selectors and the sampled fractions. Reproduces exactly the same
subcolumns as that first pass, so the two are consistent.

Mutates the subcolumn mask and scratch fields in `p.precomputed` and returns `nothing`.
The hydrometeor broadcasts handed to `consume!` alias those fields and are invalidated by
the next iteration.
"""
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

@inline _cosp_nsubcolumns(::Val{N}) where {N} = N

"""
    nogw_model_callback!(integrator)

Recompute the non-orographic gravity-wave drag tendency.

Mutates the forcing fields in `integrator.p.non_orographic_gravity_wave` and returns
`nothing`; those fields are then held fixed and applied every step by the tendencies until
the next call. Installed by `nogw_callback` on the `dt_nogw` cadence.
"""
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

"""
    ogw_model_callback!(integrator)

Recompute the orographic gravity-wave drag tendency.

Mutates the forcing fields in `integrator.p.orographic_gravity_wave` and returns
`nothing`. Installed by `ogw_callback` on the `dt_ogw` cadence.
"""
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
"""
    set_insolation_variables!(Y, p, t, insolation)

Write the cosine of the solar zenith angle and the TOA flux into the RRTMGP solver.

Called from `rrtmgp_solver_callback!` before each radiation solve, so the insolation is
refreshed on the `dt_rad` cadence rather than every step. Mutates
`RRTMGP.cos_zenith(p.radiation.rrtmgp_solver)` and `RRTMGP.toa_sw_flux_dn(...)`; the return
value is unused.

Dispatches on `p.atmos.insolation`:

  - `RCEMIPIIInsolation`: uniform values prescribed by the RCEMIP-II protocol
    [Wing2018](@cite).
  - `IdealizedInsolation`: annual-mean insolation with no diurnal cycle, as a function of
    latitude only [OGorman2008](@cite).
  - `Larcform1Insolation`: perpetual polar night, with zero TOA flux.
  - `TimeVaryingInsolation`: the full orbital calculation, via `Insolation.insolation` at
    the current date. Uses the explicit `latitude`/`longitude` override when set, otherwise
    the column coordinates, falling back to the equator on a flat space.
  - `GCMDrivenInsolation` and `ExternalTVInsolation`: values read from the external forcing.
    The latter reconstructs the TOA flux as `rsdt / coszen`.

!!! note

    RRTMGP requires a strictly positive cosine zenith angle, so the modes that can reach
    zero clamp it to `eps(FT)` rather than letting it vanish.
"""
function set_insolation_variables!(Y, p, t, ::RCEMIPIIInsolation)
    FT = Spaces.undertype(axes(Y.c))
    (; rrtmgp_solver) = p.radiation
    RRTMGP.cos_zenith(rrtmgp_solver) .= cosd(FT(42.05))
    RRTMGP.toa_sw_flux_dn(rrtmgp_solver) .= FT(551.58)
end

function set_insolation_variables!(Y, p, t, ::GCMDrivenInsolation)
    (; rrtmgp_solver) = p.radiation
    RRTMGP.cos_zenith(rrtmgp_solver) .= Fields.field2array(p.external_forcing.cos_zenith)
    RRTMGP.toa_sw_flux_dn(rrtmgp_solver) .=
        Fields.field2array(p.external_forcing.toa_flux)
end

function set_insolation_variables!(Y, p, t, ::ExternalTVInsolation)
    # unpack objects with time varying data
    (; rrtmgp_solver) = p.radiation
    (; coszen, rsdt) = p.external_forcing.surface_fields
    coszen_tv = p.external_forcing.surface_timevaryinginputs.coszen
    rsdt_tv = p.external_forcing.surface_timevaryinginputs.rsdt
    # evaluate time varying data onto temporary fields
    evaluate!(coszen, coszen_tv, t)
    evaluate!(rsdt, rsdt_tv, t)

    # set insolation variables from the values within the fields
    RRTMGP.cos_zenith(rrtmgp_solver) .= Fields.field2array(coszen)
    RRTMGP.toa_sw_flux_dn(rrtmgp_solver) .= Fields.field2array(rsdt ./ coszen)
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
    RRTMGP.toa_sw_flux_dn(rrtmgp_solver) .= 680
    cos_zenith = RRTMGP.cos_zenith(rrtmgp_solver)
    @. cos_zenith =
        (1 + FT(0.3) * (1 - 3 * sind(latitude)^2)) * FT(0.5)
end

function set_insolation_variables!(Y, p, t, ::Larcform1Insolation)
    FT = Spaces.undertype(axes(Y.c))
    (; rrtmgp_solver) = p.radiation
    RRTMGP.cos_zenith(rrtmgp_solver) .= eps(FT) # polar night; keep μ>0 for RRTMGP
    RRTMGP.toa_sw_flux_dn(rrtmgp_solver) .= FT(0)
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
        RRTMGP.toa_sw_flux_dn(rrtmgp_solver),
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

"""
    save_state_to_disk_func(integrator, output_dir)

Write a checkpoint of the prognostic state to an HDF5 file.

The file is named `day\$day.\$sec.hdf5` from the elapsed simulation time, and the state is
written under the field name `"Y"`. Two attributes are attached: `"time"`, the simulation
time in seconds, and `"atmos_model_hash"`, a hash of `p.atmos` that a restart checks
against so a checkpoint is not silently loaded into a different model configuration.

Returns `nothing`. Installed by `checkpoint_callback` when `checkpoint_frequency` is
finite.
"""
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

"""
    gc_func(integrator)

Run an incremental garbage collection and log what it reclaimed.

Emits a `@debug` record with the allocation since the previous collection, the live bytes
before and after, the time spent, and the cumulative pause and full-sweep counts, all in
MB, seconds, and counts. Returns `nothing`.

Installed by `gc_callback`, which only adds it on distributed runs: uncoordinated
collections across ranks stall the whole job at the next communication, so GC is instead
forced on a fixed step cadence. See `gc_callback` for the cadence and how to change it.
"""
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
    maybe_graceful_exit(output_dir, integrator)

Return whether the user has asked the running simulation to stop.

Checked after every timestep. To request a stop, edit `{output_dir}/graceful_exit.dat` and
change its contents from `0` to `1`; the simulation then terminates through the
integrator, so the final state and diagnostics are still written. The file is created,
holding `0`, on the first call if it does not exist, and reset to `0` if its contents
cannot be parsed.

Used as the condition of the callback built by `graceful_exit_callback`.

!!! note

    This may not be reliable for MPI jobs, where ranks poll the file independently.
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
"""
    reset_graceful_exit(output_dir)

Write `0` to `{output_dir}/graceful_exit.dat`, creating the directory if needed.

Clears any stop request left over from a previous run, so that a restart writing into the
same output directory does not exit immediately. See `maybe_graceful_exit`.
"""
function reset_graceful_exit(output_dir)
    file = joinpath(output_dir, "graceful_exit.dat")
    ispath(output_dir) || mkpath(output_dir)
    open(io -> print(io, 0), file, "w")
end

"""
    check_nans(integrator)

Abort the simulation if any prognostic variable has gone to `NaN`.

On failure, walks the state and logs which `Y.<subfield>.<variable>` contain `NaN` and how
many elements of each, then errors. Naming the affected variables is the point of the
extra pass: it usually identifies the guilty tendency without a rerun.

Returns `nothing` when the state is clean. Installed by `nan_checking_callback`, whose
`check_nan_every` cadence trades detection latency against the cost of scanning the whole
state.
"""
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
