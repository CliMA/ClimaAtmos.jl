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
import DiffEqCallbacks as DECB
import ClimaCore: InputOutput
using Dates
using Insolation: instantaneous_zenith_angle
import ClimaCore.Fields: ColumnField

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
            Δt
        if p.atmos.surface_model isa PrescribedSurfaceTemperature
            net_energy_flux_sfc[] +=
                horizontal_integral_at_boundary(ᶠradiation_flux, half) * Δt
        end
    end
    return nothing
end

function cloud_fraction_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    set_cloud_fraction!(Y, p, p.atmos.moisture_model)
end

NVTX.@annotate function rrtmgp_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    t = integrator.t

    (; ᶜts, sfc_conditions) = p.precomputed
    (; params) = p
    (; idealized_insolation, idealized_h2o, idealized_clouds) = p.radiation
    (; insolation_tuple, ᶠradiation_flux, radiation_model) = p.radiation

    FT = Spaces.undertype(axes(Y.c))
    thermo_params = CAP.thermodynamics_params(params)
    insolation_params = CAP.insolation_params(params)

    sfc_ts = sfc_conditions.ts
    sfc_T =
        RRTMGPI.array2field(radiation_model.surface_temperature, axes(sfc_ts))
    @. sfc_T = TD.air_temperature(thermo_params, sfc_ts)

    ᶜp = RRTMGPI.array2field(radiation_model.center_pressure, axes(Y.c))
    ᶜT = RRTMGPI.array2field(radiation_model.center_temperature, axes(Y.c))
    @. ᶜp = TD.air_pressure(thermo_params, ᶜts)
    @. ᶜT = TD.air_temperature(thermo_params, ᶜts)

    if !(radiation_model.radiation_mode isa RRTMGPI.GrayRadiation)
        ᶜvmr_h2o = RRTMGPI.array2field(
            radiation_model.center_volume_mixing_ratio_h2o,
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
            @. ᶜvmr_h2o = TD.shum_to_mixing_ratio(ᶜq_tot, ᶜq_tot)
        else
            @. ᶜvmr_h2o = TD.vol_vapor_mixing_ratio(
                thermo_params,
                TD.PhasePartition(thermo_params, ᶜts),
            )
        end
    end

    if !idealized_insolation
        current_datetime = p.start_date + Dates.Second(round(Int, t)) # current time
        max_zenith_angle = FT(π) / 2 - eps(FT)
        irradiance = FT(CAP.tot_solar_irrad(params))
        au = FT(CAP.astro_unit(params))
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
        if eltype(bottom_coords) <: Geometry.LatLongZPoint
            solar_zenith_angle = RRTMGPI.array2field(
                radiation_model.solar_zenith_angle,
                axes(bottom_coords),
            )
            weighted_irradiance = RRTMGPI.array2field(
                radiation_model.weighted_irradiance,
                axes(bottom_coords),
            )
            @. insolation_tuple = instantaneous_zenith_angle(
                d,
                δ,
                η_UTC,
                bottom_coords.long,
                bottom_coords.lat,
            ) # the tuple is (zenith angle, azimuthal angle, earth-sun distance)
            @. solar_zenith_angle =
                min(first(insolation_tuple), max_zenith_angle)
            @. weighted_irradiance =
                irradiance * (au / last(insolation_tuple))^2
        else
            # assume that the latitude and longitude are both 0 for flat space
            insolation_tuple =
                instantaneous_zenith_angle(d, δ, η_UTC, FT(0), FT(0))
            radiation_model.solar_zenith_angle .=
                min(first(insolation_tuple), max_zenith_angle)
            radiation_model.weighted_irradiance .=
                irradiance * (au / last(insolation_tuple))^2
        end
    end

    if !idealized_clouds && !(
        radiation_model.radiation_mode isa RRTMGPI.GrayRadiation ||
        radiation_model.radiation_mode isa RRTMGPI.ClearSkyRadiation
    )
        ᶜΔz = Fields.local_geometry_field(Y.c).∂x∂ξ.components.data.:9
        ᶜlwp = RRTMGPI.array2field(
            radiation_model.center_cloud_liquid_water_path,
            axes(Y.c),
        )
        ᶜiwp = RRTMGPI.array2field(
            radiation_model.center_cloud_ice_water_path,
            axes(Y.c),
        )
        ᶜfrac = RRTMGPI.array2field(
            radiation_model.center_cloud_fraction,
            axes(Y.c),
        )
        # multiply by 1000 to convert from kg/m^2 to g/m^2
        @. ᶜlwp =
            1000 * Y.c.ρ * TD.liquid_specific_humidity(thermo_params, ᶜts) * ᶜΔz
        @. ᶜiwp =
            1000 * Y.c.ρ * TD.ice_specific_humidity(thermo_params, ᶜts) * ᶜΔz
        @. ᶜfrac = ifelse(TD.has_condensate(thermo_params, ᶜts), 1, 0 * ᶜΔz)
    end

    RRTMGPI.update_fluxes!(radiation_model)
    RRTMGPI.field2array(ᶠradiation_flux) .= radiation_model.face_flux
    return nothing
end

function common_diagnostics(p, ᶜu, ᶜts)
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    ᶜρ = TD.air_density.(thermo_params, ᶜts)
    diagnostics = (;
        u_velocity = Geometry.UVector.(ᶜu),
        v_velocity = Geometry.VVector.(ᶜu),
        w_velocity = Geometry.WVector.(ᶜu),
        temperature = TD.air_temperature.(thermo_params, ᶜts),
        potential_temperature = TD.dry_pottemp.(thermo_params, ᶜts),
        specific_enthalpy = TD.specific_enthalpy.(thermo_params, ᶜts),
        buoyancy = CAP.grav(p.params) .* (p.core.ᶜρ_ref .- ᶜρ) ./ ᶜρ,
        density = TD.air_density.(thermo_params, ᶜts),
        cloud_fraction_gm = p.precomputed.ᶜcloud_fraction,
    )
    if !(p.atmos.moisture_model isa DryModel)
        diagnostics = (;
            diagnostics...,
            q_vap = TD.vapor_specific_humidity.(thermo_params, ᶜts),
            q_liq = TD.liquid_specific_humidity.(thermo_params, ᶜts),
            q_ice = TD.ice_specific_humidity.(thermo_params, ᶜts),
            q_tot = TD.total_specific_humidity.(thermo_params, ᶜts),
            relative_humidity = TD.relative_humidity.(thermo_params, ᶜts),
        )
    end
    return diagnostics
end

# Adds a prefix to the front of each name in the named tuple. This function
# is not type stable, but that probably doesn't matter for diagnostics.
add_prefix(diagnostics::NamedTuple{names}, prefix) where {names} =
    NamedTuple{Symbol.(prefix, names)}(values(diagnostics))

# TODO - temporary crutch untill we move this to the new diagnostics
# and start using quadratures in EDMFX
draft_cloud_fraction(thermo_params, ts, area::FT) where {FT} =
    TD.has_condensate(thermo_params, ts) && area > 1e-3 ? FT(1) : FT(0)

NVTX.@annotate function compute_diagnostics(integrator)
    (; t, u, p) = integrator
    Y = u
    (; params) = p
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)

    (; ᶜu, ᶜK, ᶜts, ᶜp, sfc_conditions) = p.precomputed
    dycore_diagnostic = (;
        common_diagnostics(p, ᶜu, ᶜts)...,
        pressure = ᶜp,
        kinetic_energy = ᶜK,
        sfc_temperature = TD.air_temperature.(thermo_params, sfc_conditions.ts),
        sfc_qt = TD.total_specific_humidity.(thermo_params, sfc_conditions.ts),
    )

    if eltype(Fields.coordinate_field(axes(Y.c))) <: Geometry.Abstract3DPoint
        ᶜvort = @. Geometry.WVector(curlₕ(Y.c.uₕ))
        if p.do_dss
            Spaces.weighted_dss!(ᶜvort)
        end
        dycore_diagnostic = (; dycore_diagnostic..., vorticity = ᶜvort)
    end

    if p.atmos.precip_model isa Microphysics0Moment
        (; ᶜS_ρq_tot, col_integrated_rain, col_integrated_snow) =
            p.precipitation
        Fields.bycolumn(axes(Y.c)) do colidx
            precipitation_tendency!(
                nothing,
                Y,
                p,
                t,
                colidx,
                p.atmos.precip_model,
            )
        end # TODO: Set the diagnostics without computing the tendency.
        precip_diagnostic = (;
            precipitation_removal = ᶜS_ρq_tot,
            column_integrated_rain = col_integrated_rain,
            column_integrated_snow = col_integrated_snow,
        )
    elseif p.atmos.precip_model isa Microphysics1Moment
        # TODO: Get column integrals for the land model.
        precip_diagnostic =
            (; q_rai = Y.c.ρq_rai ./ Y.c.ρ, q_sno = Y.c.ρq_sno ./ Y.c.ρ)
    else
        precip_diagnostic = NamedTuple()
    end

    if p.atmos.turbconv_model isa PrognosticEDMFX
        (; ᶜtke⁰, ᶜu⁰, ᶜts⁰, ᶜmixing_length) = p.precomputed
        (; ᶜu⁺, ᶜts⁺, ᶜa⁺, ᶜa⁰) = output_prognostic_sgs_quantities(Y, p, t)
        env_diagnostics = (;
            common_diagnostics(p, ᶜu⁰, ᶜts⁰)...,
            area = ᶜa⁰,
            cloud_fraction = draft_cloud_fraction.(thermo_params, ᶜts⁰, ᶜa⁰),
            tke = ᶜtke⁰,
            mixing_length = ᶜmixing_length,
        )
        draft_diagnostics = (;
            common_diagnostics(p, ᶜu⁺, ᶜts⁺)...,
            area = ᶜa⁺,
            cloud_fraction = draft_cloud_fraction.(thermo_params, ᶜts⁺, ᶜa⁺),
        )
        turbulence_convection_diagnostic = (;
            add_prefix(env_diagnostics, :env_)...,
            add_prefix(draft_diagnostics, :draft_)...,
            cloud_fraction = ᶜa⁰ .*
                             draft_cloud_fraction.(thermo_params, ᶜts⁰, ᶜa⁰) .+ ᶜa⁺ .*
                             draft_cloud_fraction.(thermo_params, ᶜts⁺, ᶜa⁺),
        )
    elseif p.atmos.turbconv_model isa DiagnosticEDMFX
        (; ᶜtke⁰, ᶜmixing_length) = p.precomputed
        (; ᶜu⁺, ᶜts⁺, ᶜa⁺) = output_diagnostic_sgs_quantities(Y, p, t)
        env_diagnostics = (;
            cloud_fraction = draft_cloud_fraction.(thermo_params, ᶜts, FT(1)),
            tke = ᶜtke⁰,
            mixing_length = ᶜmixing_length,
        )
        draft_diagnostics = (;
            common_diagnostics(p, ᶜu⁺, ᶜts⁺)...,
            area = ᶜa⁺,
            cloud_fraction = draft_cloud_fraction.(thermo_params, ᶜts⁺, ᶜa⁺),
        )
        turbulence_convection_diagnostic = (;
            add_prefix(env_diagnostics, :env_)...,
            add_prefix(draft_diagnostics, :draft_)...,
            cloud_fraction = draft_cloud_fraction.(thermo_params, ᶜts, FT(1)) .+ ᶜa⁺ .*
                             draft_cloud_fraction.(thermo_params, ᶜts⁺, ᶜa⁺),
        )
    else
        turbulence_convection_diagnostic = NamedTuple()
    end

    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(Y.f), Fields.half)
    surface_ct3_unit = CT3.(unit_basis_vector_data.(CT3, sfc_local_geometry))
    (; ρ_flux_uₕ, ρ_flux_h_tot) = p.precomputed.sfc_conditions
    sfc_flux_momentum =
        Geometry.UVVector.(
            adjoint.(ρ_flux_uₕ ./ Spaces.level(ᶠinterp.(Y.c.ρ), half)) .*
            surface_ct3_unit
        )
    vert_diff_diagnostic = (;
        sfc_flux_u = sfc_flux_momentum.components.data.:1,
        sfc_flux_v = sfc_flux_momentum.components.data.:2,
        sfc_flux_energy = dot.(ρ_flux_h_tot, surface_ct3_unit),
    )
    if :ρq_tot in propertynames(Y.c)
        (; ρ_flux_q_tot) = p.precomputed.sfc_conditions
        vert_diff_diagnostic = (;
            vert_diff_diagnostic...,
            sfc_evaporation = dot.(ρ_flux_q_tot, surface_ct3_unit),
        )
    end

    if p.atmos.radiation_mode isa RRTMGPI.AbstractRRTMGPMode
        (; face_lw_flux_dn, face_lw_flux_up, face_sw_flux_dn, face_sw_flux_up) =
            p.radiation.radiation_model
        rad_diagnostic = (;
            lw_flux_down = RRTMGPI.array2field(FT.(face_lw_flux_dn), axes(Y.f)),
            lw_flux_up = RRTMGPI.array2field(FT.(face_lw_flux_up), axes(Y.f)),
            sw_flux_down = RRTMGPI.array2field(FT.(face_sw_flux_dn), axes(Y.f)),
            sw_flux_up = RRTMGPI.array2field(FT.(face_sw_flux_up), axes(Y.f)),
        )
        if p.atmos.radiation_mode isa
           RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics
            (;
                face_clear_lw_flux_dn,
                face_clear_lw_flux_up,
                face_clear_sw_flux_dn,
                face_clear_sw_flux_up,
            ) = p.radiation.radiation_model
            rad_clear_diagnostic = (;
                clear_lw_flux_down = RRTMGPI.array2field(
                    FT.(face_clear_lw_flux_dn),
                    axes(Y.f),
                ),
                clear_lw_flux_up = RRTMGPI.array2field(
                    FT.(face_clear_lw_flux_up),
                    axes(Y.f),
                ),
                clear_sw_flux_down = RRTMGPI.array2field(
                    FT.(face_clear_sw_flux_dn),
                    axes(Y.f),
                ),
                clear_sw_flux_up = RRTMGPI.array2field(
                    FT.(face_clear_sw_flux_up),
                    axes(Y.f),
                ),
            )
        else
            rad_clear_diagnostic = NamedTuple()
        end
    elseif p.atmos.radiation_mode isa RadiationDYCOMS_RF01
        # TODO: add radiation diagnostics
        rad_diagnostic = NamedTuple()
        rad_clear_diagnostic = NamedTuple()
    elseif p.atmos.radiation_mode isa RadiationTRMM_LBA
        # TODO: add radiation diagnostics
        rad_diagnostic = NamedTuple()
        rad_clear_diagnostic = NamedTuple()
    else
        rad_diagnostic = NamedTuple()
        rad_clear_diagnostic = NamedTuple()
    end

    diagnostic = merge(
        dycore_diagnostic,
        precip_diagnostic,
        vert_diff_diagnostic,
        rad_diagnostic,
        rad_clear_diagnostic,
        turbulence_convection_diagnostic,
    )
    return diagnostic
end

function save_to_disk_func(integrator, output_dir)
    (; t, u, p) = integrator
    Y = u
    diagnostic = compute_diagnostics(integrator)

    day = floor(Int, t / (60 * 60 * 24))
    sec = floor(Int, t % (60 * 60 * 24))
    @info "Saving diagnostics to HDF5 file on day $day second $sec"
    output_file = joinpath(output_dir, "day$day.$sec.hdf5")
    comms_ctx = ClimaComms.context(integrator.u.c)
    hdfwriter = InputOutput.HDF5Writer(output_file, comms_ctx)
    InputOutput.HDF5.write_attribute(hdfwriter.file, "time", t) # TODO: a better way to write metadata
    InputOutput.write!(hdfwriter, Y, "Y")
    FT = Spaces.undertype(axes(Y.c))
    values = map(Fields.wrap, diagnostic)
    fv = Fields.FieldVector{FT}(values)
    InputOutput.write!(hdfwriter, fv, "diagnostics")
    Base.close(hdfwriter)
    return nothing
end

function save_restart_func(integrator, output_dir)
    (; t, u, p) = integrator

    Y = u
    day = floor(Int, t / (60 * 60 * 24))
    sec = floor(Int, t % (60 * 60 * 24))
    @info "Saving restart file to HDF5 file on day $day second $sec"
    mkpath(joinpath(output_dir, "restart"))
    output_file = joinpath(output_dir, "restart", "day$day.$sec.hdf5")
    comms_ctx = ClimaComms.context(integrator.u.c)
    hdfwriter = InputOutput.HDF5Writer(output_file, comms_ctx)
    InputOutput.HDF5.write_attribute(hdfwriter.file, "time", t) # TODO: a better way to write metadata
    InputOutput.write!(hdfwriter, Y, "Y")
    Base.close(hdfwriter)
    return nothing
end

Base.@kwdef mutable struct WallTimeEstimate
    """Number of calls to the callback"""
    n_calls::Int = 0
    """Int indicating next time the callback will print to the log"""
    n_next::Int = 1
    """Wall time of previous call to update `WallTimeEstimate`"""
    t_wall_last::Float64 = -1
    """Sum of elapsed walltime over calls to `step!`"""
    ∑Δt_wall::Float64 = 0
    """Fixed increment to increase n_next by after 5% completion"""
    n_fixed_increment::Float64 = -1
end
import Dates
function print_walltime_estimate(integrator)
    (; walltime_estimate, dt, t_end) = integrator.p
    wte = walltime_estimate

    # Notes on `ready_to_report`
    #   - The very first call (when `n_calls == 0`), there's no elapsed
    #     times to report (and this is called during initialization,
    #     before `step!` has been called).
    #   - The second call (`n_calls == 1`) is after `step!` is called
    #     for the first time, but we don't want to report this since it
    #     includes compilation time.
    #   - Calls after that (`n_calls > 1`) exclude compilation and provide
    #     the best wall time estimates

    ready_to_report = wte.n_calls > 1
    if ready_to_report
        # We need to account for skipping cost of `Δt_wall` when `n_calls == 1`:
        factor = wte.n_calls == 2 ? 2 : 1
        Δt_wall = factor * (time() - wte.t_wall_last)
    else
        wte.n_calls == 1 && @info "Progress: Completed first step"
        Δt_wall = Float64(0)
        wte.n_next = wte.n_calls + 1
    end
    wte.∑Δt_wall += Δt_wall
    wte.t_wall_last = time()

    if wte.n_calls == wte.n_next && ready_to_report
        t = integrator.t
        n_steps_total = ceil(Int, t_end / dt)
        n_steps = ceil(Int, t / dt)
        wall_time_ave_per_step = wte.∑Δt_wall / n_steps
        wall_time_ave_per_step_str = time_and_units_str(wall_time_ave_per_step)
        percent_complete = round(t / t_end * 100; digits = 1)
        n_steps_remaining = n_steps_total - n_steps
        wall_time_remaining = wall_time_ave_per_step * n_steps_remaining
        wall_time_remaining_str = time_and_units_str(wall_time_remaining)
        wall_time_total =
            time_and_units_str(wall_time_ave_per_step * n_steps_total)
        wall_time_spent = time_and_units_str(wte.∑Δt_wall)
        simulation_time = time_and_units_str(Float64(t))
        sypd = round(
            simulated_years_per_day(
                EfficiencyStats((zero(t), t), wte.∑Δt_wall),
            );
            digits = 3,
        )
        estimated_finish_date =
            Dates.now() + compound_period(wall_time_remaining, Dates.Second)
        @info "Progress" simulation_time = simulation_time n_steps_completed =
            n_steps wall_time_per_step = wall_time_ave_per_step_str wall_time_total =
            wall_time_total wall_time_remaining = wall_time_remaining_str wall_time_spent =
            wall_time_spent percent_complete = "$percent_complete%" sypd = sypd date_now =
            Dates.now() estimated_finish_date = estimated_finish_date

        # the first fixed increment is equivalent to
        # doubling (which puts us at 10%), so we check
        # if we're below 5%.
        if percent_complete < 5
            # doubling factor (to reduce log noise)
            wte.n_next *= 2
        else
            if wte.n_fixed_increment == -1
                wte.n_fixed_increment = wte.n_next
            end
            # increase by fixed increment after 10%
            # completion to maintain logs after 50%.
            wte.n_next += wte.n_fixed_increment
        end
    end
    wte.n_calls += 1

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
