import ClimaCore.DataLayouts as DL
import .RRTMGPInterface as RRTMGPI
import Thermodynamics as TD
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
            Δt
        if p.atmos.surface_model isa PrescribedSurfaceTemperature
            net_energy_flux_sfc[] +=
                horizontal_integral_at_boundary(ᶠradiation_flux, half) * Δt
        end
    end
    return nothing
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

# TODO: Move this somewhere else
update_o3!(_, _, _) = nothing
function update_o3!(p, t, ::PrescribedOzone)
    evaluate!(p.tracers.o3, p.tracers.prescribed_o3_timevaryinginput, t)
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
    if :prescribed_aerosols_field in propertynames(p.tracers)
        for (key, tv) in pairs(p.tracers.prescribed_aerosol_timevaryinginputs)
            field = getproperty(p.tracers.prescribed_aerosols_field, key)
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
    # When add_isothermal_boundary_layer is true, we add a layer in rrtmgp and set the
    # pressure to half of the sum of top level pressure and rrtmgp minimum pressure. 
    # Here we limit the pressure to p_min multiplied by a small number to prevent 
    # pressure from decreasing or being constant with height.
    p_min = RRTMGPI.get_p_min(rrtmgp_model)
    @. ᶜp = max(TD.air_pressure(thermo_params, ᶜts), p_min * FT(1.01))
    # TODO: move this to RRTMGP
    @. ᶜT =
        min(max(TD.air_temperature(thermo_params, ᶜts), FT(T_min)), FT(T_max))

    if !(radiation_mode isa RRTMGPI.GrayRadiation)
        ᶜrh =
            Fields.array2field(rrtmgp_model.center_relative_humidity, axes(Y.c))
        ᶜvmr_h2o = Fields.array2field(
            rrtmgp_model.center_volume_mixing_ratio_h2o,
            axes(Y.c),
        )
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
            ᶜlwp = Fields.array2field(
                rrtmgp_model.center_cloud_liquid_water_path,
                axes(Y.c),
            )
            ᶜiwp = Fields.array2field(
                rrtmgp_model.center_cloud_ice_water_path,
                axes(Y.c),
            )
            ᶜfrac = Fields.array2field(
                rrtmgp_model.center_cloud_fraction,
                axes(Y.c),
            )
            # RRTMGP needs lwp and iwp in g/m^2
            kg_to_g_factor = 1000
            @. ᶜlwp =
                kg_to_g_factor * Y.c.ρ * cloud_diagnostics_tuple.q_liq * ᶜΔz /
                max(cloud_diagnostics_tuple.cf, eps(FT))
            @. ᶜiwp =
                kg_to_g_factor * Y.c.ρ * cloud_diagnostics_tuple.q_ice * ᶜΔz /
                max(cloud_diagnostics_tuple.cf, eps(FT))
            @. ᶜfrac = cloud_diagnostics_tuple.cf
        end
    end

    if !(radiation_mode isa RRTMGPI.GrayRadiation)
        if radiation_mode.aerosol_radiation
            ᶜΔz = Fields.Δz_field(Y.c)

            ᶜaero_conc = Fields.array2field(
                rrtmgp_model.center_dust_column_mass_density,
                axes(Y.c),
            )
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

            ᶜaero_conc = Fields.array2field(
                rrtmgp_model.center_ss_column_mass_density,
                axes(Y.c),
            )
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

            for (rrtmgp_aerosol_name, prescribed_aerosol_name) in
                aerosol_names_pair
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
            ᶜvmr_o3 = Fields.array2field(
                rrtmgp_model.center_volume_mixing_ratio_o3,
                axes(Y.c),
            )
            @. ᶜvmr_o3 = p.tracers.o3
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

function set_insolation_variables!(Y, p, t, ::GCMDrivenInsolation)
    (; rrtmgp_model) = p.radiation
    rrtmgp_model.cos_zenith .= Fields.field2array(p.external_forcing.cos_zenith)
    rrtmgp_model.weighted_irradiance .=
        Fields.field2array(p.external_forcing.insolation)
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

function set_insolation_variables!(Y, p, t, ::TimeVaryingInsolation)
    FT = Spaces.undertype(axes(Y.c))
    params = p.params
    insolation_params = CAP.insolation_params(params)
    (; insolation_tuple, rrtmgp_model) = p.radiation

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
    t_start = integrator.sol.prob.tspan[1]
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
        n_steps_total = ceil(Int, (t_end - t_start) / dt)
        n_steps = ceil(Int, (t - t_start) / dt)
        wall_time_ave_per_step = wte.∑Δt_wall / n_steps
        wall_time_ave_per_step_str = time_and_units_str(wall_time_ave_per_step)
        percent_complete = round((t - t_start) / t_end * 100; digits = 1)
        n_steps_remaining = n_steps_total - n_steps
        wall_time_remaining = wall_time_ave_per_step * n_steps_remaining
        wall_time_remaining_str = time_and_units_str(wall_time_remaining)
        wall_time_total =
            time_and_units_str(wall_time_ave_per_step * n_steps_total)
        wall_time_spent = time_and_units_str(wte.∑Δt_wall)
        simulation_time = time_and_units_str(Float64(t))
        es = EfficiencyStats((t_start, t), wte.∑Δt_wall)
        _sypd = simulated_years_per_day(es)
        _sypd_str = string(round(_sypd; digits = 3))
        sypd = _sypd_str * if _sypd < 0.01
            sdpd = round(_sypd * 365, digits = 3)
            " (sdpd = $sdpd)"
        else
            ""
        end
        estimated_finish_date =
            Dates.now() + compound_period(wall_time_remaining)
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

# TODO: Move this function to ClimaTimeSteppers.jl
get_dtγ(integrator) =
    if (
        integrator.alg isa CTS.RosenbrockAlgorithm ||
        integrator.alg.name isa CTS.IMEXARKAlgorithmName
    )
        tableau_coefficients =
            integrator.alg isa CTS.RosenbrockAlgorithm ?
            integrator.alg.tableau.Γ : integrator.alg.tableau.a_imp
        γs = unique(filter(!iszero, LinearAlgebra.diag(tableau_coefficients)))
        length(γs) == 1 || error(
            "The exact Jacobian must be updated on every Newton \
             iteration, rather than on every timestep (or every N steps), \
             because the specified IMEX algorithm has implicit stages \
             with distinct tableau coefficients (i.e., it is not an \
             SDIRK algorithm).",
        )
        FT = eltype(integrator.u)
        FT(integrator.dt * γs[1])
    else
        nothing
    end

function update_exact_jacobian!(integrator)
    dtγ = get_dtγ(integrator)
    isnothing(dtγ) && return
    (; u, p, t) = integrator
    update_exact_jacobian!(p.jacobian, u, p, dtγ, t)
end

function update_jacobian_init!(integrator)
    dtγ = get_dtγ(integrator)
    isnothing(dtγ) && return
    (; u, p, t) = integrator
    update_jacobian_init!(p.jacobian, u, p, dtγ, t)
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
function maybe_graceful_exit(integrator)
    output_dir = integrator.p.output_dir
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
