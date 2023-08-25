import ClimaCore.DataLayouts as DL
import ClimaAtmos.RRTMGPInterface as RRTMGPI
import Thermodynamics as TD
import LinearAlgebra
import ClimaCore.Fields
import ClimaComms
import ClimaCore as CC
import ClimaCore.Spaces
import OrdinaryDiffEq as ODE
import ClimaAtmos.Parameters as CAP
import DiffEqCallbacks as DEQ
import ClimaCore: InputOutput
import Dates
using Insolation: instantaneous_zenith_angle
import ClimaCore.Fields: ColumnField

include("callback_helpers.jl")

"""
    display_status_callback!()

Returns a callback to display simulation information.
Adapted from ClimaTimeSteppers.jl #89.
"""
function display_status_callback!()
    prev_t = 0.0
    t = 0.0
    t_end = 0.0

    start_time = 0.0
    prev_time = 0.0
    time = 0.0
    eta = 0.0
    speed = 0.0
    is_first_step = true

    step = 0


    function initialize(_, _, _, integrator)
    end

    function affect!(integrator)
        t_end = integrator.p.simulation.t_end
        nsteps = ceil(Int, t_end / integrator.dt)
        # speed = wallclock time / simulation time
        # Print ETA = speed * remaining simulation time
        t = integrator.t
        time = time_ns() / 1e9
        speed = (time - prev_time) / (t - prev_t)
        eta = speed * (t_end - t)
        step += 1
        if is_first_step
            # @info "Time Remaining: ..."
            is_first_step = false
            start_time = time
        else
            @info "$(Dates.format(Dates.now(), "HH:MM:SS:ss u-d")) \n\
            Timestep: $(step) / $(nsteps); Simulation Time: $(t) seconds \n\
            Walltime: $(round(time - start_time, digits=2)) seconds; \
            Time/Step: $(round(speed * integrator.dt, digits=2)) seconds"
            # Time Remaining: $(Int64(round(eta))) seconds"
        end
        prev_t = t
        prev_time = time
    end
    return initialize, affect!
end

function dss_callback!(integrator)
    Y = integrator.u
    ghost_buffer = integrator.p.ghost_buffer
    if integrator.p.do_dss
        NVTX.@range "dss callback" color = colorant"yellow" begin
            Spaces.weighted_dss_start2!(Y.c, ghost_buffer.c)
            Spaces.weighted_dss_start2!(Y.f, ghost_buffer.f)
            Spaces.weighted_dss_internal2!(Y.c, ghost_buffer.c)
            Spaces.weighted_dss_internal2!(Y.f, ghost_buffer.f)
            Spaces.weighted_dss_ghost2!(Y.c, ghost_buffer.c)
            Spaces.weighted_dss_ghost2!(Y.f, ghost_buffer.f)
        end
    end
    return nothing
end

horizontal_integral_at_boundary(f, lev) = sum(
    Spaces.level(f, lev) ./ Fields.dz_field(axes(Spaces.level(f, lev))) .* 2,
)

function flux_accumulation!(integrator)
    Y = integrator.u
    p = integrator.p
    if !isnothing(p.radiation_model)
        (; ᶠradiation_flux, net_energy_flux_toa, net_energy_flux_sfc, Δt) = p
        nlevels = Spaces.nlevels(axes(Y.c))
        net_energy_flux_toa[] +=
            horizontal_integral_at_boundary(ᶠradiation_flux, nlevels + half) *
            Δt
        net_energy_flux_sfc[] +=
            horizontal_integral_at_boundary(ᶠradiation_flux, half) * Δt
    end
    return nothing
end

function turb_conv_affect_filter!(integrator)
    p = integrator.p
    (; edmf_cache) = p
    (; edmf, param_set) = edmf_cache
    t = integrator.t
    Y = integrator.u
    tc_params = CAP.turbconv_params(param_set)

    set_precomputed_quantities!(Y, p, t) # sets ᶜts for set_edmf_surface_bc
    Fields.bycolumn(axes(Y.c)) do colidx
        state = TC.tc_column_state(Y, p, nothing, colidx, t)
        grid = TC.Grid(state)
        TC.affect_filter!(edmf, grid, state, tc_params, t)
    end

    # We're lying to OrdinaryDiffEq.jl, in order to avoid
    # paying for an additional `∑tendencies!` call, which is required
    # to support supplying a continuous representation of the
    # solution.
    ODE.u_modified!(integrator, false)
end

function rrtmgp_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    t = integrator.t

    set_precomputed_quantities!(Y, p, t) # sets ᶜts and sfc_conditions

    (; ᶜts, sfc_conditions, params, env_thermo_quad) = p
    (; idealized_insolation, idealized_h2o, idealized_clouds) = p
    (; insolation_tuple, ᶠradiation_flux, radiation_model) = p

    FT = eltype(params)
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
        current_datetime = p.simulation.start_date + Dates.Second(round(Int, t)) # current time
        max_zenith_angle = FT(π) / 2 - eps(FT)
        irradiance = FT(CAP.tot_solar_irrad(params))
        au = FT(CAP.astro_unit(params))
        (; orbital_data) = p

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
            ref_insolation_params = tuple(insolation_params)
            @. insolation_tuple = instantaneous_zenith_angle(
                current_datetime,
                orbital_data,
                Float64(bottom_coords.long),
                Float64(bottom_coords.lat),
                ref_insolation_params,
            ) # the tuple is (zenith angle, azimuthal angle, earth-sun distance)
            @. solar_zenith_angle =
                min(first(insolation_tuple), max_zenith_angle)
            @. weighted_irradiance =
                irradiance * (au / last(insolation_tuple))^2
        else
            # assume that the latitude and longitude are both 0 for flat space
            insolation_tuple = instantaneous_zenith_angle(
                current_datetime,
                orbital_data,
                0.0,
                0.0,
                insolation_params,
            )
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
        @. ᶜfrac =
            get_cloud_fraction(thermo_params, env_thermo_quad, FT(ᶜp), ᶜts)
    end

    RRTMGPI.update_fluxes!(radiation_model)
    RRTMGPI.field2array(ᶠradiation_flux) .= radiation_model.face_flux
    return nothing
end

function compute_diagnostics(integrator)
    (; t, u, p) = integrator
    Y = u
    (; params, env_thermo_quad) = p
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)

    function common_diagnostics(ᶜu, ᶜts)
        ᶜρ = TD.air_density.(thermo_params, ᶜts)
        diagnostics = (;
            u_velocity = Geometry.UVector.(ᶜu),
            v_velocity = Geometry.VVector.(ᶜu),
            w_velocity = Geometry.WVector.(ᶜu),
            temperature = TD.air_temperature.(thermo_params, ᶜts),
            potential_temperature = TD.dry_pottemp.(thermo_params, ᶜts),
            specific_enthalpy = TD.specific_enthalpy.(thermo_params, ᶜts),
            buoyancy = CAP.grav(params) .* (p.ᶜρ_ref .- ᶜρ) ./ ᶜρ,
        )
        if !(p.atmos.moisture_model isa DryModel)
            diagnostics = (;
                diagnostics...,
                q_vap = TD.vapor_specific_humidity.(thermo_params, ᶜts),
                q_liq = TD.liquid_specific_humidity.(thermo_params, ᶜts),
                q_ice = TD.ice_specific_humidity.(thermo_params, ᶜts),
                q_tot = TD.total_specific_humidity.(thermo_params, ᶜts),
                relative_humidity = TD.relative_humidity.(thermo_params, ᶜts),
                cloud_fraction_gm = get_cloud_fraction.(
                    thermo_params,
                    env_thermo_quad,
                    ᶜp,
                    ᶜts,
                ),
            )
        end
        return diagnostics
    end

    set_precomputed_quantities!(Y, p, t) # sets ᶜu, ᶜK, ᶜts, ᶜp, & SGS analogues

    (; ᶜu, ᶜK, ᶜts, ᶜp, sfc_conditions) = p
    dycore_diagnostic = (;
        common_diagnostics(ᶜu, ᶜts)...,
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
        (; ᶜS_ρq_tot, col_integrated_rain, col_integrated_snow) = p
        Fields.bycolumn(axes(Y.c)) do colidx
            precipitation_tendency!(p.Yₜ, Y, p, t, colidx, p.precip_model)
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

    # Adds a prefix to the front of each name in the named tuple. This function
    # is not type stable, but that probably doesn't matter for diagnostics.
    add_prefix(diagnostics::NamedTuple{names}, prefix) where {names} =
        NamedTuple{Symbol.(prefix, names)}(values(diagnostics))

    cloud_fraction(ts, area::FT) where {FT} =
        TD.has_condensate(thermo_params, ts) && area > 1e-3 ? FT(1) : FT(0)

    if p.atmos.turbconv_model isa EDMFX
        (; ᶜspecific⁰, ᶜu⁰, ᶜts⁰, ᶜmixing_length) = p
        (; ᶜu⁺, ᶜts⁺, ᶜa⁺, ᶜa⁰) = output_sgs_quantities(Y, p, t)
        env_diagnostics = (;
            common_diagnostics(ᶜu⁰, ᶜts⁰)...,
            area = ᶜa⁰,
            cloud_fraction = get_cloud_fraction.(
                thermo_params,
                env_thermo_quad,
                ᶜp,
                ᶜts⁰,
            ),
            tke = ᶜspecific⁰.tke,
            mixing_length = ᶜmixing_length,
        )
        draft_diagnostics = (;
            common_diagnostics(ᶜu⁺, ᶜts⁺)...,
            area = ᶜa⁺,
            cloud_fraction = cloud_fraction.(ᶜts⁺, ᶜa⁺),
        )
        turbulence_convection_diagnostic = (;
            add_prefix(env_diagnostics, :env_)...,
            add_prefix(draft_diagnostics, :draft_)...,
            cloud_fraction = ᶜa⁰ .*
                             get_cloud_fraction.(
                thermo_params,
                env_thermo_quad,
                ᶜp,
                ᶜts⁰,
            ) .+ ᶜa⁺ .* cloud_fraction.(ᶜts⁺, ᶜa⁺),
        )
    elseif p.atmos.turbconv_model isa DiagnosticEDMFX
        (; ᶜtke⁰, ᶜmixing_length) = p
        (; ᶜu⁺, ᶜts⁺, ᶜa⁺) = output_diagnostic_sgs_quantities(Y, p, t)
        env_diagnostics = (;
            cloud_fraction = get_cloud_fraction.(
                thermo_params,
                env_thermo_quad,
                ᶜp,
                ᶜts,
            ),
            tke = ᶜtke⁰,
            mixing_length = ᶜmixing_length,
        )
        draft_diagnostics = (;
            common_diagnostics(ᶜu⁺, ᶜts⁺)...,
            area = ᶜa⁺,
            cloud_fraction = cloud_fraction.(ᶜts⁺, ᶜa⁺),
        )
        turbulence_convection_diagnostic = (;
            add_prefix(env_diagnostics, :env_)...,
            add_prefix(draft_diagnostics, :draft_)...,
            cloud_fraction = get_cloud_fraction.(
                thermo_params,
                env_thermo_quad,
                ᶜp,
                ᶜts,
            ) .+ ᶜa⁺ .* cloud_fraction.(ᶜts⁺, ᶜa⁺),
        )
    elseif p.atmos.turbconv_model isa TC.EDMFModel
        tc_cent(p) = p.edmf_cache.aux.cent.turbconv
        tc_face(p) = p.edmf_cache.aux.face.turbconv
        turbulence_convection_diagnostic = (;
            bulk_up_area = tc_cent(p).bulk.area,
            bulk_up_h_tot = tc_cent(p).bulk.h_tot,
            bulk_up_q_tot = tc_cent(p).bulk.q_tot,
            bulk_up_q_liq = tc_cent(p).bulk.q_liq,
            bulk_up_q_ice = tc_cent(p).bulk.q_ice,
            bulk_up_temperature = tc_cent(p).bulk.T,
            bulk_up_cloud_fraction = tc_cent(p).bulk.cloud_fraction,
            bulk_up_e_tot_tendency_precip_formation = tc_cent(
                p,
            ).bulk.e_tot_tendency_precip_formation,
            bulk_up_qt_tendency_precip_formation = tc_cent(
                p,
            ).bulk.qt_tendency_precip_formation,
            env_area = tc_cent(p).en.area,
            env_q_tot = tc_cent(p).en.q_tot,
            env_q_liq = tc_cent(p).en.q_liq,
            env_q_ice = tc_cent(p).en.q_ice,
            env_theta_liq_ice = tc_cent(p).en.θ_liq_ice,
            env_theta_virt = tc_cent(p).en.θ_virt,
            env_theta_dry = tc_cent(p).en.θ_dry,
            env_e_tot = tc_cent(p).en.e_tot,
            env_e_kin = tc_cent(p).en.e_kin,
            env_h_tot = tc_cent(p).en.h_tot,
            env_RH = tc_cent(p).en.RH,
            env_temperature = tc_cent(p).en.T,
            env_cloud_fraction = tc_cent(p).en.cloud_fraction,
            env_TKE = tc_cent(p).en.tke,
            env_e_tot_tendency_precip_formation = tc_cent(
                p,
            ).en.e_tot_tendency_precip_formation,
            env_qt_tendency_precip_formation = tc_cent(
                p,
            ).en.qt_tendency_precip_formation,
            face_env_buoyancy = tc_face(p).en.buoy,
            face_up1_buoyancy = tc_face(p).bulk.buoy_up1,
            face_bulk_w = tc_face(p).bulk.w,
            face_env_w = tc_face(p).en.w,
            bulk_up_filter_flag_1 = tc_cent(p).bulk.filter_flag_1,
            bulk_up_filter_flag_2 = tc_cent(p).bulk.filter_flag_2,
            bulk_up_filter_flag_3 = tc_cent(p).bulk.filter_flag_3,
            bulk_up_filter_flag_4 = tc_cent(p).bulk.filter_flag_4,
            env_q_vap = tc_cent(p).en.q_tot .- tc_cent(p).en.q_liq .-
                        tc_cent(p).en.q_ice,
            draft_q_vap = tc_cent(p).bulk.q_tot .- tc_cent(p).bulk.q_liq .-
                          tc_cent(p).bulk.q_ice,
            cloud_fraction = tc_cent(p).en.area .*
                             tc_cent(p).en.cloud_fraction .+
                             tc_cent(p).bulk.area .*
                             tc_cent(p).bulk.cloud_fraction,
        )
    else
        turbulence_convection_diagnostic = NamedTuple()
    end

    if p.atmos.energy_form isa TotalEnergy
        sfc_local_geometry =
            Fields.level(Fields.local_geometry_field(Y.f), Fields.half)
        surface_ct3_unit =
            CT3.(unit_basis_vector_data.(CT3, sfc_local_geometry))
        (; ρ_flux_uₕ, ρ_flux_h_tot) = p.sfc_conditions
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
            (; ρ_flux_q_tot) = p.sfc_conditions
            vert_diff_diagnostic = (;
                vert_diff_diagnostic...,
                sfc_evaporation = dot.(ρ_flux_q_tot, surface_ct3_unit),
            )
        end
    else
        vert_diff_diagnostic = NamedTuple()
    end

    if p.atmos.radiation_mode isa RRTMGPI.AbstractRRTMGPMode
        (; face_lw_flux_dn, face_lw_flux_up, face_sw_flux_dn, face_sw_flux_up) =
            p.radiation_model
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
            ) = p.radiation_model
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

function save_to_disk_func(integrator)
    (; t, u, p) = integrator
    Y = u
    diagnostic = compute_diagnostics(integrator)

    (; output_dir) = p.simulation
    day = floor(Int, t / (60 * 60 * 24))
    sec = floor(Int, t % (60 * 60 * 24))
    @info "Saving diagnostics to HDF5 file on day $day second $sec"
    output_file = joinpath(output_dir, "day$day.$sec.hdf5")
    hdfwriter = InputOutput.HDF5Writer(output_file, p.comms_ctx)
    InputOutput.HDF5.write_attribute(hdfwriter.file, "time", t) # TODO: a better way to write metadata
    InputOutput.write!(hdfwriter, Y, "Y")
    InputOutput.write!(
        hdfwriter,
        Fields.FieldVector(; pairs(diagnostic)...),
        "diagnostics",
    )
    Base.close(hdfwriter)
    return nothing
end

function save_restart_func(integrator)
    (; t, u, p) = integrator
    (; output_dir) = p.simulation
    Y = u
    day = floor(Int, t / (60 * 60 * 24))
    sec = floor(Int, t % (60 * 60 * 24))
    @info "Saving restart file to HDF5 file on day $day second $sec"
    mkpath(joinpath(output_dir, "restart"))
    output_file = joinpath(output_dir, "restart", "day$day.$sec.hdf5")
    hdfwriter = InputOutput.HDF5Writer(output_file, integrator.p.comms_ctx)
    InputOutput.HDF5.write_attribute(hdfwriter.file, "time", t) # TODO: a better way to write metadata
    InputOutput.write!(hdfwriter, Y, "Y")
    Base.close(hdfwriter)
    return nothing
end

function gc_func(integrator)
    full = true # whether to do a full GC
    num_pre = Base.gc_num()
    alloc_since_last = (num_pre.allocd + num_pre.deferred_alloc) / 2^20
    live_pre = Base.gc_live_bytes() / 2^20
    GC.gc(full)
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
