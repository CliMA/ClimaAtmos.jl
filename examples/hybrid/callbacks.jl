import ClimaCore.DataLayouts as DL
import ClimaCore.Fields
import ClimaComms
import ClimaCore as CC
import ClimaCore.Spaces
import OrdinaryDiffEq as ODE
import ClimaAtmos.Parameters as CAP
import DiffEqCallbacks as DEQ
import ClimaCore: InputOutput

function get_callbacks(parsed_args, simulation, model_spec, params)
    FT = eltype(params)
    (; dt) = simulation

    callback_filters = ODE.DiscreteCallback(
        condition_every_iter,
        affect_filter!;
        save_positions = (false, false),
    )
    tc_callbacks = ODE.DiscreteCallback(
        condition_every_iter,
        turb_conv_affect_filter!;
        save_positions = (false, false),
    )

    additional_callbacks = if !isnothing(model_spec.radiation_model)
        # TODO: better if-else criteria?
        dt_rad = if parsed_args["config"] == "column"
            dt
        else
            FT(time_to_seconds(parsed_args["dt_rad"]))
        end
        (
            DEQ.PeriodicCallback(
                rrtmgp_model_callback!,
                dt_rad; # update RRTMGPModel every dt_rad
                initial_affect = true, # run callback at t = 0
                save_positions = (false, false), # do not save Y before and after callback
            ),
        )
    else
        ()
    end

    if !isnothing(model_spec.turbconv_model)
        additional_callbacks = (additional_callbacks..., tc_callbacks)
    end
    if model_spec.moisture_model isa EquilMoistModel &&
       parsed_args["apply_moisture_filter"]
        additional_callbacks = (additional_callbacks..., callback_filters)
    end

    dt_save_to_disk = time_to_seconds(parsed_args["dt_save_to_disk"])
    dt_save_restart = time_to_seconds(parsed_args["dt_save_restart"])

    dss_cb = DEQ.FunctionCallingCallback(dss_callback, func_start = true)
    save_to_disk_callback = if dt_save_to_disk == Inf
        nothing
    else
        DEQ.PeriodicCallback(
            save_to_disk_func,
            dt_save_to_disk;
            initial_affect = true,
            save_positions = (false, false),
        )
    end

    save_restart_callback = if dt_save_restart == Inf
        nothing
    else
        DEQ.PeriodicCallback(
            save_restart_func,
            dt_save_restart;
            initial_affect = true,
            save_positions = (false, false),
        )
    end
    return ODE.CallbackSet(
        dss_cb,
        save_to_disk_callback,
        save_restart_callback,
        additional_callbacks...,
    )
end


condition_every_iter(u, t, integrator) = true

function affect_filter!(Y::Fields.FieldVector)
    @. Y.c.ρq_tot = max(Y.c.ρq_tot, 0)
    return nothing
end

function dss_callback(Y, t, integrator)
    p = integrator.p
    @nvtx "dss callback" color = colorant"yellow" begin
        Spaces.weighted_dss_start!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_start!(Y.f, p.ghost_buffer.f)
        Spaces.weighted_dss_internal!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_internal!(Y.f, p.ghost_buffer.f)
        Spaces.weighted_dss_ghost!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_ghost!(Y.f, p.ghost_buffer.f)
    end
    # ODE.u_modified!(integrator, false) # TODO: try this
end


function affect_filter!(integrator)
    (; apply_moisture_filter) = integrator.p
    affect_filter!(integrator.u)
    # We're lying to OrdinaryDiffEq.jl, in order to avoid
    # paying for an additional tendency call, which is required
    # to support supplying a continuous representation of the solution.
    ODE.u_modified!(integrator, false)
end

function turb_conv_affect_filter!(integrator)
    (; edmf_cache, Δt) = p
    (; edmf, param_set, aux, case, surf_params) = edmf_cache
    t = integrator.t
    Y = integrator.u
    tc_params = CAP.turbconv_params(param_set)

    for inds in TC.iterate_columns(Y.c)
        state = TCU.tc_column_state(Y, aux, nothing, inds...)
        grid = TC.Grid(state)
        surf = TCU.get_surface(surf_params, grid, state, t, tc_params)
        TC.affect_filter!(edmf, grid, state, tc_params, surf, t)
    end

    # We're lying to OrdinaryDiffEq.jl, in order to avoid
    # paying for an additional `∑tendencies!` call, which is required
    # to support supplying a continuous representation of the
    # solution.
    ODE.u_modified!(integrator, false)
end

function save_to_disk_func(integrator)
    (; t, u, p) = integrator
    (; output_dir) = p.simulation
    Y = u

    if :ρq_tot in propertynames(Y.c)
        (; ᶜts, ᶜp, ᶜS_ρq_tot, params, ᶜK, ᶜΦ) = p
    else
        (; ᶜts, ᶜp, params, ᶜK, ᶜΦ) = p
    end
    thermo_params = CAP.thermodynamics_params(params)
    cm_params = CAP.microphysics_params(params)

    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    # kinetic
    @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2

    # thermo state
    thermo_state!(ᶜts, Y, params, ᶜinterp, ᶜK)
    @. ᶜp = TD.air_pressure(thermo_params, ᶜts)
    ᶜT = @. TD.air_temperature(thermo_params, ᶜts)
    ᶜθ = @. TD.dry_pottemp(thermo_params, ᶜts)

    # vorticity
    curl_uh = @. curlₕ(Y.c.uₕ)
    ᶜvort = Geometry.WVector.(curl_uh)
    Spaces.weighted_dss!(ᶜvort)

    dry_diagnostic = (;
        pressure = ᶜp,
        temperature = ᶜT,
        potential_temperature = ᶜθ,
        kinetic_energy = ᶜK,
        vorticity = ᶜvort,
    )

    # cloudwater (liquid and ice), watervapor, precipitation, and RH for moist simulation
    if :ρq_tot in propertynames(Y.c)
        ᶜq = @. TD.PhasePartition(thermo_params, ᶜts)
        ᶜcloud_liquid = @. ᶜq.liq
        ᶜcloud_ice = @. ᶜq.ice
        ᶜwatervapor = @. TD.vapor_specific_humidity(ᶜq)
        ᶜRH = @. TD.relative_humidity(thermo_params, ᶜts)

        # precipitation
        @. ᶜS_ρq_tot =
            Y.c.ρ * CM.Microphysics0M.remove_precipitation(
                cm_params,
                TD.PhasePartition(thermo_params, ᶜts),
            )

        # rain vs snow
        ᶜ3d_rain = @. ifelse(ᶜT >= FT(273.15), ᶜS_ρq_tot, FT(0))
        ᶜ3d_snow = @. ifelse(ᶜT < FT(273.15), ᶜS_ρq_tot, FT(0))
        col_integrated_rain =
            vertical∫_col(ᶜ3d_rain) ./ FT(CAP.ρ_cloud_liq(params))
        col_integrated_snow =
            vertical∫_col(ᶜ3d_snow) ./ FT(CAP.ρ_cloud_liq(params))

        moist_diagnostic = (;
            cloud_liquid = ᶜcloud_liquid,
            cloud_ice = ᶜcloud_ice,
            water_vapor = ᶜwatervapor,
            precipitation_removal = ᶜS_ρq_tot,
            column_integrated_rain = col_integrated_rain,
            column_integrated_snow = col_integrated_snow,
            relative_humidity = ᶜRH,
        )
    else
        moist_diagnostic = NamedTuple()
    end

    if vert_diff
        (; dif_flux_uₕ, dif_flux_energy, dif_flux_ρq_tot) = p
        vert_diff_diagnostic = (;
            sfc_flux_momentum = dif_flux_uₕ,
            sfc_flux_energy = dif_flux_energy,
            sfc_evaporation = dif_flux_ρq_tot,
        )
    else
        vert_diff_diagnostic = NamedTuple()
    end

    if !isnothing(model_spec.radiation_model)
        (; face_lw_flux_dn, face_lw_flux_up, face_sw_flux_dn, face_sw_flux_up) =
            p.rrtmgp_model
        rad_diagnostic = (;
            lw_flux_down = RRTMGPI.array2field(FT.(face_lw_flux_dn), axes(Y.f)),
            lw_flux_up = RRTMGPI.array2field(FT.(face_lw_flux_up), axes(Y.f)),
            sw_flux_down = RRTMGPI.array2field(FT.(face_sw_flux_dn), axes(Y.f)),
            sw_flux_up = RRTMGPI.array2field(FT.(face_sw_flux_up), axes(Y.f)),
        )
        if model_spec.radiation_model isa
           RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics
            (;
                face_clear_lw_flux_dn,
                face_clear_lw_flux_up,
                face_clear_sw_flux_dn,
                face_clear_sw_flux_up,
            ) = p.rrtmgp_model
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
    else
        rad_diagnostic = NamedTuple()
        rad_clear_diagnostic = NamedTuple()
    end

    diagnostic = merge(
        dry_diagnostic,
        moist_diagnostic,
        vert_diff_diagnostic,
        rad_diagnostic,
        rad_clear_diagnostic,
    )

    day = floor(Int, t / (60 * 60 * 24))
    sec = Int(mod(t, 3600 * 24))
    @info "Saving diagnostics to HDF5 file on day $day second $sec"
    output_file = joinpath(output_dir, "day$day.$sec.hdf5")
    hdfwriter = InputOutput.HDF5Writer(output_file, comms_ctx)
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
    sec = Int(mod(t, 3600 * 24))
    @info "Saving restart file to HDF5 file on day $day second $sec"
    mkpath(joinpath(output_dir, "restart"))
    output_file = joinpath(output_dir, "restart", "day$day.$sec.hdf5")
    hdfwriter = InputOutput.HDF5Writer(output_file, comms_ctx)
    InputOutput.HDF5.write_attribute(hdfwriter.file, "time", t) # TODO: a better way to write metadata
    InputOutput.write!(hdfwriter, Y, "Y")
    Base.close(hdfwriter)
    return nothing
end
