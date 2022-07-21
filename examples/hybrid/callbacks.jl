import ClimaCore.DataLayouts as DL
import ClimaCore.Fields
import ClimaComms
import ClimaCore as CC
import OrdinaryDiffEq as ODE
import ClimaAtmos.Parameters as CAP

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
        dt_rad = parsed_args["config"] == "column" ? dt : FT(6 * 60 * 60)
        (
            PeriodicCallback(
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

    dss_callback =
        FunctionCallingCallback(func_start = true) do Y, t, integrator
            p = integrator.p
            @nvtx "dss callback" color = colorant"yellow" begin
                Spaces.weighted_dss_start!(Y.c, p.ghost_buffer.c)
                Spaces.weighted_dss_start!(Y.f, p.ghost_buffer.f)
                Spaces.weighted_dss_internal!(Y.c, p.ghost_buffer.c)
                Spaces.weighted_dss_internal!(Y.f, p.ghost_buffer.f)
                Spaces.weighted_dss_ghost!(Y.c, p.ghost_buffer.c)
                Spaces.weighted_dss_ghost!(Y.f, p.ghost_buffer.f)
            end
        end
    save_to_disk_callback = if dt_save_to_disk == Inf
        nothing
    else
        PeriodicCallback(
            save_to_disk_func,
            dt_save_to_disk;
            initial_affect = true,
            save_positions = (false, false),
        )
    end
    return CallbackSet(
        dss_callback,
        save_to_disk_callback,
        additional_callbacks...,
    )
end


condition_every_iter(u, t, integrator) = true

function affect_filter!(Y::Fields.FieldVector)
    @. Y.c.ρq_tot = max(Y.c.ρq_tot, 0)
    return nothing
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
    if integrator.p.simulation.is_distributed
        save_to_disk_func_distributed(integrator)
    else
        save_to_disk_func_serial(integrator)
    end
end

# TODO: remove closures
function save_to_disk_func_distributed(integrator)
    (; t, u, p) = integrator
    (; output_dir) = p.simulation
    (; horizontal_mesh, quad, z_max, z_elem, z_stretch) = p.spaces

    if ClimaComms.iamroot(comms_ctx)
        global_h_space = make_horizontal_space(horizontal_mesh, quad, nothing)
        global_center_space, global_face_space =
            make_hybrid_spaces(global_h_space, z_max, z_elem, z_stretch)
    end
    global_Y_c = DL.gather(comms_ctx, Fields.field_values(u.c))
    global_Y_f = DL.gather(comms_ctx, Fields.field_values(u.f))

    if vert_diff
        (; dif_flux_uₕ, dif_flux_energy, dif_flux_ρq_tot) = p
        data_global_dif_flux_uₕ =
            DL.gather(comms_ctx, Fields.field_values(dif_flux_uₕ))
        data_global_dif_flux_energy =
            DL.gather(comms_ctx, Fields.field_values(dif_flux_energy))
        data_global_dif_flux_ρq_tot =
            DL.gather(comms_ctx, Fields.field_values(dif_flux_ρq_tot))
    end

    if !isnothing(model_spec.radiation_model)
        (; face_lw_flux_dn, face_lw_flux_up, face_sw_flux_dn, face_sw_flux_up) =
            p.rrtmgp_model

        # TODO: this is a heavily repeated pattern,
        # a local closure may be beneficial here
        data_global_face_lw_flux_dn = DL.gather(
            comms_ctx,
            Fields.field_values(
                RRTMGPI.array2field(FT.(face_lw_flux_dn), axes(u.f)),
            ),
        )
        data_global_face_lw_flux_up = DL.gather(
            comms_ctx,
            Fields.field_values(
                RRTMGPI.array2field(FT.(face_lw_flux_up), axes(u.f)),
            ),
        )
        data_global_face_sw_flux_dn = DL.gather(
            comms_ctx,
            Fields.field_values(
                RRTMGPI.array2field(FT.(face_sw_flux_dn), axes(u.f)),
            ),
        )
        data_global_face_sw_flux_up = DL.gather(
            comms_ctx,
            Fields.field_values(
                RRTMGPI.array2field(FT.(face_sw_flux_up), axes(u.f)),
            ),
        )
        if model_spec.radiation_model isa
           RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics
            (;
                face_clear_lw_flux_dn,
                face_clear_lw_flux_up,
                face_clear_sw_flux_dn,
                face_clear_sw_flux_up,
            ) = p.rrtmgp_model
            data_global_face_clear_lw_flux_dn = DL.gather(
                comms_ctx,
                Fields.field_values(
                    RRTMGPI.array2field(FT.(face_clear_lw_flux_dn), axes(u.f)),
                ),
            )
            data_global_face_clear_lw_flux_up = DL.gather(
                comms_ctx,
                Fields.field_values(
                    RRTMGPI.array2field(FT.(face_clear_lw_flux_up), axes(u.f)),
                ),
            )
            data_global_face_clear_sw_flux_dn = DL.gather(
                comms_ctx,
                Fields.field_values(
                    RRTMGPI.array2field(FT.(face_clear_sw_flux_dn), axes(u.f)),
                ),
            )
            data_global_face_clear_sw_flux_up = DL.gather(
                comms_ctx,
                Fields.field_values(
                    RRTMGPI.array2field(FT.(face_clear_sw_flux_up), axes(u.f)),
                ),
            )
        end
    end

    if ClimaComms.iamroot(comms_ctx)
        global_u = Fields.FieldVector(
            c = Fields.Field(global_Y_c, global_center_space),
            f = Fields.Field(global_Y_f, global_face_space),
        )
    end

    if ClimaComms.iamroot(comms_ctx)
        Y = global_u

        ᶜuₕ = Y.c.uₕ
        ᶠw = Y.f.w

        (; params) = p
        thermo_params = CAP.thermodynamics_params(params)
        cm_params = CAP.microphysics_params(params)
        # kinetic energy
        global_ᶜK = @. norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2

        # pressure, temperature, potential temperature
        global_ᶜts = thermo_state(Y, params, ᶜinterp, global_ᶜK)
        global_ᶜp = @. TD.air_pressure(thermo_params, global_ᶜts)
        global_ᶜT = @. TD.air_temperature(thermo_params, global_ᶜts)
        global_ᶜθ = @. TD.dry_pottemp(thermo_params, global_ᶜts)

        # vorticity
        global_curl_uh = @. curlₕ(Y.c.uₕ)
        global_ᶜvort = Geometry.WVector.(global_curl_uh)
        Spaces.weighted_dss!(global_ᶜvort)

        # surface flux if vertical diffusion is on
        if vert_diff
            z_bottom = Spaces.level(Fields.coordinate_field(Y.c).z, 1)

            # make sure datatype is correct
            global_dif_flux_uₕ =
                Geometry.Contravariant3Vector.(zeros(axes(z_bottom))) .⊗
                Geometry.Covariant12Vector.(
                    zeros(axes(z_bottom)),
                    zeros(axes(z_bottom)),
                )
            global_dif_flux_energy = similar(z_bottom, Geometry.WVector{FT})
            if :ρq_tot in propertynames(Y.c)
                global_dif_flux_ρq_tot = similar(z_bottom, Geometry.WVector{FT})
            else
                global_dif_flux_ρq_tot = Ref(Geometry.WVector(FT(0)))
            end
            # assign values from the gathered
            Fields.field_values(global_dif_flux_uₕ) .= data_global_dif_flux_uₕ
            Fields.field_values(global_dif_flux_energy) .=
                data_global_dif_flux_energy
            Fields.field_values(global_dif_flux_ρq_tot) .=
                data_global_dif_flux_ρq_tot

            vert_diff_diagnostic = (;
                sfc_flux_momentum = global_dif_flux_uₕ,
                sfc_flux_energy = global_dif_flux_energy,
                sfc_evaporation = global_dif_flux_ρq_tot,
            )
        else
            vert_diff_diagnostic = NamedTuple()
        end

        if !isnothing(model_spec.radiation_model)
            ᶠz_field = Fields.coordinate_field(Y.f).z

            # make sure datatype is correct
            global_face_lw_flux_dn = similar(ᶠz_field)
            global_face_lw_flux_up = similar(ᶠz_field)
            global_face_sw_flux_dn = similar(ᶠz_field)
            global_face_sw_flux_up = similar(ᶠz_field)
            # assign values from the gathered
            Fields.field_values(global_face_lw_flux_dn) .=
                data_global_face_lw_flux_dn
            Fields.field_values(global_face_lw_flux_up) .=
                data_global_face_lw_flux_up
            Fields.field_values(global_face_sw_flux_dn) .=
                data_global_face_sw_flux_dn
            Fields.field_values(global_face_sw_flux_up) .=
                data_global_face_sw_flux_up
            rad_diagnostic = (;
                lw_flux_down = global_face_lw_flux_dn,
                lw_flux_up = global_face_lw_flux_up,
                sw_flux_down = global_face_sw_flux_dn,
                sw_flux_up = global_face_sw_flux_up,
            )
            if model_spec.radiation_model isa
               RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics

                # make sure datatype is correct
                global_face_clear_lw_flux_dn = similar(ᶠz_field)
                global_face_clear_lw_flux_up = similar(ᶠz_field)
                global_face_clear_sw_flux_dn = similar(ᶠz_field)
                global_face_clear_sw_flux_up = similar(ᶠz_field)

                # assign values from the gathered
                Fields.field_values(global_face_clear_lw_flux_dn) .=
                    data_global_face_clear_lw_flux_dn
                Fields.field_values(global_face_clear_lw_flux_up) .=
                    data_global_face_clear_lw_flux_up
                Fields.field_values(global_face_clear_sw_flux_dn) .=
                    data_global_face_clear_sw_flux_dn
                Fields.field_values(global_face_clear_sw_flux_up) .=
                    data_global_face_clear_sw_flux_up
                rad_clear_diagnostic = (;
                    clear_lw_flux_down = global_face_clear_lw_flux_dn,
                    clear_lw_flux_up = global_face_clear_lw_flux_up,
                    clear_sw_flux_down = global_face_clear_sw_flux_dn,
                    clear_sw_flux_up = global_face_clear_sw_flux_up,
                )
            else
                rad_clear_diagnostic = NamedTuple()
            end
        else
            rad_diagnostic = NamedTuple()
            rad_clear_diagnostic = NamedTuple()
        end

        dry_diagnostic = (;
            pressure = global_ᶜp,
            temperature = global_ᶜT,
            potential_temperature = global_ᶜθ,
            kinetic_energy = global_ᶜK,
            vorticity = global_ᶜvort,
        )

        # cloudwater (liquid and ice), watervapor, precipitation, and RH for moist simulation
        if :ρq_tot in propertynames(Y.c)
            global_ᶜq = @. TD.PhasePartition(thermo_params, global_ᶜts)
            global_ᶜcloud_liquid = @. global_ᶜq.liq
            global_ᶜcloud_ice = @. global_ᶜq.ice
            global_ᶜwatervapor = @. TD.vapor_specific_humidity(global_ᶜq)
            global_ᶜRH = @. TD.relative_humidity(thermo_params, global_ᶜts)

            # precipitation
            global_ᶜS_ρq_tot =
                @. Y.c.ρ * CM.Microphysics0M.remove_precipitation(
                    cm_params,
                    TD.PhasePartition(thermo_params, global_ᶜts),
                )

            # rain vs snow
            global_ᶜ3d_rain =
                @. ifelse(global_ᶜT >= FT(273.15), global_ᶜS_ρq_tot, FT(0))
            global_ᶜ3d_snow =
                @. ifelse(global_ᶜT < FT(273.15), global_ᶜS_ρq_tot, FT(0))
            global_col_integrated_rain =
                vertical∫_col(global_ᶜ3d_rain) ./ FT(CAP.ρ_cloud_liq(params))
            global_col_integrated_snow =
                vertical∫_col(global_ᶜ3d_snow) ./ FT(CAP.ρ_cloud_liq(params))

            moist_diagnostic = (;
                cloud_liquid = global_ᶜcloud_liquid,
                cloud_ice = global_ᶜcloud_ice,
                water_vapor = global_ᶜwatervapor,
                precipitation_removal = global_ᶜS_ρq_tot,
                column_integrated_rain = global_col_integrated_rain,
                column_integrated_snow = global_col_integrated_snow,
                relative_humidity = global_ᶜRH,
            )
        else
            moist_diagnostic = NamedTuple()
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
        @info "Saving prognostic variables to JLD2 file on day $day second $sec"
        suffix = ".jld2"
        output_file = joinpath(output_dir, "day$day.$sec$suffix")
        jldsave(output_file; t, Y, diagnostic)
    end
end

function save_to_disk_func_serial(integrator)
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
    @info "Saving prognostic variables to JLD2 file on day $day second $sec"
    output_file = joinpath(output_dir, "day$day.$sec.jld2")
    jldsave(output_file; t, Y, diagnostic)
    return nothing
end
