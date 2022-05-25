
space_string(::Spaces.FaceExtrudedFiniteDifferenceSpace) = "(Face field)"
space_string(::Spaces.CenterExtrudedFiniteDifferenceSpace) = "(Center field)"

function process_name(s::AbstractString)
    # "c_ρ", "c_ρe", "c_uₕ_1", "c_uₕ_2", "f_w_1"
    s = replace(s, "components_data_" => "")
    s = replace(s, "ₕ" => "_h")
    s = replace(s, "ρ" => "rho")
    return s
end
processed_varname(pc::Tuple) = process_name(join(pc, "_"))

# TODO: Make this a RecipesBase.@recipe
function profile_animation(sol, output_dir, fps)
    # Column animations
    Y0 = first(sol.u)
    for prop_chain in Fields.property_chains(Y0)
        var_name = processed_varname(prop_chain)
        var_space = axes(Fields.single_field(Y0, prop_chain))
        Ni, Nj, _, _, Nh = size(ClimaCore.Spaces.local_geometry_data(var_space))
        n_columns = Nh * Nj * Ni # TODO: is this correct?
        @info "Creating profile animation with `n_columns` = $n_columns, for `$var_name`"
        @info "Number of timesteps: $(length(sol.u))"
        anim = Plots.@animate for Y in sol.u
            var = Fields.single_field(Y, prop_chain)
            temporary = ClimaCore.column(var, 1, 1, 1)
            ϕ_col_ave = deepcopy(vec(temporary))
            ϕ_col_std = deepcopy(vec(temporary))
            ϕ_col_ave .= 0
            ϕ_col_std .= 0
            local_geom = Fields.local_geometry_field(axes(var))
            z_f = ClimaCore.column(local_geom, 1, 1, 1)
            z_f = z_f.coordinates.z
            z = vec(z_f)
            for h in 1:Nh, j in 1:Nj, i in 1:Ni
                ϕ_col = ClimaCore.column(var, i, j, h)
                ϕ_col_ave .+= vec(ϕ_col) ./ n_columns
            end
            for h in 1:Nh, j in 1:Nj, i in 1:Ni
                ϕ_col = ClimaCore.column(var, i, j, h)
                ϕ_col_std .+=
                    sqrt.((vec(ϕ_col) .- ϕ_col_ave) .^ 2 ./ n_columns)
            end

            # TODO: use xribbon when supported: https://github.com/JuliaPlots/Plots.jl/issues/2702
            # Plots.plot(ϕ_col_ave, z ./ 1000; label = "Mean & Variance", xerror=ϕ_col_std)
            # Plots.plot!(; ylabel = "z [km]", xlabel = "$var_name", markershape = :circle)

            Plots.plot(
                z ./ 1000,
                ϕ_col_ave;
                label = "Mean & Std",
                grid = false,
                ribbon = ϕ_col_std,
                fillalpha = 0.5,
            )
            Plots.plot!(;
                ylabel = "$var_name",
                xlabel = "z [km]",
                markershape = :circle,
            )
            Plots.title!("$(space_string(var_space))")
        end
        Plots.mp4(
            anim,
            joinpath(output_dir, "profile_$var_name.mp4"),
            fps = fps,
        )
    end
end

function contour_animations(sol, output_dir, fps)
    for prop_chain in Fields.property_chains(sol.u[end])
        var_name = processed_varname(prop_chain)
        @info "Creating animation for variable:`$(var_name)`"
        anim = Plots.@animate for Y in sol.u
            var = Fields.single_field(Y, prop_chain)
            level = 3
            # TODO: do not use ClimaCore internals
            if axes(var) isa Spaces.FaceExtrudedFiniteDifferenceSpace
                level = ClimaCore.Utilities.PlusHalf(level)
            end
            clim = (minimum(var), maximum(var))
            Plots.plot(var, level = level, clim = clim)
        end
        Plots.mp4(
            anim,
            joinpath(output_dir, "contour_$var_name.mp4"),
            fps = fps,
        )
    end
end

function postprocessing(sol, output_dir, fps)
    for prop_chain in Fields.property_chains(sol.u[1])
        var_name = processed_varname(prop_chain)
        var = Fields.single_field(sol.u[1], prop_chain)
        @info "L₂ norm of `$(var_name)` at t = $(sol.t[1]): $(norm(var))"
    end
    for prop_chain in Fields.property_chains(sol.u[end])
        var_name = processed_varname(prop_chain)
        var = Fields.single_field(sol.u[end], prop_chain)
        @info "L₂ norm of `$(var_name)` at t = $(sol.t[end]): $(norm(var))"
    end

    # contour_animations(sol, output_dir, fps) # For generic contours:

    anim = Plots.@animate for Y in sol.u
        ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
        Plots.plot(ᶜv, level = 3, clim = (-6, 6))
    end
    Plots.mp4(anim, joinpath(output_dir, "v.mp4"), fps = fps)

    prop_chains = Fields.property_chains(sol.u[1])
    if any(pc -> pc == (:c, :ρq_tot), prop_chains)
        anim = Plots.@animate for Y in sol.u
            ᶜq_tot = Y.c.ρq_tot ./ Y.c.ρ
            Plots.plot(ᶜq_tot .* FT(1e3), level = 3, clim = (0, 1))
        end
        Plots.mp4(anim, joinpath(output_dir, "contour_q_tot.mp4"), fps = fps)
    else
        @info "Moisture not found. property_chains: `$(prop_chains)`"
    end

    profile_animation(sol, output_dir, fps)
end

# Dispatcher:
# baroclinic wave
paperplots_baro_wave(args...) =
    paperplots_baro_wave(energy_form(), moisture_model(), args...)

paperplots_baro_wave(::PotentialTemperature, ::DryModel, args...) =
    paperplots_baro_wave_ρθ(args...)
paperplots_baro_wave(::TotalEnergy, ::DryModel, args...) =
    paperplots_baro_wave_ρe(args...)
paperplots_baro_wave(::TotalEnergy, ::EquilMoistModel, args...) =
    paperplots_moist_baro_wave_ρe(args...)

# held-suarez
paperplots_held_suarez(args...) =
    paperplots_held_suarez(energy_form(), moisture_model(), args...)

paperplots_held_suarez(::PotentialTemperature, ::DryModel, args...) =
    paperplots_dry_held_suarez_ρθ(args...)
paperplots_held_suarez(::TotalEnergy, ::DryModel, args...) =
    paperplots_dry_held_suarez_ρe(args...)
paperplots_held_suarez(::InternalEnergy, ::DryModel, args...) =
    paperplots_dry_held_suarez_ρe_int(args...)
paperplots_held_suarez(::TotalEnergy, ::EquilMoistModel, args...) =
    paperplots_moist_held_suarez_ρe(args...)

# plots in the Ullrish et al 2014 paper: surface pressure, 850 temperature and vorticity at day 8 and day 10
function paperplots_baro_wave_ρθ(sol, output_dir, p, nlat, nlon)
    (; ᶜts, ᶜp, params) = p
    days = [8, 10]

    # obtain pressure, temperature, and vorticity at cg points;
    # and remap them onto lat lon
    for day in days
        iu = findall(x -> x == day * 24 * 3600, sol.t)[1]
        Y = sol.u[iu]
        # compute pressure, temperature, vorticity
        @. ᶜts = thermo_state_ρθ(Y.c.ρθ, Y.c, params)
        @. ᶜp = TD.air_pressure(params, ᶜts)
        ᶜT = @. TD.air_temperature(params, ᶜts)
        curl_uh = @. curlₕ(Y.c.uₕ)
        ᶜvort = Geometry.WVector.(curl_uh)
        Spaces.weighted_dss!(ᶜvort)

        # space info to generate nc raw data
        cspace = axes(Y.c)
        hspace = cspace.horizontal_space
        Nq = Spaces.Quadratures.degrees_of_freedom(
            cspace.horizontal_space.quadrature_style,
        )

        # create a temporary dir for intermediate data
        remap_tmpdir = output_dir * "/remaptmp/"
        mkpath(remap_tmpdir)

        ### create an nc file to store raw cg data
        # create data
        datafile_cc = remap_tmpdir * "/bw-raw_day" * string(day) * ".nc"
        nc = NCDataset(datafile_cc, "c")
        # defines the appropriate dimensions and variables for a space coordinate
        def_space_coord(nc, cspace, type = "cgll")
        # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
        nc_time = def_time_coord(nc)
        # defines variables for pressure, temperature, and vorticity
        nc_p = defVar(nc, "pres", FT, cspace, ("time",))
        nc_T = defVar(nc, "T", FT, cspace, ("time",))
        nc_ω = defVar(nc, "vort", FT, cspace, ("time",))

        nc_time[1] = FT(day * 24 * 3600)
        nc_p[:, 1] = ᶜp
        nc_T[:, 1] = ᶜT
        nc_ω[:, 1] = ᶜvort

        close(nc)

        # write out our cubed sphere mesh
        meshfile_cc = remap_tmpdir * "/mesh_cubedsphere.g"
        write_exodus(meshfile_cc, hspace.topology)

        meshfile_rll = remap_tmpdir * "/mesh_rll.g"
        rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

        meshfile_overlap = remap_tmpdir * "/mesh_overlap.g"
        overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

        weightfile = remap_tmpdir * "/remap_weights.nc"
        remap_weights(
            weightfile,
            meshfile_cc,
            meshfile_rll,
            meshfile_overlap;
            in_type = "cgll",
            in_np = Nq,
        )

        datafile_latlon = output_dir * "/bw-remapped_day" * string(day) * ".nc"
        apply_remap(
            datafile_latlon,
            datafile_cc,
            weightfile,
            ["pres", "T", "vort"],
        )

        rm(remap_tmpdir, recursive = true)
    end

    # create plots as in the paper
    for day in days
        datafile_latlon = output_dir * "/bw-remapped_day" * string(day) * ".nc"
        ncdata = NCDataset(datafile_latlon, "r")
        lon = ncdata["lon"][:]
        lat = ncdata["lat"][:]

        p = ncdata["pres"][:]
        T = ncdata["T"][:]
        vort = ncdata["vort"][:] * FT(1e5)
        close(ncdata)
        latidx = findall(x -> x >= 0, lat)
        lonidx = findall(x -> 0 <= x <= 240, lon)

        plot_p = contourf(
            lon[lonidx],
            lat[latidx],
            p[lonidx, latidx, 1, 1]',
            color = :rainbow,
            title = "pressure (1500m) day " * string(day),
        )
        png(plot_p, output_dir * "/bw-pressure-day" * string(day) * ".png")

        plot_T = contourf(
            lon[lonidx],
            lat[latidx],
            T[lonidx, latidx, 1, 1]',
            color = :rainbow,
            levels = 220:10:310,
            title = "temperature (1500m) day " * string(day),
        )
        png(plot_T, output_dir * "/bw-temperature-day" * string(day) * ".png")

        plot_ω = contourf(
            lon[lonidx],
            lat[latidx],
            vort[lonidx, latidx, 1, 1]',
            color = :balance,
            title = "vorticity (1500m) day " * string(day),
        )
        png(plot_ω, output_dir * "/bw-vorticity-day" * string(day) * ".png")

        rm(datafile_latlon)
    end

end

function paperplots_baro_wave_ρe(sol, output_dir, p, nlat, nlon)
    (; ᶜts, ᶜp, params, ᶜK, ᶜΦ) = p
    days = [8, 10]

    # obtain pressure, temperature, and vorticity at cg points;
    # and remap them onto lat lon
    for day in days
        iu = findall(x -> x == day * 24 * 3600, sol.t)[1]
        Y = sol.u[iu]

        # compute pressure, temperature, vorticity
        ᶜuₕ = Y.c.uₕ
        ᶠw = Y.f.w
        @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2
        @. ᶜts = thermo_state_ρe(Y.c.ρe_tot, Y.c, ᶜK, ᶜΦ, params)
        @. ᶜp = TD.air_pressure(params, ᶜts)
        ᶜT = @. TD.air_temperature(params, ᶜts)
        curl_uh = @. curlₕ(Y.c.uₕ)
        ᶜvort = Geometry.WVector.(curl_uh)
        Spaces.weighted_dss!(ᶜvort)

        # space info to generate nc raw data
        cspace = axes(Y.c)
        hspace = cspace.horizontal_space
        Nq = Spaces.Quadratures.degrees_of_freedom(
            cspace.horizontal_space.quadrature_style,
        )

        # create a temporary dir for intermediate data
        remap_tmpdir = output_dir * "/remaptmp/"
        mkpath(remap_tmpdir)

        ### create an nc file to store raw cg data
        # create data
        datafile_cc = remap_tmpdir * "/bw-raw_day" * string(day) * ".nc"
        nc = NCDataset(datafile_cc, "c")
        # defines the appropriate dimensions and variables for a space coordinate
        def_space_coord(nc, cspace, type = "cgll")
        # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
        nc_time = def_time_coord(nc)
        # defines variables for pressure, temperature, and vorticity
        nc_p = defVar(nc, "pres", FT, cspace, ("time",))
        nc_T = defVar(nc, "T", FT, cspace, ("time",))
        nc_ω = defVar(nc, "vort", FT, cspace, ("time",))

        nc_time[1] = FT(day * 24 * 3600)
        nc_p[:, 1] = ᶜp
        nc_T[:, 1] = ᶜT
        nc_ω[:, 1] = ᶜvort

        close(nc)

        # write out our cubed sphere mesh
        meshfile_cc = remap_tmpdir * "/mesh_cubedsphere.g"
        write_exodus(meshfile_cc, hspace.topology)

        meshfile_rll = remap_tmpdir * "/mesh_rll.g"
        rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

        meshfile_overlap = remap_tmpdir * "/mesh_overlap.g"
        overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

        weightfile = remap_tmpdir * "/remap_weights.nc"
        remap_weights(
            weightfile,
            meshfile_cc,
            meshfile_rll,
            meshfile_overlap;
            in_type = "cgll",
            in_np = Nq,
        )

        datafile_latlon = output_dir * "/bw-remapped_day" * string(day) * ".nc"
        apply_remap(
            datafile_latlon,
            datafile_cc,
            weightfile,
            ["pres", "T", "vort"],
        )

        rm(remap_tmpdir, recursive = true)
    end

    # create plots as in the paper
    for day in days
        datafile_latlon = output_dir * "/bw-remapped_day" * string(day) * ".nc"
        ncdata = NCDataset(datafile_latlon, "r")
        lon = ncdata["lon"][:]
        lat = ncdata["lat"][:]

        p = ncdata["pres"][:]
        T = ncdata["T"][:]
        vort = ncdata["vort"][:] * FT(1e5)

        latidx = findall(x -> x >= 0, lat)
        lonidx = findall(x -> 0 <= x <= 240, lon)

        plot_p = contourf(
            lon[lonidx],
            lat[latidx],
            p[lonidx, latidx, 1, 1]',
            color = :rainbow,
            title = "pressure (1500m) day " * string(day),
        )
        png(plot_p, output_dir * "/bw-pressure-day" * string(day) * ".png")

        plot_T = contourf(
            lon[lonidx],
            lat[latidx],
            T[lonidx, latidx, 1, 1]',
            color = :rainbow,
            levels = 220:10:310,
            title = "temperature (1500m) day " * string(day),
        )
        png(plot_T, output_dir * "/bw-temperature-day" * string(day) * ".png")

        plot_ω = contourf(
            lon[lonidx],
            lat[latidx],
            vort[lonidx, latidx, 1, 1]',
            color = :balance,
            title = "vorticity (1500m) day " * string(day),
        )
        png(plot_ω, output_dir * "/bw-vorticity-day" * string(day) * ".png")

        rm(datafile_latlon)
    end

end

# plots for moist baroclinic wave: https://www.cesm.ucar.edu/events/wg-meetings/2018/presentations/amwg/jablonowski.pdf
function paperplots_moist_baro_wave_ρe(sol, output_dir, p, nlat, nlon)
    (; ᶜts, ᶜp, params, ᶜK, ᶜΦ) = p
    days = [8, 10]

    # obtain pressure, temperature, and vorticity at cg points;
    # and remap them onto lat lon
    for day in days
        iu = findall(x -> x == day * 24 * 3600, sol.t)[1]
        Y = sol.u[iu]

        # compute pressure, temperature, vorticity
        ᶜρ = Y.c.ρ
        ᶜuₕ = Y.c.uₕ
        ᶠw = Y.f.w
        ᶜuₕ_phy = Geometry.UVVector.(ᶜuₕ)
        ᶠw_phy = Geometry.WVector.(ᶠw)
        ᶜw_phy = ᶜinterp.(ᶠw_phy)
        @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2
        @. ᶜts = thermo_state_ρe(Y.c.ρe_tot, Y.c, ᶜK, ᶜΦ, params)
        @. ᶜp = TD.air_pressure(params, ᶜts)

        ᶜq = @. TD.PhasePartition(params, ᶜts)
        ᶜcloudwater = @. TD.condensate(ᶜq) # @. ᶜq.liq + ᶜq.ice
        ᶜwatervapor = @. TD.vapor_specific_humidity(ᶜq)

        ᶜT = @. TD.air_temperature(params, ᶜts)
        curl_uh = @. curlₕ(Y.c.uₕ)
        ᶜvort = Geometry.WVector.(curl_uh)
        Spaces.weighted_dss!(ᶜvort)

        # space info to generate nc raw data
        cspace = axes(Y.c)
        hspace = cspace.horizontal_space
        Nq = Spaces.Quadratures.degrees_of_freedom(
            cspace.horizontal_space.quadrature_style,
        )

        # create a temporary dir for intermediate data
        remap_tmpdir = output_dir * "/remaptmp/"
        mkpath(remap_tmpdir)

        ### create an nc file to store raw cg data
        # create data
        datafile_cc = remap_tmpdir * "/bw-raw_day" * string(day) * ".nc"
        nc = NCDataset(datafile_cc, "c")
        # defines the appropriate dimensions and variables for a space coordinate
        def_space_coord(nc, cspace, type = "cgll")
        # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
        nc_time = def_time_coord(nc)
        # defines variables for pressure, temperature, and vorticity
        nc_p = defVar(nc, "pres", FT, cspace, ("time",))
        nc_T = defVar(nc, "T", FT, cspace, ("time",))
        nc_ω = defVar(nc, "vort", FT, cspace, ("time",))
        nc_qt = defVar(nc, "qt", FT, cspace, ("time",))
        nc_w = defVar(nc, "w", FT, cspace, ("time",))
        nc_ρ = defVar(nc, "rho", FT, cspace, ("time",))
        nc_cloudwater = defVar(nc, "cloud_water", FT, cspace, ("time",))
        nc_watervapor = defVar(nc, "water_vapor", FT, cspace, ("time",))
        nc_u = defVar(nc, "u", FT, cspace, ("time",))
        nc_v = defVar(nc, "v", FT, cspace, ("time",))

        nc_time[1] = FT(day * 24 * 3600)
        nc_p[:, 1] = ᶜp
        nc_T[:, 1] = ᶜT
        nc_ω[:, 1] = ᶜvort
        nc_qt[:, 1] = Y.c.ρq_tot ./ Y.c.ρ
        nc_w[:, 1] = ᶜw_phy
        nc_ρ[:, 1] = ᶜρ
        nc_cloudwater[:, 1] = ᶜcloudwater
        nc_watervapor[:, 1] = ᶜwatervapor
        nc_u[:, 1] = ᶜuₕ_phy.components.data.:1
        nc_v[:, 1] = ᶜuₕ_phy.components.data.:2

        close(nc)

        # write out our cubed sphere mesh
        meshfile_cc = remap_tmpdir * "/mesh_cubedsphere.g"
        write_exodus(meshfile_cc, hspace.topology)

        meshfile_rll = remap_tmpdir * "/mesh_rll.g"
        rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

        meshfile_overlap = remap_tmpdir * "/mesh_overlap.g"
        overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

        weightfile = remap_tmpdir * "/remap_weights.nc"
        remap_weights(
            weightfile,
            meshfile_cc,
            meshfile_rll,
            meshfile_overlap;
            in_type = "cgll",
            in_np = Nq,
        )

        datafile_latlon = output_dir * "/bw-remapped_day" * string(day) * ".nc"
        apply_remap(
            datafile_latlon,
            datafile_cc,
            weightfile,
            [
                "pres",
                "T",
                "vort",
                "qt",
                "w",
                "rho",
                "cloud_water",
                "water_vapor",
                "u",
                "v",
            ],
        )

        rm(remap_tmpdir, recursive = true)
    end

    # create plots as in the reference
    for day in days
        datafile_latlon = output_dir * "/bw-remapped_day" * string(day) * ".nc"
        ncdata = NCDataset(datafile_latlon, "r")
        lon = ncdata["lon"][:]
        lat = ncdata["lat"][:]

        p = ncdata["pres"][:]
        T = ncdata["T"][:]
        vort = ncdata["vort"][:] * FT(1e5)
        qt = ncdata["qt"][:] * FT(1e3)
        cloud_water = ncdata["cloud_water"][:] * FT(1e3)
        water_vapor = ncdata["water_vapor"][:] * FT(1e3)
        rho = ncdata["rho"][:]
        w = ncdata["w"][:]
        u = ncdata["u"][:]
        v = ncdata["v"][:]

        vert_intg_cloud_water =
            sum(cloud_water .* rho, dims = 3) ./ sum(rho, dims = 3)
        vert_intg_water_vapor =
            sum(water_vapor .* rho, dims = 3) ./ sum(rho, dims = 3)

        latidx = findall(x -> x >= 0, lat)
        lonidx = findall(x -> 0 <= x <= 240, lon)

        plot_p = contourf(
            lon[lonidx],
            lat[latidx],
            p[lonidx, latidx, 1, 1]',
            color = :rainbow,
            title = "pressure (1500m) day " * string(day),
        )
        png(plot_p, output_dir * "/bw-pressure-day" * string(day) * ".png")

        plot_T = contourf(
            lon[lonidx],
            lat[latidx],
            T[lonidx, latidx, 1, 1]',
            color = :rainbow,
            levels = 220:10:310,
            title = "temperature (1500m) day " * string(day),
        )
        png(plot_T, output_dir * "/bw-temperature-day" * string(day) * ".png")

        plot_ω = contourf(
            lon[lonidx],
            lat[latidx],
            vort[lonidx, latidx, 1, 1]',
            color = :balance,
            title = "vorticity (1500m) day " * string(day),
        )
        png(
            plot_ω,
            output_dir * "/bw-vorticity_lower-day" * string(day) * ".png",
        )

        plot_ω = contourf(
            lon[lonidx],
            lat[latidx],
            vort[lonidx, latidx, 3, 1]',
            color = :balance,
            title = "vorticity (7500m) day " * string(day),
        )
        png(
            plot_ω,
            output_dir * "/bw-vorticity_uppper-day" * string(day) * ".png",
        )

        plot_qt = contourf(
            lon[lonidx],
            lat[latidx],
            qt[lonidx, latidx, 1, 1]',
            color = :balance,
            title = "qt (1500m) day " * string(day),
        )
        png(plot_qt, output_dir * "/bw-qt-day" * string(day) * ".png")

        plot_w = contourf(
            lon[lonidx],
            lat[latidx],
            w[lonidx, latidx, 1, 1]',
            color = :balance,
            title = "w (1500m) day " * string(day),
        )
        png(plot_w, output_dir * "/bw-w-day" * string(day) * ".png")

        plot_u = contourf(
            lon[lonidx],
            lat[latidx],
            u[lonidx, latidx, 3, 1]',
            color = :balance,
            title = "u (7500m) day " * string(day),
        )
        png(plot_u, output_dir * "/bw-u7500-day" * string(day) * ".png")

        plot_u = contourf(
            lon[lonidx],
            lat[latidx],
            u[lonidx, latidx, 1, 1]',
            color = :balance,
            title = "u (1500m) day " * string(day),
        )
        png(plot_u, output_dir * "/bw-u1500-day" * string(day) * ".png")

        plot_v = contourf(
            lon[lonidx],
            lat[latidx],
            v[lonidx, latidx, 3, 1]',
            color = :balance,
            title = "v (7500m) day " * string(day),
        )
        png(plot_v, output_dir * "/bw-v7500-day" * string(day) * ".png")

        plot_v = contourf(
            lon[lonidx],
            lat[latidx],
            v[lonidx, latidx, 1, 1]',
            color = :balance,
            title = "v (1500m) day " * string(day),
        )
        png(plot_v, output_dir * "/bw-v1500-day" * string(day) * ".png")

        plot_cw = contourf(
            lon[lonidx],
            lat[latidx],
            vert_intg_cloud_water[lonidx, latidx, 1, 1]',
            color = :balance,
            title = "cloud condensate (vertically integrated; g/kg) day " *
                    string(day),
        )
        png(
            plot_cw,
            output_dir * "/bw-vert_intg_cloud_water-day" * string(day) * ".png",
        )

        plot_wv = contourf(
            lon[lonidx],
            lat[latidx],
            vert_intg_water_vapor[lonidx, latidx, 1, 1]',
            color = :balance,
            title = "water vapor (vertically integrated; g/kg) day " *
                    string(day),
        )
        png(
            plot_wv,
            output_dir * "/bw-vert_intg_water_vapor-day" * string(day) * ".png",
        )

        rm(datafile_latlon)
    end

end

# plots in the held-suarez paper: https://journals.ametsoc.org/view/journals/bams/75/10/1520-0477_1994_075_1825_apftio_2_0_co_2.xml?tab_body=pdf
calc_zonalave_timeave(x) =
    dropdims(dropdims(mean(mean(x, dims = 1), dims = 4), dims = 4), dims = 1)
calc_zonalave_variance(x) = calc_zonalave_timeave((x .- mean(x, dims = 4)) .^ 2)

function paperplots_dry_held_suarez_ρθ(sol, output_dir, p, nlat, nlon)
    (; ᶜts, params) = p

    ### save raw data into nc -> in preparation for remapping
    # space info to generate nc raw data
    Y = sol.u[1]
    cspace = axes(Y.c)
    hspace = cspace.horizontal_space
    Nq = Spaces.Quadratures.degrees_of_freedom(
        cspace.horizontal_space.quadrature_style,
    )

    # create a temporary dir for intermediate data
    remap_tmpdir = output_dir * "/remaptmp/"
    mkpath(remap_tmpdir)

    # create an nc file to store raw cg data
    # create data
    datafile_cc = remap_tmpdir * "/hs-raw.nc"
    nc = NCDataset(datafile_cc, "c")
    # defines the appropriate dimensions and variables for a space coordinate
    def_space_coord(nc, cspace, type = "cgll")
    # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
    nc_time = def_time_coord(nc)
    # defines variables for pressure, temperature, and vorticity
    nc_θ = defVar(nc, "PotentialTemperature", FT, cspace, ("time",))
    nc_T = defVar(nc, "T", FT, cspace, ("time",))
    nc_u = defVar(nc, "u", FT, cspace, ("time",))

    unique_time = unique(sol.t) # for some reason, if time of save to solution coincides with time of save to disk, a duplicate copy is saved in sol

    # save raw data
    for i in 1:length(unique_time)
        nc_time[i] = unique_time[i]

        iu = findall(x -> x == unique_time[i], sol.t)[1]
        Y = sol.u[iu]

        # potential temperature
        ᶜθ = Y.c.ρθ ./ Y.c.ρ
        # temperature
        @. ᶜts = thermo_state_ρθ(Y.c.ρθ, Y.c, params)
        ᶜT = @. TD.air_temperature(params, ᶜts)
        # zonal wind
        ᶜuₕ = Y.c.uₕ
        ᶜuₕ_phy = Geometry.UVVector.(ᶜuₕ)

        # assigning to nc obj
        nc_θ[:, i] = ᶜθ
        nc_T[:, i] = ᶜT
        nc_u[:, i] = ᶜuₕ_phy.components.data.:1
    end

    close(nc)

    ### write out our cubed sphere mesh
    meshfile_cc = remap_tmpdir * "/mesh_cubedsphere.g"
    write_exodus(meshfile_cc, hspace.topology)

    meshfile_rll = remap_tmpdir * "/mesh_rll.g"
    rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

    meshfile_overlap = remap_tmpdir * "/mesh_overlap.g"
    overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

    weightfile = remap_tmpdir * "/remap_weights.nc"
    remap_weights(
        weightfile,
        meshfile_cc,
        meshfile_rll,
        meshfile_overlap;
        in_type = "cgll",
        in_np = Nq,
    )

    ### remap to lat/lon
    datafile_latlon = output_dir * "/hs-remapped.nc"
    apply_remap(
        datafile_latlon,
        datafile_cc,
        weightfile,
        ["PotentialTemperature", "T", "u"],
    )

    rm(remap_tmpdir, recursive = true)

    ### load remapped data and create statistics for plots
    datafile_latlon = output_dir * "/hs-remapped.nc"
    ncdata = NCDataset(datafile_latlon, "r")
    lat = ncdata["lat"][:]
    z = ncdata["z"][:]
    time = ncdata["time"][:]

    T = ncdata["T"][:, :, :, time .> 3600 * 24 * 200]
    θ = ncdata["PotentialTemperature"][:, :, :, time .> 3600 * 24 * 200]
    u = ncdata["u"][:, :, :, time .> 3600 * 24 * 200]

    u_timeave_zonalave = calc_zonalave_timeave(u)
    T_timeave_zonalave = calc_zonalave_timeave(T)
    θ_timeave_zonalave = calc_zonalave_timeave(θ)

    T2_timeave_zonalave = calc_zonalave_variance(T)

    Plots.png(
        Plots.contourf(
            lat,
            z,
            u_timeave_zonalave',
            color = :balance,
            clim = (-30, 30),
            linewidth = 0,
            yaxis = :log,
            title = "u",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-u.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            T_timeave_zonalave',
            levels = 190:10:310,
            clim = (190, 310),
            contour_labels = true,
            yaxis = :log,
            title = "T",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-T.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            θ_timeave_zonalave',
            levels = 260:10:360,
            clim = (260, 360),
            contour_labels = true,
            yaxis = :log,
            title = "theta",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-theta.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            T2_timeave_zonalave',
            color = :balance,
            clim = (0, 40),
            linewidth = 0,
            yaxis = :log,
            title = "[T^2]",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-T2.png",
    )

end

function paperplots_dry_held_suarez_ρe(sol, output_dir, p, nlat, nlon)
    (; ᶜts, params, ᶜK, ᶜΦ) = p

    ### save raw data into nc -> in preparation for remapping
    # space info to generate nc raw data
    Y = sol.u[1]
    cspace = axes(Y.c)
    hspace = cspace.horizontal_space
    Nq = Spaces.Quadratures.degrees_of_freedom(
        cspace.horizontal_space.quadrature_style,
    )

    # create a temporary dir for intermediate data
    remap_tmpdir = output_dir * "/remaptmp/"
    mkpath(remap_tmpdir)

    # create an nc file to store raw cg data
    # create data
    datafile_cc = remap_tmpdir * "/hs-raw.nc"
    nc = NCDataset(datafile_cc, "c")
    # defines the appropriate dimensions and variables for a space coordinate
    def_space_coord(nc, cspace, type = "cgll")
    # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
    nc_time = def_time_coord(nc)
    # defines variables for pressure, temperature, and vorticity
    nc_θ = defVar(nc, "PotentialTemperature", FT, cspace, ("time",))
    nc_T = defVar(nc, "T", FT, cspace, ("time",))
    nc_u = defVar(nc, "u", FT, cspace, ("time",))

    unique_time = unique(sol.t) # for some reason, if time of save to solution coincides with time of save to disk, a duplicate copy is saved in sol

    # save raw data
    for i in 1:length(unique_time)
        nc_time[i] = unique_time[i]

        iu = findall(x -> x == unique_time[i], sol.t)[1]
        Y = sol.u[iu]

        # zonal wind
        ᶜuₕ = Y.c.uₕ
        ᶜuₕ_phy = Geometry.UVVector.(ᶜuₕ)

        # temperature
        ᶠw = Y.f.w
        @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2
        @. ᶜts = thermo_state_ρe(Y.c.ρe_tot, Y.c, ᶜK, ᶜΦ, params)
        ᶜT = @. TD.air_temperature(params, ᶜts)
        ᶜθ = @. TD.dry_pottemp(params, ᶜts)

        # assigning to nc obj
        nc_θ[:, i] = ᶜθ
        nc_T[:, i] = ᶜT
        nc_u[:, i] = ᶜuₕ_phy.components.data.:1
    end

    close(nc)

    ### write out our cubed sphere mesh
    meshfile_cc = remap_tmpdir * "/mesh_cubedsphere.g"
    write_exodus(meshfile_cc, hspace.topology)

    meshfile_rll = remap_tmpdir * "/mesh_rll.g"
    rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

    meshfile_overlap = remap_tmpdir * "/mesh_overlap.g"
    overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

    weightfile = remap_tmpdir * "/remap_weights.nc"
    remap_weights(
        weightfile,
        meshfile_cc,
        meshfile_rll,
        meshfile_overlap;
        in_type = "cgll",
        in_np = Nq,
    )

    ### remap to lat/lon
    datafile_latlon = output_dir * "/hs-remapped.nc"
    apply_remap(
        datafile_latlon,
        datafile_cc,
        weightfile,
        ["PotentialTemperature", "T", "u"],
    )

    # rm(remap_tmpdir, recursive = true)

    ### load remapped data and create statistics for plots
    datafile_latlon = output_dir * "/hs-remapped.nc"
    ncdata = NCDataset(datafile_latlon, "r")
    lat = ncdata["lat"][:]
    z = ncdata["z"][:]
    time = ncdata["time"][:]

    T = ncdata["T"][:, :, :, time .> 3600 * 24 * 200]
    θ = ncdata["PotentialTemperature"][:, :, :, time .> 3600 * 24 * 200]
    u = ncdata["u"][:, :, :, time .> 3600 * 24 * 200]

    u_timeave_zonalave = calc_zonalave_timeave(u)
    T_timeave_zonalave = calc_zonalave_timeave(T)
    θ_timeave_zonalave = calc_zonalave_timeave(θ)

    T2_timeave_zonalave = calc_zonalave_variance(T)

    Plots.png(
        Plots.contourf(
            lat,
            z,
            u_timeave_zonalave',
            color = :balance,
            clim = (-30, 30),
            linewidth = 0,
            yaxis = :log,
            title = "u",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-u.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            T_timeave_zonalave',
            levels = 190:10:310,
            clim = (190, 310),
            contour_labels = true,
            yaxis = :log,
            title = "T",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-T.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            θ_timeave_zonalave',
            levels = 260:10:360,
            clim = (260, 360),
            contour_labels = true,
            yaxis = :log,
            title = "theta",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-theta.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            T2_timeave_zonalave',
            color = :balance,
            clim = (0, 40),
            linewidth = 0,
            yaxis = :log,
            title = "[T^2]",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-T2.png",
    )

end

function paperplots_dry_held_suarez_ρe_int(sol, output_dir, p, nlat, nlon)
    (; ᶜts, params) = p

    ### save raw data into nc -> in preparation for remapping
    # space info to generate nc raw data
    Y = sol.u[1]
    cspace = axes(Y.c)
    hspace = cspace.horizontal_space
    Nq = Spaces.Quadratures.degrees_of_freedom(
        cspace.horizontal_space.quadrature_style,
    )

    # create a temporary dir for intermediate data
    remap_tmpdir = output_dir * "/remaptmp/"
    mkpath(remap_tmpdir)

    # create an nc file to store raw cg data
    # create data
    datafile_cc = remap_tmpdir * "/hs-raw.nc"
    nc = NCDataset(datafile_cc, "c")
    # defines the appropriate dimensions and variables for a space coordinate
    def_space_coord(nc, cspace, type = "cgll")
    # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
    nc_time = def_time_coord(nc)
    # defines variables for pressure, temperature, and vorticity
    nc_θ = defVar(nc, "PotentialTemperature", FT, cspace, ("time",))
    nc_T = defVar(nc, "T", FT, cspace, ("time",))
    nc_u = defVar(nc, "u", FT, cspace, ("time",))

    unique_time = unique(sol.t) # for some reason, if time of save to solution coincides with time of save to disk, a duplicate copy is saved in sol

    # save raw data
    for i in 1:length(unique_time)
        nc_time[i] = unique_time[i]

        iu = findall(x -> x == unique_time[i], sol.t)[1]
        Y = sol.u[iu]

        # temperature
        @. ᶜts = thermo_state_ρe_int(Y.c.ρe_int, Y.c, params)
        ᶜT = @. TD.air_temperature(params, ᶜts)
        ᶜθ = @. TD.dry_pottemp(params, ᶜts)

        # zonal wind
        ᶜuₕ = Y.c.uₕ
        ᶜuₕ_phy = Geometry.UVVector.(ᶜuₕ)

        # assigning to nc obj
        nc_θ[:, i] = ᶜθ
        nc_T[:, i] = ᶜT
        nc_u[:, i] = ᶜuₕ_phy.components.data.:1
    end

    close(nc)

    ### write out our cubed sphere mesh
    meshfile_cc = remap_tmpdir * "/mesh_cubedsphere.g"
    write_exodus(meshfile_cc, hspace.topology)

    meshfile_rll = remap_tmpdir * "/mesh_rll.g"
    rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

    meshfile_overlap = remap_tmpdir * "/mesh_overlap.g"
    overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

    weightfile = remap_tmpdir * "/remap_weights.nc"
    remap_weights(
        weightfile,
        meshfile_cc,
        meshfile_rll,
        meshfile_overlap;
        in_type = "cgll",
        in_np = Nq,
    )

    ### remap to lat/lon
    datafile_latlon = output_dir * "/hs-remapped.nc"
    apply_remap(
        datafile_latlon,
        datafile_cc,
        weightfile,
        ["PotentialTemperature", "T", "u"],
    )

    rm(remap_tmpdir, recursive = true)

    ### load remapped data and create statistics for plots
    datafile_latlon = output_dir * "/hs-remapped.nc"
    ncdata = NCDataset(datafile_latlon, "r")
    lat = ncdata["lat"][:]
    z = ncdata["z"][:]
    time = ncdata["time"][:]

    T = ncdata["T"][:, :, :, time .> 3600 * 24 * 200]
    θ = ncdata["PotentialTemperature"][:, :, :, time .> 3600 * 24 * 200]
    u = ncdata["u"][:, :, :, time .> 3600 * 24 * 200]

    u_timeave_zonalave = calc_zonalave_timeave(u)
    T_timeave_zonalave = calc_zonalave_timeave(T)
    θ_timeave_zonalave = calc_zonalave_timeave(θ)

    T2_timeave_zonalave = calc_zonalave_variance(T)

    Plots.png(
        Plots.contourf(
            lat,
            z,
            u_timeave_zonalave',
            color = :balance,
            clim = (-30, 30),
            linewidth = 0,
            yaxis = :log,
            title = "u",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-u.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            T_timeave_zonalave',
            levels = 190:10:310,
            clim = (190, 310),
            contour_labels = true,
            yaxis = :log,
            title = "T",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-T.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            θ_timeave_zonalave',
            levels = 260:10:360,
            clim = (260, 360),
            contour_labels = true,
            yaxis = :log,
            title = "theta",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-theta.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            T2_timeave_zonalave',
            color = :balance,
            clim = (0, 40),
            linewidth = 0,
            yaxis = :log,
            title = "[T^2]",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-T2.png",
    )

end

function postprocessing_edmf(sol, output_dir, fps)
    profile_animation(sol, output_dir, fps)
end

function paperplots_moist_held_suarez_ρe(sol, output_dir, p, nlat, nlon)
    (; ᶜts, params, ᶜK, ᶜΦ) = p

    ### save raw data into nc -> in preparation for remapping
    # space info to generate nc raw data
    Y = sol.u[1]
    cspace = axes(Y.c)
    hspace = cspace.horizontal_space
    Nq = Spaces.Quadratures.degrees_of_freedom(
        cspace.horizontal_space.quadrature_style,
    )

    # create a temporary dir for intermediate data
    remap_tmpdir = output_dir * "/remaptmp/"
    mkpath(remap_tmpdir)

    # create an nc file to store raw cg data
    # create data
    datafile_cc = remap_tmpdir * "/hs-raw.nc"
    nc = NCDataset(datafile_cc, "c")
    # defines the appropriate dimensions and variables for a space coordinate
    def_space_coord(nc, cspace, type = "cgll")
    # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
    nc_time = def_time_coord(nc)
    # defines variables for pressure, temperature, and vorticity
    nc_θ = defVar(nc, "PotentialTemperature", FT, cspace, ("time",))
    nc_T = defVar(nc, "T", FT, cspace, ("time",))
    nc_u = defVar(nc, "u", FT, cspace, ("time",))
    nc_qt = defVar(nc, "qt", FT, cspace, ("time",))

    unique_time = unique(sol.t) # for some reason, if time of save to solution coincides with time of save to disk, a duplicate copy is saved in sol

    # save raw data
    for i in 1:length(unique_time)
        nc_time[i] = unique_time[i]

        iu = findall(x -> x == unique_time[i], sol.t)[1]
        Y = sol.u[iu]

        # zonal wind
        ᶜuₕ = Y.c.uₕ
        ᶜuₕ_phy = Geometry.UVVector.(ᶜuₕ)

        # temperature
        ᶠw = Y.f.w
        @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2
        @. ᶜts = thermo_state_ρe(Y.c.ρe_tot, Y.c, ᶜK, ᶜΦ, params)
        ᶜT = @. TD.air_temperature(params, ᶜts)
        ᶜθ = @. TD.dry_pottemp(params, ᶜts)

        # qt
        ᶜqt = Y.c.ρq_tot ./ Y.c.ρ

        # assigning to nc obj
        nc_θ[:, i] = ᶜθ
        nc_T[:, i] = ᶜT
        nc_u[:, i] = ᶜuₕ_phy.components.data.:1
        nc_qt[:, i] = ᶜqt
    end

    close(nc)

    ### write out our cubed sphere mesh
    meshfile_cc = remap_tmpdir * "/mesh_cubedsphere.g"
    write_exodus(meshfile_cc, hspace.topology)

    meshfile_rll = remap_tmpdir * "/mesh_rll.g"
    rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

    meshfile_overlap = remap_tmpdir * "/mesh_overlap.g"
    overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

    weightfile = remap_tmpdir * "/remap_weights.nc"
    remap_weights(
        weightfile,
        meshfile_cc,
        meshfile_rll,
        meshfile_overlap;
        in_type = "cgll",
        in_np = Nq,
    )

    ### remap to lat/lon
    datafile_latlon = output_dir * "/hs-remapped.nc"
    apply_remap(
        datafile_latlon,
        datafile_cc,
        weightfile,
        ["PotentialTemperature", "T", "u", "qt"],
    )

    rm(remap_tmpdir, recursive = true)

    ### load remapped data and create statistics for plots
    datafile_latlon = output_dir * "/hs-remapped.nc"
    ncdata = NCDataset(datafile_latlon, "r")
    lat = ncdata["lat"][:]
    z = ncdata["z"][:]
    time = ncdata["time"][:]

    T = ncdata["T"][:, :, :, time .> 3600 * 24 * 200]
    θ = ncdata["PotentialTemperature"][:, :, :, time .> 3600 * 24 * 200]
    u = ncdata["u"][:, :, :, time .> 3600 * 24 * 200]
    qt = ncdata["qt"][:, :, :, time .> 3600 * 24 * 200]

    u_timeave_zonalave = calc_zonalave_timeave(u)
    T_timeave_zonalave = calc_zonalave_timeave(T)
    θ_timeave_zonalave = calc_zonalave_timeave(θ)
    qt_timeave_zonalave = calc_zonalave_timeave(qt)

    T2_timeave_zonalave = calc_zonalave_variance(T)

    Plots.png(
        Plots.contourf(
            lat,
            z,
            u_timeave_zonalave',
            color = :balance,
            clim = (-30, 30),
            linewidth = 0,
            yaxis = :log,
            title = "u",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-u.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            T_timeave_zonalave',
            levels = 190:10:310,
            clim = (190, 310),
            contour_labels = true,
            yaxis = :log,
            title = "T",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-T.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            θ_timeave_zonalave',
            levels = 260:10:360,
            clim = (260, 360),
            contour_labels = true,
            yaxis = :log,
            title = "theta",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-theta.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            qt_timeave_zonalave' * 1000,
            color = :balance,
            levels = -10:2:30,
            linewidth = 0,
            yaxis = :log,
            title = "qt",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-qt.png",
    )

    Plots.png(
        Plots.contourf(
            lat,
            z,
            T2_timeave_zonalave',
            color = :balance,
            clim = (0, 40),
            linewidth = 0,
            yaxis = :log,
            title = "[T^2]",
            xlabel = "lat (deg N)",
            ylabel = "z (m)",
        ),
        output_dir * "/hs-T2.png",
    )
end

function custom_postprocessing(sol, output_dir)
    get_var(i, var) = Fields.single_field(sol.u[i], var)
    n = length(sol.u)
    #! format: off
    get_row(var) = [
        "Y.$(join(var, '.'))";;
        "$(norm(get_var(1, var), 2)) → $(norm(get_var(n, var), 2))";;
        "$(mean(get_var(1, var))) → $(mean(get_var(n, var)))";;
        "$(maximum(abs, get_var(1, var))) → $(maximum(abs, get_var(n, var)))";;
        "$(minimum(abs, get_var(1, var))) → $(minimum(abs, get_var(n, var)))";;
    ]
    #! format: on
    pretty_table(
        vcat(map(get_row, Fields.property_chains(sol.u[1]))...);
        title = "Change in Y from t = $(sol.t[1]) to t = $(sol.t[n]):",
        header = ["var", "‖var‖₂", "mean(var)", "max(∣var∣)", "min(∣var∣)"],
        alignment = :c,
    )

    anim = @animate for Y in sol.u
        if :ρθ in propertynames(Y.c)
            ᶜts = @. thermo_state_ρθ(Y.c.ρθ, Y.c, params)
        elseif :ρe_tot in propertynames(Y.c)
            grav = FT(Planet.grav(params))
            ᶜK = @. norm_sqr(C123(Y.c.uₕ) + C123(ᶜinterp(Y.f.w))) / 2
            ᶜΦ = grav .* Fields.coordinate_field(Y.c).z
            ᶜts = @. thermo_state_ρe(Y.c.ρe_tot, Y.c, ᶜK, ᶜΦ, params)
        elseif :ρe_int in propertynames(Y.c)
            ᶜts = @. thermo_state_ρe_int(Y.c.ρe_int, Y.c, params)
        end
        plot(
            vec(TD.air_temperature.(params, ᶜts)),
            vec(Fields.coordinate_field(Y.c).z ./ 1000);
            xlabel = "T [K]",
            ylabel = "z [km]",
            xlims = (190, 310),
            legend = false,
        )
    end
    Plots.mp4(anim, joinpath(output_dir, "T.mp4"), fps = 10)
end
