
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
function profile_animation(sol, output_dir)
    # Column animations
    Y0 = first(sol.u)
    for prop_chain in Fields.property_chains(Y0)
        var_name = processed_varname(prop_chain)
        var_space = axes(Fields.single_field(Y0, prop_chain))
        Ni, Nj, _, _, Nh = size(ClimaCore.Spaces.local_geometry_data(var_space))
        n_columns = Nh * Nj * Ni # TODO: is this correct?
        @info "Creating profile animation with `n_columns` = $n_columns, for `$var_name`"
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
                ϕ_col_std .+= sqrt.((vec(ϕ_col) .- ϕ_col_ave) .^ 2 ./ n_columns)
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
        Plots.mp4(anim, joinpath(output_dir, "profile_$var_name.mp4"), fps = 5)
    end
end

function contour_animations(sol, output_dir)
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
        Plots.mp4(anim, joinpath(output_dir, "contour_$var_name.mp4"), fps = 5)
    end
end

function postprocessing(sol, output_dir)
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

    # contour_animations(sol, output_dir) # For generic contours:

    anim = Plots.@animate for Y in sol.u
        ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
        Plots.plot(ᶜv, level = 3, clim = (-6, 6))
    end
    Plots.mp4(anim, joinpath(output_dir, "v.mp4"), fps = 5)

    prop_chains = Fields.property_chains(sol.u[1])
    if any(pc -> pc == (:c, :ρq_tot), prop_chains)
        anim = Plots.@animate for Y in sol.u
            ᶜq_tot = Y.c.ρq_tot ./ Y.c.ρ
            Plots.plot(ᶜq_tot .* FT(1e3), level = 3, clim = (0, 1))
        end
        Plots.mp4(anim, joinpath(output_dir, "contour_q_tot.mp4"), fps = 5)
    else
        @info "Moisture not found. property_chains: `$(prop_chains)`"
    end

    profile_animation(sol, output_dir)
end

# plots in the Ullrish et al 2014 paper: surface pressure, 850 temperature and vorticity at day 8 and day 10
function paperplots_baro_wave_ρθ(sol, output_dir, p, nlat, nlon)
    (; comms_ctx, ᶜts, ᶜp, params) = p
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
        Spaces.weighted_dss!(ᶜvort, comms_ctx)

        # space info to generate nc raw data
        cspace = axes(Y.c)
        hspace = cspace.horizontal_space
        Nq =
            Spaces.Quadratures.degrees_of_freedom(cspace.horizontal_space.quadrature_style,)

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
    (; comms_ctx, ᶜts, ᶜp, params, ᶜK, ᶜΦ) = p
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
        @. ᶜts = thermo_state_ρe(Y.c.ρe, Y.c, ᶜK, ᶜΦ, params)
        @. ᶜp = TD.air_pressure(params, ᶜts)
        ᶜT = @. TD.air_temperature(params, ᶜts)
        curl_uh = @. curlₕ(Y.c.uₕ)
        ᶜvort = Geometry.WVector.(curl_uh)
        Spaces.weighted_dss!(ᶜvort, comms_ctx)

        # space info to generate nc raw data
        cspace = axes(Y.c)
        hspace = cspace.horizontal_space
        Nq =
            Spaces.Quadratures.degrees_of_freedom(cspace.horizontal_space.quadrature_style,)

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
