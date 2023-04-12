using NCDatasets
import ClimaCoreSpectra: power_spectrum_1d, power_spectrum_2d
using Statistics

function generate_paperplots_dry_baro_wave(fig_dir, nc_files)
    for day in [8, 10, 100]
        ncfile = filter(x -> endswith(x, "day$day.0.nc"), nc_files)
        if !isempty(ncfile)
            nt = NCDataset(ncfile[1], "r") do nc
                lon = nc["lon"][:]
                lat = nc["lat"][:]
                z = nc["z"][:]
                p = nc["pressure"][:]
                T = nc["temperature"][:]
                vort = nc["vorticity"][:]
                (; lon, lat, z, p, T, vort)
            end
            (; lon, lat, z, p, T, vort) = nt

            latidx = findall(x -> x >= 0, lat)
            lonidx = findall(x -> 0 <= x <= 240, lon)

            FT = eltype(vort)

            fig = []
            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    p[lonidx, latidx, 1, 1]',
                    color = :rainbow,
                    title = "pressure day $day z $(round(z[1])) m",
                ),
            )
            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    T[lonidx, latidx, 1, 1]',
                    color = :rainbow,
                    levels = 220:10:310,
                    title = "temperature day $day z $(round(z[1])) m",
                ),
            )
            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    vort[lonidx, latidx, 1, 1]' .* FT(1e5),
                    color = :balance,
                    title = "vorticity day $day z $(round(z[1])) m",
                ),
            )
            png(
                Plots.plot(fig..., layout = (3, 1), size = (600, 800)),
                fig_dir * "/dbw_day$day.png",
            )
        else
            @warn "day$day.0.nc DOES NOT EXIST!!!"
        end
    end
end

function generate_paperplots_moist_baro_wave(fig_dir, nc_files)
    for day in [8, 10, 100]
        ncfile = filter(x -> endswith(x, "day$day.0.nc"), nc_files)
        if !isempty(ncfile)
            nt = NCDataset(ncfile[1], "r") do nc
                lon = nc["lon"][:]
                lat = nc["lat"][:]
                z = nc["z"][:]
                p = nc["pressure"][:]
                T = nc["temperature"][:]
                vort = nc["vorticity"][:]
                qt = nc["qt"][:]
                cloud_water = nc["cloud_liquid"][:] + nc["cloud_ice"][:]
                water_vapor = nc["water_vapor"][:]
                rho = nc["rho"][:]
                w = nc["w"][:]
                u = nc["u"][:]
                v = nc["v"][:]
                (;
                    lon,
                    lat,
                    z,
                    p,
                    T,
                    vort,
                    qt,
                    cloud_water,
                    water_vapor,
                    rho,
                    w,
                    u,
                    v,
                )
            end
            (;
                lon,
                lat,
                z,
                p,
                T,
                vort,
                qt,
                cloud_water,
                water_vapor,
                rho,
                w,
                u,
                v,
            ) = nt

            vert_intg_cloud_water =
                sum(cloud_water .* rho, dims = 3) ./ sum(rho, dims = 3)
            vert_intg_water_vapor =
                sum(water_vapor .* rho, dims = 3) ./ sum(rho, dims = 3)

            latidx = findall(x -> x >= 0, lat)
            lonidx = findall(x -> 0 <= x <= 240, lon)

            FT = eltype(vort)

            fig = []
            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    p[lonidx, latidx, 1, 1]',
                    color = :rainbow,
                    title = "pressure day $day z $(round(z[1])) m",
                ),
            )

            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    T[lonidx, latidx, 1, 1]',
                    color = :rainbow,
                    levels = 220:10:310,
                    title = "temperature day $day z $(round(z[1])) m",
                ),
            )

            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    vort[lonidx, latidx, 1, 1]' * FT(1e5),
                    color = :balance,
                    title = "vorticity day $day z $(round(z[1])) m",
                ),
            )

            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    vort[lonidx, latidx, 3, 1]' * FT(1e5),
                    color = :balance,
                    title = "vorticity day $day z $(round(z[3])) m",
                ),
            )

            png(
                Plots.plot(fig..., layout = (4, 1), size = (600, 1000)),
                fig_dir * "/mbw_basic_day$day.png",
            )

            fig = []

            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    w[lonidx, latidx, 1, 1]',
                    color = :balance,
                    title = "w day $day z $(round(z[1])) m",
                ),
            )

            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    w[lonidx, latidx, 3, 1]',
                    color = :balance,
                    title = "w day $day z $(round(z[3])) m",
                ),
            )

            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    u[lonidx, latidx, 3, 1]',
                    color = :balance,
                    title = "u day $day z $(round(z[1])) m",
                ),
            )

            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    u[lonidx, latidx, 1, 1]',
                    color = :balance,
                    title = "u day $day z $(round(z[3])) m",
                ),
            )

            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    v[lonidx, latidx, 3, 1]',
                    color = :balance,
                    title = "v day $day z $(round(z[1])) m",
                ),
            )

            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    v[lonidx, latidx, 1, 1]',
                    color = :balance,
                    title = "v day $day z $(round(z[3])) m",
                ),
            )

            png(
                Plots.plot(fig..., layout = (3, 2), size = (800, 800)),
                fig_dir * "/mbw_velocity_day$day.png",
            )

            fig = []

            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    qt[lonidx, latidx, 1, 1]' * FT(1e3),
                    color = :balance,
                    title = "qt day $day z $(round(z[1])) m",
                ),
            )

            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    vert_intg_cloud_water[lonidx, latidx, 1, 1]' * FT(1e3),
                    color = :balance,
                    title = "cloud water (g/kg) day $day z $(round(z[1])) m",
                ),
            )

            push!(
                fig,
                Plots.contourf(
                    lon[lonidx],
                    lat[latidx],
                    vert_intg_water_vapor[lonidx, latidx, 1, 1]' * FT(1e3),
                    color = :balance,
                    title = "water vapor (g/kg) day $day z $(round(z[1])) m",
                ),
            )

            png(
                Plots.plot(fig..., layout = (3, 1), size = (600, 800)),
                fig_dir * "/mbw_moisture_day$day.png",
            )

            # Spectrum calculation
            mass_weight = ones(FT, 1) # only 1 level

            # Compute 2d spectrum of u-component
            u_2d_spectrum, wave_numbers, spherical, mesh_info =
                power_spectrum_2d(FT, u[:, :, 1, 1], mass_weight) # use first level, for default CI config
            # size(u) = (180, 90, 10, 1) (10 vert levels for CI default config, 1 component)

            # Compute 2d spectrum of v-component
            v_2d_spectrum, wave_numbers, spherical, mesh_info =
                power_spectrum_2d(FT, v[:, :, 1, 1], mass_weight) # use first level

            spectrum_2d = 0.5 .* (u_2d_spectrum + v_2d_spectrum)

            fig = []

            push!(
                fig,
                Plots.contourf(
                    collect(0:1:(mesh_info.num_fourier))[:],
                    collect(0:1:(mesh_info.num_spherical))[:],
                    (spectrum_2d[:, :, 1])', # dimensions of spectrum_2d are [m,n,k], where in this case there is only one vert level
                    xlabel = "m",
                    ylabel = "n",
                    color = :balance,
                    title = "2d spectrum day $day z $(round(z[1])) m",
                ),
            )

            push!(
                fig,
                Plots.plot(
                    collect(0:1:(mesh_info.num_fourier))[:], # plot against the zonal wavenumber, m
                    sum(spectrum_2d[:, :, 1], dims = 2), # sum along the total wavenumber, n
                    yaxis = :log,
                    xlabel = "zonal wavenumber (m)",
                    ylabel = "KE spectrum",
                    color = :balance,
                    title = "zonal wavenumber KE spectrum day $day z $(round(z[1])) m",
                ),
            )

            push!(
                fig,
                Plots.plot(
                    collect(0:1:(mesh_info.num_spherical))[:], # plot against the total wavenumber, n
                    (log.(sum(spectrum_2d[:, :, 1], dims = 1)))', # sum along the zonal wavenumber, m
                    # yaxis = :log,
                    xlabel = "total wavenumber (n)",
                    ylabel = "KE spectrum (log)",
                    color = :balance,
                    title = "total wavenumber KE spectrum day $day z $(round(z[1])) m",
                ),
            )

            png(
                Plots.plot(fig..., layout = (3, 1), size = (600, 800)),
                fig_dir * "/mbw_spectrum_day$day.png",
            )

        else
            @warn "day$day.0.nc DOES NOT EXIST!!!"
        end
    end
end

# plots in the held-suarez paper: https://journals.ametsoc.org/view/journals/bams/75/10/1520-0477_1994_075_1825_apftio_2_0_co_2.xml?tab_body=pdf
calc_zonalave_timeave(x) =
    dropdims(dropdims(mean(mean(x, dims = 1), dims = 4), dims = 4), dims = 1)
calc_zonalave_variance(x) = calc_zonalave_timeave((x .- mean(x, dims = 1)) .^ 2)

function generate_paperplots_held_suarez(fig_dir, nc_files; moist)
    days = map(x -> parse(Int, split(basename(x), ".")[1][4:end]), nc_files)
    # filter days for plotting:
    # if simulated time is more than 200 days, days after day 200 will be collected and used for longterm average
    # if simulated time is less than 200 days, the last day will be used to create a napshot
    if maximum(days) > 200
        filter!(x -> x > 200, days)
    else
        filter!(x -> x == maximum(days), days)
    end

    # identify nc files for plotting 
    nc_file = []
    for day in unique(days)
        push!(nc_file, filter(x -> occursin(string(day), x), nc_files)...)
    end

    println("files to generate the plots:")
    println.(nc_file)

    # collect data from all nc files and combine data from different time step into one array
    for (i, ifile) in enumerate(nc_file)
        nc = NCDataset(ifile, "r")
        if i == 1
            global lat = nc["lat"][:]
            global z = nc["z"][:]
            global u = nc["u"][:]
            global T = nc["temperature"][:]
            global θ = nc["potential_temperature"][:]
            if moist
                global qt = nc["qt"][:]
            end
        else
            u = cat(u, nc["u"][:], dims = 4)
            T = cat(T, nc["temperature"][:], dims = 4)
            θ = cat(θ, nc["potential_temperature"][:], dims = 4)
            if moist
                qt = cat(qt, nc["qt"][:], dims = 4)
            end
        end
    end

    # compute longterm zonal mean statistics
    u_timeave_zonalave = calc_zonalave_timeave(u)
    T_timeave_zonalave = calc_zonalave_timeave(T)
    θ_timeave_zonalave = calc_zonalave_timeave(θ)
    T2_timeave_zonalave = calc_zonalave_variance(T)
    @info T2_timeave_zonalave

    if moist
        qt_timeave_zonalave = calc_zonalave_timeave(qt)
    end

    # plot!!
    fig = []
    push!(
        fig,
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
    )

    push!(
        fig,
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
    )

    push!(
        fig,
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
    )

    push!(
        fig,
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
    )

    if moist
        push!(
            fig,
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
        )
    end

    if !moist
        png(
            Plots.plot(fig..., layout = (2, 2), size = (800, 800)),
            fig_dir * "/diagnostics.png",
        )
    else
        png(
            Plots.plot(fig..., layout = (2, 3), size = (1200, 800)),
            fig_dir * "/diagnostics.png",
        )
    end
end
