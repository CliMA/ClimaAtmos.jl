using NCDatasets
import ClimaCoreSpectra: power_spectrum_1d, power_spectrum_2d
using Statistics

"""
    create_plot!(fig; X,Y,Z, args...)

Create and save a Makie plot object, which consists 
of either a line-plot (X,Y) or a surface plot ((X,Y),Z),
Basic functionality is included, this can be extended with
`args` compatible with the Makie plot package.  
"""
function create_plot!(
    fig::Figure;
    X = lon,
    Y = lat,
    Z = nothing,
    p_loc::Tuple = (1, 1),
    title = "",
    xlabel = "Longitude",
    ylabel = "Latitude",
    xscale = identity,
    yscale = identity,
    linewidth = 6,
    level::Int = 1,
    timeind::Int = 1,
)
    if Z == nothing
        generic_axis = fig[p_loc[1], p_loc[2]] = GridLayout()
        Axis(generic_axis[1, 1]; title, xlabel, ylabel, xscale, yscale)
        CairoMakie.lines!(X, Y; title, linewidth)
    else
        generic_axis = fig[p_loc[1], p_loc[2]] = GridLayout() # Generic Axis Layout
        Axis(generic_axis[1, 1]; title, xlabel, ylabel, yscale) # Plot specific attributes
        # custom_levels is a workaround for plotting constant fields with CairoMakie
        custom_levels =
            minimum(Z) ≈ maximum(Z) ? (minimum(Z):0.1:(minimum(Z) + 0.2)) : 25
        generic_plot = CairoMakie.contourf!(X, Y, Z, levels = custom_levels) # Add plot contents
        Colorbar(generic_axis[1, 2], generic_plot)
    end
end

"""
    generate_empty_figure(;resolution, bgcolor, fontsize)

Generate an empty `Makie.Figure` object, with specified
`resolution`, background color `bgcolor` and `fontsize`.
Returns the empty figure which can then be populated with 
the `create_plot!(...)`. 
"""
function generate_empty_figure(;
    resolution::Tuple = (4098, 2048),
    bgcolor::Tuple = (0.98, 0.98, 0.98),
    fontsize = 36,
)
    fig = Figure(;
        backgroundcolor = RGBf(bgcolor[1], bgcolor[2], bgcolor[3]),
        resolution,
        fontsize,
    )
    return fig
end

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

            fig = generate_empty_figure()
            create_plot!(
                fig;
                X = lon[lonidx],
                Y = lat[latidx],
                Z = p[lonidx, latidx, 1, 1],
                title = "Pressure: Day $day; z $(round(z[1])) m",
            )
            create_plot!(
                fig;
                p_loc = (2, 1),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = T[lonidx, latidx, 1, 1],
                title = "Temperature: Day $day; z $(round(z[1])) m",
            )
            create_plot!(
                fig;
                p_loc = (3, 1),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = vort[lonidx, latidx, 1, 1] .* FT(1e5),
                title = "Vorticity(× 10⁵): Day $day; z $(round(z[1])) m",
            )
            CairoMakie.save(fig_dir * "/dbw_day$day.png", fig)
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
                z_sfc = nc["sfc_elevation"][:]
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
                    z_sfc,
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
                z_sfc,
            ) = nt

            vert_intg_cloud_water =
                sum(cloud_water .* rho, dims = 3) ./ sum(rho, dims = 3)
            vert_intg_water_vapor =
                sum(water_vapor .* rho, dims = 3) ./ sum(rho, dims = 3)

            latidx = findall(x -> x >= 0, lat)
            lonidx = findall(x -> 0 <= x <= 240, lon)

            FT = eltype(vort)

            # Basic
            fig = generate_empty_figure()

            create_plot!(
                fig;
                p_loc = (1, 1),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = p[lonidx, latidx, 1, 1],
                title = "Pressure: Day $day; z $(round(z[1])) m",
            )
            create_plot!(
                fig;
                p_loc = (2, 1),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = T[lonidx, latidx, 1, 1],
                title = "Temperature: Day $day; z $(round(z[1])) m",
            )
            create_plot!(
                fig;
                p_loc = (3, 1),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = vort[lonidx, latidx, 1, 1] .* FT(1e5),
                title = "Vorticity(×10⁵): Day $day; z $(round(z[1])) m",
            )
            CairoMakie.save(fig_dir * "/mbw_day$day.png", fig)

            fig = generate_empty_figure()
            create_plot!(
                fig;
                p_loc = (1, 1),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = w[lonidx, latidx, 1, 1],
                title = "Vertical Velocity: Day $day; z $(round(z[1])) m",
            )
            create_plot!(
                fig;
                p_loc = (2, 1),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = w[lonidx, latidx, 3, 1],
                title = "Vertical Velocity: Day $day; z $(round(z[3])) m",
            )
            create_plot!(
                fig;
                p_loc = (1, 2),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = u[lonidx, latidx, 1, 1],
                title = "Zonal Velocity: Day $day; z $(round(z[1])) m",
            )
            create_plot!(
                fig;
                p_loc = (2, 2),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = u[lonidx, latidx, 3, 1],
                title = "Zonal Velocity: Day $day; z $(round(z[3])) m",
            )
            create_plot!(
                fig;
                p_loc = (1, 3),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = v[lonidx, latidx, 1, 1],
                title = "Meridional Velocity: Day $day; z $(round(z[1])) m",
            )
            create_plot!(
                fig;
                p_loc = (2, 3),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = v[lonidx, latidx, 3, 1],
                title = "Meridional Velocity: Day $day; z $(round(z[3])) m",
            )
            CairoMakie.save(fig_dir * "/mbw_velocity_day$day.png", fig)

            # Moisture 
            fig = generate_empty_figure()
            create_plot!(
                fig;
                p_loc = (1, 1),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = qt[lonidx, latidx, 1, 1] .* 1e3,
                title = "Tot. Specific Moisture (qt×10³): Day $day; z $(round(z[1])) m",
            )
            create_plot!(
                fig;
                p_loc = (2, 1),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = vert_intg_cloud_water[lonidx, latidx, 1, 1] .* 1e3,
                title = "Vert Int Cloud Water [g/kg]: Day $day; z $(round(z[1])) m",
            )
            create_plot!(
                fig;
                p_loc = (3, 1),
                X = lon[lonidx],
                Y = lat[latidx],
                Z = vert_intg_water_vapor[lonidx, latidx, 1, 1] .* 1e3,
                title = "Vert Int Cloud Vapor [g/kg]: Day $day; z $(round(z[1])) m",
            )
            CairoMakie.save(fig_dir * "/mbw_moisture_day$day.png", fig)

            # Spectrum calculation
            mass_weight = ones(FT, 1) # only 1 level

            # Compute 2d spectrum of u-component
            u_2d_spectrum, wave_numbers, spherical, mesh_info =
                power_spectrum_2d(FT, u[:, :, 1, 1], mass_weight) # use first level, for default CI config
            # size(u) = (180, 90, 10, 1) (10 vert levels for CI default config, 1 component)

            # Compute 2d spectrum of v-component
            v_2d_spectrum, wave_numbers, spherical, mesh_info =
                power_spectrum_2d(FT, v[:, :, 1, 1], mass_weight) # use first level

            orography_spectrum, wave_numbers, spherical, mesh_info =
                power_spectrum_2d(FT, z_sfc[:, :, 1, 1], mass_weight) # use first level, for default CI config

            spectrum_2d = 0.5 .* (u_2d_spectrum + v_2d_spectrum)

            # Spectra
            fig = generate_empty_figure()
            create_plot!(
                fig;
                p_loc = (1, 1),
                X = collect(0:1:(mesh_info.num_fourier))[:],
                Y = collect(0:1:(mesh_info.num_spherical))[:],
                Z = (spectrum_2d[:, :, 1]),
                xlabel = "",
                ylabel = "",
                title = "2D KE (horz) Spectrum: Day $day; z $(round(z[1])) m",
            )
            create_plot!(
                fig;
                p_loc = (2, 1),
                X = collect(0:1:(mesh_info.num_fourier))[2:(end - 1)], # plot against the zonal wavenumber, m
                Y = sum(spectrum_2d[:, :, 1], dims = 2)[2:(end - 1), 1], # sum along the total wavenumber, n
                xlabel = "",
                ylabel = "",
                title = "Σₙ KE (horz) Spectrum: Day $day; z $(round(z[1])) m",
            )
            create_plot!(
                fig;
                p_loc = (3, 1),
                X = collect(0:1:(mesh_info.num_spherical))[2:(end - 1)], # plot against the total wavenumber, n
                Y = log.(sum(spectrum_2d[:, :, 1], dims = 1))'[2:(end - 1), 1], # sum along the zonal wavenumber, m
                xlabel = "",
                ylabel = "",
                title = "Zonal Wavenumber (m) KE (horz) Spectrum: Day $day; z $(round(z[1])) m",
            )
            CairoMakie.save(fig_dir * "/mbw_spectrum_day$day.png", fig)
        else
            @warn "day$day.0.nc DOES NOT EXIST!!!"
        end
    end
end

function generate_elevation_spectra(fig_dir, nc_files)
    ncfile = filter(x -> endswith(x, "day0.0.nc"), nc_files)
    if !isempty(ncfile)
        nt = NCDataset(ncfile[1], "r") do nc
            z_sfc = nc["sfc_elevation"][:]
            lon = nc["lon"][:]
            lat = nc["lat"][:]
            (; lon, lat, z_sfc)
        end
        (; lon, lat, z_sfc) = nt

        FT = eltype(lat)
        # Spectrum calculation
        mass_weight = ones(FT, 1) # only 1 level

        orography_spectrum, wave_numbers, spherical, mesh_info =
            power_spectrum_2d(FT, z_sfc[:, :, 1, 1], mass_weight) # use first level, for default CI config

        fig = generate_empty_figure()
        create_plot!(
            fig;
            p_loc = (1, 1),
            X = collect(0:1:(mesh_info.num_fourier))[2:end], # plot against the zonal wavenumber, m
            Y = log.(sum(orography_spectrum[:, :, 1], dims = 2))[2:end], # sum along the total wavenumber, n
            title = "Diagnostic: Surface Elevation Spectrum",
            xlabel = "Zonal wavenumber",
            ylabel = "Log surface elevation spectrum",
        )

        CairoMakie.save(fig_dir * "/surface_elev_spectrum.png", fig)

    else
        @warn "day0.0.nc DOES NOT EXIST!!!"
    end
end

# plots in the held-suarez paper: https://journals.ametsoc.org/view/journals/bams/75/10/1520-0477_1994_075_1825_apftio_2_0_co_2.xml?tab_body=pdf
calc_zonalave_timeave(x) =
    dropdims(dropdims(mean(mean(x, dims = 1), dims = 4), dims = 4), dims = 1)
calc_zonalave_variance(x) = calc_zonalave_timeave((x .- mean(x, dims = 1)) .^ 2)
calc_zonalave_covariance(x, y) =
    calc_zonalave_timeave((x .- mean(x, dims = 1)) .* (y .- mean(y, dims = 1)))

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
            global lon = nc["lon"][:]
            global z = nc["z"][:]
            global u = nc["u"][:]
            global v = nc["v"][:]
            global w = nc["w"][:]
            global T = nc["temperature"][:]
            global θ = nc["potential_temperature"][:]
            if moist
                global qt = nc["qt"][:]
            end
        else
            u = cat(u, nc["u"][:], dims = 4)
            v = cat(v, nc["v"][:], dims = 4)
            w = cat(w, nc["w"][:], dims = 4)
            T = cat(T, nc["temperature"][:], dims = 4)
            θ = cat(θ, nc["potential_temperature"][:], dims = 4)
            if moist
                qt = cat(qt, nc["qt"][:], dims = 4)
            end
        end
    end

    # compute longterm zonal mean statistics
    u_timeave_zonalave = calc_zonalave_timeave(u)
    v_timeave_zonalave = calc_zonalave_timeave(v)
    w_timeave_zonalave = calc_zonalave_timeave(w)
    T_timeave_zonalave = calc_zonalave_timeave(T)
    θ_timeave_zonalave = calc_zonalave_timeave(θ)
    T2_timeave_zonalave = calc_zonalave_variance(T)
    u2_timeave_zonalave = calc_zonalave_variance(u)
    v2_timeave_zonalave = calc_zonalave_variance(v)
    w2_timeave_zonalave = calc_zonalave_variance(w)
    uT_cov_timeave_zonalave = calc_zonalave_covariance(u, T)
    wT_cov_timeave_zonalave = calc_zonalave_covariance(w, T)

    if moist
        qt_timeave_zonalave = calc_zonalave_timeave(qt)
        uqt_cov_timeave_zonalave = calc_zonalave_covariance(u, qt)
        wqt_cov_timeave_zonalave = calc_zonalave_covariance(w, qt)
    end

    # plot!!
    fig = generate_empty_figure(; resolution = (4098, 4098))
    create_plot!(
        fig;
        p_loc = (1, 1),
        X = lat,
        Y = z,
        Z = u_timeave_zonalave,
        yscale = log10,
        title = "Zonal Velocity",
        xlabel = "Latitude",
        ylabel = "Altitude[m]",
    ) # TODO Make lims (-30:30)
    create_plot!(
        fig;
        p_loc = (1, 2),
        X = lat,
        Y = z,
        Z = T_timeave_zonalave,
        yscale = log10,
        title = "Temperature T",
        xlabel = "Latitude",
        ylabel = "Altitude[m]",
    ) # TODO Make lims (190:10:310)
    create_plot!(
        fig;
        p_loc = (1, 3),
        X = lat,
        Y = z,
        Z = θ_timeave_zonalave,
        yscale = log10,
        title = "Potential Temperature θ",
        xlabel = "Latitude",
        ylabel = "Altitude[m]",
    ) # TODO Make lims (190:10:310)
    create_plot!(
        fig;
        p_loc = (2, 1),
        X = lat,
        Y = z,
        Z = u2_timeave_zonalave,
        yscale = log10,
        title = "Zonal Vel Variance u′²",
        xlabel = "Latitude",
        ylabel = "Altitude[m]",
    ) # TODO Make lims (0:40)
    create_plot!(
        fig;
        p_loc = (2, 2),
        X = lat,
        Y = z,
        Z = T2_timeave_zonalave,
        yscale = log10,
        title = "Temp Variance T′²",
        xlabel = "Latitude",
        ylabel = "Altitude[m]",
    ) # TODO Make lims (0:40)
    create_plot!(
        fig;
        p_loc = (3, 1),
        X = lat,
        Y = z,
        Z = uT_cov_timeave_zonalave,
        yscale = log10,
        title = "Covariance u′T′",
        xlabel = "Latitude",
        ylabel = "Altitude[m]",
    ) # TODO Make lims (0:40)
    create_plot!(
        fig;
        p_loc = (3, 2),
        X = lon,
        Y = lat,
        Z = T[:, :, 1, end],
        title = "Temperature: Day $(days[end]); z $(round(z[1])) m",
        xlabel = "Longitude",
        ylabel = "Latitude",
    ) # TODO Make lims (0:40)
    create_plot!(
        fig;
        p_loc = (3, 3),
        X = lat,
        Y = z,
        Z = wT_cov_timeave_zonalave,
        yscale = log10,
        title = "Covariance w′T′",
        xlabel = "Latitude",
        ylabel = "Altitude[m]",
    ) # TODO Make lims (0:40)
    if moist
        create_plot!(
            fig;
            p_loc = (4, 1),
            X = lat,
            Y = z,
            Z = qt_timeave_zonalave .* 1e3,
            yscale = log10,
            title = "qt × 10³ (Zonal-Time-Averaged)",
            xlabel = "Latitude",
            ylabel = "Altitude[m]",
        ) # TODO Make lims (0:40)
        create_plot!(
            fig;
            p_loc = (4, 2),
            X = lon,
            Y = lat,
            Z = qt[:, :, 1, end] .* 1e3,
            title = "qt × 10³: Day $(days[end]); z $(round(z[1])) m",
            xlabel = "Longitude",
            ylabel = "Latitude",
        ) # TODO Make lims (0:40)
        create_plot!(
            fig;
            p_loc = (4, 3),
            X = lat,
            Y = z,
            Z = uqt_cov_timeave_zonalave,
            yscale = log10,
            title = "Covariance u′q′ (Zonal-Time-Averaged)",
            xlabel = "Latitude",
            ylabel = "Altitude[m]",
        ) # TODO Make lims (0:40)
        create_plot!(
            fig;
            p_loc = (2, 3),
            X = lat,
            Y = z,
            Z = wqt_cov_timeave_zonalave,
            yscale = log10,
            title = "Covariance w′qt′",
            xlabel = "Latitude",
            ylabel = "Altitude[m]",
        ) # TODO Make lims (0:40)
    end

    CairoMakie.save(fig_dir * "/diagnostics.png", fig)
end
