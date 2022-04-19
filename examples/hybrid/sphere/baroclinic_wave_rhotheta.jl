using ClimaCorePlots, Plots
using ClimaCore.DataLayouts
using NCDatasets
using ClimaCoreTempestRemap

include("baroclinic_wave_utilities.jl")

const sponge = false

# Variables required for driver.jl (modify as needed)
params = BaroclinicWaveParameterSet()
horizontal_mesh = baroclinic_wave_mesh(; params, h_elem = 4)
npoly = 4
z_max = FT(30e3)
z_elem = 10
dt_save_to_disk = FT(0) # 0 means don't save to disk
ode_algorithm = OrdinaryDiffEq.Rosenbrock23
jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

additional_cache(Y, params, dt) = merge(
    hyperdiffusion_cache(Y; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(Y, dt) : NamedTuple(),
)
function additional_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
end

center_initial_condition(local_geometry, params) =
    center_initial_condition(local_geometry, params, Val(:ρθ))

# plots in the Ullrish et al 2014 paper: surface pressure, 850 temperature and vorticity at day 8 and day 10
function paperplots(sol, output_dir, p, nlat, nlon)
    (; comms_ctx, ᶜts, ᶜp, params) = p
    days = [8, 10]

    # obtain pressure, temperature, and vorticity at cg points;
    # and remap them onto lat lon
    for day in days
        iu = findall(x -> x == day * 24 * 3600, sol.t)[1]
        Y = sol.u[iu]
        # compute pressure, temperature, vorticity
        @. ᶜts = thermo_state_ρθ(Y.c.ρθ, Y.c, params)
        @. ᶜp = TD.air_pressure(ᶜts)
        ᶜT = @. TD.air_temperature(ᶜts)
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
