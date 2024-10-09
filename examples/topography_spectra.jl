using LazyArtifacts
using ClimaAtmos.AtmosArtifacts
import ClimaAtmos as CA
import ClimaCore as CC
import ClimaComms
import ClimaParams as CP
import ClimaCore.Quadratures
using ClimaUtilities: SpaceVaryingInputs
using ClimaCore.Hypsography: diffuse_surface_elevation!
using ClimaCore: Remapping, Geometry, Operators, Spaces
using Plots, ClimaCorePlots

import ClimaCoreSpectra: power_spectrum_2d

const AA = AtmosArtifacts

function generate_spaces(; h_elem = 16, diffiter = 32)
    FT = Float32
    param_dict = CP.create_toml_dict(FT)
    params = CP.get_parameter_values(param_dict, ["planet_radius"])
    cubed_sphere_mesh =
        CA.cubed_sphere_mesh(; radius = params.planet_radius, h_elem)
    quad = Quadratures.GLL{4}()
    comms_ctx = ClimaComms.SingletonCommsContext()
    h_space = CA.make_horizontal_space(cubed_sphere_mesh, quad, comms_ctx, true)
    Δh_scale = Spaces.node_horizontal_length_scale(h_space)
    @assert h_space isa CC.Spaces.SpectralElementSpace2D
    coords = CC.Fields.coordinate_field(h_space)
    target_field = CC.Fields.zeros(h_space)
    elev_from_file = SpaceVaryingInputs.SpaceVaryingInput(
        AA.earth_orography_file_path(; context = comms_ctx),
        "z",
        h_space,
    )
    parent(elev_from_file) .=
        ifelse.(parent(elev_from_file) .< FT(0), FT(0), parent(elev_from_file))
    diffuse_surface_elevation_biharmonic!(
        elev_from_file,
        κ = FT((Δh_scale)^4 / 1e3),
        maxiter = diffiter,
    )
    #diffuse_surface_elevation!(elev_from_file, κ=FT((Δh_scale)^2/100), dt=FT(1), maxiter=diffiter)
    parent(elev_from_file) .=
        ifelse.(parent(elev_from_file) .< FT(0), FT(0), parent(elev_from_file))
    return elev_from_file
end

function diffuse_surface_elevation_biharmonic!(
    f::CC.Fields.Field;
    κ::T = 5e8,
    maxiter::Int = 100,
) where {T}
    if eltype(f) <: Real
        f_z = f
    elseif eltype(f) <: Geometry.ZPoint
        f_z = f.z
    end
    # Define required ops
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()
    # Create dss buffer
    ghost_buffer = (bf = Spaces.create_dss_buffer(f_z),)
    # Apply smoothing
    χf = @. wdiv(grad(f_z))
    Spaces.weighted_dss!(χf, ghost_buffer.bf)
    @. χf = wdiv(grad(χf))
    for iter in 1:maxiter
        # Euler steps
        if iter ≠ 1
            @. χf = wdiv(grad(f_z))
            Spaces.weighted_dss!(χf, ghost_buffer.bf)
            @. χf = wdiv(grad(χf))
        end
        Spaces.weighted_dss!(χf, ghost_buffer.bf)
        @. f_z -= κ * χf
    end
    # Return mutated surface elevation profile
    return f
end

function remap_to_array(var, hcoords)
    remapper = Remapping.Remapper(CC.axes(var), hcoords)
    orog = Array(Remapping.interpolate(remapper, var))
end

function gen_spectra(var)
    # Remap onto 1° resolution horizontal coordinates. 
    longpts = range(-180.0, 180.0, 360)
    latpts = range(-90.0, 90.0, 180)
    hcoords =
        [Geometry.LatLongPoint(lat, long) for long in longpts, lat in latpts]
    var = remap_to_array(var, hcoords)
    # Use ClimaCoreSpectra to generate power spectra for orography data. 
    len1 = size(var)[1]
    len2 = size(var)[2]
    @assert len1 == 2len2
    FT = eltype(var)
    mass_weight = FT(1) # No weighting applied 
    spectrum_data, wave_numbers, _spherical, mesh_info =
        power_spectrum_2d(FT, var, mass_weight)
    power_spectrum =
        dropdims(sum(spectrum_data, dims = 1), dims = 1)[begin:(end - 1), :]
    w_numbers = collect(0:1:(mesh_info.num_spherical - 1))
    @info "Returning wavenumber array (spherical) and orography power spectrum"
    return w_numbers, power_spectrum
end

function generate_all_spectra(; h_elem = 32)
    fig = Plots.plot()
    for ii in (0, 4, 8, 16, 32, 64, 128, 256)
        var = generate_spaces(; h_elem, diffiter = ii)
        sph_wn, psd = gen_spectra(var)
        fig = Plots.plot!(
            sph_wn[2:end],
            log.(psd)[2:end];
            lw = 2,
            xlabel = "Spherical wavenumber",
            ylabel = "log(elevation power spectra)",
            label = "$(ii)×",
            xaxis = :log,
            xticks = (
                [2 4 8 16 32 64 128],
                ["2", "4", "8", "16", "32", "64", "128"],
            ),
        )
    end
    fig = Plots.plot!(;
        legend = :bottomleft,
        grid = :off,
        fontsize = 20,
        title = "Biharmonic smoothing",
    )
    return fig, var
end
