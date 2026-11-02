import CairoMakie as Makie
import ClimaAtmos.AtmosphericProfilesLibrary as APL

# Units of each profile variable, keyed by the profile name without its case prefix
const VARIABLE_UNITS = Dict(
    "θ_liq_ice" => "K",
    "T" => "K",
    "p" => "Pa",
    "q_tot" => "kg/kg",
    "RH" => "%",
    "u" => "m/s",
    "v" => "m/s",
    "u0" => "m/s",
    "v0" => "m/s",
    "geostrophic_u" => "m/s",
    "geostrophic_v" => "m/s",
    "geostrophic_ug" => "m/s",
    "geostrophic_vg" => "m/s",
    "subsidence" => "m/s",
    "tke" => "m²/s²",
    "tke_prescribed" => "m²/s²",
    "dTdt" => "K/s",
    "dqtdt" => "kg/kg/s",
    "radiation" => "K/s",
    "shf" => "W/m²",
    "lhf" => "W/m²",
    "inv_τ_scalar" => "1/s",
    "inv_τ_wind" => "1/s",
    "updrafts_θ_liq_ice" => "K",
    "updrafts_T" => "K",
    "updrafts_w" => "m/s",
    "updrafts_area" => "-",
)

# Units of profiles that differ from VARIABLE_UNITS
const PROFILE_UNITS = Dict(:Larcform1_RH => "-")

"""
    case_profiles(case)

`(name, profile)` pairs of the `Float64` profiles whose names start with `case_`, in
alphabetical order. Functions that return raw data instead of a profile are skipped.
"""
function case_profiles(case)
    prefix = case * "_"
    profiles = Tuple{Symbol, APL.AbstractProfile}[]
    for name in names(APL; all = true)
        startswith(string(name), prefix) || continue
        f = getfield(APL, name)
        f isa Function && hasmethod(f, Tuple{Type{Float64}}) || continue
        profile = f(Float64)
        profile isa APL.AbstractProfile && push!(profiles, (name, profile))
    end
    return profiles
end

"""
    axis_label(case, name)

`"variable [units]"` label of profile `name` from `case`.
"""
function axis_label(case, name)
    variable = chopprefix(string(name), case * "_")
    units = if haskey(PROFILE_UNITS, name)
        PROFILE_UNITS[name]
    elseif haskey(VARIABLE_UNITS, variable)
        VARIABLE_UNITS[variable]
    else
        error("Add the units of `$name` to atmospheric_profiles_library_plots.jl")
    end
    return "$variable [$units]"
end

"""
    plot_case(case; z_max = 4000, t_max = 15 * 3600, ncols = 3)

Figure with one panel per profile of `case`, over heights up to `z_max` [m] and times
up to `t_max` [s]. Height and time profiles are lines, and time-height profiles are
heatmaps. Profiles that also take the Exner function are skipped.
"""
function plot_case(case; z_max = 4000, t_max = 15 * 3600, ncols = 3)
    profiles = case_profiles(case)
    line_profiles = filter(p -> last(p) isa Union{APL.ZProfile, APL.TimeProfile}, profiles)
    heatmap_profiles = filter(p -> last(p) isa APL.TimeZProfile, profiles)

    z = range(0, z_max; length = 200)
    t = range(0, t_max; length = 200)
    z_km = z ./ 1000
    t_hours = t ./ 3600

    line_rows = cld(length(line_profiles), ncols)
    nrows = line_rows + length(heatmap_profiles)
    fig = Makie.Figure(size = (300 * ncols, 260 * nrows))
    for (i, (name, prof)) in enumerate(line_profiles)
        row, col = fldmod1(i, ncols)
        label = axis_label(case, name)
        if prof isa APL.ZProfile
            ax = Makie.Axis(
                fig[row, col];
                title = string(name),
                titlesize = 12,
                xlabel = label,
                ylabel = "z [km]",
            )
            Makie.lines!(ax, prof.(z), z_km)
        else
            ax = Makie.Axis(
                fig[row, col];
                title = string(name),
                titlesize = 12,
                xlabel = "t [h]",
                ylabel = label,
            )
            Makie.lines!(ax, t_hours, prof.(t))
        end
    end
    for (i, (name, prof)) in enumerate(heatmap_profiles)
        row = line_rows + i
        ax = Makie.Axis(
            fig[row, 1:ncols];
            title = string(name),
            titlesize = 12,
            xlabel = "t [h]",
            ylabel = "z [km]",
        )
        hm = Makie.heatmap!(ax, t_hours, z_km, prof.(t, z'))
        Makie.Colorbar(fig[row, ncols + 1], hm; label = axis_label(case, name))
    end
    return fig
end
