using CairoMakie
import CairoMakie.Makie: ColorSchemes



"""
    spliced_cmap(left_cs, right_cs, lo, hi; mid, kwargs...)

Create a spliced colormap from two source colormaps `left_cs` and `right_cs`, meeting at `mid`.

This function constructs a `Makie.ColorGradient` that maps `left_cs` to the interval `[lo, mid]` 
and `right_cs` to `[mid, hi]`. The colormaps are oriented such that they diverge from `mid` 
(lightest color) towards `lo` and `hi` (darkest colors).

# Arguments
- `left_cs`: Colormap (or symbol) for the left side (values < `mid`).
- `right_cs`: Colormap (or symbol) for the right side (values > `mid`).
- `lo`, `hi`: The data range covered by the full colormap.

# Keyword Arguments
- `mid`: The data value where the two colormaps meet (default: 0).
- `symmetrize_color_ranges`: If `true`, ensures that color intensity is proportional to the distance from `mid`.
  If the range `[lo, hi]` is asymmetric around `mid`, the side with the smaller range will only use 
  a fraction of its colormap (up to the equivalent intensity of the other side).
- `categorical`: If `true`, returns a categorical color gradient (default: `true`).
- `n`: Number of colors to sample per side (default: 5 if categorical, 100 otherwise).
    - for more fine-grained control, use `n_left` and `n_right` to control the number of colors per side.

# Returns
- A `Makie.ColorGradient` object.
"""
function spliced_cmap(
    left_cs, right_cs, lo, hi;
    mid = 0.0,
    symmetrize_color_ranges = true,
    categorical = true,
    n = categorical ? 5 : 100,
    n_left = n,
    n_right = n,
)
    # 1. Calculate the split point in the final (0, 1) output space
    # This determines "where the middle is" on the colorbar.
    s_mid = clamp((mid - lo) / (hi - lo), 0, 1)

    # 2. Determine the "intensity" ranges for sampling the source maps.
    # We generally want `mid` to map to the lightest color (index 0.0 for standard maps)
    # and the outer bounds to map to darker colors (index 1.0).

    if symmetrize_color_ranges && 0 < s_mid < 1  # only symmetrize if mid ∈ (lo, hi)
        # Ideally, -10 should be as "dark" as +10 (for mid=0).
        # We find the max deviation from mid to normalize intensities.
        max_dev = max(abs(lo - mid), abs(hi - mid))

        # Intensity at the boundaries (0.0 = light, 1.0 = dark)
        # If lo is -5 and max_dev is 10, left side only goes to 0.5 intensity (for mid=0).
        left_max_intensity = abs(lo - mid) / max_dev
        right_max_intensity = abs(hi - mid) / max_dev
    else
        # Stretch full colormaps to cover the available space
        left_max_intensity = 1.0
        right_max_intensity = 1.0
    end

    # If e.g. `lo > mid`, then we don't want any `left` colors, so set `n_left` to 0
    n_left = lo > mid ? 0 : n_left
    n_right = hi < mid ? 0 : n_right

    # 3. Create sampling indices for the source colormaps
    # Left side: lo -> mid
    # We want to transition from Dark (Max Intensity) -> Light (0.0)
    # Note: If passing `reverse(:blues)`, then index 0=Dark, 1=Light.
    # Adjust this logic based on your preferred accumulation direction.
    # Assuming standard :blues (0=Light, 1=Dark), we want 1.0 -> 0.0.
    left_sample_indices = range(left_max_intensity, 0.0; length = n_left)

    # Right side: mid -> hi
    # We want Light (0.0) -> Dark (Max Intensity)
    right_sample_indices = range(0.0, right_max_intensity; length = n_right)

    # 4. Fetch the colors
    # Note: `left_cs` logic depends on if you pass `:blues` or `reverse(:blues)`.
    # Based on your file, you might need `reverse(left_cs)` if passing standard maps,
    # or just sample appropriately. 
    left_colors = get(left_cs, left_sample_indices)
    right_colors = get(right_cs, right_sample_indices)

    # 5. Construct the final gradient
    # The left side covers output space 0.0 -> s_mid
    # The right side covers output space s_mid -> 1.0
    left_pos = range(0.0, s_mid; length = n_left)
    right_pos = range(s_mid, 1.0; length = n_right)

    # Combine
    all_colors = vcat(left_colors, right_colors)
    all_pos = vcat(left_pos, right_pos)

    return Makie.cgrad(all_colors, all_pos; categorical)
end
spliced_cmap(left::Symbol, right::Symbol, lo, hi; kwargs...) =
    spliced_cmap(Makie.cgrad(left).colors, Makie.cgrad(right).colors, lo, hi; kwargs...)

import ClimaAtmos as CA
FT = Float64

thermo_params = CA.TD.Parameters.ThermodynamicsParameters(FT)
title = "Condensation/evaporation rate (g/kg/s) \n"
g_kg⁻¹ = 1e-3

# qₜ_gkg = 2; qₜ = qₜ_gkg * g_kg⁻¹; title *= "qₜ=$(qₜ_gkg)g/kg, "
y = qₜ = -40g_kg⁻¹:(0.1g_kg⁻¹):40g_kg⁻¹; ylabel = "total water humidity, q_tot (g/kg)"; y_suf = "qt"

T = 295; title *= "T=$(T)K, "
# y = T = 240:0.01:273.15; ylabel = "Temperature (K)"; y_suf = "T"

qᵢ = qᵣ = qₛ = 0g_kg⁻¹; title *= "qᵢ=qᵣ=qₛ=0, "
ρ = 1; title *= "ρ=$(ρ)kg/m³, "
dt = 1; title *= "dt=$(dt)s, "
τ_relax = 1; title *= "τ_relax=$(τ_relax)s, "
cm_params = CA.CM.Parameters.CloudLiquid(FT)
cm_params = CA.CM.Parameters.CloudLiquid(FT(τ_relax), cm_params.ρw, cm_params.r_eff) # overwrite τ_relax

qₗ = -40g_kg⁻¹:(0.1g_kg⁻¹):40g_kg⁻¹; xlabel = "liquid water humidity, q_liq (g/kg)"; x_suf = "ql"
# qₗ = -2g_kg⁻¹:(0.01g_kg⁻¹):2g_kg⁻¹; xlabel = "liquid water humidity, q_liq (g/kg)"; x_suf = "ql"

x_sc = qₗ / g_kg⁻¹
y_sc = qₜ / g_kg⁻¹
# y_sc = T

qᵥ = @. (qₜ' - qₗ - qᵣ - qᵢ - qₛ) / g_kg⁻¹

function calc_δ(thp, T, ρ, q_tot, q_liq, q_ice)
    qᵥ = q_tot - q_liq - q_ice
    pₛᵥ = CA.TD.saturation_vapor_pressure(thp, T, CA.TD.Liquid())
    qₛₗ = CA.TD.q_vap_from_p_vap(thp, T, ρ, pₛᵥ)
    return qᵥ - qₛₗ
end
# δ = @. calc_δ(thermo_params, T', ρ, qₜ, qₗ + qᵣ, qᵢ + qₛ) / g_kg⁻¹
δ = @. calc_δ(thermo_params, T, ρ, qₜ', qₗ + qᵣ, qᵢ + qₛ) / g_kg⁻¹
# Calculate condensation/evaporation term
# S_cl = CA.cloud_sources.(cm_params, thermo_params, qₜ, qₗ, qᵢ, qᵣ, qₛ, ρ, T', dt) * 1e3
S_cl = CA.cloud_sources.(cm_params, thermo_params, qₜ', qₗ, qᵢ, qᵣ, qₛ, ρ, T, dt) * 1e3
colorrange = extrema(S_cl)
# colorrange = (0.2, 1.0)
S_cl[iszero.(S_cl)] .= NaN  # set zero values to NaN, then use `nan_color=:gray` to show them as gray. These are clipped values.
# Plotting
cmap = spliced_cmap(:blues, :reds, colorrange...; mid=0, categorical=true, symmetrize_color_ranges=true)
fig = Figure()
ax = Axis(fig[1,1]; xlabel, ylabel, title)
hm = heatmap!(ax, x_sc, y_sc, S_cl; colorrange, colormap = cmap, nan_color=:gray)
contour!(ax, x_sc, y_sc, qᵥ; color=:black, levels = [-60,0,60], labels=true, labelformatter = l -> "qᵥ = " * Makie.contour_label_formatter(l) * " g/kg")
contour!(ax, x_sc, y_sc, δ; labels=true, colormap=:matter, labelcolor=:black,
    # levels = -3:3,
    levels = [-10, 0, 10],
    labelformatter = l -> "δ = " * Makie.contour_label_formatter(l) * " g/kg",
)
label = rich("Condensation(+)",color=:red) * " / " * rich("Evaporation(−)",color=:blue) * " rate (g/kg/s) " * rich("(gray≡0)",color=:gray)
Colorbar(fig[1,2], hm; colorrange, label)
save("cloud_sources_$(x_suf)_$(y_suf).png", fig)