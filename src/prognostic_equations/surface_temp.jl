#####
##### Couples the atmospheric model with a slab surface model
#####

using .SurfaceConditions: SurfaceTemperature, SlabOceanTemperature

"""
    surface_temp_tendency!(Yₜ, Y, p, t, temperature::SurfaceTemperature)
    surface_temp_tendency!(Yₜ, Y, p, t, temperature::SlabOceanTemperature)

Add the explicit tendencies of the prognostic surface temperature `Y.sfc.T`.

Fluxes are positive when directed upward, from the surface to the atmosphere,
and are subtracted from the tendencies. For `SurfaceTemperature` other than a
slab ocean, `T` is prescribed or diagnosed. For `SlabOceanTemperature`,
`Yₜ.sfc.T` is decremented by the net upward radiative flux at the surface plus
the idealized ocean heat-flux divergence `Q` (when `slab.q_flux` is active),
divided by `ρ_ocean cp_ocean depth_ocean`. Turbulent and precipitation fluxes
between the atmosphere and slab ocean are applied post-ARK in column physics.
"""
surface_temp_tendency!(Yₜ, Y, p, t, ::SurfaceTemperature) = nothing

function surface_temp_tendency!(Yₜ, Y, p, t, slab::SlabOceanTemperature)
    FT = eltype(Y)

    depth_ocean = slab.depth_ocean
    ρ_ocean = slab.ρ_ocean
    cp_ocean = slab.cp_ocean
    q_flux_enabled = slab.q_flux

    # --- ENERGY BALANCE ---
    surface_heat_capacity_per_area = ρ_ocean * cp_ocean * depth_ocean

    # 1. Radiative energy surface fluxes
    if !isnothing(p.atmos.radiation_mode)
        # ᶠradiation_flux is positive for net upward flux at the surface
        # (SW_up - SW_down + LW_up - LW_down)
        (; ᶠradiation_flux) = p.radiation
        sfc_rad_e_flux = Spaces.level(ᶠradiation_flux, half).components.data.:1
    else
        sfc_rad_e_flux = 0
    end

    # 2. Idealized Q-fluxes (parameterization of horizontal ocean energy flux divergence),
    # following Merlis et al. (2013), "Hadley Circulation Response to Orbital Precession.
    # Part II: Subtropical Continent.", J. Climate, 26, https://doi.org/10.1175/JCLI-D-12-00149.1
    if q_flux_enabled
        ϕ₀ = slab.ϕ₀
        Q₀ = slab.Q₀
        ϕ = deg2rad.(Fields.level(Fields.coordinate_field(Y.f).lat, half))
        ϕ₀ʳ = FT(deg2rad(ϕ₀))
        Q = @. Q₀ * (1 - 2ϕ^2 / ϕ₀ʳ^2) * exp(-(ϕ^2 / ϕ₀ʳ^2)) / cos(ϕ)
    else
        Q = FT(0)
    end

    @. Yₜ.sfc.T -= (sfc_rad_e_flux + Q) / surface_heat_capacity_per_area
    return nothing
end
