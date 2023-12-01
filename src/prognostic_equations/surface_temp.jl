surface_temp_tendency!(Yₜ, Y, p, t, ::PrescribedSurfaceTemperature) = nothing

function surface_temp_tendency!(Yₜ, Y, p, t, slab::PrognosticSurfaceTemperature)
    FT = eltype(Y)
    params = p.params

    depth_ocean = slab.depth_ocean
    ρ_ocean = slab.ρ_ocean
    cp_ocean = slab.cp_ocean
    q_flux = slab.q_flux


    # ENERGY
    # radiative energy surface fluxes
    if !isnothing(p.atmos.radiation_mode)
        (; ᶠradiation_flux) = p.radiation
        sfc_rad_e_flux = Spaces.level(ᶠradiation_flux, half).components.data.:1
    else
        sfc_rad_e_flux = 0
    end

    # turbulent surface fluxes: energy (sensible + latent heat)
    sfc_turb_e_flux =
        Geometry.WVector.(
            p.precomputed.sfc_conditions.ρ_flux_h_tot
        ).components.data.:1

    # Q-fluxes (parameterization of horizontal ocean mixing of energy)
    # as in Zurita-Gotor et al., 2023
    if q_flux
        ϕ₀ = slab.ϕ₀
        Q₀ = slab.Q₀
        ϕ = deg2rad.(Fields.level(Fields.coordinate_field(Y.f).lat, half))
        ϕ₀ʳ = FT(deg2rad(ϕ₀))
        Q = @. Q₀ * (1 - 2ϕ^2 / ϕ₀ʳ^2) * exp(-(ϕ^2 / ϕ₀ʳ^2)) / cos(ϕ)
    else
        Q = FT(0)
    end

    # TODO: add an energy contribution from melting snow at the surface

    @. Yₜ.sfc.T -=
        (sfc_rad_e_flux + sfc_turb_e_flux + Q) /
        (ρ_ocean * cp_ocean * depth_ocean)

    if !(p.atmos.moisture_model isa DryModel)
        # WATER
        # turbulent surface fluxes: water (evaporation)
        sfc_turb_w_flux = p.precomputed.sfc_conditions.ρ_flux_q_tot

        # precipitation
        ρ_liq = CAP.ρ_cloud_liq(params)
        ρ_ice = CAP.ρ_cloud_ice(params)
        P_liq = p.precipitation.col_integrated_rain
        P_snow = p.precipitation.col_integrated_snow
        @. Yₜ.sfc.water -=
            P_liq * ρ_liq +
            P_snow * ρ_ice +
            Geometry.WVector.(sfc_turb_w_flux).components.data.:1 # d(water)/dt = P - E [kg/m²/s]
    end
end
