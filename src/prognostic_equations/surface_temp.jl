surface_temp_tendency!(Yₜ, Y, p, t, ::PrescribedSurfaceTemperature) = nothing

function surface_temp_tendency!(Yₜ, Y, p, t, ::PrognosticSurfaceTemperature)
    depth = 10 # ocean mixed layer depth [m]
    ρ_ocean = 1020 # ocean density [kg / m³]
    cp_ocean = 4184 # ocean heat capacity [J/(kg * K)]

    (; ᶠradiation_flux) = p.radiation
    sfc_rad_flux = Spaces.level(ᶠradiation_flux, half).components.data.:1

    # Merlis et al., 2013 eq(9)
    @. Yₜ.sfc.T -= sfc_rad_flux / (ρ_ocean * cp_ocean * depth)
end
