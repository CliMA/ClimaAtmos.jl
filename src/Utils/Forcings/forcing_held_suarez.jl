# Held-Suarez forcing
struct HeldSuarezForcing{S} <: AbstractAtmosForcing
    parameters::S
end

function rhs_forcing!(dY, Y, Ya, p, hs::HeldSuarezForcing, ::TotalEnergy)
    # # unpack cache 
    # TODO: make cache capability work for 3D model
    # p = Ya.p

    # unpack hs parameters
    @unpack k_a, k_f, k_s, ΔT_y, Δθ_z, T_equator, T_min, σ_b = hs.parameters

    R_d = CLIMAParameters.Planet.R_d(hs.parameters)
    cp_d = CLIMAParameters.Planet.cp_d(hs.parameters)  
    cv_d = CLIMAParameters.Planet.cv_d(hs.parameters)
    p_0 = CLIMAParameters.Planet.MSLP(hs.parameters)

    # obtain latitude
    c_local_geometry = Fields.local_geometry_field(Y.base.ρ)
    φ = deg2rad.(c_local_geometry.coordinates.lat)

    # forcing
    σ = p / hs.parameters.p0
    height_factor = max(0, (σ - σ_b) / (1 - σ_b))
    ΔρT =
        (k_a + (k_s - k_a) * height_factor * cos(φ)^4) *
        Y.base.ρ *
        ( # ᶜT - ᶜT_equil
            p / (Y.base.ρ * R_d) - max(
                T_min,
                (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) *
                σ^(R_d / cp_d),
            )
        )

    dY.base.uh -= (k_f * height_factor) * Y.base.uh
    dY.thermodynamics.ρe_tot -= ΔρT * cv_d
end

function rhs_forcing!(dY, Y, Ya, p, hs::HeldSuarezForcing, ::PotentialTemperature)
    # # unpack cache
    # TODO: make cache capability work for 3D model
    # p = Ya.p

    # unpack hs parameters
    @unpack k_a, k_f, k_s, ΔT_y, Δθ_z, T_equator, T_min, σ_b = hs.parameters

    R_d = CLIMAParameters.Planet.R_d(hs.parameters)
    cp_d = CLIMAParameters.Planet.cp_d(hs.parameters)  
    cv_d = CLIMAParameters.Planet.cv_d(hs.parameters)
    p_0 = CLIMAParameters.Planet.MSLP(hs.parameters)

    # obtain latitude
    c_local_geometry = Fields.local_geometry_field(Y.base.ρ)
    φ = deg2rad.(c_local_geometry.coordinates.lat)

    # forcing
    σ = p / hs.parameters.p0
    height_factor = max(0, (σ - σ_b) / (1 - σ_b))
    ΔρT =
        (k_a + (k_s - k_a) * height_factor * cos(φ)^4) *
        Y.base.ρ *
        ( # ᶜT - ᶜT_equil
            p / (Y.base.ρ * R_d) - max(
                T_min,
                (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) *
                σ^(R_d / cp_d),
            )
        )

    dY.base.uh -= (k_f * height_factor) * Y.base.uh
    dY.thermodynamics.ρθ -= ΔρT * (p_0 / p)^κ
end
