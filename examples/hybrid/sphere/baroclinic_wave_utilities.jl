using CloudMicrophysics
const CM = CloudMicrophysics

include("../staggered_nonhydrostatic_model.jl")

struct BaroclinicWaveParameterSet <: AbstractEarthParameterSet end
Planet.R_d(::BaroclinicWaveParameterSet) = 287.0
Planet.MSLP(::BaroclinicWaveParameterSet) = 1.0e5
Planet.grav(::BaroclinicWaveParameterSet) = 9.80616
Planet.Omega(::BaroclinicWaveParameterSet) = 7.29212e-5
Planet.planet_radius(::BaroclinicWaveParameterSet) = 6.371229e6

baroclinic_wave_mesh(; params, h_elem) =
    cubed_sphere_mesh(; radius = FT(Planet.planet_radius(params)), h_elem)

##
## Initial conditions
##

function center_initial_condition(
    local_geometry,
    params,
    ᶜ𝔼_name;
    is_balanced_flow = false,
    moisture_mode = Val(:dry),
)
    # Constants from CLIMAParameters
    R_d = FT(Planet.R_d(params))
    MSLP = FT(Planet.MSLP(params))
    grav = FT(Planet.grav(params))
    Ω = FT(Planet.Omega(params))
    R = FT(Planet.planet_radius(params))

    # Constants required for dry initial conditions
    k = 3
    T_e = FT(310) # temperature at the equator
    T_p = FT(240) # temperature at the pole
    T_0 = FT(0.5) * (T_e + T_p)
    Γ = FT(0.005)
    A = 1 / Γ
    B = (T_0 - T_p) / T_0 / T_p
    C = FT(0.5) * (k + 2) * (T_e - T_p) / T_e / T_p
    b = 2
    H = R_d * T_0 / grav
    z_t = FT(15e3)
    λ_c = FT(20)
    ϕ_c = FT(40)
    d_0 = R / 6
    V_p = FT(1)

    # Constants required for moist initial conditions
    p_w = FT(3.4e4)
    q_t = FT(1e-12)
    q_0 = FT(0.018)
    ϕ_w = FT(2 * π / 9)
    ε = FT(0.608)

    # Coordinates
    z = local_geometry.coordinates.z
    ϕ = local_geometry.coordinates.lat
    λ = local_geometry.coordinates.long

    # Initial virtual temperature and pressure
    τ_z_1 = exp(Γ * z / T_0)
    τ_z_2 = 1 - 2 * (z / b / H)^2
    τ_z_3 = exp(-(z / b / H)^2)
    τ_1 = 1 / T_0 * τ_z_1 + B * τ_z_2 * τ_z_3
    τ_2 = C * τ_z_2 * τ_z_3
    τ_int_1 = A * (τ_z_1 - 1) + B * z * τ_z_3
    τ_int_2 = C * z * τ_z_3
    I_T = cosd(ϕ)^k - k * (cosd(ϕ))^(k + 2) / (k + 2)
    T_v = (τ_1 - τ_2 * I_T)^(-1)
    p = MSLP * exp(-grav / R_d * (τ_int_1 - τ_int_2 * I_T))

    # Initial velocity
    U = grav * k / R * τ_int_2 * T_v * (cosd(ϕ)^(k - 1) - cosd(ϕ)^(k + 1))
    u = -Ω * R * cosd(ϕ) + sqrt((Ω * R * cosd(ϕ))^2 + R * cosd(ϕ) * U)
    v = FT(0)
    if !is_balanced_flow
        F_z = (1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3) * (z ≤ z_t)
        r = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))
        c3 = cos(π * r / 2 / d_0)^3
        s1 = sin(π * r / 2 / d_0)
        cond = (0 < r < d_0) * (r != R * pi)
        u +=
            -16 * V_p / 3 / sqrt(FT(3)) *
            F_z *
            c3 *
            s1 *
            (-sind(ϕ_c) * cosd(ϕ) + cosd(ϕ_c) * sind(ϕ) * cosd(λ - λ_c)) /
            sin(r / R) * cond
        v +=
            16 * V_p / 3 / sqrt(FT(3)) *
            F_z *
            c3 *
            s1 *
            cosd(ϕ_c) *
            sind(λ - λ_c) / sin(r / R) * cond
    end
    uₕ_local = Geometry.UVVector(u, v)
    uₕ = Geometry.Covariant12Vector(uₕ_local, local_geometry)

    # Initial moisture and temperature
    if moisture_mode === Val(:dry)
        q_tot = FT(0)
    else
        q_tot = (p <= p_w) ? q_t :
            q_0 * exp(-(ϕ / ϕ_w)^4) * exp(-((p - MSLP) / p_w)^2)
    end
    T = T_v / (1 + ε * q_tot) # This is the formula used in the paper.
    # T = T_v * (1 + q_tot) / (1 + q_tot * Planet.molmass_ratio(params))
    # This is the actual formula, which would be consistent with TD.

    # Initial values computed from the thermodynamic state
    ts = TD.PhaseEquil_pTq(params, p, T, q_tot)
    ρ = TD.air_density(ts)
    if ᶜ𝔼_name === Val(:ρθ)
        ᶜ𝔼_kwarg = (; ρθ = ρ * TD.liquid_ice_pottemp(ts))
    elseif ᶜ𝔼_name === Val(:ρe)
        K = norm_sqr(uₕ_local) / 2
        ᶜ𝔼_kwarg = (; ρe = ρ * (TD.internal_energy(ts) + K + grav * z))
    elseif ᶜ𝔼_name === Val(:ρe_int)
        ᶜ𝔼_kwarg = (; ρe_int = ρ * TD.internal_energy(ts))
    end
    if moisture_mode === Val(:dry)
        moisture_kwargs = NamedTuple()
    elseif moisture_mode === Val(:equil)
        moisture_kwargs = (; ρq_tot = ρ * q_tot)
    elseif moisture_mode === Val(:nonequil)
        moisture_kwargs = (;
            ρq_tot = ρ * q_tot,
            ρq_liq = ρ * TD.liquid_specific_humidity(ts),
            ρq_ice = ρ * TD.ice_specific_humidity(ts),
        )
    end
    # TODO: Include ability to handle nonzero initial cloud condensate

    return (; ρ, ᶜ𝔼_kwarg..., uₕ, moisture_kwargs...)
end

face_initial_condition(local_geometry, params) =
    (; w = Geometry.Covariant3Vector(FT(0)))

##
## Additional tendencies
##

# Rayleigh sponge 

function rayleigh_sponge_cache(Y, dt)
    z_D = FT(15e3)
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜαₘ = @. ifelse(ᶜz > z_D, 1 / (20 * dt), FT(0))
    ᶠαₘ = @. ifelse(ᶠz > z_D, 1 / (20 * dt), FT(0))
    zmax = maximum(ᶠz)
    ᶜβ = @. ᶜαₘ * sin(π / 2 * (ᶜz - z_D) / (zmax - z_D))^2
    ᶠβ = @. ᶠαₘ * sin(π / 2 * (ᶠz - z_D) / (zmax - z_D))^2
    return (; ᶜβ, ᶠβ)
end

function rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    (; ᶜβ, ᶠβ) = p
    @. Yₜ.c.uₕ -= ᶜβ * Y.c.uₕ
    @. Yₜ.f.w -= ᶠβ * Y.f.w
end

# Held-Suarez forcing

held_suarez_cache(Y) = (;
    ᶜσ = similar(Y.c, FT),
    ᶜheight_factor = similar(Y.c, FT),
    ᶜΔρT = similar(Y.c, FT),
    ᶜφ = deg2rad.(Fields.coordinate_field(Y.c).lat),
)

function held_suarez_tendency!(Yₜ, Y, p, t)
    (; ᶜp, ᶜσ, ᶜheight_factor, ᶜΔρT, ᶜφ, params) = p # assume ᶜp has been updated

    R_d = FT(Planet.R_d(params))
    κ_d = FT(Planet.kappa_d(params))
    cv_d = FT(Planet.cv_d(params))
    day = FT(Planet.day(params))
    MSLP = FT(Planet.MSLP(params))

    σ_b = FT(7 / 10)
    k_a = 1 / (40 * day)
    k_s = 1 / (4 * day)
    k_f = 1 / day
    ΔT_y = FT(60)
    Δθ_z = FT(10)
    T_equator = FT(315)
    T_min = FT(200)

    @. ᶜσ = ᶜp / MSLP
    @. ᶜheight_factor = max(0, (ᶜσ - σ_b) / (1 - σ_b))
    @. ᶜΔρT =
        (k_a + (k_s - k_a) * ᶜheight_factor * cos(ᶜφ)^4) *
        Y.c.ρ *
        ( # ᶜT - ᶜT_equil
            ᶜp / (Y.c.ρ * R_d) - max(
                T_min,
                (T_equator - ΔT_y * sin(ᶜφ)^2 - Δθ_z * log(ᶜσ) * cos(ᶜφ)^2) *
                ᶜσ^κ_d,
            )
        )

    @. Yₜ.c.uₕ -= (k_f * ᶜheight_factor) * Y.c.uₕ
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ -= ᶜΔρT * (MSLP / ᶜp)^κ_d
    elseif :ρe in propertynames(Y.c)
        @. Yₜ.c.ρe -= ᶜΔρT * cv_d
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int -= ᶜΔρT * cv_d
    end
end

# 0-Moment Microphysics

zero_moment_microphysics_cache(Y) =
    (ᶜS_ρq_tot = similar(Y.c, FT), ᶜλ = similar(Y.c, FT))

function zero_moment_microphysics_tendency!(Yₜ, Y, p, t)
    (; ᶜts, ᶜΦ, ᶜS_ρq_tot, ᶜλ, params) = p # assume ᶜts has been updated

    # _qc_0 - set it to 0 to remove immediately after supersat
    # _τ_precip - make it super short to get behavior similar to instantaneous

    @. ᶜS_ρq_tot =
        Y.c.ρ *
        CM.Microphysics_0M.remove_precipitation(params, TD.PhasePartition(ᶜts))
    @. Yₜ.c.ρq_tot += ᶜS_ρq_tot
    @. Yₜ.c.ρ += ᶜS_ρq_tot

    @. ᶜλ = TD.liquid_fraction(ᶜts)

    if :ρθ in propertynames(Y.c)
        L_v0 = Planet.LH_v0(params)
        L_s0 = Planet.LH_s0(params)
        @. Yₜ.c.ρθ -=
            ᶜS_ρq_tot / TD.exner(ᶜts) / TD.cp_m(ᶜts) *
            (L_v0 * ᶜλ + L_s0 * (1 - ᶜλ))
    elseif :ρe in propertynames(Y.c)
        @. Yₜ.c.ρe +=
            ᶜS_ρq_tot * (
                ᶜλ * TD.internal_energy_liquid(ᶜts) +
                (1 - ᶜλ) * TD.internal_energy_ice(ᶜts) +
                ᶜΦ
            )
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int +=
            ᶜS_ρq_tot * (
                ᶜλ * TD.internal_energy_liquid(ᶜts) +
                (1 - ᶜλ) * TD.internal_energy_ice(ᶜts)
            )
    end
end

# Vertical diffusion boundary layer parameterization

# Apply on potential temperature and moisture
# 1) turn the liquid_theta into theta version
# 2) have a total energy version (primary goal)

# Note: ᶠv_a and ᶠz_a are 3D projections of 2D Fields (the values of uₕ and z at
#       the first cell center of every column, respectively).
# TODO: Allow ClimaCore to handle both 2D and 3D Fields in a single broadcast.
#       This currently results in a mismatched spaces error.
function vertical_diffusion_boundary_layer_cache(Y)
    ᶠz_a = similar(Y.f, FT)
    Fields.field_values(ᶠz_a) .=
        Fields.field_values(Spaces.level(Fields.coordinate_field(Y.c).z, 1)) .*
        one.(Fields.field_values(ᶠz_a)) # TODO: fix VIJFH copyto! to remove this
    return (;
        ᶠv_a = similar(Y.f, eltype(Y.c.uₕ)), ᶠz_a, ᶠK_E = similar(Y.f, FT),
    )
end

function eddy_diffusivity_coefficient(norm_v_a, z_a, p)
    C_E = FT(0.0044)
    p_pbl = FT(85000)
    p_strato = FT(10000)
    K_E = C_E * norm_v_a * z_a
    return p > p_pbl ? K_E : K_E * exp(-((p_pbl - p) / p_strato)^2)
end

function vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    (; ᶜp, ᶠv_a, ᶠz_a, ᶠK_E) = p # assume ᶜp has been updated

    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ

    Fields.field_values(ᶠv_a) .=
        Fields.field_values(Spaces.level(Y.c.uₕ, 1)) .*
        one.(Fields.field_values(ᶠz_a)) # TODO: fix VIJFH copyto! to remove this
    @. ᶠK_E = eddy_diffusivity_coefficient(norm(ᶠv_a), ᶠz_a, ᶠinterp(ᶜp))

    # diffusion scheme for boundary layer
    if :ρθ in propertynames(Y.c)
        F₋ = Geometry.Contravariant3Vector(FT(0)) # TODO: Make real :)
        F₊ = Geometry.Contravariant3Vector(FT(0))
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(F₊),
            bottom = Operators.SetValue(F₋),
        )
        @. Yₜ.c.ρθ += ᶜdivᵥ(ᶠK_E * ᶠinterp(ᶜρ) * ᶠgradᵥ(Y.c.ρθ / ᶜρ))
    elseif :ρe in propertynames(Y.c)
        F₋ = Geometry.Contravariant3Vector(FT(0)) # TODO: Make real :)
        F₊ = Geometry.Contravariant3Vector(FT(0))
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(F₊),
            bottom = Operators.SetValue(F₋),
        )
        # θ = TD.dry_pottemp(ts)
        # Δθ = ᶜdivᵥ(ᶜK_E * ᶠinterp(ᶜρ) * ᶠgradᵥ( θ ))
        # T = θ ^ (cp_m / cv_m) * (ᶜρ * R_m / p_0) ^ (R_m/cv_m)
        # ΔT = Δθ part + Δρ part lol
        # alternatively, applying diffusion on e_int
        # e_int = TD.internal_energy(ᶜts)
        @. Yₜ.c.ρe += ᶜdivᵥ(ᶠK_E * ᶠinterp(ᶜρ) * ᶠgradᵥ((Y.c.ρe + ᶜp) / ᶜρ))
    elseif :ρe_int in propertynames(Y.c)
        F₋ = Geometry.Contravariant3Vector(FT(0)) # TODO: Make real :)
        F₊ = Geometry.Contravariant3Vector(FT(0))
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(F₊),
            bottom = Operators.SetValue(F₋),
        )
        @. Yₜ.c.ρe_int +=
            ᶜdivᵥ(ᶠK_E * ᶠinterp(ᶜρ) * ᶠgradᵥ((Y.c.ρe_int + ᶜp) / ᶜρ))
    end

    if :ρq_tot in propertynames(Y.c)
        F₋ = Geometry.Contravariant3Vector(FT(0)) # TODO: Make real :)
        F₊ = Geometry.Contravariant3Vector(FT(0))
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(F₊),
            bottom = Operators.SetValue(F₋),
        )
        @. Yₜ.c.ρq_tot += ᶜdivᵥ(ᶠK_E * ᶠinterp(ᶜρ) * ᶠgradᵥ(Y.c.ρq_tot / ᶜρ))
    end
end
