# Constants required by "staggered_nonhydrostatic_model.jl"
# const FT = ? # specified in each test file
const p_0 = FT(1.0e5)
const R_d = FT(287.0)
const κ = FT(2 / 7)
const T_tri = FT(273.16)
const grav = FT(9.80616)
const Ω = FT(7.29212e-5)
include("../staggered_nonhydrostatic_model.jl")

# Constants required for balanced flow and baroclinic wave initial conditions
const R = FT(6.371229e6)
const k = 3
const T_e = FT(310) # temperature at the equator
const T_p = FT(240) # temperature at the pole
const T_0 = FT(0.5) * (T_e + T_p)
const Γ = FT(0.005)
const A = 1 / Γ
const B = (T_0 - T_p) / T_0 / T_p
const C = FT(0.5) * (k + 2) * (T_e - T_p) / T_e / T_p
const b = 2
const H = R_d * T_0 / grav
const z_t = FT(15e3)
const λ_c = FT(20)
const ϕ_c = FT(40)
const d_0 = R / 6
const V_p = FT(1)

# Constants required for Rayleigh sponge layer
const z_D = FT(15e3)

# Constants required for Held-Suarez forcing
const day = FT(3600 * 24)
const k_a = 1 / (40 * day)
const k_f = 1 / day
const k_s = 1 / (4 * day)
const ΔT_y = FT(60)
const Δθ_z = FT(10)
const T_equator = FT(315)
const T_min = FT(200)
const σ_b = FT(7 / 10)

##
## Initial conditions
##

τ_z_1(z) = exp(Γ * z / T_0)
τ_z_2(z) = 1 - 2 * (z / b / H)^2
τ_z_3(z) = exp(-(z / b / H)^2)
τ_1(z) = 1 / T_0 * τ_z_1(z) + B * τ_z_2(z) * τ_z_3(z)
τ_2(z) = C * τ_z_2(z) * τ_z_3(z)
τ_int_1(z) = A * (τ_z_1(z) - 1) + B * z * τ_z_3(z)
τ_int_2(z) = C * z * τ_z_3(z)
F_z(z) = (1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3) * (z ≤ z_t)
I_T(ϕ) = cosd(ϕ)^k - k * (cosd(ϕ))^(k + 2) / (k + 2)
temp(ϕ, z) = (τ_1(z) - τ_2(z) * I_T(ϕ))^(-1)
pres(ϕ, z) = p_0 * exp(-grav / R_d * (τ_int_1(z) - τ_int_2(z) * I_T(ϕ)))
θ(ϕ, z) = temp(ϕ, z) * (p_0 / pres(ϕ, z))^κ
r(λ, ϕ) = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))
U(ϕ, z) =
    grav * k / R * τ_int_2(z) * temp(ϕ, z) * (cosd(ϕ)^(k - 1) - cosd(ϕ)^(k + 1))
u(ϕ, z) = -Ω * R * cosd(ϕ) + sqrt((Ω * R * cosd(ϕ))^2 + R * cosd(ϕ) * U(ϕ, z))
v(ϕ, z) = zero(z)
c3(λ, ϕ) = cos(π * r(λ, ϕ) / 2 / d_0)^3
s1(λ, ϕ) = sin(π * r(λ, ϕ) / 2 / d_0)
cond(λ, ϕ) = (0 < r(λ, ϕ) < d_0) * (r(λ, ϕ) != R * pi)
δu(λ, ϕ, z) =
    -16 * V_p / 3 / sqrt(FT(3)) *
    F_z(z) *
    c3(λ, ϕ) *
    s1(λ, ϕ) *
    (-sind(ϕ_c) * cosd(ϕ) + cosd(ϕ_c) * sind(ϕ) * cosd(λ - λ_c)) /
    sin(r(λ, ϕ) / R) * cond(λ, ϕ)
δv(λ, ϕ, z) =
    16 * V_p / 3 / sqrt(FT(3)) *
    F_z(z) *
    c3(λ, ϕ) *
    s1(λ, ϕ) *
    cosd(ϕ_c) *
    sind(λ - λ_c) / sin(r(λ, ϕ) / R) * cond(λ, ϕ)

function center_initial_condition(
    local_geometry,
    ᶜ𝔼_name;
    is_balanced_flow = false,
)
    (; lat, long, z) = local_geometry.coordinates
    ρ = pres(lat, z) / R_d / temp(lat, z)
    u₀ = u(lat, z)
    v₀ = v(lat, z)
    if !is_balanced_flow
        u₀ += δu(long, lat, z)
        v₀ += δv(long, lat, z)
    end
    uₕ_local = Geometry.UVVector(u₀, v₀)
    uₕ = Geometry.Covariant12Vector(uₕ_local, local_geometry)
    if ᶜ𝔼_name === Val(:ρθ)
        ρθ = ρ * θ(lat, z)
        return (; ρ, ρθ, uₕ)
    elseif ᶜ𝔼_name === Val(:ρe)
        ρe =
            ρ *
            (cv_d * (temp(lat, z) - T_tri) + norm_sqr(uₕ_local) / 2 + grav * z)
        return (; ρ, ρe, uₕ)
    elseif ᶜ𝔼_name === Val(:ρe_int)
        ρe_int = ρ * cv_d * (temp(lat, z) - T_tri)
        return (; ρ, ρe_int, uₕ)
    end
end
face_initial_condition(local_geometry) =
    (; w = Geometry.Covariant3Vector(FT(0)))

##
## Additional tendencies
##

function rayleigh_sponge_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt)
    ᶜz = ᶜlocal_geometry.coordinates.z
    ᶠz = ᶠlocal_geometry.coordinates.z
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

held_suarez_cache(ᶜlocal_geometry) = (;
    ᶜσ = similar(ᶜlocal_geometry, FT),
    ᶜheight_factor = similar(ᶜlocal_geometry, FT),
    ᶜΔρT = similar(ᶜlocal_geometry, FT),
    ᶜφ = deg2rad.(ᶜlocal_geometry.coordinates.lat),
)

function held_suarez_tendency!(Yₜ, Y, p, t)
    (; ᶜp, ᶜσ, ᶜheight_factor, ᶜΔρT, ᶜφ) = p # assume that ᶜp has been updated

    @. ᶜσ = ᶜp / p_0
    @. ᶜheight_factor = max(0, (ᶜσ - σ_b) / (1 - σ_b))
    @. ᶜΔρT =
        (k_a + (k_s - k_a) * ᶜheight_factor * cos(ᶜφ)^4) *
        Y.c.ρ *
        ( # ᶜT - ᶜT_equil
            ᶜp / (Y.c.ρ * R_d) - max(
                T_min,
                (T_equator - ΔT_y * sin(ᶜφ)^2 - Δθ_z * log(ᶜσ) * cos(ᶜφ)^2) *
                ᶜσ^(R_d / cp_d),
            )
        )

    @. Yₜ.c.uₕ -= (k_f * ᶜheight_factor) * Y.c.uₕ
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ -= ᶜΔρT * (p_0 / ᶜp)^κ
    elseif :ρe in propertynames(Y.c)
        @. Yₜ.c.ρe -= ᶜΔρT * cv_d
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int -= ᶜΔρT * cv_d
    end
end
