"""
    InitialCondition

A mechanism for specifying the `LocalState` of an `AtmosModel` at every point in
the domain. Given some `initial_condition`, calling `initial_condition(params)`
returns a function of the form `local_state(local_geometry)::LocalState`.
"""
abstract type InitialCondition end

# Perturbation coefficient for the initial conditions
# It would be better to be able to specify the wavenumbers
# but we don't have access to the domain size here

perturb_coeff(p::Geometry.AbstractPoint{FT}) where {FT} = FT(0)
perturb_coeff(p::Geometry.LatLongZPoint{FT}) where {FT} = sind(p.long)
perturb_coeff(p::Geometry.XZPoint{FT}) where {FT} = sin(p.x)
perturb_coeff(p::Geometry.XYZPoint{FT}) where {FT} = sin(p.x)

"""
    ColumnInterpolatableField(::Fields.ColumnField)

A column field object that can be interpolated
in the z-coordinate. For example:

```julia
cif = ColumnInterpolatableField(column_field)
z = 1.0
column_field_at_z = cif(z)
```

!!! warn
    This function allocates and is not GPU-compatible
    so please avoid using this inside `step!` only use
    this for initialization.
"""
struct ColumnInterpolatableField{F, D}
    f::F
    data::D
    function ColumnInterpolatableField(f::Fields.ColumnField)
        zdata = vec(parent(Fields.Fields.coordinate_field(f).z))
        fdata = vec(parent(f))
        data = Intp.extrapolate(
            Intp.interpolate((zdata,), fdata, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
        return new{typeof(f), typeof(data)}(f, data)
    end
end
(f::ColumnInterpolatableField)(z) = Spaces.undertype(axes(f.f))(f.data(z))

import ClimaComms
import ClimaCore.Domains as Domains
import ClimaCore.Meshes as Meshes
import ClimaCore.Geometry as Geometry
import ClimaCore.Operators as Operators
import ClimaCore.Topologies as Topologies
import ClimaCore.Spaces as Spaces

"""
    column_indefinite_integral(f, ϕ₀, zspan; nelems = 100)

The column integral, returned as
an interpolate-able field.
"""
function column_indefinite_integral(
    f::Function,
    ϕ₀::FT,
    zspan::Tuple{FT, FT};
    nelems = 100, # sets resolution for integration
) where {FT <: Real}
    # --- Make a space for integration:
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(first(zspan)),
        Geometry.ZPoint(last(zspan));
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain; nelems)
    context = ClimaComms.SingletonCommsContext()
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    cspace = Spaces.CenterFiniteDifferenceSpace(z_topology)
    fspace = Spaces.FaceFiniteDifferenceSpace(z_topology)
    # ---
    zc = Fields.coordinate_field(cspace)
    ᶠintegral = Fields.Field(FT, fspace)
    Operators.column_integral_indefinite!(f, ᶠintegral, ϕ₀)
    return ColumnInterpolatableField(ᶠintegral)
end

##
## Simple Profiles
##

"""
    IsothermalProfile(; temperature = 300)

An `InitialCondition` with a uniform temperature profile.
"""
Base.@kwdef struct IsothermalProfile{T} <: InitialCondition
    temperature::T = 300
end

function (initial_condition::IsothermalProfile)(params)
    (; temperature) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        R_d = CAP.R_d(params)
        MSLP = CAP.MSLP(params)
        grav = CAP.grav(params)
        thermo_params = CAP.thermodynamics_params(params)
        T = FT(temperature)

        (; z) = local_geometry.coordinates
        p = MSLP * exp(-z * grav / (R_d * T))

        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseDry_pT(thermo_params, p, T),
        )
    end
    return local_state
end

"""
    DecayingProfile(; perturb = true)

An `InitialCondition` with a decaying temperature profile, and with an optional
perturbation to the temperature.
"""
Base.@kwdef struct DecayingProfile <: InitialCondition
    perturb::Bool = true
end

function (initial_condition::DecayingProfile)(params)
    (; perturb) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        grav = CAP.grav(params)
        thermo_params = CAP.thermodynamics_params(params)
        temp_profile = DecayingTemperatureProfile{FT}(
            thermo_params,
            FT(290),
            FT(220),
            FT(8e3),
        )

        (; z) = local_geometry.coordinates
        coeff = perturb_coeff(local_geometry.coordinates)
        T, p = temp_profile(thermo_params, z)
        if perturb
            T += coeff * FT(0.1) * (z < 5000)
        end

        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseDry_pT(thermo_params, p, T),
        )
    end
    return local_state
end

"""
    AgnesiHProfile(; perturb = false)

An `InitialCondition` with a decaying temperature profile
"""
struct AgnesiHProfile <: InitialCondition end

function (initial_condition::AgnesiHProfile)(params)
    function local_state(local_geometry)
        FT = eltype(params)
        grav = CAP.grav(params)
        thermo_params = CAP.thermodynamics_params(params)
        (; x, z) = local_geometry.coordinates
        cp_d = CAP.cp_d(params)
        cv_d = CAP.cv_d(params)
        p_0 = CAP.p_ref_theta(params)
        R_d = CAP.R_d(params)
        T_0 = CAP.T_0(params)
        # auxiliary quantities
        T_bar = FT(250)
        buoy_freq = grav / sqrt(cp_d * T_bar)
        π_exn = exp(-grav * z / cp_d / T_bar)
        p = p_0 * π_exn^(cp_d / R_d) # pressure
        ρ = p / R_d / T_bar # density
        velocity = @. Geometry.UVVector(FT(20), FT(0))
        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseDry_pT(thermo_params, p, T_bar),
            velocity = velocity,
        )
    end
    return local_state
end

"""
    ScharProfile(; perturb = false)

An `InitialCondition` with a prescribed Brunt-Vaisala Frequency
"""
Base.@kwdef struct ScharProfile <: InitialCondition end

function (initial_condition::ScharProfile)(params)
    function local_state(local_geometry)
        FT = eltype(params)

        thermo_params = CAP.thermodynamics_params(params)
        g = CAP.grav(params)
        R_d = CAP.R_d(params)
        cp_d = CAP.cp_d(params)
        cv_d = CAP.cv_d(params)
        p₀ = CAP.p_ref_theta(params)
        (; x, z) = local_geometry.coordinates
        θ₀ = FT(280.0)
        buoy_freq = FT(0.01)
        θ = θ₀ * exp(buoy_freq^2 * z / g)
        π_exner =
            1 +
            g^2 / (cp_d * θ₀ * buoy_freq^2) * (exp(-buoy_freq^2 * z / g) - 1)
        T = π_exner * θ # temperature
        ρ = p₀ / (R_d * T) * (π_exner)^(cp_d / R_d)
        p = ρ * R_d * T
        velocity = Geometry.UVVector(FT(10), FT(0))

        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseDry_pT(thermo_params, p, T),
            velocity = velocity,
        )
    end
    return local_state
end

"""
    DryDensityCurrentProfile(; perturb = false)

An `InitialCondition` with an isothermal background profile, with a negatively
buoyant bubble, and with an optional
perturbation to the temperature.
"""
Base.@kwdef struct DryDensityCurrentProfile <: InitialCondition
    perturb::Bool = false
end

function (initial_condition::DryDensityCurrentProfile)(params)
    (; perturb) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        grav = CAP.grav(params)
        thermo_params = CAP.thermodynamics_params(params)
        ndims = length(propertynames(local_geometry.coordinates))
        (; x, z) = local_geometry.coordinates
        x_c = FT(25600)
        x_r = FT(4000)
        z_c = FT(2000)
        z_r = FT(2000)
        r_c = FT(1)
        θ_b = FT(300)
        θ_c = FT(-15)
        cp_d = CAP.cp_d(params)
        cv_d = CAP.cv_d(params)
        p_0 = CAP.p_ref_theta(params)
        R_d = CAP.R_d(params)
        T_0 = CAP.T_0(params)

        # auxiliary quantities
        r² = FT(0)
        r² += ((x - x_c) / x_r)^2 + ((z - z_c) / z_r)^2
        if ndims == 3
            (; y) = local_geometry.coordinates
            y_r = FT(2000)
            y_c = FT(3200)
            r² += ((y - y_c) / y_r)^2
        end
        θ_p =
            sqrt(r²) < r_c ? FT(1 / 2) * θ_c * (FT(1) + cospi(sqrt(r²) / r_c)) :
            FT(0) # potential temperature perturbation
        θ = θ_b + θ_p # potential temperature
        π_exn = FT(1) - grav * z / cp_d / θ # exner function
        T = π_exn * θ # temperature
        p = p_0 * π_exn^(cp_d / R_d) # pressure
        ρ = p / R_d / T # density

        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseDry_pT(thermo_params, p, T),
        )
    end
    return local_state
end

"""
    RisingThermalBubbleProfile(; perturb = false)

An `InitialCondition` with an isothermal background profile, with a positively
buoyant bubble, and with an optional perturbation to the temperature.
"""
Base.@kwdef struct RisingThermalBubbleProfile <: InitialCondition
    perturb::Bool = false
end

function (initial_condition::RisingThermalBubbleProfile)(params)
    (; perturb) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        grav = CAP.grav(params)
        thermo_params = CAP.thermodynamics_params(params)
        ndims = length(propertynames(local_geometry.coordinates))
        (; x, z) = local_geometry.coordinates
        x_c = FT(500)
        x_r = FT(250)
        z_c = FT(350)
        z_r = FT(250)
        r_c = FT(1)
        θ_b = FT(300)
        θ_c = FT(0.0)
        cp_d = CAP.cp_d(params)
        cv_d = CAP.cv_d(params)
        p_0 = CAP.p_ref_theta(params)
        R_d = CAP.R_d(params)
        T_0 = CAP.T_0(params)

        # auxiliary quantities
        r² = FT(0)
        r² += ((x - x_c) / x_r)^2 + ((z - z_c) / z_r)^2
        if ndims == 3
            (; y) = local_geometry.coordinates
            y_c = FT(500)
            y_r = FT(250)
            r² += ((y - y_c) / y_r)^2
        end
        θ_p =
            sqrt(r²) < r_c ? FT(1 / 2) * θ_c * (FT(1) + cospi(sqrt(r²) / r_c)) :
            FT(0) # potential temperature perturbation
        θ = θ_b + θ_p # potential temperature
        π_exn = FT(1) - grav * z / cp_d / θ # exner function
        T = π_exn * θ # temperature
        p = p_0 * π_exn^(cp_d / R_d) # pressure
        ρ = p / R_d / T # density
        q = ρ * FT(0)

        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseEquil_pTq(thermo_params, p, T, q),
        )
    end
    return local_state
end

##
## Baroclinic Wave
##

function shallow_atmos_baroclinic_wave_values(z, ϕ, λ, params, perturb)
    FT = eltype(params)
    R_d = CAP.R_d(params)
    MSLP = CAP.MSLP(params)
    grav = CAP.grav(params)
    Ω = CAP.Omega(params)
    R = CAP.planet_radius(params)

    # Constants from paper
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

    # Virtual temperature and pressure
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

    # Horizontal velocity
    U = grav * k / R * τ_int_2 * T_v * (cosd(ϕ)^(k - 1) - cosd(ϕ)^(k + 1))
    u = -Ω * R * cosd(ϕ) + sqrt((Ω * R * cosd(ϕ))^2 + R * cosd(ϕ) * U)
    v = FT(0)
    if perturb
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

    return (; T_v, p, u, v)
end

function deep_atmos_baroclinic_wave_values(z, ϕ, λ, params, perturb)
    FT = eltype(params)
    R_d = CAP.R_d(params)
    MSLP = CAP.MSLP(params)
    grav = CAP.grav(params)
    Ω = CAP.Omega(params)
    R = CAP.planet_radius(params)

    # Constants from paper (See Table 1. in Ullrich et al (2014))
    k = 3         # Power for temperature field
    T_e = FT(310) # Surface temperature at the equator
    T_p = FT(240) # Surface temperature at the pole
    T_0 = FT(0.5) * (T_e + T_p)
    Γ = FT(0.005) # Lapse rate
    A = 1 / Γ  # (Eq 16)
    B = (T_0 - T_p) / T_0 / T_p # (Eq 17)
    C = FT(0.5) * (k + 2) * (T_e - T_p) / T_e / T_p # (Eq 17)
    b = 2 # half-width parameter
    H = R_d * T_0 / grav
    z_t = FT(15e3) # Top of perturbation domain
    λ_c = FT(20) # Geographical location (λ dim) of perturbation center
    ϕ_c = FT(40) # Geographical location (ϕ dim) of perturbation center
    d_0 = R / 6
    V_p = FT(1)

    # Virtual temperature and pressure
    τ̃₁ =
        A * Γ / T_0 * exp(Γ * z / T_0) +
        B * (1 - 2 * (z / b / H)^2) * exp(-(z / b / H)^2)# (Eq 14)
    τ̃₂ = C * (1 - 2 * (z / b / H)^2) * exp(-(z / b / H)^2) # (Eq 15)
    ∫τ̃₁ = (A * (exp(Γ * z / T_0) - 1)) + B * z * exp(-(z / b / H)^2) # (Eq A1)
    ∫τ̃₂ = C * z * exp(-(z / b / H)^2) # (Eq A2)
    I_T =
        ((z + R) / R * cosd(ϕ))^k -
        (k / (k + 2)) * ((z + R) / R * cosd(ϕ))^(k + 2)
    T_v = FT((R / (z + R))^2 * (τ̃₁ - τ̃₂ * I_T)^(-1)) # (Eq A3)
    p = FT(MSLP * exp(-grav / R_d * (∫τ̃₁ - ∫τ̃₂ * I_T))) # (Eq A6)
    # Horizontal velocity
    U =
        grav / R *
        k *
        T_v *
        ∫τ̃₂ *
        (((z + R) * cosd(ϕ) / R)^(k - 1) - ((R + z) * cosd(ϕ) / R)^(k + 1)) # wind-proxy (Eq A4)
    u = FT(
        -Ω * (R + z) * cosd(ϕ) +
        sqrt((Ω * (R + z) * cosd(ϕ))^2 + (R + z) * cosd(ϕ) * U),
    )
    v = FT(0)
    if perturb
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
    return (; T_v, p, u, v)
end

function moist_baroclinic_wave_values(z, ϕ, λ, params, perturb, deep_atmosphere)
    FT = eltype(params)
    MSLP = CAP.MSLP(params)

    # Constants from paper
    p_w = FT(3.4e4)
    p_t = FT(1e4)
    q_t = FT(1e-12)
    q_0 = FT(0.018)
    ϕ_w = FT(40)
    ε = FT(0.608)

    if deep_atmosphere
        (; p, T_v, u, v) =
            deep_atmos_baroclinic_wave_values(z, ϕ, λ, params, perturb)
    else
        (; p, T_v, u, v) =
            shallow_atmos_baroclinic_wave_values(z, ϕ, λ, params, perturb)
    end

    q_tot =
        (p <= p_t) ? q_t : q_0 * exp(-(ϕ / ϕ_w)^4) * exp(-((p - MSLP) / p_w)^2)
    T = T_v / (1 + ε * q_tot) # This is the formula used in the paper.

    # This is the actual formula, which would be consistent with TD:
    # T = T_v * (1 + q_tot) / (1 + q_tot * CAP.molmass_ratio(params))

    return (; T, p, q_tot, u, v)
end

"""
    DryBaroclinicWave(; perturb = true, deep_atmosphere = false)

An `InitialCondition` with a dry baroclinic wave, and with an optional
perturbation to the horizontal velocity.
"""
Base.@kwdef struct DryBaroclinicWave <: InitialCondition
    perturb::Bool = true
    deep_atmosphere::Bool = false
end

function (initial_condition::DryBaroclinicWave)(params)
    (; perturb, deep_atmosphere) = initial_condition
    function local_state(local_geometry)
        thermo_params = CAP.thermodynamics_params(params)
        (; z, lat, long) = local_geometry.coordinates
        if deep_atmosphere
            (; p, T_v, u, v) =
                deep_atmos_baroclinic_wave_values(z, lat, long, params, perturb)
        else
            (; p, T_v, u, v) = shallow_atmos_baroclinic_wave_values(
                z,
                lat,
                long,
                params,
                perturb,
            )
        end

        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseDry_pT(thermo_params, p, T_v),
            velocity = Geometry.UVVector(u, v),
        )
    end
    return local_state
end

"""
    MoistBaroclinicWave(; perturb = true, deep_atmosphere = false)

An `InitialCondition` with a moist baroclinic wave, and with an optional
perturbation to the horizontal velocity.
"""
Base.@kwdef struct MoistBaroclinicWave <: InitialCondition
    perturb::Bool = true
    deep_atmosphere::Bool = false
end

function (initial_condition::MoistBaroclinicWave)(params)
    (; perturb, deep_atmosphere) = initial_condition
    function local_state(local_geometry)
        thermo_params = CAP.thermodynamics_params(params)
        (; z, lat, long) = local_geometry.coordinates
        (; p, T, q_tot, u, v) = moist_baroclinic_wave_values(
            z,
            lat,
            long,
            params,
            perturb,
            deep_atmosphere,
        )
        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseEquil_pTq(thermo_params, p, T, q_tot),
            velocity = Geometry.UVVector(u, v),
        )
    end
    return local_state
end

"""
    MoistBaroclinicWaveWithEDMF(; perturb = true, deep_atmosphere = false)

The same `InitialCondition` as `MoistBaroclinicWave`, except with an initial TKE
of 0 and an initial draft area fraction of 0.2.
"""
Base.@kwdef struct MoistBaroclinicWaveWithEDMF <: InitialCondition
    perturb::Bool = true
    deep_atmosphere::Bool = false
end

function (initial_condition::MoistBaroclinicWaveWithEDMF)(params)
    (; perturb, deep_atmosphere) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        (; z, lat, long) = local_geometry.coordinates
        (; p, T, q_tot, u, v) = moist_baroclinic_wave_values(
            z,
            lat,
            long,
            params,
            perturb,
            deep_atmosphere,
        )
        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseEquil_pTq(thermo_params, p, T, q_tot),
            velocity = Geometry.UVVector(u, v),
            turbconv_state = EDMFState(; tke = FT(0), draft_area = FT(0.2)),
        )
    end
    return local_state
end

##
## EDMFX Test
##
"""
    MoistAdiabaticProfileEDMFX(; perturb = true)

An `InitialCondition` with a moist adiabatic temperature profile, and with an optional
perturbation to the temperature.
"""
Base.@kwdef struct MoistAdiabaticProfileEDMFX <: InitialCondition
    perturb::Bool = false
end

draft_area(::Type{FT}) where {FT} =
    z -> z < 0.7e4 ? FT(0.5) * exp(-(z - FT(4e3))^2 / 2 / FT(1e3)^2) : FT(0)

edmfx_q_tot(::Type{FT}) where {FT} =
    z -> z < 0.7e4 ? FT(1e-3) * exp(-(z - FT(4e3))^2 / 2 / FT(1e3)^2) : FT(0)

function (initial_condition::MoistAdiabaticProfileEDMFX)(params)
    (; perturb) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        temp_profile = DryAdiabaticProfile{FT}(thermo_params, FT(330), FT(200))

        (; z) = local_geometry.coordinates
        coeff = perturb_coeff(local_geometry.coordinates)
        T, p = temp_profile(thermo_params, z)
        if perturb
            T += coeff * FT(0.1) * (z < 5000)
        end
        q_tot = edmfx_q_tot(FT)(z)

        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseEquil_pTq(thermo_params, p, T, q_tot),
            turbconv_state = EDMFState(;
                tke = FT(0),
                draft_area = draft_area(FT)(z),
                velocity = Geometry.WVector(FT(1.0)),
            ),
        )
    end
    return local_state
end

"""
    SimplePlume(; perturb = true)

An `InitialCondition` with a moist adiabatic temperature profile
"""
Base.@kwdef struct SimplePlume <: InitialCondition
    prognostic_tke::Bool = false
end

function (initial_condition::SimplePlume)(params)
    function local_state(local_geometry)
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        temp_profile = DryAdiabaticProfile{FT}(thermo_params, FT(310), FT(290))

        (; z) = local_geometry.coordinates
        T, p = temp_profile(thermo_params, z)
        q_tot = FT(0)

        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseEquil_pTq(thermo_params, p, T, q_tot),
            turbconv_state = EDMFState(; tke = FT(0)),
        )
    end
    return local_state
end
##
## EDMF Test Cases
##
# TODO: Get rid of this
import AtmosphericProfilesLibrary as APL

const FunctionOrSpline =
    Union{Function, APL.AbstractProfile, Intp.Extrapolation}

"""
    hydrostatic_pressure_profile(; thermo_params, p_0, [T, θ, q_tot, z_max])

Solves the initial value problem `p'(z) = -g * ρ(z)` for all `z ∈ [0, z_max]`,
given `p(0)`, either `T(z)` or `θ(z)`, and optionally also `q_tot(z)`. If
`q_tot(z)` is not given, it is assumed to be 0. If `z_max` is not given, it is
assumed to be 30 km. Note that `z_max` should be the maximum elevation to which
the specified profiles T(z), θ(z), and/or q_tot(z) are valid.
"""
function hydrostatic_pressure_profile(;
    thermo_params,
    p_0,
    T = nothing,
    θ = nothing,
    q_tot = nothing,
    z_max = 30000,
)
    FT = eltype(thermo_params)
    grav = TD.Parameters.grav(thermo_params)

    ts(p, z, ::Nothing, ::Nothing, _) = error("Either T or θ must be specified")
    ts(p, z, T::FunctionOrSpline, θ::FunctionOrSpline, _) =
        error("Only one of T and θ can be specified")
    ts(p, z, T::FunctionOrSpline, ::Nothing, ::Nothing) =
        TD.PhaseDry_pT(thermo_params, p, oftype(p, T(z)))
    ts(p, z, ::Nothing, θ::FunctionOrSpline, ::Nothing) =
        TD.PhaseDry_pθ(thermo_params, p, oftype(p, θ(z)))
    ts(p, z, T::FunctionOrSpline, ::Nothing, q_tot::FunctionOrSpline) =
        TD.PhaseEquil_pTq(
            thermo_params,
            p,
            oftype(p, T(z)),
            oftype(p, q_tot(z)),
        )
    ts(p, z, ::Nothing, θ::FunctionOrSpline, q_tot::FunctionOrSpline) =
        TD.PhaseEquil_pθq(
            thermo_params,
            p,
            oftype(p, θ(z)),
            oftype(p, q_tot(z)),
        )
    dp_dz(p, z) = -grav * TD.air_density(thermo_params, ts(p, z, T, θ, q_tot))

    return column_indefinite_integral(dp_dz, p_0, (FT(0), FT(z_max)))
end

"""
    Nieuwstadt

The `InitialCondition` described in [Nieuwstadt1993](@cite), but with a
hydrostatically balanced pressure profile.
"""
Base.@kwdef struct Nieuwstadt <: InitialCondition
    prognostic_tke::Bool = false
end

"""
    GABLS

The `InitialCondition` described in [Kosovic2000](@cite), but with a hydrostatically
balanced pressure profile.
"""
Base.@kwdef struct GABLS <: InitialCondition
    prognostic_tke::Bool = false
end

for IC in (:Nieuwstadt, :GABLS)
    θ_func_name = Symbol(IC, :_θ_liq_ice)
    u_func_name = Symbol(IC, :_u)
    tke_func_name = Symbol(IC, :_tke_prescribed)
    @eval function (initial_condition::$IC)(params)
        (; prognostic_tke) = initial_condition
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        p_0 = FT(100000.0)
        θ = APL.$θ_func_name(FT)
        p = hydrostatic_pressure_profile(; thermo_params, p_0, θ)
        u = APL.$u_func_name(FT)
        tke = APL.$tke_func_name(FT)
        function local_state(local_geometry)
            (; z) = local_geometry.coordinates
            return LocalState(;
                params,
                geometry = local_geometry,
                thermo_state = TD.PhaseDry_pθ(thermo_params, p(z), θ(z)),
                velocity = Geometry.UVector(u(z)),
                turbconv_state = EDMFState(;
                    tke = prognostic_tke ? FT(0) : tke(z),
                ),
            )
        end
        return local_state
    end
end

"""
    GATE_III

The `InitialCondition` described in [Khairoutdinov2009](@cite), but with a
hydrostatically balanced pressure profile.
"""
Base.@kwdef struct GATE_III <: InitialCondition
    prognostic_tke::Bool = false
end

function (initial_condition::GATE_III)(params)
    (; prognostic_tke) = initial_condition
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = FT(101500.0)
    T = APL.GATE_III_T(FT)
    q_tot = APL.GATE_III_q_tot(FT)
    p = hydrostatic_pressure_profile(; thermo_params, p_0, T, q_tot)
    u = APL.GATE_III_u(FT)
    tke = APL.GATE_III_tke(FT)
    function local_state(local_geometry)
        (; z) = local_geometry.coordinates
        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseEquil_pTq(
                thermo_params,
                p(z),
                T(z),
                q_tot(z),
            ),
            velocity = Geometry.UVector(u(z)),
            turbconv_state = EDMFState(; tke = prognostic_tke ? FT(0) : tke(z)),
        )
    end
    return local_state
end

"""
    Soares

The `InitialCondition` described in [Soares2004](@cite), but with a
hydrostatically balanced pressure profile.
"""
Base.@kwdef struct Soares <: InitialCondition
    prognostic_tke::Bool = false
end

"""
    Bomex

The `InitialCondition` described in [Holland1973](@cite), but with a hydrostatically
balanced pressure profile.
"""
Base.@kwdef struct Bomex <: InitialCondition
    prognostic_tke::Bool = false
end

"""
    LifeCycleTan2018

The `InitialCondition` described in [Tan2018](@cite), but with a hydrostatically
balanced pressure profile.
"""
Base.@kwdef struct LifeCycleTan2018 <: InitialCondition
    prognostic_tke::Bool = false
end

"""
    ARM_SGP

The `InitialCondition` described in [Brown2002](@cite), but with a
hydrostatically balanced pressure profile.
"""
Base.@kwdef struct ARM_SGP <: InitialCondition
    prognostic_tke::Bool = false
end

for IC in (:Soares, :Bomex, :LifeCycleTan2018, :ARM_SGP)
    θ_func_name = Symbol(IC, :_θ_liq_ice)
    q_tot_func_name = Symbol(IC, :_q_tot)
    u_func_name = Symbol(IC, :_u)
    tke_func_name = Symbol(IC, :_tke_prescribed)
    @eval function (initial_condition::$IC)(params)
        (; prognostic_tke) = initial_condition
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        p_0 = FT(
            $IC <: Bomex || $IC <: LifeCycleTan2018 ? 101500.0 :
            $IC <: Soares ? 100000.0 :
            $IC <: ARM_SGP ? 97000.0 :
            error("Invalid Initial Condition : $($IC)"),
        )
        θ = APL.$θ_func_name(FT)
        q_tot = APL.$q_tot_func_name(FT)
        p = hydrostatic_pressure_profile(; thermo_params, p_0, θ, q_tot)
        u = APL.$u_func_name(FT)
        tke = APL.$tke_func_name(FT)
        function local_state(local_geometry)
            (; z) = local_geometry.coordinates
            return LocalState(;
                params,
                geometry = local_geometry,
                thermo_state = TD.PhaseEquil_pθq(
                    thermo_params,
                    p(z),
                    θ(z),
                    q_tot(z),
                ),
                velocity = Geometry.UVector(u(z)),
                turbconv_state = EDMFState(;
                    tke = prognostic_tke ? FT(0) : tke(z),
                ),
            )
        end
        return local_state
    end
end

"""
    DYCOMS_RF01

The `InitialCondition` described in [Stevens2005](@cite), but with a
hydrostatically balanced pressure profile.
"""
Base.@kwdef struct DYCOMS_RF01 <: InitialCondition
    prognostic_tke::Bool = false
end

"""
    DYCOMS_RF02

The `InitialCondition` described in [Ackerman2009](@cite), but with a
hydrostatically balanced pressure profile.
"""
Base.@kwdef struct DYCOMS_RF02 <: InitialCondition
    prognostic_tke::Bool = false
end

for IC in (:Dycoms_RF01, :Dycoms_RF02)
    IC_Type = Symbol(uppercase(string(IC)))
    θ_func_name = Symbol(IC, :_θ_liq_ice)
    q_tot_func_name = Symbol(IC, :_q_tot)
    u_func_name = Symbol(IC, IC == :Dycoms_RF01 ? :_u0 : :_u)
    v_func_name = Symbol(IC, IC == :Dycoms_RF01 ? :_v0 : :_v)
    tke_func_name = Symbol(IC, :_tke_prescribed)
    @eval function (initial_condition::$IC_Type)(params)
        (; prognostic_tke) = initial_condition
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        p_0 = FT(101780.0)
        θ = APL.$θ_func_name(FT)
        q_tot = APL.$q_tot_func_name(FT)
        p = hydrostatic_pressure_profile(; thermo_params, p_0, θ, q_tot)
        u = APL.$u_func_name(FT)
        v = APL.$v_func_name(FT)
        #tke = APL.$tke_func_name(FT)
        tke = APL.Dycoms_RF01_tke_prescribed(FT) #TODO - dont have the tke profile for Dycoms_RF02
        function local_state(local_geometry)
            (; z) = local_geometry.coordinates
            return LocalState(;
                params,
                geometry = local_geometry,
                thermo_state = TD.PhaseEquil_pθq(
                    thermo_params,
                    p(z),
                    θ(z),
                    q_tot(z),
                ),
                velocity = Geometry.UVVector(u(z), v(z)),
                turbconv_state = EDMFState(;
                    tke = prognostic_tke ? FT(0) : tke(z),
                ),
            )
        end
        return local_state
    end
end

"""
    Rico

The `InitialCondition` described in [Rauber2007](@cite), but with a hydrostatically
balanced pressure profile.
"""
Base.@kwdef struct Rico <: InitialCondition
    prognostic_tke::Bool = false
end

function (initial_condition::Rico)(params)
    (; prognostic_tke) = initial_condition
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = FT(101540.0)
    θ = APL.Rico_θ_liq_ice(FT)
    q_tot = APL.Rico_q_tot(FT)
    p = hydrostatic_pressure_profile(; thermo_params, p_0, θ, q_tot)
    u = APL.Rico_u(FT)
    v = APL.Rico_v(FT)
    tke = APL.Rico_tke_prescribed(FT)
    #tke = z -> z < 2980 ? 1 - z / 2980 : FT(0) # TODO: Move this to APL.
    function local_state(local_geometry)
        (; z) = local_geometry.coordinates
        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseEquil_pθq(
                thermo_params,
                p(z),
                θ(z),
                q_tot(z),
            ),
            velocity = Geometry.UVVector(u(z), v(z)),
            turbconv_state = EDMFState(; tke = prognostic_tke ? FT(0) : tke(z)),
        )
    end
    return local_state
end

"""
    TRMM_LBA

The `InitialCondition` described in [Grabowski2006](@cite), but with a
hydrostatically balanced pressure profile.
"""
Base.@kwdef struct TRMM_LBA <: InitialCondition
    prognostic_tke::Bool = false
end

function (initial_condition::TRMM_LBA)(params)
    (; prognostic_tke) = initial_condition
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = FT(99130.0)
    T = APL.TRMM_LBA_T(FT)

    # Set q_tot to the value implied by the measured pressure and relative
    # humidity profiles (see the definition of relative humidity and equation 37
    # in Pressel et al.). Note that the measured profiles are different from the
    # ones required for hydrostatic balance.
    # TODO: Move this to APL.
    molmass_ratio = TD.Parameters.molmass_ratio(thermo_params)
    measured_p = APL.TRMM_LBA_p(FT)
    measured_RH = APL.TRMM_LBA_RH(FT)
    measured_z_values = APL.TRMM_LBA_z(FT)
    measured_q_tot_values = map(measured_z_values) do z
        p_v_sat = TD.saturation_vapor_pressure(thermo_params, T(z), TD.Liquid())
        denominator =
            measured_p(z) - p_v_sat +
            (1 / molmass_ratio) * p_v_sat * measured_RH(z) / 100
        q_v_sat = p_v_sat * (1 / molmass_ratio) / denominator
        return q_v_sat * measured_RH(z) / 100
    end
    q_tot = Intp.extrapolate(
        Intp.interpolate(
            (measured_z_values,),
            measured_q_tot_values,
            Intp.Gridded(Intp.Linear()),
        ),
        Intp.Flat(),
    )

    p = hydrostatic_pressure_profile(; thermo_params, p_0, T, q_tot)
    u = APL.TRMM_LBA_u(FT)
    v = APL.TRMM_LBA_v(FT)
    tke = APL.TRMM_LBA_tke_prescribed(FT)
    function local_state(local_geometry)
        (; z) = local_geometry.coordinates
        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseEquil_pTq(
                thermo_params,
                p(z),
                T(z),
                q_tot(z),
            ),
            velocity = Geometry.UVVector(u(z), v(z)),
            turbconv_state = EDMFState(; tke = prognostic_tke ? FT(0) : tke(z)),
        )
    end
    return local_state
end

"""
    PrecipitatingColumn

A 1-dimensional precipitating column test
"""
struct PrecipitatingColumn <: InitialCondition end

prescribed_prof(::Type{FT}, z_mid, z_max, val) where {FT} =
    z -> z < z_max ? FT(val) * exp(-(z - FT(z_mid))^2 / 2 / FT(1e3)^2) : FT(0)

function (initial_condition::PrecipitatingColumn)(params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = FT(101300.0)
    qᵣ = prescribed_prof(FT, 2000, 5000, 1e-6)
    qₛ = prescribed_prof(FT, 5000, 8000, 2e-6)
    qₗ = prescribed_prof(FT, 4000, 5500, 2e-5)
    qᵢ = prescribed_prof(FT, 6000, 9000, 1e-5)
    θ = APL.Rico_θ_liq_ice(FT)
    q_tot = APL.Rico_q_tot(FT)
    u = prescribed_prof(FT, 0, Inf, 0)
    v = prescribed_prof(FT, 0, Inf, 0)
    p = hydrostatic_pressure_profile(; thermo_params, p_0, θ, q_tot)
    function local_state(local_geometry)
        (; z) = local_geometry.coordinates
        ts = TD.PhaseNonEquil_pθq(
            thermo_params,
            p(z),
            θ(z),
            TD.PhasePartition(q_tot(z), qₗ(z), qᵢ(z)),
        )
        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = ts,
            velocity = Geometry.UVVector(u(z), v(z)),
            turbconv_state = nothing,
            precip_state = PrecipState1M(; q_rai = qᵣ(z), q_sno = qₛ(z)),
        )
    end
    return local_state
end

"""
    GCMDriven <: InitialCondition

The `InitialCondition` from a provided GCM forcing file, with data type `DType`.
"""
struct GCMDriven <: InitialCondition
    external_forcing_file::String
    cfsite_number::String
end

function (initial_condition::GCMDriven)(params)
    (; external_forcing_file, cfsite_number) = initial_condition
    thermo_params = CAP.thermodynamics_params(params)

    # Read forcing file
    z_gcm = NC.NCDataset(external_forcing_file) do ds
        vec(gcm_height(ds.group[cfsite_number]))
    end
    vars = gcm_initial_conditions(external_forcing_file, cfsite_number)
    T, u, v, q_tot, ρ₀ = map(vars) do value
        Intp.extrapolate(
            Intp.interpolate((z_gcm,), value, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
    end

    function local_state(local_geometry)
        (; z) = local_geometry.coordinates
        FT = typeof(z)
        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = ts = TD.PhaseEquil_ρTq(
                thermo_params,
                FT(ρ₀(z)),
                FT(T(z)),
                FT(q_tot(z)),
            ),
            velocity = Geometry.UVVector(FT(u(z)), FT(v(z))),
            turbconv_state = EDMFState(; tke = FT(0)),
        )
    end
    return local_state
end

function gcm_initial_conditions(external_forcing_file, cfsite_number)
    NC.NCDataset(external_forcing_file) do ds
        (  # TODO: Cast to CuVector for GPU compatibility
            gcm_driven_profile_tmean(ds.group[cfsite_number], "ta"),
            gcm_driven_profile_tmean(ds.group[cfsite_number], "ua"),
            gcm_driven_profile_tmean(ds.group[cfsite_number], "va"),
            gcm_driven_profile_tmean(ds.group[cfsite_number], "hus"),
            vec(mean(1 ./ ds.group[cfsite_number]["alpha"][:, :], dims = 2)), # convert alpha to rho using rho=1/alpha, take average profile
        )
    end
end
