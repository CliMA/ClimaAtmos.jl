"""
    InitialCondition

A mechanism for specifying the `LocalState` of an `AtmosModel` at every point in
the domain. Given some `initial_condition`, calling `initial_condition(params)`
returns a function of the form `local_state(local_geometry)::LocalState`.
"""
abstract type InitialCondition end

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
        T, p = temp_profile(thermo_params, z)
        if perturb
            T += rand(FT) * FT(0.1) * (z < 5000)
        end

        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseDry_pT(thermo_params, p, T),
        )
    end
    return local_state
end

##
## Baroclinic Wave
##

function baroclinic_wave_values(z, ϕ, λ, params, perturb)
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

function moist_baroclinic_wave_values(z, ϕ, λ, params, perturb)
    FT = eltype(params)
    MSLP = CAP.MSLP(params)

    # Constants from paper
    p_w = FT(3.4e4)
    p_t = FT(1e4)
    q_t = FT(1e-12)
    q_0 = FT(0.018)
    ϕ_w = FT(40)
    ε = FT(0.608)

    (; T_v, p, u, v) = baroclinic_wave_values(z, ϕ, λ, params, perturb)
    q_tot =
        (p <= p_t) ? q_t : q_0 * exp(-(ϕ / ϕ_w)^4) * exp(-((p - MSLP) / p_w)^2)
    T = T_v / (1 + ε * q_tot) # This is the formula used in the paper.

    # This is the actual formula, which would be consistent with TD:
    # T = T_v * (1 + q_tot) / (1 + q_tot * CAP.molmass_ratio(params))

    return (; T, p, q_tot, u, v)
end

"""
    DryBaroclinicWave(; perturb = true)

An `InitialCondition` with a dry baroclinic wave, and with an optional
perturbation to the horizontal velocity.
"""
Base.@kwdef struct DryBaroclinicWave <: InitialCondition
    perturb::Bool = true
end

function (initial_condition::DryBaroclinicWave)(params)
    (; perturb) = initial_condition
    function local_state(local_geometry)
        thermo_params = CAP.thermodynamics_params(params)
        (; z, lat, long) = local_geometry.coordinates
        (; p, T_v, u, v) = baroclinic_wave_values(z, lat, long, params, perturb)
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
    MoistBaroclinicWave(; perturb = true)

An `InitialCondition` with a moist baroclinic wave, and with an optional
perturbation to the horizontal velocity.
"""
Base.@kwdef struct MoistBaroclinicWave <: InitialCondition
    perturb::Bool = true
end

function (initial_condition::MoistBaroclinicWave)(params)
    (; perturb) = initial_condition
    function local_state(local_geometry)
        thermo_params = CAP.thermodynamics_params(params)
        (; z, lat, long) = local_geometry.coordinates
        (; p, T, q_tot, u, v) =
            moist_baroclinic_wave_values(z, lat, long, params, perturb)
        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseEquil_pTq(thermo_params, p, T, q_tot),
            velocity = Geometry.UVVector(u, v),
        )
    end
    return local_state
end

##
## EDMF Test Cases
##

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
    ts(p, z, T, θ, _) = error("Only one of T and θ can be specified")
    ts(p, z, T, ::Nothing, ::Nothing) = TD.PhaseDry_pT(thermo_params, p, T(z))
    ts(p, z, ::Nothing, θ, ::Nothing) = TD.PhaseDry_pθ(thermo_params, p, θ(z))
    ts(p, z, T, ::Nothing, q_tot) =
        TD.PhaseEquil_pTq(thermo_params, p, T(z), q_tot(z))
    ts(p, z, ::Nothing, θ, q_tot) =
        TD.PhaseEquil_pθq(thermo_params, p, θ(z), q_tot(z))
    dp_dz(p, _, z) =
        -grav * TD.air_density(thermo_params, ts(p, z, T, θ, q_tot))

    prob = ODE.ODEProblem(dp_dz, p_0, (FT(0), z_max))
    return ODE.solve(prob, ODE.Tsit5(), reltol = 10eps(FT), abstol = 10eps(FT))
end

"""
    Nieuwstadt

The `InitialCondition` described in [Nieuwstadt1993](@cite), but with a
hydrostatically balanced pressure profile.
"""
struct Nieuwstadt <: InitialCondition end

"""
    GABLS

The `InitialCondition` described in [???](@cite), but with a hydrostatically
balanced pressure profile.
"""
struct GABLS <: InitialCondition end

for IC in (:Nieuwstadt, :GABLS)
    θ_func_name = Symbol(IC, :_θ_liq_ice)
    u_func_name = Symbol(IC, :_u)
    tke_func_name = Symbol(IC, :_tke_prescribed)
    @eval function (initial_condition::$IC)(params)
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        p_0 = initial_surface_pressure(initial_condition, thermo_params)
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
                turbconv_state = EDMFState(; tke = tke(z)),
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
struct GATE_III <: InitialCondition end

function (initial_condition::GATE_III)(params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = initial_surface_pressure(initial_condition, thermo_params)
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
            turbconv_state = EDMFState(; tke = tke(z)),
        )
    end
    return local_state
end

"""
    Soares

The `InitialCondition` described in [Soares2004](@cite), but with a
hydrostatically balanced pressure profile.
"""
struct Soares <: InitialCondition end

"""
    Bomex

The `InitialCondition` described in [???](@cite), but with a hydrostatically
balanced pressure profile.
"""
struct Bomex <: InitialCondition end

"""
    LifeCycleTan2018

The `InitialCondition` described in [Tan2018](@cite), but with a hydrostatically
balanced pressure profile.
"""
struct LifeCycleTan2018 <: InitialCondition end

"""
    ARM_SGP

The `InitialCondition` described in [Brown2002](@cite), but with a
hydrostatically balanced pressure profile.
"""
struct ARM_SGP <: InitialCondition end

for IC in (:Soares, :Bomex, :LifeCycleTan2018, :ARM_SGP)
    θ_func_name = Symbol(IC, :_θ_liq_ice)
    q_tot_func_name = Symbol(IC, :_q_tot)
    u_func_name = Symbol(IC, :_u)
    tke_func_name = Symbol(IC, :_tke_prescribed)
    @eval function (initial_condition::$IC)(params)
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        p_0 = initial_surface_pressure(initial_condition, thermo_params)
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
                turbconv_state = EDMFState(; tke = tke(z)),
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
struct DYCOMS_RF01 <: InitialCondition end

"""
    DYCOMS_RF02

The `InitialCondition` described in [Ackerman2009](@cite), but with a
hydrostatically balanced pressure profile.
"""
struct DYCOMS_RF02 <: InitialCondition end

for IC in (:Dycoms_RF01, :Dycoms_RF02)
    IC_Type = Symbol(uppercase(string(IC)))
    θ_func_name = Symbol(IC, :_θ_liq_ice)
    q_tot_func_name = Symbol(IC, :_q_tot)
    u_func_name = Symbol(IC, IC == :Dycoms_RF01 ? :_u0 : :_u)
    v_func_name = Symbol(IC, IC == :Dycoms_RF01 ? :_v0 : :_v)
    tke_func_name = Symbol(IC, :_tke_prescribed)
    @eval function (initial_condition::$IC_Type)(params)
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        p_0 = initial_surface_pressure(initial_condition, thermo_params)
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
                turbconv_state = EDMFState(; tke = tke(z)),
            )
        end
        return local_state
    end
end

"""
    Rico

The `InitialCondition` described in [???](@cite), but with a hydrostatically
balanced pressure profile.
"""
struct Rico <: InitialCondition end

function (initial_condition::Rico)(params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = initial_surface_pressure(initial_condition, thermo_params)
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
            turbconv_state = EDMFState(; tke = tke(z)),
        )
    end
    return local_state
end

"""
    TRMM_LBA

The `InitialCondition` described in [Grabowski2006](@cite), but with a
hydrostatically balanced pressure profile.
"""
struct TRMM_LBA <: InitialCondition end

function (initial_condition::TRMM_LBA)(params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = initial_surface_pressure(initial_condition, thermo_params)
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
    q_tot = Dierckx.Spline1D(measured_z_values, measured_q_tot_values; k = 1)

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
            turbconv_state = EDMFState(; tke = tke(z)),
        )
    end
    return local_state
end

##
## TODO: Temporary workaround for TC using its own surface parametrization.
##

# By default, use the dycore's surface parametrization.
surface_params(::InitialCondition, thermo_params) = nothing

function surface_params(::Nieuwstadt, thermo_params)
    FT = eltype(thermo_params)
    zrough::FT = 0.16 #1.0e-4 0.16 is the value specified in the Nieuwstadt paper.
    psurface::FT = 1000 * 100
    Tsurface::FT = 300.0
    θ_flux::FT = 6.0e-2
    lhf::FT = 0.0 # It would be 0.0 if we follow Nieuwstadt.
    ts = TD.PhaseDry_pT(thermo_params, psurface, Tsurface)
    shf =
        θ_flux * TD.air_density(thermo_params, ts) * TD.cp_m(thermo_params, ts)
    return TC.FixedSurfaceFlux(zrough, ts, shf, lhf)
end

function surface_params(::GABLS, thermo_params)
    FT = eltype(thermo_params)
    psurface::FT = 1.0e5
    Tsurface = t -> 265 - (FT(0.25) / 3600) * t
    zrough::FT = 0.1
    ts = t -> TD.PhaseDry_pT(thermo_params, psurface, Tsurface(t))
    return TC.MoninObukhovSurface(; ts, zrough)
end

# TODO: The paper only specifies that Tsurface = 299.88. Where did all of these
# values come from?
function surface_params(::GATE_III, thermo_params)
    FT = eltype(thermo_params)
    psurface::FT = 1013 * 100
    qsurface::FT = 16.5 / 1000.0 # kg/kg
    cm = zc_surf -> FT(0.0012)
    ch = zc_surf -> FT(0.0034337)
    cq = zc_surf -> FT(0.0034337)
    Tsurface::FT = 299.184

    # For GATE_III we provide values of transfer coefficients
    ts = TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    return TC.FixedSurfaceCoeffs(; zrough = FT(0), ts, ch, cm)
end

function surface_params(::Soares, thermo_params)
    FT = eltype(thermo_params)
    zrough::FT = 0.16 #1.0e-4 0.16 is the value specified in the Nieuwstadt paper.
    psurface::FT = 1000 * 100
    Tsurface::FT = 300.0
    qsurface::FT = 5.0e-3
    θ_flux::FT = 6.0e-2
    qt_flux::FT = 2.5e-5
    ts = TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    ρsurface = TD.air_density(thermo_params, ts)
    lhf = qt_flux * ρsurface * TD.latent_heat_vapor(thermo_params, ts)
    shf = θ_flux * ρsurface * TD.cp_m(thermo_params, ts)
    return TC.FixedSurfaceFlux(zrough, ts, shf, lhf)
end

function surface_params(::Bomex, thermo_params)
    FT = eltype(thermo_params)
    zrough::FT = 1.0e-4
    psurface::FT = 1.015e5
    qsurface::FT = 22.45e-3 # kg/kg
    Tsurface::FT = 300.4 # Equivalent to θsurface = 299.1
    θ_flux::FT = 8.0e-3
    qt_flux::FT = 5.2e-5
    ts = TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    ρsurface = TD.air_density(thermo_params, ts)
    lhf = qt_flux * ρsurface * TD.latent_heat_vapor(thermo_params, ts)
    shf = θ_flux * ρsurface * TD.cp_m(thermo_params, ts)
    ustar::FT = 0.28 # m/s
    return TC.FixedSurfaceFluxAndFrictionVelocity(zrough, ts, shf, lhf, ustar)
end

function surface_params(::LifeCycleTan2018, thermo_params)
    FT = eltype(thermo_params)
    zrough::FT = 1.0e-4 # not actually used, but initialized to reasonable value
    psurface::FT = 1.015e5
    qsurface::FT = 22.45e-3 # kg/kg
    Tsurface::FT = 300.4 # equivalent to θsurface = 299.1
    θ_flux::FT = 8.0e-3
    qt_flux::FT = 5.2e-5
    ts = TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    ρsurface = TD.air_density(thermo_params, ts)
    lhf0 = qt_flux * ρsurface * TD.latent_heat_vapor(thermo_params, ts)
    shf0 = θ_flux * ρsurface * TD.cp_m(thermo_params, ts)

    weight_factor(t) = FT(0.01) + FT(0.99) * (cos(2 * FT(π) * t / 3600) + 1) / 2
    weight::FT = 1.0
    lhf = t -> lhf0 * (weight * weight_factor(t))
    shf = t -> shf0 * (weight * weight_factor(t))

    ustar::FT = 0.28 # m/s
    return TC.FixedSurfaceFluxAndFrictionVelocity(zrough, ts, shf, lhf, ustar)
end

function surface_params(::ARM_SGP, thermo_params)
    FT = eltype(thermo_params)
    psurface::FT = 970 * 100
    qsurface::FT = 15.2e-3 # kg/kg
    θ_surface::FT = 299.0
    ts = TD.PhaseEquil_pθq(thermo_params, psurface, θ_surface, qsurface)
    ustar::FT = 0.28 # this is taken from Bomex -- better option is to approximate from LES tke above the surface

    t_Sur_in = FT[0.0, 4.0, 6.5, 7.5, 10.0, 12.5, 14.5] .* 3600 #LES time is in sec
    SH = FT[-30.0, 90.0, 140.0, 140.0, 100.0, -10, -10] # W/m^2
    LH = FT[5.0, 250.0, 450.0, 500.0, 420.0, 180.0, 0.0] # W/m^2
    shf = Dierckx.Spline1D(t_Sur_in, SH; k = 1)
    lhf = Dierckx.Spline1D(t_Sur_in, LH; k = 1)
    zrough::FT = 0

    return TC.FixedSurfaceFluxAndFrictionVelocity(zrough, ts, shf, lhf, ustar)
end

function surface_params(::DYCOMS_RF01, thermo_params)
    FT = eltype(thermo_params)
    zrough::FT = 1.0e-4
    ustar::FT = 0.28 # just to initialize grid mean covariances
    shf::FT = 15.0 # sensible heat flux
    lhf::FT = 115.0 # latent heat flux
    psurface::FT = 1017.8 * 100
    Tsurface::FT = 292.5    # K      # i.e. the SST from DYCOMS setup
    qsurface::FT = 13.84e-3 # kg/kg  # TODO - taken from Pycles, maybe it would be better to calculate the q_star(sst) for TurbulenceConvection?
    #density_surface  = 1.22     # kg/m^3
    ts = TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    return TC.FixedSurfaceFlux(zrough, ts, shf, lhf)
end

function surface_params(::DYCOMS_RF02, thermo_params)
    FT = eltype(thermo_params)
    zrough::FT = 1.0e-4  #TODO - not needed?
    ustar::FT = 0.25
    shf::FT = 16.0 # sensible heat flux
    lhf::FT = 93.0 # latent heat flux
    psurface::FT = 1017.8 * 100
    Tsurface::FT = 292.5    # K      # i.e. the SST from DYCOMS setup
    qsurface::FT = 13.84e-3 # kg/kg  # TODO - taken from Pycles, maybe it would be better to calculate the q_star(sst) for TurbulenceConvection?
    ts = TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    return TC.FixedSurfaceFluxAndFrictionVelocity(zrough, ts, shf, lhf, ustar)
end

function surface_params(::Rico, thermo_params)
    FT = eltype(thermo_params)
    zrough::FT = 0.00015
    cm0::FT = 0.001229
    ch0::FT = 0.001094
    cq0::FT = 0.001133
    # Adjust for non-IC grid spacing
    grid_adjust(zc_surf) = (log(20 / zrough) / log(zc_surf / zrough))^2
    cm = zc_surf -> cm0 * grid_adjust(zc_surf)
    ch = zc_surf -> ch0 * grid_adjust(zc_surf)
    cq = zc_surf -> cq0 * grid_adjust(zc_surf) # TODO: not yet used..
    psurface::FT = 1.0154e5
    Tsurface::FT = 299.8

    # Saturated surface condtions for a given surface temperature and pressure
    p_sat_surface =
        TD.saturation_vapor_pressure(thermo_params, Tsurface, TD.Liquid())
    ϵ_v = TD.Parameters.R_d(thermo_params) / TD.Parameters.R_v(thermo_params)
    qsurface = ϵ_v * p_sat_surface / (psurface - p_sat_surface * (1 - ϵ_v))
    ts = TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)

    # For Rico we provide values of transfer coefficients
    return TC.FixedSurfaceCoeffs(; zrough, ts, ch, cm)
end

function surface_params(::TRMM_LBA, thermo_params)
    FT = eltype(thermo_params)
    # zrough = 1.0e-4 # not actually used, but initialized to reasonable value
    zrough::FT = 0 # actually, used, TODO: should we be using the value above?
    psurface::FT = 991.3 * 100
    qsurface::FT = 22.45e-3 # kg/kg
    Tsurface::FT = 273.15 + 23.7
    ts = TD.PhaseEquil_pTq(thermo_params, psurface, Tsurface, qsurface)
    ustar::FT = 0.28 # this is taken from Bomex -- better option is to approximate from LES tke above the surface
    lhf =
        t ->
            554 *
            max(
                0,
                cos(FT(π) / 2 * ((FT(5.25) * 3600 - t) / FT(5.25) / 3600)),
            )^FT(1.3)
    shf =
        t ->
            270 *
            max(
                0,
                cos(FT(π) / 2 * ((FT(5.25) * 3600 - t) / FT(5.25) / 3600)),
            )^FT(1.5)
    return TC.FixedSurfaceFluxAndFrictionVelocity(zrough, ts, shf, lhf, ustar)
end

# This function is only called by the TC initial conditions.
function initial_surface_pressure(initial_condition, thermo_params)
    FT = eltype(thermo_params)
    surf_params = surface_params(initial_condition, thermo_params)
    surf_ts = TC.surface_thermo_state(surf_params, thermo_params, FT(0))
    return TD.air_pressure(thermo_params, surf_ts)
end
