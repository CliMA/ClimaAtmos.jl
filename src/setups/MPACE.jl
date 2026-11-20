"""
    MPACE

The M-PACE (Mixed-Phase Arctic Cloud Experiment) single-layer cloud setup
(Case B) described in [Klein2009](@cite), with a hydrostatically balanced
pressure profile integrated from the pressure-coordinate specification of the
liquid-ice potential temperature and total water mixing ratio.

To support full-column radiative transfer (RRTMGP) with model tops up to 60 km,
the [Klein2009](@cite) tropospheric sounding and large-scale forcings are
extended into the upper atmosphere following Arctic standard-sounding
conventions (cf. [`Larcform1`](@ref)):

  - Temperature follows [Klein2009](@cite) Eq. 1 up to an Arctic tropopause at
    `p_trop = 300` hPa (`T ≈ 224` K) and is held isothermal above `300` hPa.
  - Total specific humidity follows [Klein2009](@cite) Eq. 2 with a
    stratospheric floor `q_top = 3 × 10⁻⁶` kg/kg (`p ≲ 449` hPa).
  - The tropospheric large-scale subsidence (`ω`) and horizontal advective
    tendencies (`∂T/∂t`, `∂q_t/∂t`) follow [Klein2009](@cite) Eqs. 4–6 below
    `550` hPa (`z ≲ 4.7` km) and taper smoothly to zero between `550` and
    `450` hPa so stratospheric humidity is not depleted and `w_sub = -ω/(ρ g)`
    remains bounded aloft.
  - Horizontal winds are nudged toward `(-13, -3)` m/s via [`MPACEForcing`](@ref)
    ([Klein2009](@cite), Section 4).

# Examples

```julia
setup = MPACE(; prognostic_tke = true, thermo_params)
```
"""
struct MPACE{P}
    prognostic_tke::Bool
    profiles::P
end

MPACE(; prognostic_tke::Bool, thermo_params) =
    MPACE(prognostic_tke, mpace_profiles(thermo_params))

"""
    mpace_θ_liq_ice_of_p(p::FT, thermo_params) where {FT}

Return the initial liquid-ice potential temperature [K] at pressure `p` [Pa] for
the M-PACE single-layer cloud case [Klein2009](@cite) (Eq. 1) below the Arctic
tropopause (`p ≥ 300` hPa):

```math
θ_{li}(p) = \\begin{cases}
269.2 \\text{ K}, & p > 850 \\text{ hPa} \\\\
275.33 \\text{ K} + 0.0791 \\text{ K hPa}^{-1} (815 \\text{ hPa} - p), & 300 \\text{ hPa} \\le p \\le 850 \\text{ hPa}
\\end{cases}
```

Above `p_trop = 300` hPa, temperature is held isothermal at the `300` hPa value
(`T_trop ≈ 224.2` K) so upper-tropospheric and stratospheric temperatures remain
within RRTMGP's valid lookup range on deep columns.
"""
function mpace_θ_liq_ice_of_p(
    p::P,
    thermo_params::TD.Parameters.ThermodynamicsParameters{FT},
) where {P <: Real, FT}
    p_trop = FT(30000)
    if p >= p_trop
        p_hPa = p / FT(100)
        return p_hPa > FT(850) ? oftype(p, FT(269.2)) :
               oftype(p, FT(275.33) + FT(0.0791) * (FT(815) - p_hPa))
    else
        p_trop_hPa = p_trop / FT(100)
        θ_trop = FT(275.33) + FT(0.0791) * (FT(815) - p_trop_hPa)
        q_top = mpace_q_tot_of_p(p_trop, FT)
        T_trop = TD.air_temperature(
            thermo_params,
            TD.pθ_li(),
            p_trop,
            θ_trop,
            q_top,
        )
        return oftype(
            p,
            TD.liquid_ice_pottemp_given_pressure(
                thermo_params,
                T_trop,
                p,
                q_top,
                FT(0),
                FT(0),
            ),
        )
    end
end

"""
    mpace_q_tot_of_p(p::Real)

Return the initial total specific humidity [kg/kg] at pressure `p` [Pa] for the
M-PACE single-layer cloud case [Klein2009](@cite) (Eq. 2). Following
[Klein2009](@cite), the linear profile above the inversion (`p ≤ 850` hPa) is
defined in terms of the total water mixing ratio `r_t` [g/kg]:

```math
r_t(p) = \\begin{cases}
1.95 \\text{ g kg}^{-1}, & p > 850 \\text{ hPa} \\\\
\\max(0,\\; 0.291 + 0.00204 (p - 590 \\text{ hPa})) \\text{ g kg}^{-1}, & p \\le 850 \\text{ hPa}
\\end{cases}
```

converted via `q_tot = r_t / (1 + r_t)` and floored at the Arctic stratospheric
value `q_top = 3 × 10⁻⁶` kg/kg (`p ≲ 449` hPa).
"""
function mpace_q_tot_of_p(p::P, ::Type{FT} = P) where {P <: Real, FT <: Real}
    p_hPa = p / FT(100)
    q_top = FT(3e-6)
    r_tot =
        p_hPa > FT(850) ? oftype(p, FT(1.95e-3)) :
        oftype(
            p,
            max(FT(0), (FT(0.291) + FT(0.00204) * (p_hPa - FT(590))) * FT(1e-3)),
        )
    return oftype(p, max(q_top, r_tot / (FT(1) + r_tot)))
end

"""
    mpace_upper_taper(p_hPa::FT) where {FT}

Smooth raised-cosine taper that equals `1` in the [Klein2009](@cite) domain
(`p ≥ 550` hPa, `z ≲ 4.7` km) and vanishes above `450` hPa (`z ≳ 6.1` km).
"""
function mpace_upper_taper(p_hPa::FT) where {FT}
    p_full = FT(550)
    p_zero = FT(450)
    if p_hPa >= p_full
        return FT(1)
    elseif p_hPa <= p_zero
        return FT(0)
    else
        return FT(0.5) * (1 - cos(FT(π) * (p_hPa - p_zero) / (p_full - p_zero)))
    end
end

"""
    mpace_profiles(thermo_params)

Precompute the atmospheric profiles for the M-PACE single-layer cloud case,
given thermodynamic parameters `thermo_params`.

Integrates `dp/dz = -g ρ(p)` upward from `p_0 = 101000` Pa to `z_max = 65000` m
using `mpace_θ_liq_ice_of_p` and `mpace_q_tot_of_p`. Returns a `NamedTuple` of
profile functions of height `z`, with `isbits` GPU-compatible forcing closures
`w_sub`, `prof_dTdt`, and `prof_dqtdt`.
"""
function mpace_profiles(
    thermo_params::TD.Parameters.ThermodynamicsParameters{FT},
) where {FT}
    p_0 = FT(101000)
    z_max = FT(65000)
    grav = TD.Parameters.grav(thermo_params)

    function dp_dz(p, _)
        p_pos = max(p, FT(1))
        θ_val = mpace_θ_liq_ice_of_p(p_pos, thermo_params)
        q_val = mpace_q_tot_of_p(p_pos, FT)
        T_val = TD.air_temperature(
            thermo_params,
            TD.pθ_li(),
            p_pos,
            θ_val,
            q_val,
        )
        ρ_val = TD.air_density(thermo_params, T_val, p_pos, q_val, FT(0), FT(0))
        return -grav * ρ_val
    end

    p = column_indefinite_integral(dp_dz, p_0, (FT(0), z_max); nelems = 6500)
    θ(z) = mpace_θ_liq_ice_of_p(p(z), thermo_params)
    q_tot(z) = mpace_q_tot_of_p(p(z))
    u = Returns(FT(-13))
    v = Returns(FT(-3))
    tke = Returns(FT(0.1))

    # Build a compact `isbits` StaticArrays interpolant on [0, 7000] m for the
    # tendency closures (which vanish above 450 hPa ≈ 6.1 km) so `w_sub`,
    # `prof_dTdt`, and `prof_dqtdt` execute directly on CPU and GPU kernels.
    z_knots = collect(FT(0):FT(250):FT(7000))
    p_low = APL.linear_interp(z_knots, p.(z_knots))

    function w_sub(z)
        z_f = FT(z)
        z_f >= FT(7000) && return FT(0)
        p_val = p_low(z_f)
        p_hPa = p_val / FT(100)
        taper = mpace_upper_taper(p_hPa)
        taper == FT(0) && return FT(0)
        θ_val = mpace_θ_liq_ice_of_p(p_val, thermo_params)
        q_val = mpace_q_tot_of_p(p_val)
        T_val =
            TD.air_temperature(thermo_params, TD.pθ_li(), p_val, θ_val, q_val)
        ρ_val =
            TD.air_density(thermo_params, T_val, p_val, q_val, FT(0), FT(0))
        # Large-scale pressure velocity ω = min(D (p_s - p), D (p_s - p_inv)) [Pa/s]
        # with D = 5.8e-6 s⁻¹, p_s = 101000 Pa, p_inv = 85000 Pa (Klein et al. 2009, Eq. 6).
        D = FT(5.8e-6)
        p_inv = FT(85000)
        ω = taper * min(D * (p_0 - p_val), D * (p_0 - p_inv))
        return -ω / (ρ_val * grav)
    end

    function prof_dTdt(_, z)
        z_f = FT(z)
        z_f >= FT(7000) && return FT(0)
        p_hPa = p_low(z_f) / FT(100)
        taper = mpace_upper_taper(p_hPa)
        taper == FT(0) && return FT(0)
        p_s_hPa = FT(1010)
        # Klein et al. (2009), Eq. 4: [K/day] converted to [K/s]
        dTdt_day = min(FT(-4), FT(-15) * (1 - (p_s_hPa - p_hPa) / FT(218.18)))
        return taper * dTdt_day / FT(86400)
    end

    function prof_dqtdt(z)
        z_f = FT(z)
        z_f >= FT(7000) && return FT(0)
        p_hPa = p_low(z_f) / FT(100)
        taper = mpace_upper_taper(p_hPa)
        taper == FT(0) && return FT(0)
        p_s_hPa = FT(1010)
        # Klein et al. (2009), Eq. 5: [g/kg/day] converted to [kg/kg/s]
        dqdt_g_day =
            min(FT(-0.164), FT(-3) * (1 - (p_s_hPa - p_hPa) / FT(151.71)))
        return taper * dqdt_g_day * FT(1e-3) / FT(86400)
    end

    return (; θ, q_tot, p, u, v, tke, w_sub, prof_dTdt, prof_dqtdt)
end

function center_initial_condition(setup::MPACE, local_geometry, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; z) = local_geometry.coordinates
    (; prognostic_tke, profiles) = setup

    p_val = profiles.p(z)
    θ_val = profiles.θ(z)
    q_tot_val = profiles.q_tot(z)
    T = TD.air_temperature(thermo_params, TD.pθ_li(), p_val, θ_val, q_tot_val)
    tke_val = prognostic_tke ? FT(0) : profiles.tke(z)

    return physical_state(;
        T,
        p = p_val,
        q_tot = q_tot_val,
        u = profiles.u(z),
        v = profiles.v(z),
        tke = tke_val,
    )
end

function surface_condition(::MPACE, params)
    FT = eltype(params)
    return (;
        flux_scheme = MoninObukhov(;
            z0 = FT(1e-4),
            shf = FT(136.5),
            lhf = FT(107.7),
        ),
        temperature = AnalyticTemperature(Returns(FT(274.01))),
        overrides = SurfaceBoundaryOverrides(),
    )
end

subsidence_forcing(setup::MPACE, ::Type{FT}) where {FT} = setup.profiles.w_sub

large_scale_advection_forcing(setup::MPACE, ::Type{FT}) where {FT} =
    (; prof_dTdt = setup.profiles.prof_dTdt, prof_dqtdt = setup.profiles.prof_dqtdt)

external_forcing(::MPACE, ::Type{FT}) where {FT} = MPACEForcing(FT)

# Note: `AtmosSimulation` uses `ITime` anchored at the simulation's `start_date`
# config key (`"20041009-1700"` in `prognostic_edmfx_mpace_column.yml`, matching
# Klein et al. 2009 Section 4); `start_date` here is a fallback for non-`ITime`
# callers.
insolation_model(::MPACE) = TimeVaryingInsolation(;
    start_date = Dates.DateTime(2004, 10, 9, 17),
    latitude = 70.5,
    longitude = -156.0,
)
