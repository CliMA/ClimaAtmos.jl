"""
    MPACE

The M-PACE (Mixed-Phase Arctic Cloud Experiment) single-layer cloud setup
(Case B) described in [Klein2009](@cite), with a hydrostatically balanced
pressure profile integrated from the pressure-coordinate specification of the
liquid-ice potential temperature and total water mixing ratio.

# Examples

```julia
setup = MPACE(; prognostic_tke = true, thermo_params)
```
"""
struct MPACE{P}
    prognostic_tke::Bool
    profiles::P
end

MPACE(; prognostic_tke::Bool = true, thermo_params) =
    MPACE(prognostic_tke, mpace_profiles(thermo_params))

"""
    mpace_θ_liq_ice_of_p(p::FT) where {FT}

Return the initial liquid-ice potential temperature [K] at pressure `p` [Pa] for
the M-PACE single-layer cloud case [Klein2009](@cite) (Eq. 1):

```math
θ_{li}(p) = \\begin{cases}
269.2 \\text{ K}, & p > 850 \\text{ hPa} \\\\
275.33 \\text{ K} + 0.0791 \\text{ K hPa}^{-1} (815 \\text{ hPa} - p), & p \\le 850 \\text{ hPa}
\\end{cases}
```
"""
function mpace_θ_liq_ice_of_p(p::FT) where {FT}
    p_hPa = p / FT(100)
    return p_hPa > FT(850) ? FT(269.2) :
           FT(275.33) + FT(0.0791) * (FT(815) - p_hPa)
end

"""
    mpace_q_tot_of_p(p::FT) where {FT}

Return the initial total specific humidity [kg/kg] at pressure `p` [Pa] for the
M-PACE single-layer cloud case [Klein2009](@cite) (Eq. 2). Following
[Klein2009](@cite), the linear profile above the inversion (`p ≤ 850 hPa`) is
defined in terms of the total water mixing ratio `r_t` [g/kg]:

```math
r_t(p) = \\begin{cases}
1.95 \\text{ g kg}^{-1}, & p > 850 \\text{ hPa} \\\\
\\max(0,\\; 0.291 + 0.00204 (p - 590 \\text{ hPa})) \\text{ g kg}^{-1}, & p \\le 850 \\text{ hPa}
\\end{cases}
```

and converted to specific humidity via `q_tot = r_t / (1 + r_t)`.
"""
function mpace_q_tot_of_p(p::FT) where {FT}
    p_hPa = p / FT(100)
    r_tot =
        p_hPa > FT(850) ? FT(1.95e-3) :
        max(FT(0), (FT(0.291) + FT(0.00204) * (p_hPa - FT(590))) * FT(1e-3))
    return r_tot / (1 + r_tot)
end

"""
    mpace_profiles(thermo_params)

Precompute the atmospheric profiles for the M-PACE single-layer cloud case,
given thermodynamic parameters `thermo_params`.

Integrates `dp/dz = -g ρ(p)` upward from `p_0 = 101000` Pa to `z_max = 5000` m
using the pressure-coordinate profiles `mpace_θ_liq_ice_of_p` and
`mpace_q_tot_of_p`. Returns a `NamedTuple` of profile functions of height `z`.
"""
function mpace_profiles(
    thermo_params::TD.Parameters.ThermodynamicsParameters{FT},
) where {FT}
    p_0 = FT(101000)
    z_max = FT(5000)
    grav = TD.Parameters.grav(thermo_params)

    function dp_dz(p, _)
        θ_val = mpace_θ_liq_ice_of_p(p)
        q_val = mpace_q_tot_of_p(p)
        T_val = TD.air_temperature(thermo_params, TD.pθ_li(), p, θ_val, q_val)
        ρ_val = TD.air_density(thermo_params, T_val, p, q_val, FT(0), FT(0))
        return -grav * ρ_val
    end

    p = column_indefinite_integral(dp_dz, p_0, (FT(0), z_max))
    θ(z) = mpace_θ_liq_ice_of_p(p(z))
    q_tot(z) = mpace_q_tot_of_p(p(z))
    u = Returns(FT(-13))
    v = Returns(FT(-3))
    tke = Returns(FT(0.1))

    function w_sub(z)
        p_val = p(z)
        θ_val = mpace_θ_liq_ice_of_p(p_val)
        q_val = mpace_q_tot_of_p(p_val)
        T_val =
            TD.air_temperature(thermo_params, TD.pθ_li(), p_val, θ_val, q_val)
        ρ_val =
            TD.air_density(thermo_params, T_val, p_val, q_val, FT(0), FT(0))
        # Large-scale pressure velocity ω = min(D (p_s - p), D (p_s - p_inv)) [Pa/s]
        # with D = 5.8e-6 s⁻¹, p_s = 101000 Pa, p_inv = 85000 Pa (Klein et al. 2009, Eq. 6).
        D = FT(5.8e-6)
        p_inv = FT(85000)
        ω = min(D * (p_0 - p_val), D * (p_0 - p_inv))
        return -ω / (ρ_val * grav)
    end

    function prof_dTdt(_, z)
        p_hPa = p(z) / FT(100)
        p_s_hPa = FT(1010)
        # Klein et al. (2009), Eq. 4: [K/day] converted to [K/s]
        dTdt_day = min(FT(-4), FT(-15) * (1 - (p_s_hPa - p_hPa) / FT(218.18)))
        return dTdt_day / FT(86400)
    end

    function prof_dqtdt(z)
        p_hPa = p(z) / FT(100)
        p_s_hPa = FT(1010)
        # Klein et al. (2009), Eq. 5: [g/kg/day] converted to [kg/kg/s]
        dqdt_g_day =
            min(FT(-0.164), FT(-3) * (1 - (p_s_hPa - p_hPa) / FT(151.71)))
        return dqdt_g_day * FT(1e-3) / FT(86400)
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

coriolis_forcing(::MPACE, ::Type{FT}) where {FT} =
    (;
        prof_ug = Returns(FT(-13)),
        prof_vg = Returns(FT(-3)),
        # f = 2Ω sin(70.5°N) at the North Slope of Alaska (Klein et al. 2009)
        coriolis_param = FT(1.37e-4),
    )

insolation_model(::MPACE) = TimeVaryingInsolation(;
    start_date = Dates.DateTime(2004, 10, 9, 17),
    latitude = 70.5,
    longitude = -156.0,
)
