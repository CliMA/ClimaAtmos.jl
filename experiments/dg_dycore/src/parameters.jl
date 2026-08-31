#=
Scalar physical constants, bound once at model build time and carried in the
DGModel — the tendency kernels only ever see plain scalars (no params-struct
access in hot loops).

Two modes:
  :parity       — the literal parity-TOML constants, for field-level
                  reproducibility.
  :clima_params — from ClimaParams via ClimaAtmos (differs from parity in the
                  last digits: grav 9.81 vs 9.80616, Ω 7.2921159e-5 vs
                  7.29212e-5, MSLP 101325 vs 1e5, R_d via molar mass vs 287.0).
=#

struct DGConstants{FT <: AbstractFloat} # broadcast as a scalar (see below)
    p_0::FT     # reference surface pressure [Pa]
    R_d::FT     # dry gas constant [J/kg/K]
    T_tri::FT   # triple-point reference temperature [K]
    grav::FT    # gravitational acceleration [m/s²]
    Ω::FT       # planetary angular velocity [1/s]
    cp_d::FT    # isobaric specific heat, dry air
    cv_d::FT    # isochoric specific heat, dry air
    γ::FT       # cp_d / cv_d
    R::FT       # planet radius [m]
end
Base.broadcastable(c::DGConstants) = tuple(c)

function DGConstants{FT}(; mode::Symbol = :parity) where {FT}
    if mode == :parity
        p_0 = FT(1.0e5)
        R_d = FT(287.0)
        κ_gas = FT(2 / 7)
        cp_d = R_d / κ_gas
        cv_d = cp_d - R_d
        return DGConstants{FT}(
            p_0,
            R_d,
            FT(273.16),
            FT(9.80616),
            FT(7.29212e-5),
            cp_d,
            cv_d,
            cp_d / cv_d,
            FT(6.371229e6),
        )
    elseif mode == :clima_params
        return DGConstants(dg_params(FT, mode))
    else
        error("unknown DGConstants mode $mode")
    end
end

const PARITY_TOML =
    joinpath(@__DIR__, "..", "configs", "parity", "dg_parity_params.toml")

"""
    dg_params(FT, constants_mode) -> CAP.ClimaAtmosParameters

The ClimaAtmos parameter set consumed by the reused components (Setups IC
values, Held–Suarez forcing). In `:parity` mode ClimaParams is overridden with
the parity-TOML constants (configs/parity/).
"""
dg_params(::Type{FT}, constants_mode::Symbol) where {FT} =
    constants_mode == :parity ?
    CA.ClimaAtmosParameters(CP.create_toml_dict(FT; override_file = PARITY_TOML)) :
    CA.ClimaAtmosParameters(FT)

"""
    DGConstants(params::CAP.ClimaAtmosParameters)

Scalar constants bound from a ClimaAtmos parameter set (`:clima_params`
mode; also what `:parity` mode must agree with under the parity TOML —
asserted in test/).
"""
function DGConstants(params)
    FT = eltype(params)
    cp_d = FT(CAP.cp_d(params))
    cv_d = FT(CAP.cv_d(params))
    return DGConstants{FT}(
        FT(CAP.MSLP(params)),
        FT(CAP.R_d(params)),
        FT(CAP.T_0(params)),
        FT(CAP.grav(params)),
        FT(CAP.Omega(params)),
        cp_d,
        cv_d,
        cp_d / cv_d,
        FT(CAP.planet_radius(params)),
    )
end

# Pointwise diagnostic pressure from total energy density (closed form —
# the HEVI Jacobians need its analytic derivatives, so this is NOT routed
# through Thermodynamics.jl; test/test_pressure.jl asserts equality with
# TD.air_pressure(PhaseDry_ρe) to machine ε).
pres_ρe(c::DGConstants, ρe, K, Φ, ρ) =
    ρ * c.R_d * ((ρe / ρ - K - Φ) / c.cv_d + c.T_tri)

# Moist dynamics pressure/temperature. Closed-form TD.air_temperature (no Newton
# iteration) — the exact inverse of the TD.internal_energy the moist IC builds ρe
# with, so the moist hydrostatic balance closes. Condensate treated as vapor for
# the dynamics p (keeps the virtual-temperature effect via q_tot in R_m). Used
# identically in the explicit and implicit tendencies so the HEVI split is exact.
# No T clamp: a genuinely negative T/p surfaces as a NaN rather than being masked.
@inline function moist_p_dyn(thermo_params, ρ, e_int, q_tot)
    z = zero(q_tot)
    T = TD.air_temperature(thermo_params, e_int, q_tot, z, z)
    p = TD.air_pressure(thermo_params, T, ρ, q_tot, z, z)
    return (; T, p)
end
