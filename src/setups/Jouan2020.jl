"""
    Jouan2020

Initial condition for the Jouan et al. (2020) Arctic mixed-phase cloud
case. Uses a 15-km radiosonde sounding (`data/Jouan_initial_condition.txt`)
to define T(z), q_v(z), and p(z). Intended as the ClimaAtmos counterpart
to `init_profile_jouan` in KinematicDriver.jl
(src/Common/initial_condition.jl:26).

C. Jouan, E. Girard, F. Pelon, I. Gultepe, J. Delanoe, J.-P. Blanchet.
Characterization of Arctic Ice Cloud Properties Observed During ISDAC.
Weather and Forecasting 35, 1889-1909 (2020).

!!! warning "Prescribed-flow wiring is a core-code change"
    A height-varying prescribed updraft (half-sine in z to Htop = 5 km,
    ramped 2 -> 5 m/s over tscale1 = 5400 s, then zero after T_UP = 3600 s)
    is required to match the KinematicDriver.jl/kin1d reference.
    This cannot be wired without modifying:

      * `src/solver/types.jl`   (add a `Jouan2020VelocityProfile <:
        PrescribedFlow`, register in `ATMOS_MODEL_GROUPS`, and add a
        `get_ρu₃qₜ_surface` method); and
      * `src/prognostic_equations/constrain_state.jl::prescribe_flow!`,
        which currently hardcodes the ShipwayHill2012 initial profile
        when re-pinning ρ and ρe_tot every time step.

    Those files are outside the setup-module scope. The Jouan-specific
    shape/time functions below are kept here for convenience and can be
    spliced into a new `Jouan2020VelocityProfile` when a companion PR
    touches `types.jl` and `constrain_state.jl`. See
    `issues/009-climaatmos-kid-jouan.md` for the full plan.
"""
struct Jouan2020{P}
    profiles::P
end

# ----------------------------------------------------------------------------
# Height-varying w-profile (kin1d half-sine) — reference for a future
# Jouan2020VelocityProfile. Parameters mirror kid_jouan_ou/run_baseline.jl
# (cld1d.f90 Round-4 alignment).
# ----------------------------------------------------------------------------

# Half-sine in z on [0, Htop], zero elsewhere.
@inline function jouan_w_shape(z::FT, Htop::FT) where {FT}
    if z <= zero(FT) || z >= Htop
        return zero(FT)
    end
    return sin(FT(pi) * z / Htop)
end

# wmax(t): ramp AMPA -> AMPB over tscale1 (via raised-cosine),
# then zero for t > t_up. Matches kin1d_wmax.
@inline function jouan_wmax(t::FT,
    AMPA::FT = FT(2.0), AMPB::FT = FT(5.0),
    tscale1::FT = FT(5400.0), t_up::FT = FT(3600.0),
) where {FT}
    if t > t_up
        return zero(FT)
    end
    return AMPA + (AMPB - AMPA) * FT(0.5) *
           (cos((t / tscale1) * FT(2pi) + FT(pi)) + one(FT))
end

# Combined w(z, t) for reference use.
@inline jouan_w(z::FT, t::FT; Htop::FT = FT(5000.0)) where {FT} =
    jouan_wmax(t) * jouan_w_shape(z, Htop)

# ----------------------------------------------------------------------------
# Sounding reader
# ----------------------------------------------------------------------------

# Read the Jouan sounding shipped with ClimaAtmos.
# Columns: T (K), Td (K), Qv (specific), Qsat, p (Pa), -, -, -, -, z (m).
# File is top-down; reverse to bottom-up for interpolation.
function _read_jouan_sounding(::Type{FT}) where {FT}
    path = joinpath(@__DIR__, "data", "Jouan_initial_condition.txt")
    lines = readlines(path)
    # Skip 1-line header; parse the remainder as whitespace-separated floats.
    rows = [parse.(Float64, split(strip(ln))) for ln in lines[2:end] if !isempty(strip(ln))]
    data = reduce(vcat, (row' for row in rows))
    input_T = reverse(data[:, 1])
    input_qv = reverse(data[:, 3])
    input_p = reverse(data[:, 5])
    input_z = reverse(data[:, 10])
    make_interp(xs, ys) = Intp.extrapolate(
        Intp.interpolate((FT.(xs),), FT.(ys), Intp.Gridded(Intp.Linear())),
        Intp.Flat(),
    )
    return (;
        T = make_interp(input_z, input_T),
        qv = make_interp(input_z, input_qv),
        p_raw = make_interp(input_z, input_p),
        z_max = FT(last(input_z)),
    )
end

# ----------------------------------------------------------------------------
# Setup interface
# ----------------------------------------------------------------------------

function Jouan2020(; thermo_params)
    FT = eltype(thermo_params)
    sounding = _read_jouan_sounding(FT)

    q_tot(z) = max(sounding.qv(z), zero(FT))
    T(z) = sounding.T(z)

    p_0 = sounding.p_raw(FT(0))
    p = hydrostatic_pressure_profile(;
        thermo_params, p_0, T, q_tot, z_max = FT(15_000),
    )

    profiles = (; T, q_tot, p, p_raw = sounding.p_raw)
    return Jouan2020{typeof(profiles)}(profiles)
end

function center_initial_condition(setup::Jouan2020, local_geometry, params)
    (; T, q_tot, p) = setup.profiles
    (; z) = local_geometry.coordinates
    q_tot_z = q_tot(z)
    T_z = T(z)
    return physical_state(; T = T_z, p = p(z), q_tot = q_tot_z)
end

function surface_condition(setup::Jouan2020, params)
    # Capture profiles in a closure so we don't rebuild the setup on every
    # surface-state callback.
    (; T, q_tot, p_raw) = setup.profiles
    function surface_state(surface_coordinates, interior_z, t)
        FT = eltype(surface_coordinates)
        parameterization = SurfaceConditions.ExchangeCoefficients(; Cd = FT(0), Ch = FT(0))
        return SurfaceState(;
            parameterization,
            T = FT(T(FT(0))),
            p = FT(p_raw(FT(0))),
            q_vap = FT(q_tot(FT(0))),
        )
    end
    return surface_state
end

prescribed_flow_model(::Jouan2020, ::Type{FT}) where {FT} = Jouan2020VelocityProfile{FT}()
