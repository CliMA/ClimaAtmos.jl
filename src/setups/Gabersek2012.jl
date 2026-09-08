"""
    Gabersek2012

The 2.5D squall-line initial condition described in [Gabersek2012](@cite): a
tabulated mid-latitude sounding (weakly stable troposphere with a saturated
boundary layer, more stable above the 12 km tropopause) with low-level wind
shear in x, in hydrostatic balance, plus a y-independent warm bubble
(Δθ = 3 K) that triggers the storm. The surface is free-slip and thermally
insulating.

The sounding is tabulated in the appendix of Tissaoui et al. (2023), which
follows Gaberšek et al. (2012); water vapor is removed above 14 km as in the
reference implementations.

S. Gaberšek, F. X. Giraldo and J. D. Doyle.
Dry and moist idealized experiments with a two-dimensional spectral element
model. Monthly Weather Review 140, 3163-3182 (2012).

# Examples

```julia
setup = Gabersek2012()          # Float32 defaults
setup = Gabersek2012(Float64)   # specify floating-point type
```

To use thermodynamics parameters from a non-default `ClimaAtmosParameters`,
pass them explicitly via `thermo_params`.
"""
struct Gabersek2012{P}
    profiles::P
end

function Gabersek2012(
    ::Type{FT} = Float32;
    thermo_params = ThermodynamicsParameters(FT),
) where {FT}
    return Gabersek2012(gabersek_profiles(thermo_params))
end

"""
    gabersek_profiles(thermo_params)

Precompute the atmospheric profiles for the Gaberšek et al. (2012) squall-line
case, given thermodynamic parameters `thermo_params`.

Returns a NamedTuple of profile functions of height `z`: liquid-ice potential
temperature `θ`, total specific humidity `q_tot`, zonal wind `u`, and the
hydrostatically balanced pressure `p`.
"""
function gabersek_profiles(thermo_params)
    FT = eltype(thermo_params)
    # Squall-line sounding (Tissaoui et al. 2023, Table 2): height [m],
    # potential temperature [K], water vapor mixing ratio [g/kg], u wind [m/s]
    z_values = FT[
        0, 480, 960, 1440, 1920, 2400, 2880, 3360, 3840, 4320, 4800, 5280,
        5760, 6240, 6720, 7200, 7680, 8160, 8640, 9120, 9600, 10080, 10560,
        11520, 12000, 12480, 12960, 13440, 13920, 14400, 15360, 15840, 16320,
        16800, 17280, 17760, 18720, 19200, 19680, 20160, 20640, 21120, 21600,
        22560, 23040, 23520, 24000,
    ]
    θ_values = FT[
        303.025079, 303.337272, 304.402985, 305.397187, 306.306214,
        307.365269, 308.550318, 309.845257, 311.235047, 312.708238,
        314.255743, 315.869985, 317.544512, 319.273784, 321.052868,
        322.877588, 324.744235, 326.649534, 328.590559, 330.565013,
        332.571020, 334.606102, 336.668475, 340.869535, 343.712008,
        350.647306, 358.453724, 366.433620, 374.591035, 382.929618,
        400.170355, 409.081924, 418.191751, 427.504224, 437.023716,
        446.755038, 466.871821, 477.267160, 487.891998, 498.742611,
        509.643457, 520.544304, 531.445151, 553.246845, 564.147692,
        575.048539, 585.949386,
    ]
    rv_values =
        FT[
            14.000, 14.000, 14.000, 12.796, 10.556, 8.678, 7.104, 5.788,
            4.691, 3.777, 3.020, 2.396, 1.885, 1.469, 1.134, 0.866, 0.653,
            0.487, 0.357, 0.259, 0.184, 0.129, 0.088, 0.038, 0.026, 0.026,
            0.029, 0.031, 0.034, 0.037, 0.044, 0.049, 0.053, 0.058, 0.063,
            0.069, 0.083, 0.091, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094,
            0.094, 0.094, 0.094,
        ] .* FT(1e-3)
    u_values = FT[
        12.0, 9.696, 7.392, 5.088, 2.784, 0.54, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]

    linear_profile(zs, vals) = CI1D.Interpolate1D(
        SA.SVector{length(zs)}(zs), SA.SVector{length(vals)}(vals);
        interpolationorder = CI1D.Linear(),
        extrapolationorder = CI1D.Flat(),
    )
    θ_itp = linear_profile(z_values, θ_values)
    rv_itp = linear_profile(z_values, rv_values)
    u_itp = linear_profile(z_values, u_values)
    # Mixing ratio -> specific humidity; dry above 14 km
    rv(z) = z > 14_000 ? zero(z) : max(rv_itp(z), zero(z))
    q_tot(z) = rv(z) / (1 + rv(z))
    θ(z) = θ_itp(z)

    p_0 = FT(100_000)
    p = hydrostatic_pressure_profile(;
        thermo_params,
        p_0,
        θ,
        q_tot,
        z_max = 24_000,
    )
    return (; θ, q_tot, u = u_itp, p)
end

function center_initial_condition(setup::Gabersek2012, local_geometry, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; θ, q_tot, u, p) = setup.profiles
    # y-independent, so the same setup works on plane (XZ) and box (XYZ)
    (; x, z) = local_geometry.coordinates

    # Warm bubble trigger: Δθ = θ_c cos²(π r / 2) for r ≤ 1
    x_c, x_r = FT(75_000), FT(10_000)
    z_c, z_r = FT(2_000), FT(1_500)
    θ_c = FT(3)
    r = sqrt(((x - x_c) / x_r)^2 + ((z - z_c) / z_r)^2)
    Δθ = r < 1 ? θ_c * cospi(r / 2)^2 : FT(0)

    q_tot_z = q_tot(z)
    T = TD.air_temperature(thermo_params, TD.pθ_li(), p(z), θ(z) + Δθ, q_tot_z)

    return physical_state(; T, p = p(z), q_tot = q_tot_z, u = u(z))
end

function surface_condition(::Gabersek2012, params)
    FT = eltype(params)
    rv_0 = FT(0.014)
    q_vap = rv_0 / (1 + rv_0)
    return (;
        # Free-slip, thermally insulating wall
        flux_scheme = ExchangeCoefficients(; Cd = FT(0), Ch = FT(0)),
        temperature = AnalyticTemperature(Returns(FT(303.025079))),
        overrides = SurfaceBoundaryOverrides(p = FT(100_000), q_vap = q_vap),
    )
end
