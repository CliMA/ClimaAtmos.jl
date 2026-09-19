# Shared infrastructure for file-based initial conditions, used by the setups
# that initialize the prognostic state from NetCDF files (MoistFromFile,
# WeatherModel, AMIPFromERA5). The entry point is `overwrite_from_file!`, which
# regrids file data onto the model grid and populates every prognostic
# variable.

# ============================================================================
# Topographic pressure correction
# ============================================================================

"""
    correct_surface_pressure_for_topography!(
        p_sfc, file_path, face_space, Y, ᶜT, ᶜq_tot,
        thermo_params, regridder_kwargs;
        surface_altitude_var = "z_sfc",
    )

Adjust the surface pressure field `p_sfc` in place for the mismatch between the
file's surface altitude and the model orography, and return `true`.

The correction is hydrostatic over the altitude difference
`Δz = z_model_surface - z_sfc`:

```math
p_{sfc} ← p_{sfc} \\exp(-Δz \\, g / (R_m T_{sfc}))
```

The caller is responsible for checking that the file carries
`surface_altitude_var`; reading a variable the file lacks throws.
"""
function correct_surface_pressure_for_topography!(
    p_sfc,
    file_path,
    face_space,
    Y,
    ᶜT,
    ᶜq_tot,
    thermo_params,
    regridder_kwargs;
    surface_altitude_var = "z_sfc",
)
    regridder_type = :InterpolationsRegridder
    ᶠz_surface = Fields.level(
        SpaceVaryingInputs.SpaceVaryingInput(
            file_path,
            surface_altitude_var,
            face_space;
            regridder_type,
            regridder_kwargs = regridder_kwargs,
        ),
        Fields.half,
    )

    if ᶠz_surface === nothing
        return false
    end

    grav = thermo_params.grav

    ᶠz_model_surface = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶠΔz = @. ᶠz_model_surface - ᶠz_surface

    ᶠR_m = ᶠinterp.(TD.gas_constant_air.(thermo_params, ᶜq_tot))
    ᶠR_m_sfc = Fields.level(ᶠR_m, Fields.half)

    ᶠT = ᶠinterp.(ᶜT)
    ᶠT_sfc = Fields.level(ᶠT, Fields.half)

    @. p_sfc = p_sfc * exp(-(ᶠΔz) * grav / (ᶠR_m_sfc * ᶠT_sfc))

    @info "Adjusted surface pressure to account for ERA5/model surface-height differences."
    return true
end

# ============================================================================
# Internal helpers (shared by overwrite_from_file! and WeatherModel)
# ============================================================================

"""
    hydrostatic_pressure(p_sfc, ᶜT, ᶜq_tot, face_space, thermo_params)

Compute face pressure by hydrostatic integration from surface pressure.
Solves ∂(ln p)/∂z = -g/(Rₘ(q)T) using `column_integral_indefinite!`.
"""
function hydrostatic_pressure(p_sfc, ᶜT, ᶜq_tot, face_space, thermo_params)
    ᶜ∂lnp∂z = @. -thermo_params.grav /
                 (TD.gas_constant_air(thermo_params, ᶜq_tot) * ᶜT)
    ᶠlnp_over_psfc = zeros(face_space)
    Operators.column_integral_indefinite!(ᶠlnp_over_psfc, ᶜ∂lnp∂z)
    return p_sfc .* exp.(ᶠlnp_over_psfc)
end

"""
    rebalance_hydrostatic_pressure(ᶜp_anchor, ᶜT, ᶜq_tot, ᶜΦ, thermo_params)

Compute the center pressure that is in *discrete* hydrostatic balance with the
model's Exner-split vertical pressure gradient (the `Yₜ.f.u₃` PGF in
`implicit_vertical_advection_tendency!`),

```math
∇ᵥΦ - ∇ᵥΦ_r(p) + c_{p,d}\\,\\overline{(θ_v - θ_{vr}(p))}\\,∇ᵥΠ(p) = 0
```

at every interior face, holding `ᶜT` and `ᶜq_tot` fixed. The `ln p` column
integration of [`hydrostatic_pressure`](@ref) satisfies a *different*
discretization (a face-to-face midpoint rule), which leaves an O((Δz/H)²)
residual in the model operator that peaks over steep terrain where the
vertical grid is stretched — measured as 0.56 m/s² of spurious `u₃` forcing
over the Himalaya (h8/z63 SLEVE, ~6× the analytic-IC baseline) for ERA5-type
initial conditions, independent of the file data.

Per interior face between centers `k` and `k+1` the recurrence solves

```math
F(x) = ΔΦ - (Φ_r(p) - Φ_r(p_k))
     + \\tfrac{c_{p,d}}{2}\\,(θ_v'_k + θ_v'(p))\\,(Π(p) - Π(p_k)) = 0,
     \\quad p = e^x,
```

with `θ_v' = θ_v - θ_{vr}`, by Newton iteration in `x = ln p` (fixed 8
iterations; the multiplicative update keeps `p > 0`, cf. `pref_from_phi`).
The interior-face `ᶠgradᵥ`/`ᶠinterp` stencils reduce to the difference and
arithmetic mean of adjacent centers, so zeroing `F` zeroes the covariant
tendency exactly. The first-level value is copied from `ᶜp_anchor`, preserving
the caller's (topography-corrected) surface-pressure anchor. `q_liq = q_ice = 0`
is assumed, matching the pre-cache state; where the input is supersaturated,
the first saturation adjustment shifts `T` and reintroduces a small local
residual.
"""
function rebalance_hydrostatic_pressure(
    ᶜp_anchor, ᶜT, ᶜq_tot, ᶜΦ, thermo_params,
)
    FT = eltype(ᶜT)
    R_d = TD.TP.R_d(thermo_params)
    cp_d = TD.TP.cp_d(thermo_params)
    ᶜp_bal = similar(ᶜT)
    input = Base.broadcasted(tuple, ᶜΦ, ᶜT, ᶜq_tot, ᶜp_anchor)
    # Carry: (p, Φ, θ_v′) of the level below; NaN marks "below the first level".
    init = (FT(NaN), FT(NaN), FT(NaN))
    Operators.column_accumulate!(
        ᶜp_bal,
        input;
        init,
        transform = first,
    ) do (p_prev, Φ_prev, θvp_prev), (Φ, T, q, p_anchor)
        p = if isnan(p_prev)
            p_anchor
        else
            R_m = TD.gas_constant_air(thermo_params, q, zero(q), zero(q))
            # Naive hydrostatic step as the initial guess (error O((Δz/H)²)).
            x = log(p_prev) - (Φ - Φ_prev) / (R_m * T)
            Π_prev = TD.exner_given_pressure(thermo_params, p_prev)
            Φr_prev = phi_r(thermo_params, p_prev)
            for _ in 1:8
                pn = exp(x)
                Π = TD.exner_given_pressure(thermo_params, pn)
                θvp =
                    theta_v(thermo_params, T, pn, q, zero(q), zero(q)) -
                    theta_vr(thermo_params, pn)
                F =
                    (Φ - Φ_prev) - (phi_r(thermo_params, pn) - Φr_prev) +
                    cp_d / 2 * (θvp_prev + θvp) * (Π - Π_prev)
                # dF/dx up to an O(ΔΠ/Π) term: −dΦ_r/dx = R_d T_r and
                # cp_d κ_d Π θ̄_v′ = R_d Π θ̄_v′; together ≈ R_d T̄_v > 0.
                T_r = air_temperature_reference(thermo_params, pn)
                dFdx = R_d * T_r + R_d * Π * (θvp_prev + θvp) / 2
                x -= F / dFdx
            end
            exp(x)
        end
        θvp =
            theta_v(thermo_params, T, p, q, zero(q), zero(q)) -
            theta_vr(thermo_params, p)
        return (p, Φ, θvp)
    end
    return ᶜp_bal
end

"""
    assign_velocity_energy!(Y, ᶜT, ᶜq_tot, ᶠp, thermo_params, file_path, svi_kwargs)

Regrid velocity from file, compute kinetic and total energy, and assign to Y.
"""
function assign_velocity_energy!(
    Y, ᶜT, ᶜq_tot, ᶠp, thermo_params, file_path, svi_kwargs,
)
    center_space = Fields.axes(Y.c)
    vel =
        Geometry.UVWVector.(
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path, "u", center_space; svi_kwargs...,
            ),
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path, "v", center_space; svi_kwargs...,
            ),
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path, "w", center_space; svi_kwargs...,
            ),
        )
    Y.c.uₕ .= C12.(Geometry.UVVector.(vel))
    Y.f.u₃ .= ᶠinterp.(C3.(Geometry.WVector.(vel)))
    e_kin = compute_kinetic(Y.c.uₕ, Y.f.u₃)
    e_pot = geopotential.(thermo_params.grav, Fields.coordinate_field(Y.c).z)
    Y.c.ρe_tot .=
        TD.total_energy.(thermo_params, e_kin, e_pot, ᶜT, ᶜq_tot) .* Y.c.ρ
    return e_pot
end

"""
    assign_moisture_edmf!(Y, ᶜT, ᶜq_tot, e_pot, thermo_params, ᶜq_rai, ᶜq_sno)

Assign moisture variables and EDMF subdomain initialization.

`ᶜq_rai` and `ᶜq_sno` are the rain and snow specific humidities regridded
from file, or `nothing` when the file does not contain microphysics fields.
"""
function assign_moisture_edmf!(
    Y, ᶜT, ᶜq_tot, e_pot, thermo_params, ᶜq_rai, ᶜq_sno,
)

    if hasproperty(Y.c, :ρq_tot)
        Y.c.ρq_tot .= ᶜq_tot .* Y.c.ρ
    else
        error(
            "`dry` configurations are incompatible with the interpolated initial conditions.",
        )
    end

    if hasproperty(Y.c, :ρq_lcl)
        fill!(Y.c.ρq_lcl, 0)
    end
    if hasproperty(Y.c, :ρq_icl)
        fill!(Y.c.ρq_icl, 0)
    end
    if hasproperty(Y.c, :ρq_rai) && hasproperty(Y.c, :ρq_sno)
        if !isnothing(ᶜq_rai)
            Y.c.ρq_rai .= ᶜq_rai .* Y.c.ρ
            Y.c.ρq_sno .= ᶜq_sno .* Y.c.ρ
        else
            fill!(Y.c.ρq_rai, 0)
            fill!(Y.c.ρq_sno, 0)
        end
    end

    # Initialize prognostic EDMF subdomains if present
    if hasproperty(Y.c, :sgsʲs)
        ᶜmse = TD.enthalpy.(thermo_params, ᶜT, ᶜq_tot) .+ e_pot
        for name in propertynames(Y.c.sgsʲs)
            s = getproperty(Y.c.sgsʲs, name)
            hasproperty(s, :ρa) && fill!(s.ρa, 0)
            hasproperty(s, :mse) && (s.mse .= ᶜmse)
            hasproperty(s, :q_tot) && (s.q_tot .= ᶜq_tot)
            # SGS 1M microphysics tracers
            hasproperty(s, :q_lcl) && fill!(s.q_lcl, 0)
            hasproperty(s, :q_icl) && fill!(s.q_icl, 0)
            if hasproperty(s, :q_rai) && hasproperty(s, :q_sno)
                if !isnothing(ᶜq_rai)
                    s.q_rai .= ᶜq_rai
                    s.q_sno .= ᶜq_sno
                else
                    fill!(s.q_rai, 0)
                    fill!(s.q_sno, 0)
                end
            end
        end
    end

    if hasproperty(Y.c, :ρtke)
        fill!(Y.c.ρtke, 0)
    end

    return nothing
end

# ============================================================================
# Main shared overwrite function
# ============================================================================

"""
    overwrite_from_file!(file_path, extrapolation_bc, Y, thermo_params;
                         regridder_type=nothing, interpolation_method=nothing,
                         hydrostatic_rebalance=false)

Overwrite the prognostic state `Y` with data regridded from a NetCDF file.
Recomputes vertical pressure levels assuming hydrostatic balance from
surface pressure. With `hydrostatic_rebalance = true`, the column pressure is
additionally rebalanced against the model's *discrete* Exner-split vertical
pressure gradient (see [`rebalance_hydrostatic_pressure`](@ref)), so the
initial state exerts no spurious vertical acceleration at `t = 0`.

Expected variables in the file:

  - `p`: pressure (2D surface, broadcast in z)
  - `t`: temperature (3D)
  - `q`: specific humidity (3D)
  - `u, v, w`: velocity (3D)
  - `cswc, crwc`: snow and rain water content (optional, for 1-moment microphysics)
  - `z_sfc`: surface altitude (optional, for topographic pressure correction)
"""
function overwrite_from_file!(
    file_path::String,
    extrapolation_bc,
    Y,
    thermo_params;
    regridder_type = nothing,
    interpolation_method = nothing,
    hydrostatic_rebalance = false,
)
    regridder_kwargs = filter(!isnothing, (; extrapolation_bc, interpolation_method))
    svi_kwargs =
        isnothing(regridder_type) ? (; regridder_kwargs) :
        (; regridder_type, regridder_kwargs)

    isfile(file_path) || error("$(file_path) is not a file")
    @info "Overwriting initial conditions with data from file $(file_path)"

    center_space = Fields.axes(Y.c)
    face_space = Fields.axes(Y.f)

    # Regrid temperature and humidity from file
    ᶜT = SpaceVaryingInputs.SpaceVaryingInput(
        file_path, "t", center_space; svi_kwargs...,
    )
    ᶜq_tot = SpaceVaryingInputs.SpaceVaryingInput(
        file_path, "q", center_space; svi_kwargs...,
    )

    # Surface pressure with optional topographic correction
    p_sfc = Fields.level(
        SpaceVaryingInputs.SpaceVaryingInput(
            file_path, "p", face_space; svi_kwargs...,
        ),
        Fields.half,
    )

    surface_altitude_var = "z_sfc"
    has_surface_altitude = NC.NCDataset(file_path) do ds
        haskey(ds, surface_altitude_var)
    end
    if has_surface_altitude
        correct_surface_pressure_for_topography!(
            p_sfc, file_path, face_space, Y, ᶜT, ᶜq_tot,
            thermo_params, regridder_kwargs;
            surface_altitude_var,
        )
    end

    # Hydrostatic pressure integration (also the rebalance anchor at level 1)
    ᶠp = hydrostatic_pressure(p_sfc, ᶜT, ᶜq_tot, face_space, thermo_params)
    ᶜp = if hydrostatic_rebalance
        ᶜΦ = geopotential.(
            thermo_params.grav,
            Fields.coordinate_field(Y.c).z,
        )
        rebalance_hydrostatic_pressure(
            ᶜinterp.(ᶠp), ᶜT, ᶜq_tot, ᶜΦ, thermo_params,
        )
    else
        ᶜinterp.(ᶠp)
    end

    # Density
    Y.c.ρ .= TD.air_density.(thermo_params, ᶜT, ᶜp, ᶜq_tot)

    # Velocity and energy
    e_pot = assign_velocity_energy!(
        Y, ᶜT, ᶜq_tot, ᶠp, thermo_params, file_path, svi_kwargs,
    )

    # Microphysics fields from file (rain/snow water content)
    center_space = Fields.axes(Y.c)
    has_microphysics_vars = NC.NCDataset(file_path) do ds
        haskey(ds, "cswc") && haskey(ds, "crwc")
    end
    if has_microphysics_vars
        ᶜq_rai = SpaceVaryingInputs.SpaceVaryingInput(
            file_path, "crwc", center_space; svi_kwargs...,
        )
        ᶜq_sno = SpaceVaryingInputs.SpaceVaryingInput(
            file_path, "cswc", center_space; svi_kwargs...,
        )
    else
        ᶜq_rai = nothing
        ᶜq_sno = nothing
    end

    # Moisture and EDMF
    assign_moisture_edmf!(
        Y, ᶜT, ᶜq_tot, e_pot, thermo_params, ᶜq_rai, ᶜq_sno,
    )

    return nothing
end
