"""
    sslt_dry_deposition_velocity(
        V_g, r_dry, ξ, ρ_air, T, μ, λ, z_R, L, z₀, ustar, uf_params, κ_vk, grav,
        ap,
    )

Turbulent dry-deposition velocity `V_d,turb = 1/(R_a + R_s)` (m s⁻¹), the
size-segregated scheme of Zhang et al. (2001) with the revised parameters and
functional forms of Emerson et al. (2020). Carries **only** the turbulent
removal — the gravitational contribution `V_g` is deposited by the settling
term's free-outflow boundary, so the two sum to the full deposition velocity
without double counting.

  - `R_a = F_m / (κ_vk · u★)` from the MOST momentum dimensionless profile at
    reference height `z_R` (floored at 0 to guard degenerate strongly-unstable
    profiles), evaluated with the momentum roughness the emission wind
    ([`wind_at_height`](@ref)) uses, rather than Zhang's scalar pair.
  - `R_s = 1 / [ε₀ · u★ · (E_B + E_IM + E_IN) · R₁]` with Brownian collection
    `E_B = C_B·Sc^(-γ)`, impaction `E_IM = C_Im·(St/(α+St))^β`, interception
    `E_IN = 0` over water, rebound `R₁ = exp(-√St)`, smooth-surface Stokes
    number `St = τ·u★²/ν` with the particle relaxation time `τ = V_g/g`, and
    `Sc = ν/D_B`,
    `D_B = k_B·T·Cc/(6π·μ·r_wet)` ([`brownian_diffusivity`](@ref)), with the wet settling
    radius `r_wet = ξ · r_dry`, the air viscosity `μ` and mean free path `λ`
    from the same [`_aerosol_air_state`](@ref) as the settling velocity, and
    `Cc = Cc(λ/r_wet)`.

Emerson et al. (2020) revise the Brownian and impaction terms — a prefactor
`C_B = 0.2` on a land-use-independent `γ = 2/3`, and a prefactor `C_Im = 0.4`
with a softened exponent `β = 1.7` — which together shrink the deposition
velocity of accumulation-mode particles by roughly an order of magnitude
relative to Zhang et al. (2001), in line with the measurements. The
interception term is likewise revised (`C_In = 2.5`, `υ = 0.8`, in
`emerson_interception_*`) but is unused here: it needs the surface's
characteristic collector radius `A`, which the water/ocean category does not
define.

The superseded Zhang et al. (2001) values stay in ClimaParams, deprecated,
and their forms are kept commented out in the body: uncommenting those lines
and the `zhang_*` entries of the `prognostic_aerosol_params` name map runs
the original scheme for a side-by-side comparison.

Every surface currently uses the water/ocean land-use category
(`emerson_impaction_alpha_water`) — exact over ocean, an approximation over
land (TODO: per-land-use parameters, and the interception term, from the
coupler). Zero for calm/degenerate surface states. `r_dry` and `ξ` give the
bin's wet settling radius `r_wet = ξ · r_dry`, the same working radius as the
settling term.
"""
function sslt_dry_deposition_velocity(
    V_g,
    r_dry,
    ξ,
    ρ_air,
    T,
    μ,
    λ,
    z_R,
    L,
    z₀,
    ustar,
    uf_params,
    κ_vk,
    grav,
    ap,
)
    FT = typeof(V_g)
    r_wet = r_dry * ξ

    ζ = iszero(L) ? zero(FT) : z_R / L
    F_m =
        UF.dimensionless_profile(uf_params, z_R, ζ, z₀, UF.MomentumTransport())
    R_a = max(F_m / (κ_vk * ustar), zero(FT))

    C_c = cunningham_slip_correction(λ / r_wet, ap.cunningham_C)
    ν = μ / ρ_air
    D_B = brownian_diffusivity(T, C_c, μ, r_wet, ap.k_B)
    Sc = ν / D_B

    St = V_g * ustar^2 / (grav * ν)
    E_B = ap.dep_C_B * Sc^(-ap.dep_γ)
    E_IM = ap.dep_C_Im * (St / (ap.dep_α_water + St))^ap.dep_β
    R_1 = exp(-sqrt(St))
    R_s = 1 / (ap.dep_ε0 * ustar * (E_B + E_IM) * R_1)

    # Original Zhang et al. (2001) forms, for a side-by-side comparison; the
    # `zhang_*` parameters are deprecated but still in ClimaParams, so this
    # needs only these lines and their name-map entries in
    # `prognostic_aerosol_params` uncommented:
    # E_B = Sc^(-ap.zhang_γ_water)
    # E_IM = (St / (ap.zhang_α_water + St))^ap.zhang_β
    # R_s = 1 / (ap.zhang_ε0 * ustar * (E_B + E_IM) * R_1)

    return max(1 / (R_a + R_s), zero(FT))
end

"""
    sslt_bin_dry_deposition_velocity(
        air, T, r_dry, C_kelvin, ρ_s, ρ_air, z_R, L, z₀, ustar, uf_params, κ_vk, grav, ap,
    )

Turbulent dry-deposition velocity of one bin at its dry settling radius `r_dry`,
given the cell's bin-independent air state `air = (; RH, μ, λ)`: the growth
factor is evaluated once and feeds the wet radius, the wet density, and the
(uncapped) settling speed behind the deposition Stokes number.
"""
function sslt_bin_dry_deposition_velocity(
    air, T, r_dry, C_kelvin, ρ_s, ρ_air, z_R, L, z₀, ustar, uf_params, κ_vk, grav, ap,
)
    (; μ, λ) = air
    ξ, _, V_g = bin_wet_particle(air, T, r_dry, C_kelvin, ρ_s, ρ_air, grav, ap)
    return sslt_dry_deposition_velocity(
        V_g, r_dry, ξ, ρ_air, T, μ, λ, z_R, L, z₀, ustar, uf_params, κ_vk, grav,
        ap,
    )
end

"""
    sslt_bin_budgeted_deposition_velocity(
        air, T, r_dry, C_kelvin, ρ_s, ρ_air, z_R, L, z₀, ustar, uf_params, κ_vk,
        grav, courant_budget, ap,
    )

[`sslt_bin_dry_deposition_velocity`](@ref) limited so that settling and dry
deposition TOGETHER stay inside one explicit-removal budget `courant_budget`
(`settling_courant_max · Δz₁ / Δt`), rather than each being capped at the full
budget separately.

Capping them separately lets the lowest cell lose `Courant = 1` in a single
stage — settling drains up to half of it through the free-outflow surface
boundary while dry deposition removes up to half again — which empties the cell
and, with any stage-splitting overshoot, drives it negative. Both sinks are
proportional to the tracer, so a negative concentration turns them into
sources: sea-salt bin 5 (the only bin whose velocities saturate both caps)
diverged super-exponentially from the surface upward this way. The joint budget
is also what `settling_courant_max` is documented to mean in ClimaParams:
"the explicit gravitational settling *and* turbulent dry-deposition sinks".

Settling has first claim on the budget because it is the term with the surface
outflow boundary; dry deposition receives whatever is left, which is zero in
the (rare) cells where settling alone saturates it.
"""
function sslt_bin_budgeted_deposition_velocity(
    air, T, r_dry, C_kelvin, ρ_s, ρ_air, z_R, L, z₀, ustar, uf_params, κ_vk,
    grav, courant_budget, ap,
)
    FT = typeof(courant_budget)
    (; μ, λ) = air
    ξ, _, V_g = bin_wet_particle(air, T, r_dry, C_kelvin, ρ_s, ρ_air, grav, ap)
    V_d = sslt_dry_deposition_velocity(
        V_g, r_dry, ξ, ρ_air, T, μ, λ, z_R, L, z₀, ustar, uf_params, κ_vk, grav,
        ap,
    )
    # what settling already claims in this cell, then the remainder
    remaining = courant_budget - min(V_g, courant_budget)
    return min(V_d, max(remaining, zero(FT)))
end
