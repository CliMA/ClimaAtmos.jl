#####
##### Horizontal implicit acoustic wave treatment
#####
##### Uses a linearized pressure gradient force (PGF) in the implicit tendency
##### to avoid evaluating Exner-based thermodynamic functions (which require
##### log(p)) on potentially unphysical Newton trial states. The nonlinear
##### correction (full Exner PGF minus linearized PGF) is added in the explicit
##### tendency where the state is always physical.
#####

"""
    HorizontalAcousticCache

Stores horizontally-uniform reference-state fields used by the linearized
horizontal acoustic tendency. Only algebraic quantities are stored — no
Exner/log-based functions — so the update is safe on any Newton trial state.

Fields:
- `ᶜρ_ref`: reference density (used as denominator in linearized PGF)
- `ᶜp_ref`: reference pressure (diagnostic / future Helmholtz solver)
- `ᶜcs²_ref`: reference squared sound speed (diagnostic / future Helmholtz solver)
"""
struct HorizontalAcousticCache{R, P, CS}
    ᶜρ_ref::R
    ᶜp_ref::P
    ᶜcs²_ref::CS
end

"""
    horizontal_acoustic_cache(Y, atmos)

Allocate the [`HorizontalAcousticCache`](@ref) when
`atmos.horizontal_acoustic_mode == Implicit()`, otherwise return `nothing`.
"""
function horizontal_acoustic_cache(Y, atmos)
    atmos.horizontal_acoustic_mode == Implicit() || return nothing
    FT = Spaces.undertype(axes(Y.c))
    ᶜρ_ref = similar(Y.c, FT)
    ᶜp_ref = similar(Y.c, FT)
    ᶜcs²_ref = similar(Y.c, FT)
    return HorizontalAcousticCache(ᶜρ_ref, ᶜp_ref, ᶜcs²_ref)
end

"""
    update_horizontal_acoustic_reference_state!(ha_cache, Y, p)

Update reference-state fields from the current precomputed quantities.
Only uses ideal-gas-law quantities (ρ, p = ρ R T, cs² = γ R T) — no
`exner_given_pressure` or `log(p)` — so this is safe even when Newton
trial states produce negative pressure.
"""
function update_horizontal_acoustic_reference_state!(ha_cache, Y, p)
    isnothing(ha_cache) && return nothing
    (; ᶜp, ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    R_d = CAP.R_d(p.params)

    @. ha_cache.ᶜρ_ref = Y.c.ρ
    @. ha_cache.ᶜp_ref = ᶜp
    κ_m = @. lazy(
        TD.gas_constant_air(thermo_params, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) /
        TD.cv_m(thermo_params, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno),
    )
    @. ha_cache.ᶜcs²_ref = (1 + κ_m) * R_d * ᶜT
    return nothing
end

"""
    implicit_horizontal_acoustic_tendency!(Yₜ, Y, p, t)

Add the horizontally-implicit acoustic tendencies to `Yₜ`:

1. **Mass divergence** (full nonlinear): ``-∇_h ⋅ (ρ \\mathbf{u})``
2. **Energy flux divergence** (full nonlinear): ``-∇_h ⋅ (ρ h_{tot} \\mathbf{u})``
3. **Linearized pressure gradient**: ``-(1/ρ_{ref}) ∇_h p``

   where ``p = ρ R_m T`` from the ideal gas law (no `log(p)` needed).

The nonlinear correction (full Exner split-form PGF minus linearized PGF)
is retained in the explicit tendency (`horizontal_dynamics_tendency!`).
"""
function implicit_horizontal_acoustic_tendency!(Yₜ, Y, p, t)
    (; ᶜu, ᶜp, ᶜh_tot) = p.precomputed
    ᶜρ_ref = p.horizontal_acoustic_cache.ᶜρ_ref

    @. Yₜ.c.ρ -= split_divₕ(Y.c.ρ * ᶜu, 1)

    @. Yₜ.c.ρe_tot -= split_divₕ(Y.c.ρ * ᶜu, ᶜh_tot)

    @. Yₜ.c.uₕ -= C12(gradₕ(ᶜp) / ᶜρ_ref)
    return nothing
end
