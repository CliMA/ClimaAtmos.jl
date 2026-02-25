#####
##### Horizontal implicit acoustic wave treatment
#####
##### Implements the Option B predictor-corrector for horizontal acoustic waves.
##### The Newton solver in implicit_tendency! sees only vertical coupling (column-local).
##### After Newton convergence, set_precomputed_quantities! (cache!) calls
##### horizontal_helmholtz_correction! to apply the linearized horizontal Helmholtz
##### correction directly to the state Y. constrain_state! (dss!) is NOT used for this,
##### because it is also called on stage initial guesses where Helmholtz must not run.
##### The nonlinear correction (full Exner PGF minus linearized PGF) is added in the
##### explicit tendency (horizontal_dynamics_tendency!) where the state is always physical.
#####
##### Implementation note (Option B, horizontal_implicit_acoustic_tendencies.md):
##### The residual R(Y) includes horizontal acoustic coupling (∂R_ρ/∂uₕ, ∂R_uₕ/∂ρ, etc.).
##### The assembled Jacobian (dense or sparse) is column-local and omits this coupling,
##### so Newton with direct linear solve can produce unphysical states (e.g. negative ρ).
##### Two paths to allow larger dt without reducing timestep:
##### 1) Krylov workaround: set use_krylov_method: true. The Newton–Krylov solver uses
#####    Jacobian-vector products from the full residual; in practice some runs can fail
#####    earlier than with the direct solve (e.g. ~day 0 second 76800). Not reliable.
##### 2) Full Option B (predictor-corrector): implicit_tendency! omits horizontal acoustic
#####    terms so Newton sees only vertical coupling; after Newton convergence,
#####    set_precomputed_quantities! (cache!) calls horizontal_helmholtz_correction! to
#####    solve the linearized horizontal acoustic subsystem (Helmholtz for ρ, then update
#####    uₕ and ρe_tot). constrain_state! (dss!) is NOT used so initial guesses are clean.
#####

"""
    HorizontalAcousticCache

Stores horizontally-uniform reference-state fields used by the linearized
horizontal acoustic tendency and (when using predictor-corrector) scratch
for the horizontal Helmholtz corrector.

Fields:
- `ᶜρ_ref`: reference density (used as denominator in linearized PGF)
- `ᶜp_ref`: reference pressure (diagnostic / future Helmholtz solver)
- `ᶜcs²_ref`: reference squared sound speed (diagnostic / future Helmholtz solver)
- `ᶜhelmholtz_ρ_rhs`: scratch for Helmholtz RHS (predictor-corrector only)
- `ᶜhelmholtz_ρ_sol`: scratch for Helmholtz solution iterate (predictor-corrector only)
- `ᶜhelmholtz_dss_buffer`: pre-allocated DSS buffer for the scalar iterate (nothing on
  non-SEM spaces). Required: wdivₕ and gradₕ are element-local operators; without DSS
  between Richardson iterations the effective Laplacian is block-diagonal (element-local)
  whose maximum eigenvalue scales with the GLL intra-element spacing (~p²/h²) rather than
  the element spacing (1/h²), making the spectral radius >> 1 and the iteration diverge.
"""
struct HorizontalAcousticCache{R, P, CS, HRS, HSO, HDB}
    ᶜρ_ref::R
    ᶜp_ref::P
    ᶜcs²_ref::CS
    ᶜhelmholtz_ρ_rhs::HRS
    ᶜhelmholtz_ρ_sol::HSO
    ᶜhelmholtz_dss_buffer::HDB  # Spaces.create_dss_buffer(ᶜhelmholtz_ρ_sol), or nothing
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
    ᶜhelmholtz_ρ_rhs = similar(Y.c, FT)
    ᶜhelmholtz_ρ_sol = similar(Y.c, FT)
    # Initialize from initial state so the cache is valid before first update_horizontal_acoustic_reference_state!.
    # ᶜcs²_ref must be zero-initialized so that horizontal_helmholtz_correction! is a no-op
    # during the initial set_precomputed_quantities! call at cache build time (before the
    # reference state has been set by update_horizontal_acoustic_reference_state!).
    @. ᶜρ_ref = Y.c.ρ
    fill!(parent(ᶜcs²_ref), zero(FT))
    # DSS buffer for the scalar Richardson iterate. Required on GLL (SEM) spaces so that
    # wdivₕ(gradₕ(...)) assembles the global Laplacian rather than the element-local one.
    ᶜhelmholtz_dss_buffer =
        do_dss(axes(Y.c)) ? Spaces.create_dss_buffer(ᶜhelmholtz_ρ_sol) : nothing
    return HorizontalAcousticCache(
        ᶜρ_ref,
        ᶜp_ref,
        ᶜcs²_ref,
        ᶜhelmholtz_ρ_rhs,
        ᶜhelmholtz_ρ_sol,
        ᶜhelmholtz_dss_buffer,
    )
end

"""
    update_horizontal_acoustic_reference_state!(ha_cache, Y, p)

Update reference-state fields from the current state. Only called from
`set_explicit_precomputed_quantities!` (at step start), so Y is always
physical and ρ_ref stays positive. Not called during the implicit solve,
where Newton trial states could be unphysical.
Uses ideal-gas-law quantities (ρ, p = ρ R T, cs² = γ R T) — no
`exner_given_pressure` or `log(p)`.
"""
function update_horizontal_acoustic_reference_state!(ha_cache, Y, p)
    isnothing(ha_cache) && return nothing
    (; ᶜp, ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    R_d = CAP.R_d(p.params)

    # Only ever called with physical state (from set_explicit_precomputed_quantities! at step start).
    @. ha_cache.ᶜρ_ref = Y.c.ρ
    @. ha_cache.ᶜp_ref = ᶜp
    κ_m = @. lazy(
        TD.gas_constant_air(thermo_params, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) /
        TD.cv_m(thermo_params, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno),
    )
    @. ha_cache.ᶜcs²_ref = (1 + κ_m) * R_d * ᶜT
    return nothing
end

# Number of Richardson iterations for (α - ∇_h²) solve in the corrector
const N_HELMHOLTZ_ITERATIONS = 15

"""
    horizontal_helmholtz_correction!(Y, p, t)

Predictor-corrector Option B: solve the linearized horizontal acoustic subsystem
and update Y in place. Called from set_precomputed_quantities! (cache!) when
horizontal_acoustic_mode == Implicit(). NOT called from constrain_state! (dss!),
because constrain_state! also runs on stage initial guesses where the Helmholtz
correction must not be applied.

Linearized backward-Euler system (ρ_ref, cs²_ref from step-start reference state):
  ρ^{n+1} = ρ^n - dt*ρ_ref*div_h(uₕ^{n+1})
  uₕ^{n+1} = uₕ^n - dt*(cs²_ref/ρ_ref)*grad_h(ρ^{n+1})
Eliminating uₕ yields (1 - dt²*cs²_ref*∇_h²) ρ^{n+1} = ρ^n - dt*ρ_ref*div_h(uₕ^n).
We solve this un-divided form directly via Richardson iteration
  ρ^{k+1} = RHS + dt²*cs²_ref * ∇_h² ρ^k
(iteration matrix dt²*cs²_ref*∇_h², spectral radius = horizontal acoustic CFL²),
then update uₕ and ρe_tot.
"""
function horizontal_helmholtz_correction!(Y, p, t)
    ha = p.horizontal_acoustic_cache
    isnothing(ha) && return nothing
    (; ᶜρ_ref, ᶜcs²_ref, ᶜhelmholtz_ρ_rhs, ᶜhelmholtz_ρ_sol, ᶜhelmholtz_dss_buffer) =
        ha
    dt = p.dt
    FT = Spaces.undertype(axes(Y.c))

    # Step-by-step NaN checks to pinpoint the source.
    # Check all of Y (including Y.f) so Newton-NaN in u₃ or other face fields is caught.
    if any(isnan, parent(Y.c)) || any(isnan, parent(Y.f))
        nan_c = any(isnan, parent(Y.c))
        nan_f = any(isnan, parent(Y.f))
        error(
            "horizontal_helmholtz_correction!: NaN in Y before Helmholtz " *
            "(Newton produced NaN). Y.c has NaN: $nan_c, Y.f has NaN: $nan_f. " *
            "min(ρ) = $(minimum(parent(Y.c.ρ)))",
        )
    end

    # RHS for (1 - dt²·cs²_ref·∇_h²) ρ = RHS: ρ^n - dt*ρ_ref*div_h(uₕ^n)
    @. ᶜhelmholtz_ρ_rhs = Y.c.ρ - dt * ᶜρ_ref * divₕ(Y.c.uₕ)
    if any(isnan, parent(ᶜhelmholtz_ρ_rhs))
        error(
            "horizontal_helmholtz_correction!: NaN in ᶜhelmholtz_ρ_rhs (RHS). " *
            "Likely cause: NaN from divₕ(Y.c.uₕ). " *
            "max(|uₕ|) = $(maximum(abs, parent(Y.c.uₕ)))",
        )
    end

    # Coefficient dt²·cs²_ref; clamp cs²_ref to avoid zero (no-op when reference not set)
    ᶜdt2cs2 = @. dt^2 * max(ᶜcs²_ref, eps(FT))
    # Initialize with current ρ
    @. ᶜhelmholtz_ρ_sol = Y.c.ρ
    # Richardson iteration: ρ^{k+1} = RHS + dt²·cs²_ref·∇_h²·ρ^k
    #
    # DSS is applied after each step so that wdivₕ(gradₕ(...)) assembles the *global*
    # SEM Laplacian rather than the element-local (block-diagonal) one. Without DSS the
    # effective operator's maximum eigenvalue scales with the GLL intra-element spacing
    # (~p²/h², up to ~16× larger for p=4), making the spectral radius >> 1 and the
    # iteration diverge to Inf → NaN.
    for k in 1:N_HELMHOLTZ_ITERATIONS
        @. ᶜhelmholtz_ρ_sol =
            ᶜhelmholtz_ρ_rhs + ᶜdt2cs2 * wdivₕ(gradₕ(ᶜhelmholtz_ρ_sol))
        if !isnothing(ᶜhelmholtz_dss_buffer)
            Spaces.weighted_dss!(ᶜhelmholtz_ρ_sol => ᶜhelmholtz_dss_buffer)
        end
        if any(isnan, parent(ᶜhelmholtz_ρ_sol))
            error(
                "horizontal_helmholtz_correction!: NaN in Richardson iterate at k=$k. " *
                "Spectral radius may exceed 1. " *
                "max(cs²_ref)=$(maximum(parent(ᶜcs²_ref))), dt=$dt, " *
                "DSS buffer present: $(! isnothing(ᶜhelmholtz_dss_buffer))",
            )
        end
    end

    @. Y.c.ρ = max(_implicit_ρ_min, ᶜhelmholtz_ρ_sol)

    # uₕ^{n+1} = uₕ^n - dt*(cs²_ref/ρ_ref)*grad_h(ρ^{n+1})
    @. Y.c.uₕ -= dt * C12((ᶜcs²_ref / ᶜρ_ref) * gradₕ(Y.c.ρ))
    if any(isnan, parent(Y.c.uₕ))
        error(
            "horizontal_helmholtz_correction!: NaN in Y.c.uₕ after uₕ update. " *
            "min(ρ_ref)=$(minimum(parent(ᶜρ_ref))), " *
            "max(cs²_ref)=$(maximum(parent(ᶜcs²_ref)))",
        )
    end

    # Energy: (ρe)^{n+1} = (ρe)^n - dt*ρ_ref*div_h(h_ref*uₕ^{n+1}), h_ref = cs²_ref/(γ-1)
    kappa_d = FT(CAP.kappa_d(p.params))
    ᶜh_ref = @. ᶜcs²_ref * (1 - kappa_d) / kappa_d
    @. Y.c.ρe_tot -= dt * split_divₕ(ᶜρ_ref * ᶜh_ref * Y.c.uₕ, 1)
    if any(isnan, parent(Y.c.ρe_tot))
        error(
            "horizontal_helmholtz_correction!: NaN in Y.c.ρe_tot after energy update. " *
            "max(|uₕ|)=$(maximum(abs, parent(Y.c.uₕ))), " *
            "max(h_ref)=$(maximum(parent(ᶜh_ref)))",
        )
    end

    return nothing
end
