#=
Flux-form FDDG tendencies (compute_tendency_fddg!, implicit_tendency_fddg!).

Full system in flux form: (ρ, ρe, ρu⃗) with momentum in GLOBAL CARTESIAN
components, all horizontal terms discretized with Kennedy–Gruber
flux-differencing volume fluxes + KG-Rusanov/KG-Roe interfaces (Souza et
al. 2023). The constant Cartesian basis makes component-wise flux
differencing kinetic-energy-preserving with no curvature source terms.

Shallow atmosphere: the Cartesian center-momentum tendency is projected
tangentially against r̂; ρw (faces, Covariant3) carries radial momentum.

State: Y.c = (; ρ, ρe, ρu1, ρu2, ρu3[, ρq_tot]), Y.f = (; ρw).
=#

# Shared tendency core: `vertical_transport = true` gives the full tendency;
# `false` gives the HEVI explicit part (everything except the central
# vertical mass/energy fluxes and the ρw pressure-gradient + buoyancy terms,
# which live in implicit_tendency_fddg!). The (VanLeer − central) energy
# correction and the momentum-component vertical transport stay explicit,
# so the HEVI total equals the fully explicit path.
function compute_tendency_fddg!(
    dY,
    Y,
    m::DGModel{FT},
    t,
    vertical_transport,
) where {FT}
    c = m.c
    (;
        Ic, If, vdivf2c, vdivf2c3, vvdivc2f, VanLeer, ᶠupwind1,
        ᶠgradᵥ, Bw, hwdiv, hgrad,
    ) = m.ops
    (; ᶜΦ, ᶠβ_sponge, ᶜβ_sponge) = m.fields
    (; eE1, eE2, eE3, eN1, eN2, eN3, eR1, eR2, eR3, E1, E2, E3) = m.fields
    Δt = m.Δt
    κ₄ = m.κ₄

    Yc = Y.c
    ρw = Y.f.ρw
    dYc = dY.c
    dρw = dY.f.ρw
    ρ = Yc.ρ
    ρe = Yc.ρe
    ρw_w = @. Geometry.WVector(ρw)
    lgeom_c = Fields.local_geometry_field(m.spaces.hv_center_space)
    lgeom_f = Fields.local_geometry_field(m.spaces.hv_face_space)

    # Velocities: tangential-project the state (guards roundoff drift)
    u1r = @. Yc.ρu1 / ρ
    u2r = @. Yc.ρu2 / ρ
    u3r = @. Yc.ρu3 / ρ
    ur = @. u1r * eR1 + u2r * eR2 + u3r * eR3
    u1 = @. u1r - ur * eR1
    u2 = @. u2r - ur * eR2
    u3 = @. u3r - ur * eR3
    uE = @. u1 * eE1 + u2 * eE2 + u3 * eE3
    uN = @. u1 * eN1 + u2 * eN2 + u3 * eN3
    uv = @. Geometry.UVVector(uE, uN)
    w_c = @. Ic(ρw_w).components.data.:1 / ρ

    K = @. (uE^2 + uN^2 + w_c^2) / 2
    # Moist (q_tot = ρq_tot/ρ) via the closed-form moist_p_dyn, or the dry EOS.
    # λ keeps the dry γ — a conservative sound-speed over-estimate, harmless for
    # the interface dissipation.
    moist = m.prob.moisture != :dry
    q_tot = moist ? (@. Yc.ρq_tot / ρ) : nothing
    dyn =
        moist ?
        (@. moist_p_dyn(m.fields.thermo_params, ρ, ρe / ρ - K - ᶜΦ, q_tot)) :
        nothing
    p = moist ? dyn.p : (@. pres_ρe(c, ρe, K, ᶜΦ, ρ))
    T_air = moist ? dyn.T : nothing
    e = @. ρe / ρ
    h_tot = @. (ρe + p) / ρ
    λ = @. sqrt(uE^2 + uN^2) + sqrt(c.γ * p / ρ)

    # --- Horizontal: FDDG volume + KG interfaces, full system ---
    dy_mw = map(
        _ -> (ρ = FT(0), ρe = FT(0), ρu1 = FT(0), ρu2 = FT(0), ρu3 = FT(0)),
        ρ,
    )
    # Momentum pressure `pm`: full thermodynamic p (:conservative) or the
    # perturbation p′ = p − p_ref (:conservative_pert, well-balanced over
    # terrain — differences the small p′). Energy always carries the full p.
    pm = m.prob.pgf == :conservative_pert ? (@. p - m.fields.ᶜp_ref) : p
    # State for the conservative (ρ,ρe,ρu⃗) two-point fluxes. `pm` feeds the
    # momentum pressure slot; `φ` (geopotential) feeds the Waruszewski gravity
    # fluctuation ½ρ̂⟦φ⟧ (KG/Ranocha volume + Roe/Rusanov interfaces ignore it;
    # Φ is single-valued at faces). Moist adds `q` for the tracer flux.
    y =
        moist ?
        map(
            (ρi, ρei, ei, pi, pmi, uvi, u1i, u2i, u3i, E1i, E2i, E3i, λi, φi, qi) -> (;
                ρ = ρi, ρe = ρei, e = ei, p = pi, pm = pmi, uv = uvi,
                u1 = u1i, u2 = u2i, u3 = u3i,
                E1 = E1i, E2 = E2i, E3 = E3i, λ = λi, φ = φi, q = qi,
            ),
            ρ, ρe, e, p, pm, uv, u1, u2, u3, E1, E2, E3, λ, ᶜΦ, q_tot,
        ) :
        map(
            (ρi, ρei, ei, pi, pmi, uvi, u1i, u2i, u3i, E1i, E2i, E3i, λi, φi) -> (;
                ρ = ρi, ρe = ρei, e = ei, p = pi, pm = pmi, uv = uvi,
                u1 = u1i, u2 = u2i, u3 = u3i,
                E1 = E1i, E2 = E2i, E3 = E3i, λ = λi, φ = φi,
            ),
            ρ, ρe, e, p, pm, uv, u1, u2, u3, E1, E2, E3, λ, ᶜΦ,
        )
    # Horizontal two-point VOLUME flux (KEP family): KG / Ranocha / Waruszewski.
    # The interface flux (m.interface_flux_fn) is paired to the same family in
    # model.jl so the central parts match.
    volume_flux_fn =
        m.prob.volume_flux == :waruszewski ?
        Operators.waruszewski_cartesian_flux :
        m.prob.volume_flux == :ranocha ? Operators.ranocha_cartesian_flux :
        Operators.kennedy_gruber_cartesian_flux
    Operators.add_flux_differencing_divergence!(volume_flux_fn, dy_mw, y)
    Operators.add_numerical_flux_internal!(m.interface_flux_fn, dy_mw, y)
    @. dYc.ρ = dy_mw.ρ / lgeom_c.WJ
    @. dYc.ρe = dy_mw.ρe / lgeom_c.WJ
    @. dYc.ρu1 = dy_mw.ρu1 / lgeom_c.WJ
    @. dYc.ρu2 = dy_mw.ρu2 / lgeom_c.WJ
    @. dYc.ρu3 = dy_mw.ρu3 / lgeom_c.WJ

    # --- Vertical FD (implicit under HEVI). The face flux is the TOTAL
    #     momentum: its CT3 projection carries the terrain cross-term
    #     ρuₕ·∇ₓξ³, required for the discrete GCL Σᵢ Dᵢ(Jaⁱ) = 0 to close
    #     against the horizontal FDDG rows over warped grids (w-only flux
    #     leaves a spurious source −(1/J)∂_η(J ∂ξ³∂x·ρuₕ), fatal under
    #     SLEVE). Flat topography: cross entries are exact zeros. ---
    ᶠρuE = @. If(ρ * uE)
    ᶠρuN = @. If(ρ * uN)
    ᶠM = @. Geometry.UVWVector(ᶠρuE, ᶠρuN, ρw_w.components.data.:1)
    # VERT_ENERGY — vertical h_tot (and, under HEVI, q_tot) transport:
    #   vanleer (default): 2nd-order monotone; the explicit correction's
    #     (1 − |v|Δt) factor requires vertical advective C < 1.
    #   central: central-implicit; no vertical dissipation (diagnostic only).
    #   upwind1: 1st-order upwind, fully implicit (linear at frozen flux
    #     sign): unconditionally stable and monotone — for large dt over
    #     steep warped grids.
    vert_energy = Symbol(get(ENV, "VERT_ENERGY", "vanleer"))
    if vertical_transport
        @. dYc.ρ -= vdivf2c(ᶠM)
        if vert_energy == :central
            @. dYc.ρe -= vdivf2c(ᶠM * If(h_tot))
        elseif vert_energy == :upwind1
            @. dYc.ρe -= vdivf2c(ᶠupwind1(ᶠM, h_tot))
        else
            @. dYc.ρe -= vdivf2c(VanLeer(ᶠM, h_tot, Δt))
        end
    else
        # Mass/energy transport by the full ᶠM is implicit (the cross-term
        # is frozen in ρuₕ, so the Jacobian is unchanged). Explicit
        # remainder: only the (VanLeer − central) correction (vanleer mode).
        if vert_energy == :vanleer
            @. dYc.ρe -=
                vdivf2c(VanLeer(ᶠM, h_tot, Δt)) - vdivf2c(ᶠM * If(h_tot))
        end
    end
    # Vertical momentum-component transport: central-implicit under HEVI
    # (tridiagonal Jacobian blocks; removes the explicit dt ≲ Δz_min/w
    # limit); Lin-VanLeer under the explicit stepper (C < 1 regime).
    vert_imp = m.prob.stepper == :hevi
    if !vert_imp
        @. dYc.ρu1 -= vdivf2c(VanLeer(ᶠM, u1, Δt))
        @. dYc.ρu2 -= vdivf2c(VanLeer(ᶠM, u2, Δt))
        @. dYc.ρu3 -= vdivf2c(VanLeer(ᶠM, u3, Δt))
    elseif vertical_transport
        @. dYc.ρu1 -= vdivf2c(ᶠM * If(u1))
        @. dYc.ρu2 -= vdivf2c(ᶠM * If(u2))
        @. dYc.ρu3 -= vdivf2c(ᶠM * If(u3))
    end # HEVI remaining tendency: momentum transport is fully implicit

    # --- Moisture: ρq_tot transport (all explicit — never in the implicit
    #     acoustic subsystem). Horizontal: KG tracer flux riding the same mass
    #     flux as continuity + Rusanov penalty. Vertical: VanLeer. Then 0-moment
    #     precipitation. ---
    if moist
        dy_q = map(_ -> (ρq = FT(0),), ρ)
        Operators.add_flux_differencing_divergence!(
            Operators.kennedy_gruber_tracer_flux,
            dy_q,
            y,
        )
        Operators.add_numerical_flux_internal!(
            Operators.kennedy_gruber_rusanov_tracer,
            dy_q,
            y,
        )
        @. dYc.ρq_tot = dy_q.ρq / lgeom_c.WJ
        # Vertical: implicit under HEVI (upwind1 mode is monotone; central
        # mode relies on the smooth moisture IC — monitor minρq);
        # Lin-VanLeer under the explicit stepper.
        if !vert_imp
            @. dYc.ρq_tot -= vdivf2c(VanLeer(ᶠM, q_tot, Δt))
        elseif vertical_transport
            if vert_energy == :upwind1
                @. dYc.ρq_tot -= vdivf2c(ᶠupwind1(ᶠM, q_tot))
            else
                @. dYc.ρq_tot -= vdivf2c(ᶠM * If(q_tot))
            end
        end
        m.prob.microphysics == :zero_moment &&
            microphysics_0m_tendency!(dYc, ρ, ᶜΦ, q_tot, T_air, m)
    end

    # --- Coriolis: −2Ω ẑ×u⃗, exact in the constant Cartesian frame ---
    @. dYc.ρu1 += 2 * c.Ω * ρ * u2
    @. dYc.ρu2 -= 2 * c.Ω * ρ * u1

    # --- Held–Suarez forcing (reused ClimaAtmos implementation) ---
    m.prob.held_suarez && hs_forcing_fddg!(dYc, ρ, p, u1, u2, u3, m)

    # --- κ₄ hyperdiffusion (h_tot + Cartesian velocity components) ---
    if κ₄ != 0
        τ_κ₄ = Operators.ldg_penalty_parameter(κ₄, m.spaces.hv_center_space)
        χe = similar(h_tot)
        @. χe = hwdiv(hgrad(h_tot))
        χ1 = similar(u1)
        @. χ1 = hwdiv(hgrad(u1))
        χ2 = similar(u2)
        @. χ2 = hwdiv(hgrad(u2))
        χ3 = similar(u3)
        @. χ3 = hwdiv(hgrad(u3))
        de4 = Operators.ldg_laplacian_tendency(χe, ρ, κ₄, τ_κ₄)
        du1 = Operators.ldg_laplacian_tendency(χ1, ρ, κ₄, τ_κ₄)
        du2 = Operators.ldg_laplacian_tendency(χ2, ρ, κ₄, τ_κ₄)
        du3 = Operators.ldg_laplacian_tendency(χ3, ρ, κ₄, τ_κ₄)
        @. dYc.ρe -= de4
        @. dYc.ρu1 -= du1
        @. dYc.ρu2 -= du2
        @. dYc.ρu3 -= du3
    end

    # --- Tangential projection of the momentum tendency (shallow atm.) ---
    dmr = @. dYc.ρu1 * eR1 + dYc.ρu2 * eR2 + dYc.ρu3 * eR3
    @. dYc.ρu1 -= dmr * eR1
    @. dYc.ρu2 -= dmr * eR2
    @. dYc.ρu3 -= dmr * eR3

    # --- Terrain metric-defect correction (wb_metric == :metric_source):
    #     subtract p·δ, δ the IC-agnostic pure-metric GCL defect calibrated on
    #     an isothermal rest state (calibrate_wb_metric!). Removes the spurious
    #     terrain force p·δ; leaves real pressure gradients (jet drive) intact.
    #     Explicit + ungated ⇒ HEVI split rhs = imp + rem preserved. ---
    if m.prob.wb_metric === :metric_source
        (; ᶜwb_δ1, ᶜwb_δ2, ᶜwb_δ3) = m.fields
        @. dYc.ρu1 -= p * ᶜwb_δ1
        @. dYc.ρu2 -= p * ᶜwb_δ2
        @. dYc.ρu3 -= p * ᶜwb_δ3
    end

    # --- ρw: pressure gradient + buoyancy (discretely balanced pair,
    #     implicit under HEVI), vertical advection, horizontal DG
    #     advection, sponge ---
    w = @. ρw_w / If(ρ)
    if vertical_transport
        if m.prob.pgf == :conservative_pert
            # Stratified: −∂_ξ³ p′ − (ρ − ρ_ref) g buoyancy pair — differences
            # the small p′ ⇒ well-balanced over terrain (pm = p − p_ref).
            @. dρw = Bw(
                -(ᶠgradᵥ(pm) + If(ρ - m.fields.ᶜρ_ref) * ᶠgradᵥ(ᶜΦ)) -
                C3(vvdivc2f(Ic(ᶠM ⊗ w)), lgeom_f),
            )
        else
            @. dρw = Bw(
                -(ᶠgradᵥ(p) + If(ρ) * ᶠgradᵥ(ᶜΦ)) -
                C3(vvdivc2f(Ic(ᶠM ⊗ w)), lgeom_f),
            )
        end
    else
        @. dρw = Bw(-C3(vvdivc2f(Ic(ᶠM ⊗ w)), lgeom_f))
    end
    ρw_sc = @. ρw_w.components.data.:1
    uvf = @. If(uv)
    λf =
        m.prob.interface_flux == :roe ? (@. If(sqrt(uE^2 + uN^2))) :
        (@. If(λ))
    y_f = map((h, uvi, λi) -> (; h = h, uv = uvi, λ = λi), ρw_sc, uvf, λf)
    dρw_mw = @. hwdiv(uvf * ρw_sc) * (-(lgeom_f.WJ))
    Operators.add_numerical_flux_internal!(
        Operators.kennedy_gruber_rusanov_height,
        dρw_mw,
        y_f,
    )
    @. dρw += C3(Geometry.WVector(dρw_mw / lgeom_f.WJ), lgeom_f)
    @. dρw -= ᶠβ_sponge * ρw
    if m.prob.sponge_uh
        @. dYc.ρu1 -= ᶜβ_sponge * Yc.ρu1
        @. dYc.ρu2 -= ᶜβ_sponge * Yc.ρu2
        @. dYc.ρu3 -= ᶜβ_sponge * Yc.ρu3
    end
    return dY
end

rhs_fddg!(dY, Y, m, t) = compute_tendency_fddg!(dY, Y, m, t, true)
remaining_tendency_fddg!(dY, Y, m, t) =
    compute_tendency_fddg!(dY, Y, m, t, false)

# HEVI implicit part: central vertical mass/energy fluxes + ρw pressure
# gradient and buoyancy (the discretely balanced pair). Linear in ρw given
# frozen h_tot; Jacobian in jacobians.jl.
function implicit_tendency_fddg!(dY, Y, m::DGModel{FT}, t) where {FT}
    c = m.c
    (; Ic, If, vdivf2c, ᶠupwind1, ᶠgradᵥ, Bw) = m.ops
    (; ᶜΦ, eE1, eE2, eE3, eN1, eN2, eN3) = m.fields

    Yc = Y.c
    ρ = Yc.ρ
    ρe = Yc.ρe
    ρw_w = @. Geometry.WVector(Y.f.ρw)

    # Same tangential projection as the explicit path (keeps the HEVI
    # split identity rhs == implicit + remaining exact).
    (; eR1, eR2, eR3) = m.fields
    u1r = @. Yc.ρu1 / ρ
    u2r = @. Yc.ρu2 / ρ
    u3r = @. Yc.ρu3 / ρ
    ur = @. u1r * eR1 + u2r * eR2 + u3r * eR3
    u1 = @. u1r - ur * eR1
    u2 = @. u2r - ur * eR2
    u3 = @. u3r - ur * eR3
    uE = @. u1 * eE1 + u2 * eE2 + u3 * eE3
    uN = @. u1 * eN1 + u2 * eN2 + u3 * eN3
    w_c = @. Ic(ρw_w).components.data.:1 / ρ
    K = @. (uE^2 + uN^2 + w_c^2) / 2
    # Moist p must match the explicit path so the HEVI split rhs = imp + rem
    # holds (the column Jacobian keeps the dry-effective ∂p coefficients — an
    # approximate preconditioner, still convergent under Newton).
    moist = m.prob.moisture != :dry
    p_thermo =
        moist ?
        (@. moist_p_dyn(m.fields.thermo_params, ρ, ρe / ρ - K - ᶜΦ, Yc.ρq_tot / ρ).p) :
        (@. pres_ρe(c, ρe, K, ᶜΦ, ρ))
    h_tot = @. (ρe + p_thermo) / ρ

    # Full central vertical transport by the TOTAL flux (ρw + terrain
    # cross-term ρuₕ·∇ₓξ³), mirroring the ClimaAtmos CG implicit block
    # (implicit_tendency.jl uses the full ᶠu³). The cross-term is frozen in
    # ρuₕ (identity rows in the implicit system), so it only shifts the
    # Newton residual — the ∂/∂ρw Jacobian (jacobians.jl) is unchanged.
    ᶠρuE = @. If(ρ * uE)
    ᶠρuN = @. If(ρ * uN)
    ᶠM = @. Geometry.UVWVector(ᶠρuE, ᶠρuN, ρw_w.components.data.:1)
    @. dY.c.ρ = -vdivf2c(ᶠM)
    # Energy: central by default (VanLeer correction rides explicit);
    # monotone first-order upwind under VERT_ENERGY=upwind1.
    if Symbol(get(ENV, "VERT_ENERGY", "vanleer")) == :upwind1
        @. dY.c.ρe = -vdivf2c(ᶠupwind1(ᶠM, h_tot))
    else
        @. dY.c.ρe = -vdivf2c(ᶠM * If(h_tot))
    end
    # Momentum/moisture vertical transport by the frozen flux ᶠM: linear in
    # the transported quantity (tridiagonal Jacobian blocks in jacobians.jl).
    @. dY.c.ρu1 = -vdivf2c(ᶠM * If(u1))
    @. dY.c.ρu2 = -vdivf2c(ᶠM * If(u2))
    @. dY.c.ρu3 = -vdivf2c(ᶠM * If(u3))
    if moist
        if Symbol(get(ENV, "VERT_ENERGY", "vanleer")) == :upwind1
            @. dY.c.ρq_tot = -vdivf2c(ᶠupwind1(ᶠM, Yc.ρq_tot / ρ))
        else
            @. dY.c.ρq_tot = -vdivf2c(ᶠM * If(Yc.ρq_tot / ρ))
        end
    end
    # Pressure-gradient + buoyancy pair (must match the explicit form in
    # compute_tendency_fddg! so the HEVI split rhs = implicit + remaining). The
    # perturbation form shifts p and ρ by the frozen reference, so the Jacobian
    # (jacobians.jl) is unchanged.
    if m.prob.pgf == :conservative_pert
        pm = @. p_thermo - m.fields.ᶜp_ref
        @. dY.f.ρw = Bw(-(ᶠgradᵥ(pm) + If(ρ - m.fields.ᶜρ_ref) * ᶠgradᵥ(ᶜΦ)))
    else
        @. dY.f.ρw = Bw(-(ᶠgradᵥ(p_thermo) + If(ρ) * ᶠgradᵥ(ᶜΦ)))
    end
    return dY
end
