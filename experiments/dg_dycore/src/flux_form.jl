#=
Flux-form FDDG tendencies — port of baroclinic_wave_fddg_fluxform.jl
(compute_tendency_fddg!, implicit_tendency_fddg!) with every module-level
const replaced by fields of the DGModel `m` (the integrator parameter).

Full system in flux form: (ρ, ρe, ρu⃗) with momentum in GLOBAL CARTESIAN
components, all horizontal terms discretized with Kennedy–Gruber
flux-differencing volume fluxes + KG-Rusanov/KG-Roe interfaces (Souza et
al. 2023). The constant Cartesian basis makes component-wise flux
differencing kinetic-energy-preserving with no curvature source terms.

Shallow atmosphere: the Cartesian center-momentum tendency is projected
tangentially against r̂; ρw (faces, Covariant3) carries radial momentum.

State (ClimaAtmos naming): Y.c = (; ρ, ρe, ρu1, ρu2, ρu3), Y.f = (; ρw).

NOTE (Stage A1): per-call temporary fields follow the example verbatim;
preallocation into the model is a later perf pass (measured against the
reference driver's steps/sec).
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
    (; Ic, If, vdivf2c, vvdivc2f, VanLeer, ᶠgradᵥ, Bw, hwdiv, hgrad) = m.ops
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
    p = @. pres_ρe(c, ρe, K, ᶜΦ, ρ)
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
    # Φ is single-valued at faces).
    y = map(
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

    # --- Vertical FD (plane flux-form pattern; implicit under HEVI) ---
    if vertical_transport
        @. dYc.ρ -= vdivf2c(ρw_w)
        @. dYc.ρe -= vdivf2c(VanLeer(ρw_w, h_tot, Δt))
    else
        # mass flux is fully implicit (linear); energy gets the explicit
        # (VanLeer − central) correction so the HEVI total is Lin-VanLeer
        @. dYc.ρe -=
            vdivf2c(VanLeer(ρw_w, h_tot, Δt)) - vdivf2c(ρw_w * If(h_tot))
    end
    @. dYc.ρu1 -= vdivf2c(VanLeer(ρw_w, u1, Δt))
    @. dYc.ρu2 -= vdivf2c(VanLeer(ρw_w, u2, Δt))
    @. dYc.ρu3 -= vdivf2c(VanLeer(ρw_w, u3, Δt))

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
                C3(vvdivc2f(Ic(ρw_w ⊗ w)), lgeom_f),
            )
        else
            @. dρw = Bw(
                -(ᶠgradᵥ(p) + If(ρ) * ᶠgradᵥ(ᶜΦ)) -
                C3(vvdivc2f(Ic(ρw_w ⊗ w)), lgeom_f),
            )
        end
    else
        @. dρw = Bw(-C3(vvdivc2f(Ic(ρw_w ⊗ w)), lgeom_f))
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
    (; Ic, If, vdivf2c, ᶠgradᵥ, Bw) = m.ops
    (; ᶜΦ, eE1, eE2, eE3, eN1, eN2, eN3) = m.fields

    Yc = Y.c
    ρ = Yc.ρ
    ρe = Yc.ρe
    ρw_w = @. Geometry.WVector(Y.f.ρw)

    uE = @. (Yc.ρu1 * eE1 + Yc.ρu2 * eE2 + Yc.ρu3 * eE3) / ρ
    uN = @. (Yc.ρu1 * eN1 + Yc.ρu2 * eN2 + Yc.ρu3 * eN3) / ρ
    w_c = @. Ic(ρw_w).components.data.:1 / ρ
    K = @. (uE^2 + uN^2 + w_c^2) / 2
    p_thermo = @. pres_ρe(c, ρe, K, ᶜΦ, ρ)
    h_tot = @. (ρe + p_thermo) / ρ

    @. dY.c.ρ = -vdivf2c(ρw_w)
    @. dY.c.ρe = -vdivf2c(ρw_w * If(h_tot))
    dY.c.ρu1 .= FT(0)
    dY.c.ρu2 .= FT(0)
    dY.c.ρu3 .= FT(0)
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
