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

# Reference-subtracted curvilinear Roe interface (well-balanced dissipation).
# Bitwise-identical to `Operators.kennedy_gruber_roe_cartesian_curvilinear`
# EXCEPT the acoustic/contact wave AMPLITUDES use hydrostatic-DEVIATION jumps
#   Δp′ = [[p − p_ref]],  Δρ′ = [[ρ − ρ_ref]]
# instead of the raw [[p]], [[ρ]]. On terrain-following faces the raw jumps
# carry an O(1) HYDROSTATIC component (neighbours at different true altitude);
# damping it injects a spurious force from the rest state (change (b)). The
# central flux and the Roe linearisation point (ρ̂, ĉ, Ĥ, û) stay on the FULL
# state; velocity jumps are unchanged (the reference is at rest, u_ref = 0),
# so the Harten floor's shear/contact stabilisation is retained. Reduces to
# the raw curvilinear Roe wherever p_ref ≡ ρ_ref ≡ 0 (e.g. a flat non-terrain
# base with zero reference).  Requires state fields `p_ref`, `ρ_ref` in
# addition to those of `kennedy_gruber_roe_cartesian_curvilinear`.
function kennedy_gruber_roe_cartesian_curvilinear_wb(normal, (y⁻,), (y⁺,))
    F = Operators.kennedy_gruber_cartesian_flux_curvilinear(
        normal,
        normal,
        y⁻,
        y⁺,
    )
    γd = oftype(y⁻.ρ, Operators.γ_dry)
    # face normal in Cartesian components (includes W terrain cross-term)
    n1 = y⁻.Ec1' * normal
    n2 = y⁻.Ec2' * normal
    n3 = y⁻.Ec3' * normal
    # Roe-averaged state (full physical state = linearisation point)
    s⁻ = sqrt(y⁻.ρ)
    s⁺ = sqrt(y⁺.ρ)
    ρ̂ = s⁻ * s⁺
    a⁻ = s⁻ / (s⁻ + s⁺)
    a⁺ = 1 - a⁻
    û1 = a⁻ * y⁻.u1 + a⁺ * y⁺.u1
    û2 = a⁻ * y⁻.u2 + a⁺ * y⁺.u2
    û3 = a⁻ * y⁻.u3 + a⁺ * y⁺.u3
    ûuvw = a⁻ * y⁻.uvw + a⁺ * y⁺.uvw
    Ĥ = a⁻ * (y⁻.e + y⁻.p / y⁻.ρ) + a⁺ * (y⁺.e + y⁺.p / y⁺.ρ)
    ĉ = a⁻ * sqrt(γd * y⁻.p / y⁻.ρ) + a⁺ * sqrt(γd * y⁺.p / y⁺.ρ)
    ûn = ûuvw' * normal
    # hydrostatic-DEVIATION jumps (vanish on the reference base state)
    Δρ = (y⁺.ρ - y⁺.ρ_ref) - (y⁻.ρ - y⁻.ρ_ref)
    Δp = (y⁺.p - y⁺.p_ref) - (y⁻.p - y⁻.p_ref)
    Δu1 = y⁺.u1 - y⁻.u1
    Δu2 = y⁺.u2 - y⁻.u2
    Δu3 = y⁺.u3 - y⁻.u3
    Δuvw = y⁺.uvw - y⁻.uvw
    Δun = Δuvw' * normal
    α₊ = (Δp + ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₋ = (Δp - ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₀ = Δρ - Δp / ĉ^2
    s₊ = abs(ûn + ĉ)
    s₋ = abs(ûn - ĉ)
    s₀ = max(abs(ûn), ĉ / 20)
    Δut1 = Δu1 - Δun * n1
    Δut2 = Δu2 - Δun * n2
    Δut3 = Δu3 - Δun * n3
    B = Ĥ - ĉ^2 / (γd - 1)
    shear_E = ûuvw' * Δuvw - ûn * Δun
    Dρ = s₊ * α₊ + s₋ * α₋ + s₀ * α₀
    Dρu1 =
        s₊ * α₊ * (û1 + ĉ * n1) + s₋ * α₋ * (û1 - ĉ * n1) +
        s₀ * (α₀ * û1 + ρ̂ * Δut1)
    Dρu2 =
        s₊ * α₊ * (û2 + ĉ * n2) + s₋ * α₋ * (û2 - ĉ * n2) +
        s₀ * (α₀ * û2 + ρ̂ * Δut2)
    Dρu3 =
        s₊ * α₊ * (û3 + ĉ * n3) + s₋ * α₋ * (û3 - ĉ * n3) +
        s₀ * (α₀ * û3 + ρ̂ * Δut3)
    Dρe =
        s₊ * α₊ * (Ĥ + ĉ * ûn) + s₋ * α₋ * (Ĥ - ĉ * ûn) +
        s₀ * (α₀ * B + ρ̂ * shear_E)
    return (
        ρ = F.ρ - Dρ / 2,
        ρe = F.ρe - Dρe / 2,
        ρu1 = F.ρu1 - Dρu1 / 2,
        ρu2 = F.ρu2 - Dρu2 / 2,
        ρu3 = F.ρu3 - Dρu3 / 2,
    )
end
# Match the curvilinear metric style so `add_numerical_flux_internal!` passes a
# UVWVector face normal (terrain cross-term retained), as for the raw variant.
Operators._fd_metric_style(
    ::typeof(kennedy_gruber_roe_cartesian_curvilinear_wb),
) = Val{:curvilinear}()

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
    if m.prob.interface_flux in (:curvilinear_roe, :curvilinear_roe_wb)
        # Curvilinear metric path: full UVW metric vectors in both the volume
        # flux and the interface normal.  `uvw = UVWVector(uE, uN, w_c)` gives
        # the correct normal velocity through terrain-tilted faces; `Ec1/Ec2/Ec3
        # = UVWVector(eE_c, eN_c, eR_c)` supply the pressure flux projection.
        # `p_ref`/`ρ_ref` (hydrostatic reference) feed the well-balanced Roe
        # interface (`:curvilinear_roe_wb`); the volume flux ignores them.
        uvw = @. Geometry.UVWVector(uE, uN, w_c)
        Ec1 = @. Geometry.UVWVector(eE1, eN1, eR1)
        Ec2 = @. Geometry.UVWVector(eE2, eN2, eR2)
        Ec3 = @. Geometry.UVWVector(eE3, eN3, eR3)
        ᶜp_ref = m.fields.ᶜp_ref
        ᶜρ_ref = m.fields.ᶜρ_ref
        y = map(
            (
                ρi, ρei, ei, pi, uvwi, u1i, u2i, u3i, Ec1i, Ec2i, Ec3i, λi,
                Φi, p_refi, ρ_refi,
            ) -> (;
                ρ = ρi, ρe = ρei, e = ei, p = pi, uvw = uvwi,
                u1 = u1i, u2 = u2i, u3 = u3i,
                Ec1 = Ec1i, Ec2 = Ec2i, Ec3 = Ec3i, λ = λi, Φ = Φi,
                p_ref = p_refi, ρ_ref = ρ_refi,
            ),
            ρ, ρe, e, p, uvw, u1, u2, u3, Ec1, Ec2, Ec3, λ, ᶜΦ,
            ᶜp_ref, ᶜρ_ref,
        )
        volume_flux_curv =
            m.prob.wb_gravity ?
            Operators.kennedy_gruber_gravity_cartesian_flux_curvilinear :
            Operators.kennedy_gruber_cartesian_flux_curvilinear
        Operators.add_flux_differencing_divergence!(volume_flux_curv, dy_mw, y)
        Operators.add_numerical_flux_internal!(m.interface_flux_fn, dy_mw, y)
    else
        # wb_gravity: KG plus the well-balanced two-point geopotential
        # fluctuation (Waruszewski et al. 2022 Eq. 76) — the along-surface
        # ρ∇Φ term the Cartesian core otherwise omits over terrain. Identical
        # on flat grids ([[Φ]] ≡ 0 along levels); interfaces unchanged
        # (Φ single-valued at faces).
        y = map(
            (ρi, ρei, ei, pi, uvi, u1i, u2i, u3i, E1i, E2i, E3i, λi, Φi) -> (;
                ρ = ρi, ρe = ρei, e = ei, p = pi, uv = uvi,
                u1 = u1i, u2 = u2i, u3 = u3i,
                E1 = E1i, E2 = E2i, E3 = E3i, λ = λi, Φ = Φi,
            ),
            ρ, ρe, e, p, uv, u1, u2, u3, E1, E2, E3, λ, ᶜΦ,
        )
        volume_flux =
            m.prob.wb_gravity ? Operators.kennedy_gruber_gravity_cartesian_flux :
            Operators.kennedy_gruber_cartesian_flux
        Operators.add_flux_differencing_divergence!(volume_flux, dy_mw, y)
        Operators.add_numerical_flux_internal!(m.interface_flux_fn, dy_mw, y)
    end
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

    # --- ρw: pressure gradient + buoyancy (discretely balanced pair,
    #     implicit under HEVI), vertical advection, horizontal DG
    #     advection, sponge ---
    w = @. ρw_w / If(ρ)
    if vertical_transport
        @. dρw = Bw(
            -(ᶠgradᵥ(p) + If(ρ) * ᶠgradᵥ(ᶜΦ)) -
            C3(vvdivc2f(Ic(ρw_w ⊗ w)), lgeom_f),
        )
    else
        @. dρw = Bw(-C3(vvdivc2f(Ic(ρw_w ⊗ w)), lgeom_f))
    end
    ρw_sc = @. ρw_w.components.data.:1
    uvf = @. If(uv)
    λf =
        m.prob.interface_flux in (:roe, :curvilinear_roe, :curvilinear_roe_wb) ?
        (@. If(sqrt(uE^2 + uN^2))) : (@. If(λ))
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
    @. dY.f.ρw = Bw(-(ᶠgradᵥ(p_thermo) + If(ρ) * ᶠgradᵥ(ᶜΦ)))
    return dY
end
