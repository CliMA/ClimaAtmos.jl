#=
Vector-invariant DG-FD tendencies — port of sphere_dg_fd_model.jl's
compute_tendency!/implicit_tendency! (state: ρ, ρe flux-form scalars;
uₕ::Covariant12 centers; w::Covariant3 faces), de-globalized onto DGModel.

State (ClimaAtmos naming): Y.c = (; ρ, ρe, uₕ), Y.f = (; w).

Horizontal treatment: (ρ, ρe) via flux differencing + interface fluxes;
non-conservative gradients/curls via element-local strong operators +
central face lifting; λ-scaled jump penalties on velocity components.
Momentum: vector-invariant (ω×u + ∇K) or the Route-B mass-flux
fluctuation form (momentum_adv = :fluctuation — validated at helem = 4
ONLY; violently unstable at helem = 16).

Face sets (docs/vi_kep_face_terms.md):
- :kg (legacy): KG {ρ}{ũ} fluxes + Rusanov on (ρ, ρe) + plain λ jump
  penalties. NOT KE-compatible with the vector-invariant pairing — needs
  κ₄ = cap/10 and the tendency cutoff filter (filter_Nc = npoly).
- :kep: {ρũ} mass flux (volume + central interface, no ρ penalty),
  matching ρe flux, ρ-weighted velocity penalties. The horizontal
  advective KE production closes to roundoff (metric-transparent: exact
  over topography too), so κ₄ = 0 / filter_Nc = 0 are admissible.

Topography (hypsography-warped grids): K uses the full-metric CA form
(cross terms via C123/CT123) and the vertical transport uses the full
contravariant ᶠu³ = CT3(uₕ) + CT3(w) with CA's ρJ-weighted face
interpolation. The initial w is terrain-adapted (u³ = 0 everywhere; see
initial_conditions.jl) — starting from w = 0 violates the kinematic BC
over the mountains and kicks the first cell at O(u·slope·h_tot/Δz)
(measured 0.19 s⁻¹ relative ρe tendency → crash within 10 steps). Bw
freezes the surface-face w at its terrain-consistent initial value; the
per-stage CA surface-velocity constraint is not applied here (O(slope·Δuₕ)
drift over long runs).
=#

function compute_tendency_vi!(
    dY,
    Y,
    m::DGModel{FT},
    t,
    vertical_transport,
) where {FT}
    c = m.c
    (;
        Ic,
        If,
        wIf,
        vdivf2c,
        vdivf2c3,
        VanLeer,
        ᶠgradᵥ,
        ᶠcurlᵥ,
        Bw,
        hwdiv,
        hgrad,
        hcurl,
    ) = m.ops
    (; ᶜΦ, ᶜf_cor, ᶠβ_sponge, ᶜβ_sponge) = m.fields
    (; eE1, eE2, eE3, eN1, eN2, eN3) = m.fields
    Δt = m.Δt
    κ₄ = m.κ₄
    kep = m.prob.face_set == :kep

    ρ = Y.c.ρ
    ρe = Y.c.ρe
    uₕ = Y.c.uₕ
    w = Y.f.w
    dYc = dY.c
    duₕ = dY.c.uₕ
    dw = dY.f.w

    lgeom_c = Fields.local_geometry_field(m.spaces.hv_center_space)
    lgeom_f = Fields.local_geometry_field(m.spaces.hv_face_space)

    # --- Diagnostics ---
    uv = @. Geometry.UVVector(uₕ)          # geographic components
    u_sc = uv.components.data.:1
    v_sc = uv.components.data.:2
    # Full-metric kinetic energy (CA compute_kinetic form): the C123/CT123
    # dots carry the g¹³/g²³ cross terms on terrain-warped grids (zero on
    # flat grids, where this reduces to (|uv|² + |Ic(w)|²)/2).
    K = @. (
        dot(C123(uₕ), CT123(uₕ)) +
        Ic(dot(C123(w), CT123(w))) +
        2 * dot(CT123(uₕ), Ic(C123(w)))
    ) / 2
    p = @. pres_ρe(c, ρe, K, ᶜΦ, ρ)
    h_tot = @. (ρe + p) / ρ
    c_snd = @. sqrt(c.γ * p / ρ)
    λ_c = @. sqrt(norm_sqr(uv)) + c_snd
    λ_f = @. If(λ_c)
    ρ_f = @. If(ρ)
    w_sc = @. Geometry.WVector(w).components.data.:1

    # --- (ρ, ρe): horizontal flux-form DG (flux differencing) ---
    y = map(
        (ρi, ρei, pi, λi, uvi, ei) ->
            (; ρ = ρi, ρe = ρei, p = pi, λ = λi, uv = uvi, e = ei),
        ρ,
        ρe,
        p,
        λ_c,
        uv,
        ρe ./ ρ,
    )
    dy_mw = map(_ -> (ρ = FT(0), ρe = FT(0)), ρ)
    Operators.add_flux_differencing_divergence!(
        kep ? Operators.vi_kep_scalars_flux :
        Operators.kennedy_gruber_scalars_flux,
        dy_mw,
        y,
    )
    Operators.add_numerical_flux_internal!(
        kep ? Operators.vi_kep_interface_scalars :
        Operators.kennedy_gruber_rusanov_scalars,
        dy_mw,
        y,
    )
    @. dYc.ρ = dy_mw.ρ / lgeom_c.WJ
    @. dYc.ρe = dy_mw.ρe / lgeom_c.WJ

    # --- (ρ, ρe): vertical FD (w part implicit under HEVI) ---
    # terrain_u3 = :full (default) adds the CT3(uₕ) transport through
    # tilted ξ³ surfaces (zero on flat grids; ρJ-weighted CA ᶠwinterp
    # machinery). With the terrain-consistent surface w its divergence is
    # ~85-90% compensated by the horizontal along-surface flux (measured:
    # −27.5 vs +31.0 at the Hughes2023 peak, residual ≈ benign 3e-5 s⁻¹
    # relative); the remaining cross-discretization free-stream defect is
    # the conservative-metric follow-up of docs/vi_kep_face_terms.md §5.
    # :wonly omits the term (FDDG-style O(slope) transport approximation).
    w_vec = @. Geometry.WVector(w)
    ᶜJ = lgeom_c.J
    ᶠu³ = if m.prob.terrain_u3 == :full
        @. CT3(C123(w)) + wIf(ρ * ᶜJ, CT3(uₕ))
    else
        @. CT3(C123(w))
    end
    if vertical_transport
        @. dYc.ρ -= vdivf2c3(VanLeer(ᶠu³, ρ, Δt))
        @. dYc.ρe -= vdivf2c3(ρ_f * VanLeer(ᶠu³, h_tot, Δt))
    else
        # HEVI explicit part: full transport minus the central w-only
        # fluxes that implicit_tendency_vi! integrates implicitly
        @. dYc.ρ -= vdivf2c3(VanLeer(ᶠu³, ρ, Δt)) - vdivf2c(ρ_f * w_vec)
        @. dYc.ρe -=
            vdivf2c3(ρ_f * VanLeer(ᶠu³, h_tot, Δt)) -
            vdivf2c(ρ_f * w_vec * If(h_tot))
    end

    # --- Vorticities (element-local strong curl + central face lifting) ---
    ᶠω¹² = @. hcurl(w)
    # project (not transform): on warped grids the tangential UVVector
    # correction has an O(slope) third contravariant component (it belongs
    # to the ω³ correction), which the exact transform refuses to drop
    ᶠω¹² .+=
        Geometry.project.(
            Ref(Geometry.Contravariant12Axis()),
            Operators.lifting_correction(
                Operators.central_curl12_lift,
                Geometry.UVVector{FT},
                w_sc,
            ),
        )
    @. ᶠω¹² += ᶠcurlᵥ(uₕ)

    ᶠu¹² = @. CT12(If(uₕ))
    # (ᶠu³ defined above with the full CT3(uₕ) + CT3(w) form)

    # --- Horizontal momentum ---
    if m.prob.momentum_adv == :fluctuation
        # Route B: KE-compatible mass-flux fluctuation form on Cartesian
        # velocity components; only vertical-KE ∇ remains here, and only
        # PLANETARY vorticity enters the cross product.
        w_c = @. Ic(Geometry.WVector(w))
        Kᵥ = @. norm_sqr(w_c) / 2
        @. duₕ = -(Ic(ᶠω¹² × ᶠu³) + ᶜf_cor × CT12(uₕ))
        @. duₕ -= hgrad(p) / ρ + hgrad(Kᵥ + ᶜΦ)
        K_lift = Kᵥ

        u1 = @. u_sc * eE1 + v_sc * eN1
        u2 = @. u_sc * eE2 + v_sc * eN2
        u3 = @. u_sc * eE3 + v_sc * eN3
        y_adv = map(
            (ρi, uvi, a, b, cc) ->
                (; ρ = ρi, uv = uvi, u1 = a, u2 = b, u3 = cc),
            ρ,
            uv,
            u1,
            u2,
            u3,
        )
        adv_mw = map(_ -> (u1 = FT(0), u2 = FT(0), u3 = FT(0)), ρ)
        Operators.add_flux_differencing_divergence!(
            Operators.kg_massflux_fluctuation,
            adv_mw,
            y_adv,
        )
        sat1 = Operators.lifting_correction(
            Operators.advective_fluctuation_lift,
            FT,
            u1,
            ρ,
            uv,
        )
        sat2 = Operators.lifting_correction(
            Operators.advective_fluctuation_lift,
            FT,
            u2,
            ρ,
            uv,
        )
        sat3 = Operators.lifting_correction(
            Operators.advective_fluctuation_lift,
            FT,
            u3,
            ρ,
            uv,
        )
        du1 = @. (adv_mw.u1 / lgeom_c.WJ + sat1) / ρ
        du2 = @. (adv_mw.u2 / lgeom_c.WJ + sat2) / ρ
        du3 = @. (adv_mw.u3 / lgeom_c.WJ + sat3) / ρ
        @. duₕ += Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(
                du1 * eE1 + du2 * eE2 + du3 * eE3,
                du1 * eN1 + du2 * eN2 + du3 * eN3,
            ),
        )
    else
        # vector-invariant: relative vorticity + full-K gradient
        ω³_sc = @. Geometry.WVector(hcurl(uₕ)).components.data.:1
        ω³_sc .+= Operators.lifting_correction(
            Operators.central_curl3_lift,
            FT,
            u_sc,
            v_sc,
        )
        ω³ = @. CT3(Geometry.WVector(ω³_sc))
        @. duₕ = -(Ic(ᶠω¹² × ᶠu³) + (ᶜf_cor + ω³) × CT12(uₕ))
        @. duₕ -= hgrad(p) / ρ + hgrad(K + ᶜΦ)
        K_lift = K
    end
    # DG lifting corrections for the strong gradients (Φ is continuous)
    lift_p = Operators.lifting_correction(
        Operators.central_gradient_lift,
        Geometry.UVVector{FT},
        p,
    )
    lift_K = Operators.lifting_correction(
        Operators.central_gradient_lift,
        Geometry.UVVector{FT},
        K_lift,
    )
    @. duₕ -= Geometry.transform(
        Geometry.Covariant12Axis(),
        lift_p / ρ + lift_K,
    )
    # λ-scaled jump penalties on the geographic velocity components.
    # :kep uses the ρ-weighted form — contracted with ρu the face KE
    # tendency is exactly −max(λ)/2·{ρ}|[[u]]|² ≤ 0 (see the derivation);
    # the plain form is sign-definite only to O([[ρ]]).
    pen_u, pen_v = if kep
        Operators.lifting_correction(
            Operators.rho_weighted_jump_penalty_lift,
            FT,
            u_sc,
            ρ,
            λ_c,
        ),
        Operators.lifting_correction(
            Operators.rho_weighted_jump_penalty_lift,
            FT,
            v_sc,
            ρ,
            λ_c,
        )
    else
        Operators.lifting_correction(
            Operators.jump_penalty_lift,
            FT,
            u_sc,
            λ_c,
        ),
        Operators.lifting_correction(
            Operators.jump_penalty_lift,
            FT,
            v_sc,
            λ_c,
        )
    end
    @. duₕ += Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.UVVector(pen_u, pen_v),
    )

    # --- Vertical momentum (acoustic terms implicit under HEVI) ---
    if vertical_transport
        @. dw = -(ᶠgradᵥ(p) / If(ρ) + ᶠgradᵥ(K + ᶜΦ))
        @. dw -= ᶠω¹² × ᶠu¹²
    else
        @. dw = -(ᶠω¹² × ᶠu¹²)
    end
    # Penalize the PROGNOSTIC covariant₃ dof directly (identical to the
    # physical-w penalty on flat grids). Converting a physical (WVector)
    # penalty into a covariant increment via C3(WVector(·), lg) is a
    # flat-grid idiom: on warped grids the covariant representation of a
    # vertical vector has O(∂z/∂ξʰ) ~ 10³ cross components, and the
    # conversion amplifies the penalty by that factor (measured 0.0012 →
    # 1.60 m/s² on the Hughes2023 flanks — instant crash).
    w_cov_sc = @. w.components.data.:1
    pen_w = Operators.lifting_correction(
        Operators.jump_penalty_lift,
        FT,
        w_cov_sc,
        λ_f,
    )
    @. dw += C3(pen_w)
    @. dw -= ᶠβ_sponge * w
    @. dw = Bw(dw)
    if m.prob.sponge_uh
        @. duₕ -= ᶜβ_sponge * uₕ
    end

    # --- κ₄ hyperdiffusion (h_tot and geographic (u, v); no ρ/w) ---
    # Over terrain, diffuse PERTURBATIONS from the steady unperturbed
    # base state (m.fields ᶜh_ref/ᶜu_ref/ᶜv_ref, evaluated at the warped
    # node heights): full fields carry an O(Δz_warp) terrain signature
    # along the coordinate surfaces (g·Δz + cp·Γ·Δz for h_tot, shear·Δz
    # for velocity) that the along-surface biharmonic converts into
    # spurious dipoles at the mountains (measured O(100 J/m³/s) at
    # κ₄ = 1e16 on Hughes2023 for h_tot; the velocity version crashed
    # even faster). Flat grids keep the validated full-field form.
    if κ₄ != 0
        τ_κ₄ = Operators.ldg_penalty_parameter(κ₄, m.spaces.hv_center_space)
        terrain = m.prob.topography != :none
        (; ᶜh_ref, ᶜu_ref, ᶜv_ref) = m.fields
        χe = similar(h_tot)
        χu = similar(u_sc)
        χv = similar(v_sc)
        if terrain
            @. χe = hwdiv(hgrad(h_tot - ᶜh_ref))
            @. χu = hwdiv(hgrad(u_sc - ᶜu_ref))
            @. χv = hwdiv(hgrad(v_sc - ᶜv_ref))
        else
            @. χe = hwdiv(hgrad(h_tot))
            @. χu = hwdiv(hgrad(u_sc))
            @. χv = hwdiv(hgrad(v_sc))
        end
        de4 = Operators.ldg_laplacian_tendency(χe, ρ, κ₄, τ_κ₄)
        du4 = Operators.ldg_laplacian_tendency(χu, nothing, κ₄, τ_κ₄)
        dv4 = Operators.ldg_laplacian_tendency(χv, nothing, κ₄, τ_κ₄)
        @. dYc.ρe -= de4
        @. duₕ -= Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(du4, dv4),
        )
    end

    # --- Held–Suarez forcing (reused ClimaAtmos implementation) ---
    if m.prob.held_suarez
        params = m.params
        T_sfc = m.fields.T_sfc
        dYc.ρe .+= CA.held_suarez_forcing_tendency_ρe_tot(
            ρ,
            uₕ,
            p,
            params,
            T_sfc,
            CA.DryModel(),
            Val(:held_suarez),
        )
        # this state carries uₕ::Covariant12, so CA's drag applies directly
        duₕ .+= CA.held_suarez_forcing_tendency_uₕ(
            uₕ,
            p,
            params,
            T_sfc,
            CA.DryModel(),
            Val(:held_suarez),
        )
    end

    # --- Element-local cutoff filter on the tendencies (this formulation
    # NEEDS it; the FDDG core must never use it — KEP) ---
    if m.filter_Nc > 0
        M = Quadratures.cutoff_filter_matrix(
            FT,
            Spaces.quadrature_style(m.spaces.hv_center_space),
            m.filter_Nc,
        )
        for f in (dYc.ρ, dYc.ρe, duₕ, dw)
            data = Fields.field_values(f)
            Operators.tensor_product!(data, data, M)
        end
        @. dw = Bw(dw)
    end
    return dY
end

rhs_vi!(dY, Y, m, t) = compute_tendency_vi!(dY, Y, m, t, true)
remaining_tendency_vi!(dY, Y, m, t) = compute_tendency_vi!(dY, Y, m, t, false)

"""
    horizontal_ke_budget(Y, m::DGModel) -> (; P_adv, P_pen, KE)

Discrete KE ledger of the horizontal terms (docs/vi_kep_face_terms.md §3-4),
evaluated on the state `Y` with the model's `face_set`:

  - `P_adv`: production of ⟨ρ(K+Φ)⟩ by the exact-ledger set — the ρ
    flux-differencing volume + interface terms contracted with (K+Φ), plus
    the momentum −∇(K+Φ) (element-local strong gradient + central K lifting)
    and the (f + ω³) Lamb term, contracted with ρu. With `face_set = :kep`
    this closes to machine roundoff (relative to `KE·λ/Δx` scale) on flat
    AND terrain-warped grids; with `:kg` it is O(jump²)-finite.
  - `P_pen`: velocity jump-penalty production — must be ≤ 0 (exactly for
    the ρ-weighted `:kep` penalties, to O([[ρ]]) for `:kg`).
  - `KE`: ⟨ρK⟩ with the full-metric K.

The vertical/staggered cross terms (`Ic(ᶠω¹²×ᶠu³)` etc.) are deliberately
excluded — they close only to interpolation truncation (§7).
"""
function horizontal_ke_budget(Y, m::DGModel{FT}) where {FT}
    c = m.c
    (; Ic, If, hgrad, hcurl) = m.ops
    (; ᶜΦ, ᶜf_cor) = m.fields
    kep = m.prob.face_set == :kep

    ρ = Y.c.ρ
    ρe = Y.c.ρe
    uₕ = Y.c.uₕ
    w = Y.f.w
    lgeom_c = Fields.local_geometry_field(m.spaces.hv_center_space)

    uv = @. Geometry.UVVector(uₕ)
    u_sc = uv.components.data.:1
    v_sc = uv.components.data.:2
    K = @. (
        dot(C123(uₕ), CT123(uₕ)) +
        Ic(dot(C123(w), CT123(w))) +
        2 * dot(CT123(uₕ), Ic(C123(w)))
    ) / 2
    p = @. pres_ρe(c, ρe, K, ᶜΦ, ρ)
    c_snd = @. sqrt(c.γ * p / ρ)
    λ_c = @. sqrt(norm_sqr(uv)) + c_snd

    # ρ tendency: horizontal flux differencing + interface flux
    y = map(
        (ρi, ρei, pi, λi, uvi, ei) ->
            (; ρ = ρi, ρe = ρei, p = pi, λ = λi, uv = uvi, e = ei),
        ρ,
        ρe,
        p,
        λ_c,
        uv,
        ρe ./ ρ,
    )
    dy_mw = map(_ -> (ρ = FT(0), ρe = FT(0)), ρ)
    Operators.add_flux_differencing_divergence!(
        kep ? Operators.vi_kep_scalars_flux :
        Operators.kennedy_gruber_scalars_flux,
        dy_mw,
        y,
    )
    Operators.add_numerical_flux_internal!(
        kep ? Operators.vi_kep_interface_scalars :
        Operators.kennedy_gruber_rusanov_scalars,
        dy_mw,
        y,
    )
    dρ_h = @. dy_mw.ρ / lgeom_c.WJ

    # momentum: Lamb (ω³) + strong ∇(K+Φ) + central K lifting
    ω³_sc = @. Geometry.WVector(hcurl(uₕ)).components.data.:1
    ω³_sc .+= Operators.lifting_correction(
        Operators.central_curl3_lift,
        FT,
        u_sc,
        v_sc,
    )
    ω³ = @. CT3(Geometry.WVector(ω³_sc))
    lift_K = Operators.lifting_correction(
        Operators.central_gradient_lift,
        Geometry.UVVector{FT},
        K,
    )
    duₕ_adv = similar(uₕ)
    @. duₕ_adv = -(ᶜf_cor + ω³) × CT12(uₕ) - hgrad(K + ᶜΦ)
    @. duₕ_adv -= Geometry.transform(Geometry.Covariant12Axis(), lift_K)

    P_adv =
        sum(@. (K + ᶜΦ) * dρ_h) +
        sum(@. ρ * dot(C123(duₕ_adv), CT123(uₕ)))

    # velocity jump penalties
    pen_u, pen_v = if kep
        Operators.lifting_correction(
            Operators.rho_weighted_jump_penalty_lift,
            FT,
            u_sc,
            ρ,
            λ_c,
        ),
        Operators.lifting_correction(
            Operators.rho_weighted_jump_penalty_lift,
            FT,
            v_sc,
            ρ,
            λ_c,
        )
    else
        Operators.lifting_correction(
            Operators.jump_penalty_lift,
            FT,
            u_sc,
            λ_c,
        ),
        Operators.lifting_correction(
            Operators.jump_penalty_lift,
            FT,
            v_sc,
            λ_c,
        )
    end
    P_pen = sum(@. ρ * (u_sc * pen_u + v_sc * pen_v))

    KE = sum(@. ρ * K)
    return (; P_adv, P_pen, KE)
end

# HEVI implicit part: vertical acoustics (column-local; central implicit
# vertical energy flux, VanLeer correction explicit).
function implicit_tendency_vi!(dY, Y, m::DGModel{FT}, t) where {FT}
    c = m.c
    (; Ic, If, vdivf2c, ᶠgradᵥ) = m.ops
    (; ᶜΦ) = m.fields
    ρ = Y.c.ρ
    ρe = Y.c.ρe
    uₕ = Y.c.uₕ
    w = Y.f.w

    K = @. (
        dot(C123(uₕ), CT123(uₕ)) +
        Ic(dot(C123(w), CT123(w))) +
        2 * dot(CT123(uₕ), Ic(C123(w)))
    ) / 2
    p_thermo = @. pres_ρe(c, ρe, K, ᶜΦ, ρ)
    h_tot = @. (ρe + p_thermo) / ρ

    w_vec = @. Geometry.WVector(w)
    @. dY.c.ρ = -vdivf2c(If(ρ) * w_vec)
    @. dY.c.ρe = -vdivf2c(If(ρ) * w_vec * If(h_tot))
    dY.c.uₕ .= (zero(eltype(dY.c.uₕ)),)
    @. dY.f.w = -(ᶠgradᵥ(p_thermo) / If(ρ) + ᶠgradᵥ(K + ᶜΦ))
    return dY
end
