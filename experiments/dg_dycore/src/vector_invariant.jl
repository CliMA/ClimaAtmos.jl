#=
Vector-invariant DG-FD tendencies — port of sphere_dg_fd_model.jl's
compute_tendency!/implicit_tendency! (state: ρ, ρe flux-form scalars;
uₕ::Covariant12 centers; w::Covariant3 faces), de-globalized onto DGModel.

State (ClimaAtmos naming): Y.c = (; ρ, ρe, uₕ), Y.f = (; w).

Horizontal treatment: (ρ, ρe) via KG flux differencing + KG-Rusanov
interfaces; non-conservative gradients/curls via element-local strong
operators + central face lifting; λ-scaled jump penalties on velocity
components. Momentum: vector-invariant (ω×u + ∇K) or the Route-B
mass-flux fluctuation form (momentum_adv = :fluctuation — validated at
helem = 4 ONLY; violently unstable at helem = 16).

Unlike the FDDG core this formulation NEEDS its stabilization: κ₄ = cap/10
and the tendency cutoff filter (filter_Nc = npoly) by default.
=#

function compute_tendency_vi!(
    dY,
    Y,
    m::DGModel{FT},
    t,
    vertical_transport,
) where {FT}
    c = m.c
    (; Ic, If, vdivf2c, VanLeer, ᶠgradᵥ, ᶠcurlᵥ, Bw, hwdiv, hgrad, hcurl) =
        m.ops
    (; ᶜΦ, ᶜf_cor, ᶠβ_sponge, ᶜβ_sponge) = m.fields
    (; eE1, eE2, eE3, eN1, eN2, eN3) = m.fields
    Δt = m.Δt
    κ₄ = m.κ₄

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
    w_c = @. Ic(Geometry.WVector(w))
    uv = @. Geometry.UVVector(uₕ)          # geographic components
    u_sc = uv.components.data.:1
    v_sc = uv.components.data.:2
    K = @. (norm_sqr(uv) + norm_sqr(w_c)) / 2
    p = @. pres_ρe(c, ρe, K, ᶜΦ, ρ)
    h_tot = @. (ρe + p) / ρ
    c_snd = @. sqrt(c.γ * p / ρ)
    λ_c = @. sqrt(norm_sqr(uv)) + c_snd
    λ_f = @. If(λ_c)
    ρ_f = @. If(ρ)
    w_sc = @. Geometry.WVector(w).components.data.:1

    # --- (ρ, ρe): horizontal flux-form DG (KG flux differencing) ---
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
        Operators.kennedy_gruber_scalars_flux,
        dy_mw,
        y,
    )
    Operators.add_numerical_flux_internal!(
        Operators.kennedy_gruber_rusanov_scalars,
        dy_mw,
        y,
    )
    @. dYc.ρ = dy_mw.ρ / lgeom_c.WJ
    @. dYc.ρe = dy_mw.ρe / lgeom_c.WJ

    # --- (ρ, ρe): vertical FD (implicit under HEVI) ---
    w_vec = @. Geometry.WVector(w)
    if vertical_transport
        @. dYc.ρ -= vdivf2c(VanLeer(w_vec, ρ, Δt))
        @. dYc.ρe -= vdivf2c(ρ_f * VanLeer(w_vec, h_tot, Δt))
    else
        # HEVI explicit part: (VanLeer − central) upwind corrections
        @. dYc.ρ -= vdivf2c(VanLeer(w_vec, ρ, Δt)) - vdivf2c(ρ_f * w_vec)
        @. dYc.ρe -=
            vdivf2c(ρ_f * VanLeer(w_vec, h_tot, Δt)) -
            vdivf2c(ρ_f * w_vec * If(h_tot))
    end

    # --- Vorticities (element-local strong curl + central face lifting) ---
    ᶠω¹² = @. hcurl(w)
    ᶠω¹² .+= Geometry.transform.(
        Ref(Geometry.Contravariant12Axis()),
        Operators.lifting_correction(
            Operators.central_curl12_lift,
            Geometry.UVVector{FT},
            w_sc,
        ),
    )
    @. ᶠω¹² += ᶠcurlᵥ(uₕ)

    ᶠu¹² = @. CT12(If(uₕ))
    ᶠu³ = @. CT3(w)

    # --- Horizontal momentum ---
    if m.prob.momentum_adv == :fluctuation
        # Route B: KE-compatible mass-flux fluctuation form on Cartesian
        # velocity components; only vertical-KE ∇ remains here, and only
        # PLANETARY vorticity enters the cross product.
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
    # λ-scaled jump penalties on the geographic velocity components
    pen_u =
        Operators.lifting_correction(Operators.jump_penalty_lift, FT, u_sc, λ_c)
    pen_v =
        Operators.lifting_correction(Operators.jump_penalty_lift, FT, v_sc, λ_c)
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
    pen_w =
        Operators.lifting_correction(Operators.jump_penalty_lift, FT, w_sc, λ_f)
    @. dw += C3(Geometry.WVector(pen_w), lgeom_f)
    @. dw -= ᶠβ_sponge * w
    @. dw = Bw(dw)
    if m.prob.sponge_uh
        @. duₕ -= ᶜβ_sponge * uₕ
    end

    # --- κ₄ hyperdiffusion (h_tot and geographic (u, v); no ρ/w) ---
    if κ₄ != 0
        τ_κ₄ = Operators.ldg_penalty_parameter(κ₄, m.spaces.hv_center_space)
        χe = similar(h_tot)
        @. χe = hwdiv(hgrad(h_tot))
        χu = similar(u_sc)
        @. χu = hwdiv(hgrad(u_sc))
        χv = similar(v_sc)
        @. χv = hwdiv(hgrad(v_sc))
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

    uv = @. Geometry.UVVector(uₕ)
    w_c = @. Ic(Geometry.WVector(w))
    K = @. (norm_sqr(uv) + norm_sqr(w_c)) / 2
    p_thermo = @. pres_ρe(c, ρe, K, ᶜΦ, ρ)
    h_tot = @. (ρe + p_thermo) / ρ

    w_vec = @. Geometry.WVector(w)
    @. dY.c.ρ = -vdivf2c(If(ρ) * w_vec)
    @. dY.c.ρe = -vdivf2c(If(ρ) * w_vec * If(h_tot))
    dY.c.uₕ .= (zero(eltype(dY.c.uₕ)),)
    @. dY.f.w = -(ᶠgradᵥ(p_thermo) / If(ρ) + ᶠgradᵥ(K + ᶜΦ))
    return dY
end
