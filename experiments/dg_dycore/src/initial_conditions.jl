#=
Initial conditions: Ullrich et al. (2014) dry baroclinic wave, shallow
atmosphere — Stage A1 carries the ClimaCore example's formulas verbatim
(sphere_dg_fd_model.jl lines 239–330) for parity runs; Stage A2 swaps the
analytic values for ClimaAtmos's Setups.shallow_atmos_barowave_values
(verified formula-identical) while KEEPING the discrete-hydrostatic ρ
correction below, which no ClimaAtmos component reproduces.

All formula constants live in `JWParams` (de-globalized from the example's
module consts).
=#

struct JWParams{FT}
    kb::Int
    T_e::FT
    T_p::FT
    T_0::FT
    Γ::FT
    A::FT
    B::FT
    C::FT
    b::FT
    H::FT
    z_t::FT
    λ_c::FT
    ϕ_c::FT
    d_0::FT
    V_p::FT
    R::FT
    Ω::FT
    grav::FT
    R_d::FT
    p_0::FT
end
Base.broadcastable(j::JWParams) = tuple(j)

function JWParams(c::DGConstants{FT}) where {FT}
    kb = 3
    T_e = FT(310)
    T_p = FT(240)
    T_0 = FT(0.5) * (T_e + T_p)
    Γ = FT(0.005)
    return JWParams{FT}(
        kb,
        T_e,
        T_p,
        T_0,
        Γ,
        1 / Γ,                                       # A
        (T_0 - T_p) / T_0 / T_p,                     # B
        FT(0.5) * (kb + 2) * (T_e - T_p) / T_e / T_p, # C
        FT(2),                                       # b
        c.R_d * T_0 / c.grav,                        # H
        FT(15e3),                                    # z_t
        FT(20),                                      # λ_c
        FT(40),                                      # ϕ_c
        c.R / 6,                                     # d_0
        FT(1),                                       # V_p
        c.R,
        c.Ω,
        c.grav,
        c.R_d,
        c.p_0,
    )
end

τ_z_1(j, z) = exp(j.Γ * z / j.T_0)
τ_z_2(j, z) = 1 - 2 * (z / j.b / j.H)^2
τ_z_3(j, z) = exp(-(z / j.b / j.H)^2)
τ_1(j, z) = 1 / j.T_0 * τ_z_1(j, z) + j.B * τ_z_2(j, z) * τ_z_3(j, z)
τ_2(j, z) = j.C * τ_z_2(j, z) * τ_z_3(j, z)
τ_int_1(j, z) = j.A * (τ_z_1(j, z) - 1) + j.B * z * τ_z_3(j, z)
τ_int_2(j, z) = j.C * z * τ_z_3(j, z)
F_z(j, z) = (1 - 3 * (z / j.z_t)^2 + 2 * (z / j.z_t)^3) * (z ≤ j.z_t)
I_T(j, ϕ) = cosd(ϕ)^j.kb - j.kb * (cosd(ϕ))^(j.kb + 2) / (j.kb + 2)
jw_temp(j, ϕ, z) = (τ_1(j, z) - τ_2(j, z) * I_T(j, ϕ))^(-1)
jw_pres(j, ϕ, z) =
    j.p_0 * exp(-j.grav / j.R_d * (τ_int_1(j, z) - τ_int_2(j, z) * I_T(j, ϕ)))
r_gc(j, λ, ϕ) =
    j.R * acos(sind(j.ϕ_c) * sind(ϕ) + cosd(j.ϕ_c) * cosd(ϕ) * cosd(λ - j.λ_c))
jw_U(j, ϕ, z) =
    j.grav * j.kb / j.R *
    τ_int_2(j, z) *
    jw_temp(j, ϕ, z) *
    (cosd(ϕ)^(j.kb - 1) - cosd(ϕ)^(j.kb + 1))
jw_u_base(j, ϕ, z) =
    -j.Ω * j.R * cosd(ϕ) +
    sqrt((j.Ω * j.R * cosd(ϕ))^2 + j.R * cosd(ϕ) * jw_U(j, ϕ, z))
c3_pert(j, λ, ϕ) = cos(oftype(j.T_0, π) * r_gc(j, λ, ϕ) / 2 / j.d_0)^3
s1_pert(j, λ, ϕ) = sin(oftype(j.T_0, π) * r_gc(j, λ, ϕ) / 2 / j.d_0)
pert_cond(j, λ, ϕ) =
    (0 < r_gc(j, λ, ϕ) < j.d_0) * (r_gc(j, λ, ϕ) != j.R * pi)
jw_δu(j, λ, ϕ, z) =
    -16 * j.V_p / 3 / sqrt(oftype(j.T_0, 3)) *
    F_z(j, z) *
    c3_pert(j, λ, ϕ) *
    s1_pert(j, λ, ϕ) *
    (-sind(j.ϕ_c) * cosd(ϕ) + cosd(j.ϕ_c) * sind(ϕ) * cosd(λ - j.λ_c)) /
    sin(r_gc(j, λ, ϕ) / j.R) * pert_cond(j, λ, ϕ)
jw_δv(j, λ, ϕ, z) =
    16 * j.V_p / 3 / sqrt(oftype(j.T_0, 3)) *
    F_z(j, z) *
    c3_pert(j, λ, ϕ) *
    s1_pert(j, λ, ϕ) *
    cosd(j.ϕ_c) *
    sind(λ - j.λ_c) / sin(r_gc(j, λ, ϕ) / j.R) * pert_cond(j, λ, ϕ)

"""
    discrete_hydrostatic_ρ!(ᶜρ, ᶜp, ᶜz, grav)

Column-wise discrete hydrostatic balance: keep the analytic p at cell
centers and correct ρ so the centered face balance
(p[v+1] − p[v])/Δz = −g (ρ[v] + ρ[v+1])/2 holds exactly. Without this the
O(Δz²) residual of ᶠgradᵥ(p) + If(ρ)·ᶠgradᵥ(Φ) projects onto gravity modes
(O(10 m/s) spurious w) and its latitude dependence seeds hemispherically
symmetric drift. Per-interface Δz from the actual center heights (supports
stretched grids).
"""
function discrete_hydrostatic_ρ!(ᶜρ, ᶜp, ᶜz, grav)
    ρ_par = parent(ᶜρ)
    p_par = parent(ᶜp)
    z_par = parent(ᶜz)
    for v in 1:(size(ρ_par, 1) - 1)
        @views @. ρ_par[v + 1, :, :, :, :] =
            -ρ_par[v, :, :, :, :] -
            2 * (p_par[v + 1, :, :, :, :] - p_par[v, :, :, :, :]) /
            (z_par[v + 1, :, :, :, :] - z_par[v, :, :, :, :]) / grav
    end
    return ᶜρ
end

"""
    initial_state_fddg(m::DGModel) -> FieldVector

Flux-form state, ClimaAtmos naming: `Y.c = (; ρ, ρe, ρu1, ρu2, ρu3)` at
centers (momentum in global Cartesian components), `Y.f = (; ρw)` with
ρw::Covariant3 at faces.
"""
# (T, p, u, v) values on centers, from either source. Both are the same
# Ullrich et al. shallow-atmosphere expressions; :setups reuses ClimaAtmos's
# implementation (constants from ClimaParams — identical under the parity
# TOML), :formulas keeps the examples' own copy (literal constants).
function jw_values(m::DGModel{FT}) where {FT}
    (; ccoords) = m.fields
    lat = ccoords.lat
    long = ccoords.long
    z = ccoords.z
    if m.prob.ic_source == :setups
        vals =
            CA.Setups.shallow_atmos_barowave_values.(
                z,
                lat,
                long,
                Ref(m.params),
                m.prob.perturb,
            )
        return (; T = vals.T, p = vals.p, uE = vals.u, uN = vals.v)
    else # :formulas
        c = m.c
        j = JWParams(c)
        T = @. jw_temp(j, lat, z)
        p = @. jw_pres(j, lat, z)
        uE = @. jw_u_base(j, lat, z)
        uN = @. 0 * z
        if m.prob.perturb
            @. uE += jw_δu(j, long, lat, z)
            @. uN += jw_δv(j, long, lat, z)
        end
        return (; T, p, uE, uN)
    end
end

function initial_state_fddg(m::DGModel{FT}) where {FT}
    c = m.c
    (; ccoords, fcoords, eE1, eE2, eE3, eN1, eN2, eN3) = m.fields
    z = ccoords.z

    (; T, p, uE, uN) = jw_values(m)
    ᶜp_ana = p
    ᶜρ = @. ᶜp_ana / c.R_d / T

    discrete_hydrostatic_ρ!(ᶜρ, ᶜp_ana, z, c.grav)

    # ρe such that the diagnosed pressure is exactly the analytic p
    ᶜK = @. (uE^2 + uN^2) / 2
    ᶜρe = @. c.cv_d * ᶜp_ana / c.R_d +
             ᶜρ * (ᶜK + c.grav * z - c.cv_d * c.T_tri)

    # u_c = uE (ê_E)_c + uN (ê_N)_c — Cartesian momentum components
    Yc = map(
        (ρi, ρei, uEi, uNi, e1E, e1N, e2E, e2N, e3E, e3N) -> (;
            ρ = ρi,
            ρe = ρei,
            ρu1 = ρi * (uEi * e1E + uNi * e1N),
            ρu2 = ρi * (uEi * e2E + uNi * e2N),
            ρu3 = ρi * (uEi * e3E + uNi * e3N),
        ),
        ᶜρ,
        ᶜρe,
        uE,
        uN,
        eE1, eN1, eE2, eN2, eE3, eN3,
    )
    # ρw in Covariant3 so the HEVI Jacobian reuses the MatrixFields
    # machinery (g³³ pairings) verbatim
    Yf = map(_ -> (; ρw = C3(FT(0))), fcoords)
    return Fields.FieldVector(c = Yc, f = Yf)
end

"""
    initial_state_vi(m::DGModel) -> FieldVector

Vector-invariant state: `Y.c = (; ρ, ρe, uₕ::Covariant12)`, `Y.f = (; w)`.
"""
function initial_state_vi(m::DGModel{FT}) where {FT}
    c = m.c
    (; ccoords, fcoords) = m.fields
    z = ccoords.z
    lgeom_c = Fields.local_geometry_field(m.spaces.hv_center_space)

    (; T, p, uE, uN) = jw_values(m)
    ᶜp_ana = p
    ᶜρ = @. ᶜp_ana / c.R_d / T

    discrete_hydrostatic_ρ!(ᶜρ, ᶜp_ana, z, c.grav)

    ᶜuₕ_local = @. Geometry.UVVector(uE, uN)
    ᶜuₕ = @. C12(ᶜuₕ_local, lgeom_c)
    ᶜK = @. norm_sqr(ᶜuₕ_local) / 2
    ᶜρe = @. c.cv_d * ᶜp_ana / c.R_d +
             ᶜρ * (ᶜK + c.grav * z - c.cv_d * c.T_tri)

    Yc = map(
        (ρi, ρei, uₕi) -> (; ρ = ρi, ρe = ρei, uₕ = uₕi),
        ᶜρ,
        ᶜρe,
        ᶜuₕ,
    )
    # Terrain-consistent surface w (CA's surface-velocity constraint,
    # applied statically): at the BOTTOM FACE ONLY, choose w₃ so
    # u³ = g³ʰuₕ + g³³w₃ = 0 — the kinematic no-normal-flow BC; Bw then
    # freezes this value. Without it the JW06 wind violates the BC over
    # the mountains and the zero-flux operator BCs kick the first cell at
    # O(u·slope·h_tot/Δz) (measured 0.19 s⁻¹ relative ρe tendency).
    # INTERIOR w stays 0: adapting w everywhere (u³ ≡ 0, flow following
    # all warped surfaces) is wrong — interior flow SHOULD cross the
    # coordinate surfaces, and a structured interior w at t = 0 breaks
    # the staggered ∇K/Lamb shear cancellation at O(u·w/Δz) (measured
    # 1.7 m/s² dw on the Hughes2023 flanks — instant crash).
    # (Identically w ≡ 0 on flat grids either way.)
    lgeom_f = Fields.local_geometry_field(m.spaces.hv_face_space)
    ᶠu³ₕ_sc = @. CT3(C123(m.ops.If(ᶜuₕ))).components.data.:1
    ᶠg³³_sc = @. CT3(C3(FT(1)), lgeom_f).components.data.:1
    w_sc = @. -(ᶠu³ₕ_sc) / ᶠg³³_sc
    parent(w_sc)[2:end, :, :, :, :] .= FT(0)   # bottom face only (v-dim first)
    Yf = map(ws -> (; w = C3(ws)), w_sc)
    return Fields.FieldVector(c = Yc, f = Yf)
end

initial_state(m::DGModel{FT, <:BaroclinicWaveFDDG}) where {FT} =
    initial_state_fddg(m)
initial_state(m::DGModel{FT, <:BaroclinicWaveDG}) where {FT} =
    initial_state_vi(m)
