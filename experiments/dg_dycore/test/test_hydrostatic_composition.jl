#=
Sphere hydrostatic composition (discrete_hydrostatic_p!): the VI initial
state over Hughes2023 terrain must satisfy the discrete vertical balance
ᶠgradᵥ(p) = −If(ρ)·ᶠgradᵥ(Φ) to roundoff at interior faces, with a
CHECKERBOARD-FREE ρ. The legacy ρ-correction (eigenvalue −1 recursion)
satisfies the same balance but deposits an oscillating δρ whose
along-surface variation is a form-independent horizontal PGF kick — the
dominant terrain imbalance over marginally resolved ridges
(docs/vi_kep_face_terms.md §8). Stretched vertical grid on purpose: the
recursion must be exact per interval, not just for uniform Δz.
=#

import LinearAlgebra: dot

@testset "Sphere hydrostatic composition (discrete_hydrostatic_p!)" begin
    prob = DG.BaroclinicWaveDG(;
        helem = 4,
        zelem = 10,
        zstretch = (300.0, 3000.0),
        topography = :hughes2023,
        face_set = :es,
        perturb = false,
        dt = 60.0,
        t_end = 60.0,
    )
    sim = DG.DGSimulation(prob)
    m = sim.model
    Y = sim.Y₀
    c = m.c
    (; Ic, If, ᶠgradᵥ) = m.ops
    (; ᶜΦ) = m.fields

    # p exactly as the tendency diagnoses it (full-metric K + pres_ρe)
    ρ = Y.c.ρ
    uₕ = Y.c.uₕ
    w = Y.f.w
    K = @. (
        dot(DG.C123(uₕ), DG.CT123(uₕ)) +
        Ic(dot(DG.C123(w), DG.CT123(w))) +
        2 * dot(DG.CT123(uₕ), Ic(DG.C123(w)))
    ) / 2
    p = @. DG.pres_ρe(c, Y.c.ρe, K, ᶜΦ, ρ)

    # discrete vertical balance at interior faces (covariant3 components;
    # bottom/top faces carry the SetGradient(0) BCs — excluded)
    g_p = parent(@. ᶠgradᵥ(p).components.data.:1)
    g_Φρ = parent(@. If(ρ) * ᶠgradᵥ(ᶜΦ).components.data.:1)
    resid = g_p .+ g_Φρ
    rel =
        maximum(abs, resid[2:(end - 1), :, :, :, :]) /
        maximum(abs, g_p[2:(end - 1), :, :, :, :])
    @info "sphere composition: vertical balance residual (rel)" rel
    @test rel < 1e-10

    # checkerboard diagnostic: second index-difference of the deviation
    # from the analytic (uncomposed) ρ. The legacy correction alternates
    # sign level-to-level (δ² ≈ 4× its amplitude); the product recursion's
    # deviation is column-smooth.
    base = DG.base_values(m)
    p_ana = base.p
    ᶜρ_ana = @. p_ana / (c.R_d * base.T)
    ρ_old = copy(ᶜρ_ana)
    DG.discrete_hydrostatic_ρ!(ρ_old, p_ana, m.fields.ccoords.z, c.grav)
    δ²_of_deviation(ρ_c) = begin
        d = (parent(ρ_c) .- parent(ᶜρ_ana)) ./ parent(ᶜρ_ana)
        v =
            d[3:end, :, :, :, :] .- 2 .* d[2:(end - 1), :, :, :, :] .+
            d[1:(end - 2), :, :, :, :]
        maximum(abs, v)
    end
    cb_new = δ²_of_deviation(ρ)
    cb_old = δ²_of_deviation(ρ_old)
    @info "sphere composition: checkerboard δ²(δρ/ρ)" cb_new cb_old
    @test cb_new < 0.1 * cb_old

    # t = 0 tendency snapshot (context for the terrain PGF residual; the
    # run-level A/B lives in the validation script, not the test suite)
    dY = similar(Y)
    DG.rhs_vi!(dY, Y, m, 0.0)
    dw_max = maximum(abs, parent(dY.f.w))
    @info "sphere composition: t = 0 max |dw/dt| (covariant)" dw_max
    @test !any(isnan, parent(dY.c))
end
