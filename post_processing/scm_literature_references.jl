# post_processing/scm_literature_references.jl
#
# Digitized multi-model LES / CRM reference profiles and water-path statistics
# from the intercomparison literature for cases without PyCLES NetCDF datasets:
#
#   - ISDAC: Ovchinnikov et al. (2014), J. Adv. Model. Earth Syst., 6, 223–248,
#     doi:10.1002/2013MS000282. Values represent the 11-model LES ensemble mean
#     and standard deviation for the baseline mixed-phase simulation (`ice1`,
#     target ice concentration Ni0 = 1 L⁻¹), with profiles averaged over hours 6–8
#     (Figures 1, 2, 3b, 3c, 5, 7, 8, 18, Section 3, and Appendix A).
#
#   - MPACE: Klein et al. (2009), Q. J. R. Meteorol. Soc., 135, 979–1002,
#     doi:10.1002/qj.416. Values represent the 9-model cloud-resolving model (CRM)
#     ensemble median and interquartile half-spread (25th–75th percentile) for
#     Case B averaged over hours 4–12 (Figures 4, 8, 9, 11, and Table 4; note
#     Table 3 in Klein et al. reports observations rather than model output).
#
# Note: Neither paper separates total ice into suspended cloud ice (q_i / IWP)
# and precipitating snow (q_s / SWP). Accordingly, `combine_ice_snow = true`
# signals `edmf_scm_intercomparison.jl` to plot ClimaAtmos q_i and q_s (IWP and
# SWP) as two separate lines on the first ice panel, and compare their sum
# q_i + q_s (IWP + SWP) against the paper's total ice reference on the second
# ice panel.

import ClimaAtmos.AtmosphericProfilesLibrary as APL

function _interp1d(xk::AbstractVector, yk::AbstractVector, xq::AbstractVector)
    @assert length(xk) == length(yk)
    @assert issorted(xk)
    yq = similar(xq, Float64)
    for (j, x) in enumerate(xq)
        if x <= xk[1]
            yq[j] = Float64(yk[1])
        elseif x >= xk[end]
            yq[j] = Float64(yk[end])
        else
            i = searchsortedlast(xk, x)
            i = clamp(i, 1, length(xk) - 1)
            w = (x - xk[i]) / (xk[i + 1] - xk[i])
            yq[j] = Float64((1 - w) * yk[i] + w * yk[i + 1])
        end
    end
    return yq
end

function literature_reference(case_name::AbstractString)
    cname = uppercase(strip(case_name))
    if cname == "ISDAC"
        return _isdac_literature_reference()
    elseif cname == "MPACE"
        return _mpace_literature_reference()
    else
        return nothing
    end
end

function _isdac_literature_reference()
    # Plotting vertical grid: 0 to 2500 m (matches prognostic_edmfx_isdac_column.yml).
    # Note: the 3D LES domain in Ovchinnikov et al. (2014) extends to 1.5 km
    # (Section 2.1), with Newtonian scalar nudging toward `ISDAC_θ_liq_ice` and
    # `ISDAC_q_tot` starting at 1200 m and ramping to 1/3600 s⁻¹ above 1500 m
    # (Appendix A3). Above the inversion (z ≥ 850 m), θ_l and q_t are evaluated
    # directly from `APL.ISDAC_θ_liq_ice` and `APL.ISDAC_q_tot`.
    z = collect(0.0:25.0:2500.0)

    θ_init = APL.ISDAC_θ_liq_ice(Float64)
    qt_init = APL.ISDAC_q_tot(Float64)
    u_init = APL.ISDAC_u(Float64)
    v_init = APL.ISDAC_v(Float64)

    # 1. Liquid-ice potential temperature θ_l (K) — Ovchinnikov et al. (2014),
    #    Section 3.1, Fig. 1a, and Appendix A1. By hours 6–8, radiative cooling
    #    drives a well-mixed boundary layer from the surface to the inversion base
    #    at z ≈ 825 m (θ_l ≈ 264.5 ± 0.3 K); above 850 m we evaluate the exact
    #    curved initial/nudging sounding `APL.ISDAC_θ_liq_ice`.
    thetali = [
        zi <= 800.0 ? 264.5 :
        zi < 850.0 ? 264.5 + (θ_init(850.0) - 264.5) * (zi - 800.0) / 50.0 :
        θ_init(zi) for zi in z
    ]
    z_th_sig = [0.0, 800.0, 825.0, 850.0, 1200.0, 1500.0, 2500.0]
    th_sig = [0.30, 0.30, 0.45, 0.15, 0.10, 0.0, 0.0]

    # 2. Total water specific humidity q_t (g/kg) — Fig. 1b, Fig. 2, Appendix A1.
    qt = [
        zi <= 800.0 ? 1.49 :
        zi < 850.0 ? 1.49 + (qt_init(850.0) * 1e3 - 1.49) * (zi - 800.0) / 50.0 :
        qt_init(zi) * 1e3 for zi in z
    ]
    z_qt_sig = [0.0, 800.0, 825.0, 850.0, 1200.0, 1500.0, 2500.0]
    qt_sig = [0.04, 0.04, 0.06, 0.02, 0.02, 0.0, 0.0]

    # 3. Liquid cloud water mixing ratio q_l (g/kg) — Fig. 2a & Fig. 5 (`ice1`
    #    hours 6–8). Near-adiabatic linear increase from cloud base (515 ± 35 m)
    #    to cloud top (825 ± 10 m), integrating to LWP ≈ 45.5 g/m².
    z_ql =
        [0.0, 475.0, 515.0, 550.0, 600.0, 650.0, 700.0, 750.0, 795.0, 825.0, 850.0, 2500.0]
    ql_mu = [0.0, 0.0, 0.012, 0.040, 0.082, 0.125, 0.168, 0.210, 0.240, 0.080, 0.0, 0.0]
    ql_sig = [0.0, 0.0, 0.012, 0.018, 0.020, 0.022, 0.024, 0.025, 0.025, 0.030, 0.0, 0.0]

    # 4. Total ice mass mixing ratio q_i + q_s (g/kg) — Digitized from Fig. 18
    #    (`ice1` red curves across the 11 LES models, hours 6–8, z ∈ [0, 1000] m).
    z_qi = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 840.0, 2500.0]
    qi_mu =
        [0.0018, 0.0028, 0.0038, 0.0050, 0.0062, 0.0070, 0.0065, 0.0048, 0.0015, 0.0, 0.0]
    qi_sig =
        [0.0007, 0.0009, 0.0011, 0.0014, 0.0016, 0.0018, 0.0016, 0.0013, 0.0006, 0.0, 0.0]

    # 5. Turbulent kinetic energy TKE (m²/s²) — Derived from Fig. 2c vertical
    #    velocity variance <w'²> and Fig. 8 in-cloud σ_w ≈ 0.44 ± 0.05 m/s.
    z_tke = [0.0, 100.0, 250.0, 400.0, 550.0, 700.0, 800.0, 840.0, 2500.0]
    tke_mu = [0.02, 0.10, 0.18, 0.26, 0.31, 0.25, 0.10, 0.0, 0.0]
    tke_sig = [0.01, 0.03, 0.05, 0.06, 0.07, 0.06, 0.04, 0.0, 0.0]

    # 6. Cloud fraction CF (–) — Overcast stratocumulus layer between 515 ± 35 m
    #    and 825 ± 10 m (Fig. 2a & Section 3.1).
    z_cf = [0.0, 475.0, 515.0, 555.0, 810.0, 830.0, 850.0, 2500.0]
    cf_mu = [0.0, 0.0, 0.50, 1.00, 1.00, 0.50, 0.0, 0.0]
    cf_sig = [0.0, 0.0, 0.25, 0.05, 0.02, 0.20, 0.0, 0.0]

    # 7. Horizontal velocities u, v (m/s) — Ovchinnikov et al. (2014) does not
    #    report 6–8 h LES horizontal wind profiles or ensemble spread; Appendix A1
    #    specifies initial/target profiles U₀(z) = -7 m/s (`APL.ISDAC_u`) and
    #    V₀(z) = -2 + 0.003 z m/s (`APL.ISDAC_v`), toward which winds are nudged
    #    with timescale τ = 2 h above 825 m and a raised-cosine ramp below 825 m
    #    (Appendix A3; surface fluxes are zero, with z₀ = 4e-4 m momentum drag).
    umean = [u_init(zi) for zi in z]
    vmean = [v_init(zi) for zi in z]

    # 8. Time series (t ∈ [0, 8] h) — Digitized from Fig. 3b (LWP) and Fig. 3c
    #    (total IWP) for `ice1` across the 11 LES models. In the LES protocol,
    #    ice nucleation is activated after the 2 h spinup.
    t_h_knots = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    lwp_k_mu = [13.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 35.0, 41.0, 44.0, 45.0, 45.5]
    lwp_k_sig = [1.0, 1.5, 2.0, 2.8, 3.5, 4.0, 4.5, 5.5, 5.5, 5.0, 4.5, 4.5]
    iwp_k_mu = [0.0, 0.0, 0.0, 0.0, 0.0, 2.2, 3.6, 4.2, 4.5, 4.7, 4.9, 5.0]
    iwp_k_sig = [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 1.5]

    t_h = collect(0.0:0.1:8.0)
    t = t_h .* 3600.0

    return (
        label = "Ovchinnikov et al. (2014) LES",
        combine_ice_snow = true,
        nsigma = 1.0,
        t_avg_window = (6.0, 8.0),
        z = z,
        t = t,
        t_h = t_h,
        thetali = thetali,
        thetali_s = _interp1d(z_th_sig, th_sig, z),
        qt = qt,
        qt_s = _interp1d(z_qt_sig, qt_sig, z),
        ql = _interp1d(z_ql, ql_mu, z),
        ql_s = _interp1d(z_ql, ql_sig, z),
        q_tot_ice = _interp1d(z_qi, qi_mu, z),
        q_tot_ice_s = _interp1d(z_qi, qi_sig, z),
        aup = nothing,
        aup_s = nothing,
        wup = nothing,
        wup_s = nothing,
        tke = _interp1d(z_tke, tke_mu, z),
        tke_s = _interp1d(z_tke, tke_sig, z),
        cf = _interp1d(z_cf, cf_mu, z),
        cf_s = _interp1d(z_cf, cf_sig, z),
        qr = nothing,
        qr_s = nothing,
        qi = nothing,
        qi_s = nothing,
        qs = nothing,
        qs_s = nothing,
        umean = umean,
        umean_s = nothing,
        vmean = vmean,
        vmean_s = nothing,
        lwp = _interp1d(t_h_knots, lwp_k_mu, t_h),
        lwp_s = _interp1d(t_h_knots, lwp_k_sig, t_h),
        lwp_lo = nothing,
        lwp_hi = nothing,
        rwp = nothing,
        iwp_ts = nothing,
        swp_ts = nothing,
        tot_iwp = _interp1d(t_h_knots, iwp_k_mu, t_h),
        tot_iwp_s = _interp1d(t_h_knots, iwp_k_sig, t_h),
        tot_iwp_lo = nothing,
        tot_iwp_hi = nothing,
    )
end

function _mpace_literature_reference()
    # Lower-tropospheric plotting grid: 0 to 4000 m (while prognostic_edmfx_mpace_column.yml
    # extends to 60 km on the AMIP stretched grid for RRTMGP radiation).
    z = collect(0.0:50.0:4000.0)

    # 1. Liquid-ice potential temperature θ_li (K) — Digitized from Klein et al.
    #    (2009) Fig. 11 (bottom-right, CRM median ± interquartile half-spread for
    #    z ∈ [0, 2000] m, hours 4–12) and Section 4.1 sounding above 2000 m.
    z_th = [
        0.0,
        250.0,
        500.0,
        750.0,
        1000.0,
        1250.0,
        1380.0,
        1500.0,
        1750.0,
        2000.0,
        3000.0,
        4000.0,
    ]
    th_mu = [
        269.0,
        269.05,
        269.1,
        269.4,
        269.8,
        270.6,
        271.8,
        274.2,
        276.6,
        278.5,
        285.8,
        292.8,
    ]
    th_sig = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 0.60, 0.30, 0.25, 0.20, 0.20]

    # 2. Total water specific humidity q_t (g/kg) — Digitized from Fig. 11
    #    (top-right, CRM median ± interquartile half-spread, hours 4–12).
    z_qt = [
        0.0,
        250.0,
        500.0,
        750.0,
        1000.0,
        1250.0,
        1380.0,
        1500.0,
        1750.0,
        2000.0,
        3000.0,
        4000.0,
    ]
    qt_mu = [2.05, 1.96, 1.88, 1.78, 1.68, 1.50, 1.32, 0.85, 0.75, 0.68, 0.45, 0.26]
    qt_sig = [0.12, 0.12, 0.12, 0.15, 0.18, 0.22, 0.25, 0.10, 0.05, 0.04, 0.03, 0.02]

    # Note on condensate height mapping (q_l and q_i + q_s below):
    # Klein et al. (2009) Figures 8 and 9 plot water contents against normalized
    # height z_n (-1 at the surface, 0 at cloud base, +1 at cloud top) rather than
    # physical altitude. Here z_n is mapped to meters using representative CRM
    # median cloud boundaries z_cb ≈ 550 m and z_ct ≈ 1450 m inferred from the
    # CRM median cloud-fraction profile in Figure 4 (with 500 m and 1500 m
    # bracketing the transition edges).

    # 3. Liquid cloud water mixing ratio q_l (g/kg) — Digitized from Fig. 8
    #    (bottom-right, CRM median ± interquartile half-spread LWC in normalized
    #    height mapped via z_cb ≈ 550 m, z_ct ≈ 1450 m and converted via
    #    q_l = LWC / ρ(z), consistent with the Table 4 CRM median LWP = 57.3 g/m²).
    z_ql = [0.0, 500.0, 600.0, 800.0, 1000.0, 1180.0, 1300.0, 1400.0, 1500.0, 4000.0]
    ql_mu = [0.0, 0.0, 0.015, 0.045, 0.090, 0.135, 0.115, 0.030, 0.0, 0.0]
    ql_sig = [0.0, 0.0, 0.015, 0.035, 0.065, 0.095, 0.085, 0.030, 0.0, 0.0]

    # 4. Total ice mass mixing ratio q_i + q_s (g/kg) — Digitized from Fig. 9
    #    (bottom-right, CRM median ± interquartile half-spread IWC in normalized
    #    height mapped via z_cb ≈ 550 m, z_ct ≈ 1450 m and converted via
    #    (q_i + q_s) = IWC / ρ(z), consistent with the Table 4 CRM median total
    #    IWP = 17.1 g/m²).
    z_qi = [0.0, 300.0, 600.0, 850.0, 1100.0, 1300.0, 1420.0, 1500.0, 4000.0]
    qi_mu = [0.008, 0.011, 0.016, 0.019, 0.016, 0.009, 0.002, 0.0, 0.0]
    qi_sig = [0.005, 0.006, 0.008, 0.009, 0.008, 0.005, 0.002, 0.0, 0.0]

    # 5. Cloud fraction CF (–) — Digitized from Fig. 4 (bottom-right, CRM median
    #    ± interquartile half-spread for z ∈ [0, 2000] m, hours 4–12).
    z_cf = [
        0.0,
        400.0,
        500.0,
        600.0,
        700.0,
        850.0,
        1000.0,
        1250.0,
        1350.0,
        1450.0,
        1550.0,
        4000.0,
    ]
    cf_mu = [0.0, 0.0, 0.15, 0.40, 0.68, 0.95, 0.99, 0.99, 0.92, 0.45, 0.0, 0.0]
    cf_sig = [0.0, 0.0, 0.10, 0.15, 0.15, 0.05, 0.02, 0.02, 0.06, 0.20, 0.0, 0.0]

    # 6. Horizontal velocities u, v (m/s) — Klein et al. (2009) does not report
    #    simulated wind profiles or ensemble spread; Section 4 specifies constant
    #    initial/target horizontal wind components (u, v) = (-13.0, -3.0) m/s
    #    maintained by nudging (Coriolis is omitted in the intercomparison).
    umean = fill(-13.0, length(z))
    vmean = fill(-3.0, length(z))

    # 7. Water-path statistics over the 4–12 h averaging window — Klein et al.
    #    (2009) does not publish LWP or IWP time series; Table 4 reports 4–12 h
    #    time-averaged values for the 9 participating CRMs (whereas Table 3
    #    reports observations). For the 9 CRMs in Table 4:
    #      - LWP median = 57.3 g/m²; empirical quartiles (3rd and 7th of 9 CRMs,
    #        COAMPS and DHARMA) = 24.1–135.7 g/m².
    #      - Total IWP median = 17.1 g/m²; empirical quartiles (3rd and 7th of
    #        9 CRMs, UCLA-LARC and COAMPS) = 8.4–25.7 g/m².
    #    Plotted strictly across t ∈ [4, 12] h as horizontal median lines with
    #    asymmetric interquartile bands (no synthetic 0–4 h spinup curve).
    t_h = [4.0, 12.0]
    t = t_h .* 3600.0

    return (
        label = "Klein et al. (2009) CRM",
        combine_ice_snow = true,
        nsigma = 1.0,
        t_avg_window = (4.0, 12.0),
        z = z,
        t = t,
        t_h = t_h,
        thetali = _interp1d(z_th, th_mu, z),
        thetali_s = _interp1d(z_th, th_sig, z),
        qt = _interp1d(z_qt, qt_mu, z),
        qt_s = _interp1d(z_qt, qt_sig, z),
        ql = _interp1d(z_ql, ql_mu, z),
        ql_s = _interp1d(z_ql, ql_sig, z),
        q_tot_ice = _interp1d(z_qi, qi_mu, z),
        q_tot_ice_s = _interp1d(z_qi, qi_sig, z),
        aup = nothing,
        aup_s = nothing,
        wup = nothing,
        wup_s = nothing,
        tke = nothing,
        tke_s = nothing,
        cf = _interp1d(z_cf, cf_mu, z),
        cf_s = _interp1d(z_cf, cf_sig, z),
        qr = nothing,
        qr_s = nothing,
        qi = nothing,
        qi_s = nothing,
        qs = nothing,
        qs_s = nothing,
        umean = umean,
        umean_s = nothing,
        vmean = vmean,
        vmean_s = nothing,
        lwp = fill(57.3, 2),
        lwp_s = nothing,
        lwp_lo = fill(24.1, 2),
        lwp_hi = fill(135.7, 2),
        rwp = nothing,
        iwp_ts = nothing,
        swp_ts = nothing,
        tot_iwp = fill(17.1, 2),
        tot_iwp_s = nothing,
        tot_iwp_lo = fill(8.4, 2),
        tot_iwp_hi = fill(25.7, 2),
    )
end
