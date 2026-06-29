# Frozen-state alpha ladder for beres_scale_factor calibration.
#
# Motivation: differencing free-running sweep members against an AD99 baseline
# failed — after 10 days the trajectories diverge and the chaos noise floor
# (~0.06 m/s/day in the tropical mean) swamps the Beres signal for all but the
# largest alpha. Here we instead run ONE simulation, then re-evaluate the full
# NOGW forcing repeatedly on the SAME final state: once with the Beres source
# disabled (exact AD99 baseline) and once per alpha. The Beres-only forcing is
# then an exact difference on identical convection — zero dynamical noise.
# Breaking is nonlinear in alpha, so each ladder member reruns the propagation
# kernel (seconds per evaluation).
#
# Usage (repo root, branch rc/gw-beres-incloud; GPU recommended):
#   export CLIMACOMMS_DEVICE=CUDA
#   julia --project=.buildkite test/parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/calibrate_beres_alpha.jl
#
# Output: text tables + calibrate_*.pdf figures in the run's output dir.
# Run from the repo root (relative config/toml paths).

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
const TD = CA.TD  # Thermodynamics, via ClimaAtmos (not a direct .buildkite dep)
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import ClimaCore.Geometry as Geometry
import CairoMakie
using Statistics
using Printf

const ALPHAS = [1.0e-8, 2.0e-8, 4.0e-8, 1.0e-7, 2.0e-7, 5.0e-7, 1.0e-6, 2.0e-6]
const LAT_BAND = 15.0     # tropical band |lat| <= 15°
const Z_LO, Z_HI = 15e3, 45e3  # calibration band: stratosphere below the sponge

# --- 1. one free-running simulation (mid-ladder alpha for the trajectory) ---
# The steady (ν=0) mechanical source is on by default and is gated implicitly by the
# phase-speed grid: the production grid (nogw_steady_grid.toml, cmax=100/dc=0.8 → 251) has
# an exact c=0 bin, so the frozen trajectory is steady-FORCED. There is no separate
# steady/transient trajectory toggle — the steady contribution is read off exactly,
# free of any trajectory confound, by the same-frozen-state attribution in §4b below
# (it differences make_bs(α, steady=false) against make_bs(α, steady=true) on identical
# convection). beres_steady_source is retained on BeresSourceParams precisely so that
# attribution can flip it at a fixed grid.
config_files = [
    "config/common_configs/numerics_sphere_he16ze63.yml",
    "config/longrun_configs/longrun_aquaplanet_allsky_progedmf_0M.yml",
    joinpath(@__DIR__, "beres_calibration_override.yml"),
]
comms_ctx = ClimaComms.SingletonCommsContext()
config = CA.AtmosConfig(config_files; job_id = "beres_calibrate", comms_ctx)
# Spin-up alpha (4.0e-8) is set via toml/beres_calibration_alpha.toml in the override
# YAML's `toml:` chain; the ladder below sweeps alpha on the frozen final state.

dev = ClimaComms.device(comms_ctx)
dev isa ClimaComms.CPUSingleThreaded &&
    @warn "single CPU core — the 10-day spin-up will take ~10 h; use CLIMACOMMS_DEVICE=CUDA"

simulation = CA.get_simulation(config)
CA.solve_atmos!(simulation)

p = simulation.integrator.p
Y = simulation.integrator.u
FT = eltype(Y.c.ρ)
gw = p.atmos.non_orographic_gravity_wave
nogw = p.non_orographic_gravity_wave

# --- 2. refresh the GW cache consistently at the final state ----------------
# (extraction fields, source/damp levels, launch levels; uforcing from this
# call uses the run alpha and is discarded)
CA.non_orographic_gravity_wave_compute_tendency!(Y, p, gw)

# Rebuild the forcing-call locals exactly as compute_tendency! does.
(; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
thermo_params = CAP.thermodynamics_params(p.params)
grav = CAP.grav(p.params)
ᶜz = Fields.coordinate_field(Y.c).z
ᶜρ = Y.c.ρ
ᶜdTdz = similar(Y.c.ρ)
ᶜdTdz .= Geometry.WVector.(CA.ᶜgradᵥ.(CA.ᶠinterp.(ᶜT))).components.data.:1
ᶜcp_m = @. TD.cp_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
ᶜN2 = @. (grav / ᶜT) * (ᶜdTdz + grav / ᶜcp_m)
ᶜN = @. ifelse(ᶜN2 < FT(2.5e-5), FT(sqrt(2.5e-5)), sqrt(abs(ᶜN2)))
ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
(; source_p_ρ_z_u_v_level, damp_level, ᶜlevel, gw_ncval) = nogw
(; uforcing, vforcing, u_waveforcing, v_waveforcing) = nogw
ᶜρ_source = source_p_ρ_z_u_v_level.:2
ᶜu_source = source_p_ρ_z_u_v_level.:4
ᶜv_source = source_p_ρ_z_u_v_level.:5
source_level = source_p_ρ_z_u_v_level.:6

# --- 3. ladder evaluation on the frozen state -------------------------------
bs = nogw.gw_beres_source
bs_fields = (; (n => getfield(bs, n) for n in fieldnames(typeof(bs)))...)

function eval_forcing!(beres_source)
    nogw2 = merge(nogw, (; gw_beres_source = beres_source))
    p2 = (; non_orographic_gravity_wave = nogw2, scratch = p.scratch)
    uforcing .= 0
    vforcing .= 0
    CA.non_orographic_gravity_wave_forcing(
        ᶜu, ᶜv, ᶜN, ᶜρ, ᶜz, ᶜlevel,
        source_level, damp_level, ᶜρ_source, ᶜu_source, ᶜv_source,
        uforcing, vforcing, gw_ncval, u_waveforcing, v_waveforcing, p2,
    )
    return copy(Array(parent(uforcing)))
end

# tropical-band column mask and z profile helpers
latp = vec(Array(parent(Fields.level(Fields.coordinate_field(Y.c).lat, 1))))
mask = abs.(latp) .<= LAT_BAND
zcol = reshape(Array(parent(ᶜz)), size(parent(ᶜz), 1), :)[:, 1]
band = findall(z -> Z_LO <= z <= Z_HI, zcol)
tropical_profile(u3d) = [
    mean(@view reshape(u3d, size(u3d, 1), :)[k, mask]) for
    k in 1:size(u3d, 1)
]

# --- flux-budget machinery ---------------------------------------------------
# Launched Beres momentum flux per column [Pa]: ā·Σ|B₀| — exact, via the pure
# spectrum function on the frozen per-column scalars (the kernel's ε =
# ā/(ρ_source·nk) and the ρ_source multiplier in wave_forcing cancel; summing
# the per-azimuth launches restores the full ā·Σ|B₀|).
# Deposited flux per layer [Pa]: ρ·|X|·Δz from the Beres-only forcing field.
# Budget identity: launched = deposited(everywhere) + removed-without-deposit,
# where the latter is (a) source-level instability/critical rejection
# (α-DEPENDENT: the breaking test uses B₀) and (b) internal reflection
# (α-INDEPENDENT criterion ⇒ a constant fraction of launched). The growth of
# the undeposited fraction with α therefore isolates mechanism (a).
# Caveat: |X| cannot distinguish sign-cancellation inside one layer; treated
# as deposition. v-component ignored (v_heat ≈ 0 in this setup).
Nv = size(parent(ᶜρ), 1)
ρr = reshape(Array(parent(ᶜρ)), Nv, :)
dzr = reshape(Array(parent(Fields.Δz_field(axes(Y.c)))), Nv, :)
k_low = findall(z -> z < Z_LO, zcol)
k_in = band                                     # full 15–45 km band
k_qbo = findall(z -> Z_LO <= z < 30e3, zcol)   # QBO-relevant lower band
k_upper = findall(z -> 30e3 <= z <= Z_HI, zcol) # upper stratosphere band
k_high = findall(z -> z > Z_HI, zcol)

activev = vec(Array(parent(nogw.gw_beres_active))) .> 0.5
acovv = Float64.(vec(Array(parent(nogw.gw_a_cover))))
Q0v = Float64.(vec(Array(parent(nogw.gw_Q0))))
hv = Float64.(vec(Array(parent(nogw.gw_h_heat))))
uheatv = Float64.(vec(Array(parent(nogw.gw_u_heat))))
Nsrcv = Float64.(vec(Array(parent(nogw.gw_N_source))))
cgrid = nogw.gw_c

S = findall(activev .& mask)                     # tropical active columns
Q0_med_S = median(Q0v[S])
S_hi = [i for i in S if Q0v[i] > 2 * Q0_med_S]   # heavy-tail columns
S_lo = setdiff(S, S_hi)

launched_col(i, bsα) =
    acovv[i] * sum(
        abs,
        CA.wave_source(
            cgrid, FT(uheatv[i]), FT(Q0v[i]), FT(hv[i]), FT(Nsrcv[i]),
            bsα, gw_ncval,
        ),
    )

function budget(ub3d, bsα)
    ubr = reshape(ub3d, Nv, :)
    flux = ρr .* abs.(ubr) .* dzr                # [Pa] per layer
    dep(ks, cols) = sum(Float64(flux[k, i]) for k in ks, i in cols; init = 0.0)
    L(cols) = sum(launched_col(i, bsα) for i in cols; init = 0.0)
    dep_all(cols) = dep(1:Nv, cols)
    zmean =
        let num = sum(Float64(zcol[k]) * flux[k, i] for k in 1:Nv, i in S; init = 0.0),
            den = dep_all(S)

            den > 0 ? num / den : NaN
        end
    return (;
        launched = L(S),
        dep_low = dep(k_low, S), dep_in = dep(k_in, S), dep_high = dep(k_high, S),
        dep_qbo = dep(k_qbo, S), dep_upper = dep(k_upper, S),
        dep_total = dep_all(S),
        zmean_dep = zmean,
        eff_hi = isempty(S_hi) ? NaN : dep_all(S_hi) / L(S_hi),
        eff_lo = isempty(S_lo) ? NaN : dep_all(S_lo) / L(S_lo),
    )
end

@info "frozen-state ladder: AD99 baseline + $(length(ALPHAS)) alphas"
uf_ad99 = eval_forcing!(nothing)
prof_ad99 = tropical_profile(uf_ad99)
results = map(ALPHAS) do α
    bsnt = merge(bs_fields, (; beres_scale_factor = FT(α)))
    bsα = CA.BeresSourceParams{FT}(; bsnt...)
    uf = eval_forcing!(bsα)
    prof = tropical_profile(uf)
    dprof = prof .- prof_ad99
    clampfrac = count(>=(0.9 * 3e-3), abs.(uf)) / length(uf)
    bud = budget(uf .- uf_ad99, bsα)
    (; α, prof, dprof, clampfrac, uf_max = maximum(abs, uf), bud)
end

# --- 4. report ---------------------------------------------------------------
println()
println("Tropical (|lat| ≤ $(LAT_BAND)°) Beres-only drag, exact on frozen state;")
println("calibration band $(Z_LO/1e3)–$(Z_HI/1e3) km. AD99 baseline in band: ",
    @sprintf("%.3g", maximum(abs, prof_ad99[band]) * 86400), " m/s/day")
@printf("%-10s %22s %12s %16s %12s\n",
    "alpha", "max|Beres-only| m/s/day", "z@max km", "Beres/AD99 @max", "clamp-frac")
for r in results
    imax = band[argmax(abs.(r.dprof[band]))]
    ratio = abs(r.dprof[imax]) / max(abs(prof_ad99[imax]), eps(Float64))
    @printf("%-10.2g %22.3g %12.1f %16.3g %12.3g\n",
        r.α, abs(r.dprof[imax]) * 86400, zcol[imax] / 1e3, ratio, r.clampfrac)
end
println("\nPick the alpha whose in-band Beres-only drag sits in your target range",
    "\n(~0.1–1 m/s/day) with clamp-frac = 0 and a sane Beres/AD99 ratio.")

# --- flux budget table -------------------------------------------------------
println("\nBeres flux budget over tropical ACTIVE columns ($(length(S)) cols, ",
    "$(length(S_hi)) with Q0 > 2×median):")
println("(percentages of launched ā·Σ|B0|. NOTE: undeposited = reflection +")
println("source rejection; reflection COMPETES with breaking (waves that break")
println("never reach their reflection level), so it falls with alpha while")
println("source rejection rises — the net slope is state-dependent. The robust")
println("source-rejection signature is the late-alpha upturn + eff(hiQ0) collapse.)")
@printf("%-10s %12s %8s %9s %9s %8s %12s %9s %9s %9s\n",
    "alpha", "launched Pa", "dep<15", "15-30km", "30-45km", "dep>45",
    "undeposited", "z_dep km", "eff(hiQ0)", "eff(loQ0)")
for r in results
    b = r.bud
    pct(x) = 100 * x / max(b.launched, eps())
    @printf("%-10.2g %12.3g %7.1f%% %8.1f%% %8.1f%% %7.1f%% %11.1f%% %9.1f %9.2f %9.2f\n",
        r.α, b.launched, pct(b.dep_low), pct(b.dep_qbo), pct(b.dep_upper),
        pct(b.dep_high), 100 - pct(b.dep_total), b.zmean_dep / 1e3,
        b.eff_hi, b.eff_lo)
end

# --- 4b. same-frozen-state steady (ν=0) attribution -------------------------
# Exact steady contribution, free of the trajectory confound between separate
# steady-ON / transient-only runs: on THIS single frozen state, evaluate the
# transient-only spectrum (beres_steady_source = false) and the transient+steady
# spectrum (= true) at each α and difference them. steady-only = (tr+st) − tr, on
# identical convection. The frozen state is itself steady-forced (the calibration grid
# has a c=0 bin), but the FORCING difference is exact regardless of which trajectory
# produced the state — that is the whole point of attributing on a frozen state.
function deposition_summary(ub3d, cols)
    ubr = reshape(ub3d, Nv, :)
    flux = ρr .* abs.(ubr) .* dzr                  # [Pa] per layer
    dep(ks) = sum(Float64(flux[k, i]) for k in ks, i in cols; init = 0.0)
    tot = dep(1:Nv)
    zmean =
        tot > 0 ?
        sum(Float64(zcol[k]) * flux[k, i] for k in 1:Nv, i in cols; init = 0.0) /
        tot : NaN
    return (;
        low = dep(k_low), qbo = dep(k_qbo), upper = dep(k_upper),
        high = dep(k_high), total = tot, zmean = zmean,
    )
end
inband_max(uf) = maximum(abs.(tropical_profile(uf)[band])) * 86400  # m/s/day

make_bs(α, steady) = CA.BeresSourceParams{FT}(;
    merge(
        bs_fields,
        (; beres_scale_factor = FT(α), beres_steady_source = steady),
    )...,
)
attribution = map(ALPHAS) do α
    uf_off = eval_forcing!(make_bs(α, false))      # AD99 + transient Beres
    uf_on = eval_forcing!(make_bs(α, true))        # AD99 + transient + steady
    uf_steady = uf_on .- uf_off                     # exact steady-only field
    (;
        α,
        prof_steady = tropical_profile(uf_steady),
        tr = inband_max(uf_off .- uf_ad99),
        st = inband_max(uf_steady),
        cm = inband_max(uf_on .- uf_ad99),
        dep = deposition_summary(uf_steady, S),
    )
end

println(
    "\nSame-frozen-state steady (ν=0) attribution (frozen state is steady-forced; ",
    "steady-only is the EXACT (tr+st)−tr difference on identical convection):",
)
println("in-band = max |drag| over $(Z_LO/1e3)–$(Z_HI/1e3) km (m/s/day); ",
    "st<15/15-30/30-45 = % of steady-only DEPOSITED flux in each band.")
@printf("%-10s %10s %10s %10s %10s %9s %8s %8s %8s\n",
    "alpha", "transient", "steady", "combined", "steady/tr",
    "st z_dep", "st<15", "st15-30", "st30-45")
for a in attribution
    d = a.dep
    pct(x) = 100 * x / max(d.total, eps())
    @printf("%-10.2g %10.3g %10.3g %10.3g %10.3g %9.1f %7.1f%% %7.1f%% %7.1f%%\n",
        a.α, a.tr, a.st, a.cm, a.st / max(a.tr, eps()), d.zmean / 1e3,
        pct(d.low), pct(d.qbo), pct(d.upper))
end
println("steady-only deposits where `st z_dep` says; `steady/tr` is the in-band ",
    "amplitude ratio. The steady piece is α-shared, so read it at your chosen α.")

# steady-only profiles figure
figs = CairoMakie.Figure(size = (560, 450))
axs = CairoMakie.Axis(figs[1, 1];
    xlabel = "steady-only utendnogw (m/s/day)", ylabel = "z (km)",
    title = "Same-frozen-state steady (ν=0) contribution, tropical mean")
for a in attribution
    CairoMakie.lines!(axs, a.prof_steady .* 86400, zcol ./ 1e3;
        label = @sprintf("%.1e", a.α))
end
CairoMakie.axislegend(axs; position = :rb)
outs = joinpath(simulation.output_dir, "calibrate_steady_attribution.pdf")
CairoMakie.save(outs, figs)
println("figure: ", outs)

# --- 5. figure ----------------------------------------------------------------
fig = CairoMakie.Figure(size = (900, 450))
ax1 = CairoMakie.Axis(fig[1, 1]; xlabel = "Beres-only utendnogw (m/s/day)",
    ylabel = "z (km)", title = "Frozen-state ladder, tropical mean")
ax2 = CairoMakie.Axis(fig[1, 2]; xlabel = "utendnogw (m/s/day)",
    ylabel = "z (km)", title = "Total (AD99 + Beres)")
CairoMakie.lines!(ax2, prof_ad99 .* 86400, zcol ./ 1e3;
    color = :black, linestyle = :dash, label = "AD99 only")
for r in results
    lbl = @sprintf("%.1e", r.α)
    CairoMakie.lines!(ax1, r.dprof .* 86400, zcol ./ 1e3; label = lbl)
    CairoMakie.lines!(ax2, r.prof .* 86400, zcol ./ 1e3; label = lbl)
end
CairoMakie.axislegend(ax1; position = :rb)
CairoMakie.axislegend(ax2; position = :rb)
out = joinpath(simulation.output_dir, "calibrate_profiles.pdf")
CairoMakie.save(out, fig)
println("figure: ", out)

# budget fractions vs alpha
figb = CairoMakie.Figure(size = (560, 420))
axb = CairoMakie.Axis(figb[1, 1]; xlabel = "beres_scale_factor α",
    ylabel = "% of launched flux", xscale = log10,
    title = "Beres flux budget (tropical active columns)")
αs = [r.α for r in results]
frac(f) = [100 * f(r.bud) / max(r.bud.launched, eps()) for r in results]
CairoMakie.lines!(axb, αs, frac(b -> b.dep_low); label = "deposited z<15 km")
CairoMakie.lines!(axb, αs, frac(b -> b.dep_in); label = "deposited 15–45 km")
CairoMakie.lines!(axb, αs, frac(b -> b.dep_high); label = "deposited z>45 km")
CairoMakie.lines!(axb, αs,
    [100 - 100 * r.bud.dep_total / max(r.bud.launched, eps()) for r in results];
    label = "undeposited (reflected + source-rejected)", linestyle = :dash)
CairoMakie.axislegend(axb; position = :rc)
outb = joinpath(simulation.output_dir, "calibrate_budget.pdf")
CairoMakie.save(outb, figb)
println("figure: ", outb)
