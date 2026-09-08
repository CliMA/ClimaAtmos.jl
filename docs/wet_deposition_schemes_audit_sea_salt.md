# Audit of Wet Deposition Schemes for Sea-Salt Aerosol in Global Models

*Prepared for a new-generation climate model development effort. Scope: wet removal (in-cloud nucleation/rainout, below-cloud impaction/washout, and convective scavenging) as applied to sea-salt aerosol. Evaluated on (1) simplicity of implementation, (2) physical vs. numerical/empirical basis, and (3) number and tunability of parameters.*

*Revision note: §4 and §9 incorporate quantitative findings from the review/intercomparison literature; §10 surveys that literature (including Jones et al. 2022, supplied by the user); §11 documents what the MERRA-2 reanalysis (GOCART) actually runs, with the exact below-cloud formula verified from a NASA GEOS source.*

---

## 1. Why sea salt is a special case

Before comparing schemes it is worth stating what makes sea salt different from a generic aerosol tracer, because it changes which parts of a wet-deposition scheme actually matter.

**Size distribution spans coarse and giant modes.** Most sea-salt *mass* sits in the coarse (>1 µm) and super-coarse/giant modes. This is the single most important fact for scheme selection. Below-cloud impaction scavenging is efficient for coarse particles (they sit well above the "Greenfield gap" — the collection-efficiency minimum near 0.1–1 µm where neither Brownian diffusion nor inertial impaction is effective), so washout is a genuinely important sink for sea salt, unlike for accumulation-mode aerosol where it is often negligible. A bulk scheme calibrated to accumulation-mode aerosol will badly mis-handle sea salt unless coefficients are integrated over the coarse mode.

**Highly hygroscopic, efficient CCN and giant CCN.** Sea-salt particles activate readily. In-cloud nucleation scavenging of the soluble/activated fraction is therefore close to complete, which is why many models simply assume ~100% in-cloud removal of the activated sea-salt fraction rather than solving activation explicitly. The in-cloud part is thus *easy to approximate well*; the harder physics is below-cloud collection and the treatment of re-evaporation.

**Short lifetime, deposition near source.** Sea-salt lifetime against deposition is short (order 0.5–1 day; AeroCom multi-model range roughly 7–11 h for the modeled size range in some studies), so the vertical profile is steep and dominated by the marine boundary layer, where high relative humidity drives strong hygroscopic growth. Removal is concentrated near the source region over the ocean.

**Wet-removal is the dominant, and most uncertain, term.** In the AeroCom Phase I diversity analysis (Textor et al., 2006), sea salt and dust showed the *highest inter-model diversity* in removal-rate coefficients and in the wet/dry deposition split of any species — models did not even agree on the partition between wet and dry deposition, nor between sedimentation and other dry processes. Wet deposition increases with solubility across species (DU < BC < POM < SO4 < SS), placing sea salt at the high-solubility, wet-dominated end. This is the headline motivation for treating the sea-salt wet-deposition scheme as a first-order design decision, not an afterthought.

**Practical implication.** For sea salt the priority ordering is: (a) get the coarse-mode below-cloud collection right, (b) approximate in-cloud nucleation scavenging of the activated fraction (near-complete is a defensible zeroth order), (c) handle re-evaporation/release back to the particle phase, and (d) treat convective scavenging consistently with convective transport.

---

## 2. How the three axes are scored

Throughout, each scheme is rated **Low / Medium / High** on:

- **Implementation simplicity** — High = a few lines and one precipitation field; Low = requires size-resolved microphysics, look-up tables, or coupling to prognostic cloud/aerosol microphysics.
- **Physical basis** — High = derived from collection-efficiency / activation physics with quantities that have direct physical meaning; Low = tuned first-order rate constants with little mechanistic content ("numerical/empirical").
- **Parameter count & tunability** — approximate number of free/adjustable parameters and how much they are typically tuned. Fewer, physically-anchored parameters are generally preferable for a model you want to be able to reason about and constrain.

There is an inherent tension: the simplest schemes are the most tunable (few knobs) but the least physical; the most physical schemes have more parameters but most of them are measured constants rather than free tuning knobs.

---

## 3. In-cloud (nucleation / rainout) schemes

### 3.1 Giorgi & Chameides (1986) first-order rainout
The canonical simple scheme. Rainout is a first-order loss, `dC/dt = −k·C`, with `k` derived from the local rate of precipitation formation divided by an assumed **in-cloud condensed water (ICCW)** content, times a soluble fraction. Classically ICCW is held **constant**.

- Simplicity: **High** — the workhorse for a beginning project.
- Physical basis: **Low–Medium** — the "conversion of cloudwater to precipitation" idea is physical, but constant ICCW and a prescribed soluble fraction are strong simplifications.
- Parameters: few (ICCW, soluble/retained fraction, sometimes a temperature threshold). Highly tunable, and in practice *heavily* tuned. Luo et al. (2019) showed the constant-ICCW assumption drives large biases and is much improved by using time/space-varying ICCW from meteorology and a cloud-fraction-weighted in-cloud precipitation rate.

For sea salt the soluble fraction is essentially 1, so this reduces to "rain out the activated fraction as precipitation forms" — a reasonable zeroth-order treatment.

### 3.2 Harvard/GMI scheme — Jacob et al. (2000), Liu et al. (2001), as used in GEOS-Chem
The most widely used *integrated* implementation of the Giorgi–Chameides philosophy, originally built for 210Pb/7Be constraints. In-cloud: **100% of water-soluble aerosol is assumed to be incorporated into cloud water and rained out** at the local precipitation-formation rate; for snow, only IN species (dust, hydrophobic BC, HNO3 monolayer) are rained out. Wang et al. (2011) refined the geometry so rainout is applied to the precipitating area `F_k` and washout to the newly-precipitating area `max(0, F_{k+1}−F_k)`, avoiding double counting.

- Simplicity: **High** — precipitation fields taken directly from meteorology; no microphysical activation solved.
- Physical basis: **Medium** — near-complete scavenging of solubles is physically justified for hygroscopic species like sea salt; the treatment is bulk, not size-resolved, in-cloud.
- Parameters: modest; retention/soluble fractions per species, temperature thresholds (e.g., 258 K, 237 K). For sea salt the "100% soluble → rained out" assumption is well-suited.

This is the recommended reference point: simple, documented, and battle-tested for sea salt.

### 3.3 Croft et al. (2010) — physically-detailed in-cloud scavenging (ECHAM5-HAM)
Ties in-cloud scavenging to prognostic aerosol number/size and cloud-droplet activation, using nucleation- and impaction-scavenging ratios that depend on aerosol size, composition, and cloud type. Uses look-up tables of scavenged fractions.

- Simplicity: **Low** — requires a modal/sectional prognostic aerosol scheme and coupling to cloud microphysics.
- Physical basis: **High** — scavenging follows from activation physics rather than a prescribed fraction.
- Parameters: many, but most physically-grounded rather than free knobs. Best once the model has interactive aerosol microphysics; overkill for an initial bulk sea-salt tracer.

### 3.4 Neu & Prather (2012) — cloud overlap + ice-phase uptake (UCI-CTM)
A large-scale scavenging algorithm emphasizing **sub-grid cloud fraction and column cloud overlap**, plus explicit ice-phase uptake of soluble species. Developed to fix vertical profiles of soluble tracers and tropospheric ozone.

- Simplicity: **Medium–Low** — overlap bookkeeping is more involved than a first-order rate but does not require full aerosol microphysics.
- Physical basis: **High** for the cloud-geometry and ice treatment.
- Parameters: moderate. Relevant to sea salt mainly aloft; less so in the marine boundary layer where most removal occurs.

### 3.5 SALSA2.0 sectional in-cloud scheme (Tonttila et al., 2020) and ECHAM-HAM family
For sectional aerosol modules: activation-based in-cloud scavenging resolved per size bin. Physically detailed (**High** physical basis, **Low** simplicity, many but physical parameters). Appropriate only for a sectional microphysical model.

---

## 4. Below-cloud (impaction / washout) schemes — the axis that matters most for sea salt

### 4.1 Dana & Hales (1976) bulk washout, `k = 0.1·P`
The classic bulk below-cloud coefficient (`k` in h⁻¹ with precipitation rate `P` in mm h⁻¹), integrating collection efficiency over a typical aerosol size distribution. This is the *standard* GEOS-Chem (historically) **and standard MERRA-2/GOCART** below-cloud parameterization (see §11).

- Simplicity: **Very High** — one constant and a precipitation rate.
- Physical basis: **Low** — a single number for all sizes.
- Parameters: **one** (0.1), fully tunable. **Caveat for sea salt:** because it integrates over a "typical" distribution, it does not capture the preferential removal of coarse particles; applied naively to sea salt it *underestimates* washout of the coarse mode. Wang et al. (2011) explicitly replaced it for coarse dust and sea salt.

An early sea-salt-specific antecedent is the NaCl below-cloud washout model of the mid-1970s (JGR, doi:10.1029/JC080i024p03410).

### 4.2 Slinn (1977; 1984) semi-empirical collection efficiency
The foundational **physical** below-cloud framework. The scavenging coefficient is an integral over the raindrop size distribution of `(collection efficiency) × (drop cross-section) × (fall speed)`, with `E(d_particle, d_drop)` summing **Brownian diffusion, interception, and inertial impaction** (plus thermophoresis/diffusiophoresis/electrostatics in extended forms). This is the physics every size-resolved scheme below discretizes.

- Simplicity: **Medium** — requires numerical integration over drop and particle spectra (usually pre-tabulated).
- Physical basis: **High** — every term is mechanistic; reproduces the Greenfield-gap minimum and the efficient coarse-particle collection sea salt needs.
- Parameters: the *functional form* has essentially no free tuning parameters. Known limitation: theory *underpredicts* observed coefficients in the accumulation mode by 1–2 orders of magnitude (turbulence, charge, non-sphericity), a discrepancy less severe for the coarse sea-salt mode. The Met Office review you supplied (Jones et al., 2022) quantifies this — theoretical Slinn-type rates run 1–2 orders of magnitude below field-measured coefficients — and the uncertainty assessment of Wang et al. (2010) found current size-resolved rain-washout parameterizations disagree by up to ~1–2 orders of magnitude at a given size and rain rate. §10 collects these bounds.

### 4.3 Feng (2007, 2009) size-resolved `k = a·P^b`
A practical parameterization: fit the Slinn-type integral to a power law in precipitation rate, with mode-specific `(a, b)` for Aitken/accumulation/coarse modes, and separate fits for **rain and snow** (Feng, 2009 gives the snowfall model). This is what GEOS-Chem adopted (Wang et al., 2011): accumulation-mode coefficients for most aerosols, **coarse-mode coefficients for coarse dust and sea salt.**

- Simplicity: **High** — power law with tabulated `(a,b)` per mode/hydrometeor.
- Physical basis: **Medium–High** — inherits Slinn physics via the fit while staying cheap.
- Parameters: a handful of `(a,b)` pairs, weakly tuned. **This is the best simplicity/physics trade-off for a coarse-mode sea-salt tracer** and is the recommended below-cloud choice for the first iteration.

### 4.4 Croft et al. (2009) size-dependent below-cloud, rain and snow (ECHAM5-HAM)
Full size-resolved look-up tables of below-cloud scavenging coefficients for rain and snow, computed from collection-efficiency theory. (This is the size-dependent scheme NASA tested experimentally in GEOS; see §11.)

- Simplicity: **Low–Medium** — needs size-resolved aerosol and the look-up tables.
- Physical basis: **High** — the reference implementation for size-resolved washout in a GCM.
- Parameters: many tabulated coefficients, but physically derived. Snow coefficients exceed rain.

### 4.5 GEM-MACH, WRF-Chem, and Met Office UM below-cloud work
Recent model-development studies are useful benchmarks: size-resolved GEM-MACHv3.1 (2024), the WRF-Chem below-cloud + coarse-particle dry-deposition update (Ryu & Min, 2022), and the Met Office UM review + sensitivity study (Jones et al., 2022; §10). Valuable for validation planning rather than as a first implementation.

---

## 5. Convective scavenging schemes

### 5.1 Balkanski et al. (1993)
Established the principle — validated on 210Pb — that convective scavenging should be handled **inside the convective transport operator** (scavenging in the updraft as the parcel rises), not as a separate first-order loss on grid-mean concentrations. Most modern models (including GOCART/MERRA-2) follow this.

- Simplicity: **Medium** — must be embedded in the convection scheme.
- Physical basis: **Medium–High** — consistent updraft transport + removal.
- Parameters: scavenging efficiency per updraft, retention on detrainment. For sea salt, convective scavenging is secondary (boundary-layer dominated), but consistency matters for the fraction lofted.

### 5.2 Model-specific convective treatments
GEOS-Chem (Wang et al., 2014) adds impaction scavenging of hydrophobic aerosol in updrafts and homogeneous-freezing removal at T < 237 K; ECHAM-HAM and CAM embed convective scavenging within their convection schemes. Giannakopoulos et al. (1999; §10) found the variant that couples wet removal to convective vertical transport performed best (−4% bias vs. −40% for a humidity-gradient scheme).

---

## 6. Integrated, model-level scheme packages (what "leading GCMs" actually run)

| Model / package | In-cloud | Below-cloud | Convective | Sea-salt handling |
|---|---|---|---|---|
| **GEOS-Chem** (Jacob 2000; Liu 2001; Wang 2011/2014; Luo 2019/2020/2023 optional) | 100% solubles rained out; variable-ICCW option | Feng `aP^b`, **coarse-mode for sea salt**; Dana–Hales legacy | Balkanski-type in convection; impaction + homogeneous freezing | Fully soluble; coarse-mode washout coefficients |
| **MERRA-2 / GEOS-5 + GOCART** (Chin 2000/2002; Colarco 2010; Randles 2017) | Giorgi–Chameides first-order, solubles rained out | **Bulk `k=0.1P` (Dana–Hales)**; size-dependent optional/experimental | Balkanski, coupled to archived cloud mass flux | 5 bins (= your SSLTxx), fully soluble/hydrophilic |
| **ECHAM(5/6)-HAM** (Croft 2009/2010; Stier) | Activation-based, size-resolved | Size-resolved look-up (rain & snow) | Embedded in convection | Prognostic modal aerosol; physically resolved |
| **CAM / CESM** (Rasch 2000 lineage) | First-order, prescribed solubility | Bulk/size-resolved by version | In convection | Bulk-modal |
| **UCI-CTM** (Neu & Prather 2012) | Cloud-overlap + ice uptake | — (focus on large-scale) | Cloud-overlap based | Soluble-tracer focus |
| **Met Office UM / UKCA** (Jones 2022) | UKCA-mode / CLASSIC | Slinn-based & empirical options (high sensitivity) | In convection | Modal / six-bin |
| **IFS-AER**, **MOCAGE**, **NGAC/GOCART** | First-order rainout | Bulk/size-resolved | In convection | Bulk sea-salt bins, prescribed solubility |
| **SALSA2.0** (Tonttila 2020) | Sectional activation | Sectional size-resolved | In convection | Sectional, physically resolved |
| **FLEXPART** (Van Leuven 2023) | Cloud-water partitioning | Below-cloud coefficients (optimised) | Lagrangian | Tracer-generic |

---

## 7. Master comparison matrix

| Scheme | Process | Implementation simplicity | Physical basis | ~Free params | Tunability | Fit for a *starting* sea-salt model |
|---|---|---|---|---|---|---|
| Giorgi & Chameides (1986) | In-cloud | High | Low–Med | ~2–3 | High | Good starting point |
| Jacob/Liu (GEOS-Chem / GOCART default) | In-cloud (+washout geometry) | High | Med | ~3–5 | Med | **Recommended baseline** |
| Croft (2010) | In-cloud | Low | High | many (physical) | Low | Later, if microphysics added |
| Neu & Prather (2012) | In-cloud + ice/overlap | Med–Low | High | moderate | Med | Optional upgrade (aloft) |
| Dana & Hales (1976) `0.1P` | Below-cloud | Very High | Low | 1 | High | Avoid for sea salt (coarse bias) — but note it's MERRA-2's default |
| Slinn (1977/1984) | Below-cloud | Med | High | ~0 free | Very Low | Physics reference |
| **Feng (2007/2009) `aP^b`** | Below-cloud | High | Med–High | few `(a,b)` | Low–Med | **Recommended baseline** |
| Croft (2009) | Below-cloud | Low–Med | High | many (physical) | Low | Later, if size-resolved |
| Balkanski (1993) | Convective | Med | Med–High | ~2 | Med | Adopt the principle early |

---

## 8. Recommendation for a beginning project

**Stage 1 (minimum viable, mostly physical enough):**
- In-cloud: Jacob/Liu-style "solubles rained out at the precipitation-formation rate," sea salt fully soluble (`f_soluble = 1`). Or Giorgi–Chameides first-order rainout — but use **variable ICCW** from your meteorology from the outset (Luo et al., 2019). Apply rainout to the precipitating fraction and washout to the newly-precipitating fraction (Wang et al., 2011).
- Below-cloud: Feng (2007/2009) `k = a·P^b` with **coarse-mode `(a,b)`** for sea salt (not the Dana–Hales bulk `0.1P`). Include the separate snow fit if you run high latitudes.
- Convective: embed a Balkanski-style scavenging fraction inside your convection operator.

~6–10 physically-anchored parameters total; the natural tuning knobs are soluble fraction, ICCW treatment, and the coarse-mode `a`.

**Stage 2 (with prognostic size information):** move below-cloud to Croft (2009) size-resolved look-up tables and in-cloud to activation-based scavenging (Croft, 2010); consider Neu & Prather (2012) cloud-overlap/ice physics for the free-troposphere profile.

**Validation targets:** 210Pb/7Be for the overall deposition timescale (GEOS-Chem finds ~8.6-day 210Pb lifetime), plus sea-salt surface-concentration and AOD networks. Build diagnostics that separate rainout, washout, convective, and dry/sedimentation fluxes from day one (the wet/dry split is the largest structural uncertainty — Textor et al., 2006).

---

## 9. Key caveats

- **The theory–observation gap in below-cloud scavenging** (Slinn theory under-predicts measured coefficients, especially 0.1–1 µm) is real but *less severe for the coarse sea-salt mode*. The review literature (§10) puts the gap at 1–2 orders of magnitude and the inter-scheme spread at up to ~3–4 orders of magnitude — i.e. the BCS scheme choice can swing a coarse-mode lifetime by a factor of several (Jones et al., 2022, report coarse-mode dust lifetime 0.9–4 d across schemes; sea salt sits in the same sensitive regime).
- **Mode-dependence of the lever.** For the *accumulation* mode (number, CCN, ACI), the dominant sink is in-cloud *nucleation* scavenging via the activated fraction, not below-cloud impaction (weak in the Greenfield gap). So a flat "fully soluble → rained out" in-cloud treatment is adequate for coarse mass but inadequate for accumulation-mode number; activation-based in-cloud scavenging is the accuracy lever there.
- **Re-evaporation / resuspension** of scavenged sea salt below cloud base returns particles to the atmosphere; more important for accumulation-mode number/CCN than for coarse mass. Handled inconsistently across models.
- **Bulk vs. size-resolved mass is the crux for sea salt.** If you carry only bulk mass, pick coefficients representative of the coarse mode where the mass lives; a generic accumulation-mode coefficient is wrong by a large factor.

---

## 10. Review and intercomparison literature

The "review-of-schemes" and multi-scheme-intercomparison papers — the class of article you supplied — and what each contributes.

**Jones et al. (2022), ACP — *the article you supplied*.** A review of below-cloud scavenging (BCS) approaches plus dust sensitivity simulations in the Met Office UM. (i) Theoretical Slinn-type BCS rates are **1–2 orders of magnitude smaller than field-observed**; (ii) simulated **coarse-mode dust lifetime ranges 0.9–4 days** purely from the BCS scheme chosen. Best single reference for justifying a size-resolved coarse-mode treatment.

**Wang, Zhang & Moran (2010), ACP — uncertainty assessment of size-resolved below-cloud rain scavenging.** Quantifies how far current size-resolved washout parameterizations disagree (up to ~1–2 orders of magnitude) and which processes account for the theory–observation gap.

**Giannakopoulos et al. (1999), JGR — validation & intercomparison with 210Pb (TOMCAT).** Three conceptually different wet-removal schemes in one CTM; the convective-transport-coupled scheme gave the best performance (−4% bias vs. −40% for a humidity-gradient scheme).

**Guelle et al. (1998), JGR (two-part) — wet deposition in a size-dependent aerosol transport model.** Isolates the influence of the scavenging scheme on 210Pb vertical profiles, surface concentrations, and deposition.

**Rasch et al. (2000), Tellus B — WCRP Cambridge Workshop intercomparison.** Fifteen models on radon, lead, SO2, sulfate; agreement within a factor of ~2 near continental sources, larger divergence over remote oceanic/polar regions.

**Textor et al. (2006), ACP — AeroCom Phase I life-cycle diversity.** Sea salt shows the highest inter-model diversity in removal-rate coefficients and wet/dry split of any species.

**Van Leuven et al. (2023), GMD — optimisation of wet deposition (FLEXPART).** Motivated by the wide inter-scheme spread (up to ~4 orders of magnitude); presents an optimisation approach to constrain scavenging rates — a template for calibrating the Feng coarse-mode coefficients.

**Ryu & Min (2022), JAMES — WRF-Chem wet & dry deposition update.** A worked example of moving from bulk to a more size-aware coarse-particle treatment.

**"Development of Wet Scavenging Process of Particles in Air Quality Modeling" (2024), Atmosphere.** A recent development/review of wet scavenging formulations in a regional CTM.

**See also (earlier intercomparison):** *An intercomparison of four wet deposition schemes used in dust transport modeling*, Atmospheric Research (2006) — scavenging coefficients differ by up to ~3 orders of magnitude across four widely-used dust schemes. (Full author/DOI not independently re-verified here.)

**Net message.** Three independent bounds recur: the theory–observation gap for below-cloud rates is **~1–2 orders of magnitude**; the inter-scheme spread is **up to ~3–4 orders of magnitude**; and for a coarse, wet-dominated species this means **factor-of-several swings in lifetime and burden**. This argues for a coarse-mode-resolved below-cloud treatment, a separable wet/dry/rainout/washout budget, and calibrating the few coarse-mode coefficients against 210Pb and sea-salt observations.

---

## 11. What MERRA-2 / GOCART actually runs

Because your sea-salt bins are GOCART-derived, it is worth stating precisely what the MERRA-2 reanalysis itself uses — which is **not** the GEOS-Chem Jacob/Liu scheme most of this audit's implementation discussion assumes. MERRA-2 computes aerosols with the **GOCART** module coupled online into GEOS-5 (Colarco et al., 2010), with AOD assimilation on top (Randles et al., 2017; Buchard et al., 2017). GEOS-Chem is a *separate* CTM that is often *driven by* MERRA-2 meteorology — easy to conflate, but a different wet-removal implementation.

**Scheme lineage.** GOCART wet removal is, per the model documentation, "scavenging of aerosols in convective updrafts and rainout/washout in large-scale precipitation (Giorgi and Chameides, 1986; Balkanski et al., 1993)," anchored for deposition to Liu et al. (2001) and Colarco et al. (2010). So it is essentially the **simple end of this audit's matrix**: first-order large-scale rainout (Giorgi–Chameides) + convective-updraft scavenging (Balkanski), with sea salt treated as **fully soluble/hydrophilic** and therefore near-completely removed in warm clouds.

**Sea-salt representation.** Five bins with dry-radius bounds 0.03–0.1, 0.1–0.5, 0.5–1.5, 1.5–5, 5–10 µm (density 2200 kg m⁻³), hydrophilic (swells with RH), Gong-type source function — i.e. **the same five bins as your `SSLT01–05`** (Collow et al., 2023). This makes MERRA-2 a near drop-in validation reference for your sea-salt tracer.

**The exact below-cloud (washout) formula — verified from a NASA GEOS source** (Zhang, Liu, et al., 2022, GEOS/GOCART). Below-cloud removal of tracer `i` is a first-order sink over the precipitating fraction:

```
ΔC_i = C_i · f · (1 − exp(−B_i · Δt))
B_i  = k_i · (P / f)          with  k_i = 0.1 mm⁻¹  (constant)
```

where `C_i` = aerosol mass mixing ratio, `f` = fraction of the gridbox receiving precipitation from above, `P` = mean precipitation rate entering the gridbox (rain + snow, mm H₂O s⁻¹), `Δt` = model step. The `k_i = 0.1 mm⁻¹` constant is the **Dana & Hales (1976) bulk value** — a single coefficient "for typical raindrop and aerosol sizes," with, in the authors' own words, "the variability of BCS coefficients due to precipitation type and aerosol size … not accounted for." In other words, **standard MERRA-2 washout is exactly the `k = 0.1P` bulk row this audit flags as "avoid for sea salt" (§4.1)** — it does *not* resolve the coarse mode where sea-salt mass lives. (The size-dependent Croft-2009 washout is available in GEOS but as an *experimental* configuration, not what standard MERRA-2 ran.)

**In-cloud rainout, convective scavenging, and re-evaporation.** The large-scale rainout is the Giorgi–Chameides first-order removal of the soluble fraction as precipitation forms; convective scavenging removes soluble aerosol inside the updraft coupled to the archived cloud mass flux (Balkanski); and a fraction of scavenged aerosol is returned to the air where precipitation subsequently evaporates. These follow Chin et al. (2000) and Colarco et al. (2010). *(The below-cloud formula above is quoted directly from a NASA GEOS source; the in-cloud/convective/re-evaporation coefficients I summarize from the scheme descriptions rather than quote, as I could not access the exact expressions in Chin et al. 2000.)*

**Three takeaways for the ClimaAtmos build.**
1. **The structure validates the immediate plan.** GOCART's own removal is `ΔC = C · f · (1 − exp(−B·Δt))` — precisely the cloud-fraction-weighted, exponential first-order sink recommended in `sea_salt_wet_deposition_immediate_plan.md`. Matching MERRA-2's *form* is essentially free.
2. **The obvious improvement is the coarse mode.** MERRA-2's standard washout is the bulk `0.1P`; replacing `k_i` with a coarse-mode Feng `a·P^b` for the large sea-salt bins is exactly the cheap, high-value upgrade §8 recommends — improving on MERRA-2 where it is weakest for coarse aerosol.
3. **The accumulation-mode / ACI gap is shared.** GOCART's "fully soluble → rained out" in-cloud treatment is the flat-solubility shortcut that is inadequate for accumulation-mode number and aerosol–cloud interactions. Your activation-based in-cloud term (reusing `CMAA`) is where you would improve on MERRA-2 for CCN/ACI — not just match it.

---

## References

Balkanski, Y. J., Jacob, D. J., Gardner, G. M., Graustein, W. C., & Turekian, K. K. (1993). Transport and residence times of tropospheric aerosols inferred from a global three-dimensional simulation of 210Pb. *Journal of Geophysical Research*, 98(D11), 20573–20586. https://doi.org/10.1029/93JD02456

Buchard, V., Randles, C. A., da Silva, A. M., et al. (2017). The MERRA-2 Aerosol Reanalysis, 1980 onward, Part II: Evaluation and case studies. *Journal of Climate*, 30, 6851–6872. https://doi.org/10.1175/JCLI-D-16-0613.1

Chin, M., Rood, R. B., Lin, S.-J., Müller, J.-F., & Thompson, A. M. (2000). Atmospheric sulfur cycle simulated in the global model GOCART: Model description and global properties. *Journal of Geophysical Research*, 105(D20), 24671–24687. https://doi.org/10.1029/2000JD900384

Chin, M., Ginoux, P., Kinne, S., Torres, O., Holben, B. N., Duncan, B. N., Martin, R. V., Logan, J. A., Higurashi, A., & Nakajima, T. (2002). Tropospheric aerosol optical thickness from the GOCART model and comparisons with satellite and Sun photometer measurements. *Journal of the Atmospheric Sciences*, 59(3), 461–483. https://doi.org/10.1175/1520-0469(2002)059<0461:TAOTFT>2.0.CO;2

Colarco, P., da Silva, A., Chin, M., & Diehl, T. (2010). Online simulations of global aerosol distributions in the NASA GEOS-4 model and comparisons to satellite and ground-based aerosol optical depth. *Journal of Geophysical Research*, 115, D14207. https://doi.org/10.1029/2009JD012820

Collow, A., Buchard, V., Chin, M., Colarco, P., Darmenov, A., & da Silva, A. (2023). *Supplemental Documentation for GEOS Aerosol Products*. GMAO Office Note No. 22 (v1.0). https://gmao.gsfc.nasa.gov/pubs/docs/Collow1463.pdf

Croft, B., Lohmann, U., Martin, R. V., Stier, P., Wurzler, S., Feichter, J., Posselt, R., & Ferrachat, S. (2009). Aerosol size-dependent below-cloud scavenging by rain and snow in the ECHAM5-HAM. *Atmospheric Chemistry and Physics*, 9, 4653–4675. https://doi.org/10.5194/acp-9-4653-2009

Croft, B., Lohmann, U., Martin, R. V., Stier, P., Wurzler, S., Feichter, J., Hoose, C., Heikkilä, U., van Donkelaar, A., & Ferrachat, S. (2010). Influences of in-cloud aerosol scavenging parameterizations on aerosol concentrations and wet deposition in ECHAM5-HAM. *Atmospheric Chemistry and Physics*, 10, 1511–1543. https://doi.org/10.5194/acp-10-1511-2010

Dana, M. T., & Hales, J. M. (1976). Statistical aspects of the washout of polydisperse aerosols. *Atmospheric Environment*, 10, 45–50. https://doi.org/10.1016/0004-6981(76)90258-4

Feng, J. (2007). A 3-mode parameterization of below-cloud scavenging of aerosols for use in atmospheric dispersion models. *Atmospheric Environment*, 41(32), 6808–6822. https://doi.org/10.1016/j.atmosenv.2007.04.046

Feng, J. (2009). A size-resolved model for below-cloud scavenging of aerosols by snowfall. *Journal of Geophysical Research*, 114, D08203. https://doi.org/10.1029/2008JD011012

Giannakopoulos, C., Chipperfield, M. P., Law, K. S., & Pyle, J. A. (1999). Validation and intercomparison of wet and dry deposition schemes using 210Pb in a global three-dimensional off-line chemical transport model. *Journal of Geophysical Research*, 104(D19), 23761–23784. https://doi.org/10.1029/1999JD900392

Giorgi, F., & Chameides, W. L. (1986). Rainout lifetimes of highly soluble aerosols and gases as inferred from simulations with a general circulation model. *Journal of Geophysical Research*, 91(D13), 14367–14376. https://doi.org/10.1029/JD091iD13p14367

Gong, S. L. (2003). A parameterization of sea-salt aerosol source function for sub- and super-micron particles. *Global Biogeochemical Cycles*, 17(4), 1097. https://doi.org/10.1029/2003GB002079

Grythe, H., Ström, J., Krejci, R., Quinn, P., & Stohl, A. (2014). A review of sea-spray aerosol source functions using a large global set of sea salt aerosol concentration measurements. *Atmospheric Chemistry and Physics*, 14, 1277–1297. https://doi.org/10.5194/acp-14-1277-2014

Guelle, W., Balkanski, Y. J., Schulz, M., Dulac, F., & Monfray, P. (1998). Wet deposition in a global size-dependent aerosol transport model, 2: Influence of the scavenging scheme on 210Pb vertical profiles, surface concentrations, and deposition. *Journal of Geophysical Research*, 103(D22), 28875–28891. https://doi.org/10.1029/98JD02769

Jacob, D. J., Liu, H., Mari, C., & Yantosca, R. M. (2000). *Harvard wet deposition scheme for GMI*. Harvard University Atmospheric Chemistry Modeling Group Technical Report.

Jaeglé, L., Quinn, P. K., Bates, T. S., Alexander, B., & Lin, J.-T. (2011). Global distribution of sea salt aerosols: new constraints from in situ and remote sensing observations. *Atmospheric Chemistry and Physics*, 11, 3137–3157. https://doi.org/10.5194/acp-11-3137-2011

Jones, A. C., Hill, A., Hemmings, J., Lemaitre, P., Quérel, A., Ryder, C. L., & Woodward, S. (2022). Below-cloud scavenging of aerosol by rain: a review of numerical modelling approaches and sensitivity simulations with mineral dust in the Met Office's Unified Model. *Atmospheric Chemistry and Physics*, 22, 11381–11407. https://doi.org/10.5194/acp-22-11381-2022

Liu, H., Jacob, D. J., Bey, I., & Yantosca, R. M. (2001). Constraints from 210Pb and 7Be on wet deposition and transport in a global three-dimensional chemical tracer model driven by assimilated meteorological fields. *Journal of Geophysical Research*, 106(D11), 12109–12128. https://doi.org/10.1029/2000JD900839

Luo, G., Yu, F., & Schwab, J. (2019). Revised treatment of wet scavenging processes dramatically improves GEOS-Chem 12.0.0 simulations of surface nitric acid, nitrate, and ammonium over the United States. *Geoscientific Model Development*, 12, 3439–3447. https://doi.org/10.5194/gmd-12-3439-2019

Luo, G., Yu, F., & Moch, J. M. (2020). Further improvement of wet process treatments in GEOS-Chem v12.6.0: impact on global distributions of aerosols and aerosol precursors. *Geoscientific Model Development*, 13, 2879–2903. https://doi.org/10.5194/gmd-13-2879-2020

Luo, G., & Yu, F. (2023). Impact of air refreshing and cloud ice uptake limitations on vertical profiles and wet depositions of nitrate, ammonium, and sulfate. *Geophysical Research Letters*. https://doi.org/10.1029/2023GL104258

Neu, J. L., & Prather, M. J. (2012). Toward a more physical representation of precipitation scavenging in global chemistry models: cloud overlap and ice physics and their impact on tropospheric ozone. *Atmospheric Chemistry and Physics*, 12, 3289–3310. https://doi.org/10.5194/acp-12-3289-2012

Randles, C. A., da Silva, A. M., Buchard, V., Colarco, P. R., Darmenov, A., Govindaraju, R., Smirnov, A., Holben, B., Ferrare, R., Hair, J., Shinozuka, Y., & Flynn, C. J. (2017). The MERRA-2 Aerosol Reanalysis, 1980 onward, Part I: System description and data assimilation evaluation. *Journal of Climate*, 30, 6823–6850. https://doi.org/10.1175/JCLI-D-16-0609.1

Rasch, P. J., et al. (2000). A comparison of scavenging and deposition processes in global models: results from the WCRP Cambridge Workshop of 1995. *Tellus B*, 52(4), 1025–1056. https://doi.org/10.3402/tellusb.v52i4.17091

Ryu, Y.-H., & Min, S.-K. (2022). Improving wet and dry deposition of aerosols in WRF-Chem: updates to below-cloud scavenging and coarse-particle dry deposition. *Journal of Advances in Modeling Earth Systems*, 14, e2021MS002792. https://doi.org/10.1029/2021MS002792

Slinn, W. G. N. (1977). Some approximations for the wet and dry removal of particles and gases from the atmosphere. *Water, Air, and Soil Pollution*, 7, 513–543. https://doi.org/10.1007/BF00285550

Slinn, W. G. N. (1984). Precipitation scavenging. In *Atmospheric Science and Power Production* (D. Randerson, Ed.), U.S. Department of Energy, DOE/TIC-27601.

Spada, M., Jorba, O., Pérez García-Pando, C., Janjic, Z., & Baldasano, J. M. (2013). Modeling and evaluation of the global sea-salt aerosol distribution: sensitivity to size-resolved and sea-surface temperature dependent emission schemes. *Atmospheric Chemistry and Physics*, 13, 11735–11755. https://doi.org/10.5194/acp-13-11735-2013

Textor, C., et al. (2006). Analysis and quantification of the diversities of aerosol life cycles within AeroCom. *Atmospheric Chemistry and Physics*, 6, 1777–1813. https://doi.org/10.5194/acp-6-1777-2006

Tonttila, J., Afzalifar, A., Kokkola, H., Raatikainen, T., Korhonen, H., & Romakkaniemi, S. (2020). In-cloud scavenging scheme for sectional aerosol modules — implementation in SALSA2.0. *Geoscientific Model Development*, 13, 6215–6235. https://doi.org/10.5194/gmd-13-6215-2020

Van Leuven, S., De Meutter, P., Camps, J., Termonia, P., & Delcloo, A. (2023). An optimisation method to improve modelling of wet deposition in atmospheric transport models: applied to FLEXPART v10.4. *Geoscientific Model Development*, 16, 5323–5338. https://doi.org/10.5194/gmd-16-5323-2023

Wang, X., Zhang, L., & Moran, M. D. (2010). Uncertainty assessment of current size-resolved parameterizations for below-cloud particle scavenging by rain. *Atmospheric Chemistry and Physics*, 10, 5685–5705. https://doi.org/10.5194/acp-10-5685-2010

Wang, Q., Jacob, D. J., Fisher, J. A., Mao, J., et al. (2011). Sources of carbonaceous aerosols and deposited black carbon in the Arctic in winter-spring: implications for radiative forcing. *Atmospheric Chemistry and Physics*, 11, 12453–12473. https://doi.org/10.5194/acp-11-12453-2011

Wang, Q., Jacob, D. J., Spackman, J. R., Perring, A. E., Schwarz, J. P., Moteki, N., Marais, E. A., Ge, C., Wang, J., & Barrett, S. R. H. (2014). Global budget and radiative forcing of black carbon aerosol: constraints from pole-to-pole (HIPPO) observations across the Pacific. *Journal of Geophysical Research*, 119, 195–206. https://doi.org/10.1002/2013JD020824

Zhang, B., Liu, H., Bian, H., et al. (2022). *Size-dependent below-cloud aerosol scavenging and its feedback on clouds and transport in the NASA GEOS model during ACTIVATE 2020* (GEOS/GOCART below-cloud scavenging description). NASA Technical Reports Server, 20220016476. https://ntrs.nasa.gov/citations/20220016476

*Early sea-salt-specific below-cloud model:* A model of below-cloud precipitation scavenging of NaCl. *Journal of Geophysical Research* (Oceans), 80(24), 3410. https://doi.org/10.1029/JC080i024p03410

*Recent air-quality review:* Development of Wet Scavenging Process of Particles in Air Quality Modeling. *Atmosphere*, 15(9), 1070 (2024). https://doi.org/10.3390/atmos15091070

*Earlier dust intercomparison:* An intercomparison of four wet deposition schemes used in dust transport modeling. *Atmospheric Research* (2006). https://www.sciencedirect.com/science/article/abs/pii/S0921818106000464
