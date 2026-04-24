# Layer-mean PDF of \((T,q)\) **inside one cell** (derivation track)

This note is **only** the physics/math definition and what is (and is not) reduced in closed form today. It does **not** justify depth-wise Riemann sums as “the analytic law,” and it does **not** replace work still to be done for the **general** case you care about.

---

## 1. What “the PDF of what’s in the cell” means here

Fix one horizontal column cell. Through that cell, depth varies from the **bottom face** to the **top face**. Pick a signed coordinate \(\zeta\) that is **zero at the cell center**, negative toward the bottom face, and positive toward the top face. Let \(L\) be the distance from center to each face (half the cell thickness), so \(\zeta \in [-L, L]\).

At **each** \(\zeta\), the subgrid layer model used in `column_tensor_mu_cov_zeta` (Python) / `SubgridColumnTensor` (Julia) says: the **layer-fluctuation** state \((T,q)\) is **jointly Gaussian** with some mean vector \(\boldsymbol{\mu}(\zeta)\) and covariance \(\boldsymbol{\Sigma}(\zeta)\) that **depend on \(\zeta\)** because:

- the **mean** temperature and humidity **move linearly** with depth through the cell, but the **rate of change** can differ on the lower half (\(\zeta<0\)) and upper half (\(\zeta\ge 0\)) (different “half-cell” physics toward bottom vs top face);
- the **variances** of \(T\) and \(q\) and their **covariance** can also **change linearly** with depth on each half, with possibly different rates toward bottom vs top.

Write \(\phi(\mathbf{x};\boldsymbol{\mu},\boldsymbol{\Sigma})\) for the bivariate normal density at \(\mathbf{x}=(T,q)^\top\).

**Uniform weight through the cell** (same convention already used when one writes a layer *mean* under Lebesgue measure in \(\zeta\)): the **layer-mean marginal density** of \((T,q)\) is

\[
  p_{\mathrm{cell}}(T,q)
  \;=\;
  \frac{1}{2L}\int_{-L}^{L}
  \phi\bigl((T,q)^\top;\;\boldsymbol{\mu}(\zeta),\;\boldsymbol{\Sigma}(\zeta)\bigr)\,\mathrm{d}\zeta .
\]

That object is what the middle panel **should** show if “purple = the PDF of what’s in the cell, exactly.”

---

## 2. Piecewise split at the cell center (matches the code)

Because rates can change at \(\zeta=0\),

\[
  p_{\mathrm{cell}}(T,q)
  \;=\;
  \frac{1}{2L}\int_{-L}^{0} \phi\bigl(\mathbf{x};\boldsymbol{\mu}_{\mathrm{dn}}(\zeta),\boldsymbol{\Sigma}_{\mathrm{dn}}(\zeta)\bigr)\,\mathrm{d}\zeta
  \;+\;
  \frac{1}{2L}\int_{0}^{L} \phi\bigl(\mathbf{x};\boldsymbol{\mu}_{\mathrm{up}}(\zeta),\boldsymbol{\Sigma}_{\mathrm{up}}(\zeta)\bigr)\,\mathrm{d}\zeta ,
\]

where on each half \(\boldsymbol{\mu}_*(\zeta)=\boldsymbol{\mu}_c+\zeta\,\mathbf{m}_*\) with a **half-specific** mean-rate vector \(\mathbf{m}_*\), and \(\boldsymbol{\Sigma}_*(\zeta)\) is positive definite on that interval (code clamps / regularizes so it stays usable).

In `column_tensor_mu_cov_zeta`, the **correlation coefficient** is taken from the **cell center** and then mapped to the off-diagonal \(\rho\,\sigma_T(\zeta)\sigma_q(\zeta)\) when the marginal standard deviations change with \(\zeta\).

---

## 3. Where a **single-sheet** closed form is easy (only a teaching special case)

Suppose **hypothetically** that on one half, \(\boldsymbol{\Sigma}(\zeta)=\boldsymbol{\Sigma}_0\) **does not depend on \(\zeta\)** on that interval, while \(\boldsymbol{\mu}(\zeta)=\boldsymbol{\mu}_0+\zeta\mathbf{m}\) **does** depend linearly on \(\zeta\). (In code language: the rates of change of variance with depth are **zero** on that half; means can still slope.)

Then for fixed \((T,q)\), with \(\boldsymbol{\eta}=\mathbf{x}-\boldsymbol{\mu}_0\),

\[
  \phi(\mathbf{x};\boldsymbol{\mu}_0+\zeta\mathbf{m},\boldsymbol{\Sigma}_0)
  \propto
  \exp\Bigl(-\tfrac12\bigl(\boldsymbol{\eta}-\zeta\mathbf{m}\bigr)^\top \boldsymbol{\Sigma}_0^{-1}\bigl(\boldsymbol{\eta}-\zeta\mathbf{m}\bigr)\Bigr),
\]

and the exponent is a **quadratic polynomial in \(\zeta\)**:

\[
  -\tfrac12\bigl(A\zeta^2 + B(\mathbf{x})\,\zeta + C(\mathbf{x})\bigr),
  \qquad
  A=\mathbf{m}^\top \boldsymbol{\Sigma}_0^{-1}\mathbf{m}\;\ge 0 .
\]

Hence the \(\zeta\)-integral over an interval \([\zeta_1,\zeta_2]\) is a **Gaussian integral in \(\zeta\)** and can be written using the standard normal cdf \(\Phi\) (equivalently \(\mathrm{erf}\)) evaluated at **affine functions of** \(\mathbf{x}=(T,q)\). That yields a **closed-form** \(p\) on that half **for this special case only**.

This is **not** the general situation you described: in general the **variances and covariance change with depth**, so \(\boldsymbol{\Sigma}(\zeta)\) is **not** constant on \([-L,0]\) or \([0,L]\). I will not call that special case “frozen”; it is simply “variance not varying with depth on that half.”

---

## 4. Why the **general** case is not yet a one-line density

When \(\boldsymbol{\Sigma}(\zeta)\) varies affinely with \(\zeta\), the density contains

\[
  \det\boldsymbol{\Sigma}(\zeta)^{-1/2}
  \exp\Bigl(
    -\tfrac12\bigl(\mathbf{x}-\boldsymbol{\mu}(\zeta)\bigr)^\top \boldsymbol{\Sigma}(\zeta)^{-1}\bigl(\mathbf{x}-\boldsymbol{\mu}(\zeta)\bigr)
  \Bigr),
\]

and \(\boldsymbol{\Sigma}(\zeta)^{-1}\) is **not** affine in \(\zeta\). The integrand is **not** \(\exp(-\tfrac12(\text{quadratic in }\zeta))\) with \(\zeta\)-independent coefficients. So the depth integral \(\int_{-L}^{L}\cdots\,\mathrm{d}\zeta\) is **not** automatically the same “complete the square in \(\zeta\)” trick as for the vertical-profile quadrature reductions in `subgrid_quadrature_methodology.tex` / this README.

**What “derive it” means next (real work):**

- Choose a coordinate change that separates the mean drift from the evolving ellipse (e.g. work in whitened coordinates, or align with principal axes of \(\boldsymbol{\Sigma}_c\) at \(\zeta=0\)).
- Expand \(\boldsymbol{\Sigma}(\zeta)^{-1}\) and \(\log\det\boldsymbol{\Sigma}(\zeta)\) in \(\zeta\) and classify the resulting \(\zeta\)-integral family (it will generally be **not** elementary unless additional structure is assumed).
- Alternatively: prove an identity that rewrites the layer mean as a **convolution** of a uniform distribution on a segment in a **direction in \((T,q)\) space** with a Gaussian **with state-dependent covariance**—then study whether that convolution admits the same \(\Psi_\nu\) / incomplete-\(\Gamma\) reductions already used in the profile–Rosen pipeline (that pipeline is **not automatically** the same measure as this depth-uniform average unless one proves equivalence).

None of that last step is finished in this file yet; this document is the **starting point**, not the conclusion.

---

## 5. Relation to other objects (no “repo = truth”)

- **Profile–Rosenblatt** (two vertical half cells, center-anchored mean profile) is a **different** closed mathematical story: it produces a **different** \(p(T,q)\) in general than \(p_{\mathrm{cell}}\) above. It can still be the right object for a **specific condensate definition** in code, but it is **not** the same thing as “uniform depth average of Gaussians” unless you prove it.

- **Moment-matched** replacement of the whole cell by one big Gaussian with a tweaked covariance is explicitly criticized in `docs/subgrid_quadrature_methodology.tex` as **not faithful** to the compact-support structure along the mean direction; this derivation track should **not** use that as a substitute.

---

## 6. Deliverable status

- **Defined:** \(p_{\mathrm{cell}}(T,q)\) exactly as the depth average of local bivariate normals (§1–2).
- **Derived in closed elementary form:** only the **variance-not-varying-with-depth-on-a-half** subcase (§3), which is **too narrow** to claim as “the cell” in general.
- **Still to derive:** closed form (or proved reduction to known special functions) for **affine** \(\boldsymbol{\mu}(\zeta)\) and **affine** \(\boldsymbol{\Sigma}(\zeta)\) on each half, with a kink at \(\zeta=0\), matching the actual column-tensor assumptions.

**Honest numerical reference (dashboard purple today):** the code evaluates the same §1–2 integral by **Gauss–Legendre** on \(\zeta \in [-L,L]\) with an **exact** bivariate normal PDF at each node (`mixture_marginal_uniform_z_on_grid` with `z_quadrature="gauss_legendre"`). That is **not** a substitute for the closed-form §6 bullet; it is only a **high-accuracy quadrature** of the defining integral so the UI matches the physics definition until an elementary formula exists.

---

## 7. Next derivation step (affine \(\boldsymbol{\Sigma}(\zeta)\), no claim of closure)

Write \(\boldsymbol{\Sigma}(\zeta)=\boldsymbol{\Sigma}_c+\zeta\boldsymbol{S}\) on one half (symmetric \(\boldsymbol{S}\), PSD along the interval). For fixed \(\mathbf{x}\),

\[
  (\mathbf{x}-\boldsymbol{\mu}(\zeta))^\top \boldsymbol{\Sigma}(\zeta)^{-1}(\mathbf{x}-\boldsymbol{\mu}(\zeta))
\]

is a **rational function of \(\zeta\)** (via Cramer’s rule on \(2\times2\)) multiplied by quadratics from \(\boldsymbol{\mu}(\zeta)=\boldsymbol{\mu}_c+\zeta\mathbf{m}\). The determinant prefactor \(\det\boldsymbol{\Sigma}(\zeta)^{-1/2}\) is **not** polynomial. Classifying \(\int \mathrm{polynomial}(\zeta)\,\exp(-\tfrac12 R(\zeta)/D(\zeta))\,\mathrm{d}\zeta\) for low-degree \(R,D\) is the local obstruction; whether it reduces to dilogarithms, Appell functions, or stays non-elementary needs a careful case analysis—not asserted here.
