# Horizontal Implicit Treatment of Acoustic Wave Components in ClimaAtmos.jl

## 1. Overview

This document describes the equations, current IMEX splitting, and the changes
required to treat the horizontal acoustic wave components (pressure gradient
force and mass divergence) implicitly in ClimaAtmos.jl. The goal is to remove
the horizontal acoustic CFL constraint on the timestep, which currently limits
$\Delta t$ to

$$
\Delta t \leq \frac{\Delta x}{c_s}
$$

where $c_s \approx 340$ m/s is the speed of sound and $\Delta x$ is the
effective horizontal grid spacing.

---

## 2. Governing Equations

ClimaAtmos.jl solves the compressible Euler equations on a rotating sphere with
terrain-following coordinates. The prognostic variables are density $\rho$,
horizontal covariant velocity $\mathbf{u}_h$ (a `Covariant12Vector`),
vertical covariant velocity $u_3$ (a `Covariant3Vector`, stored as `Y.f.u₃`),
total energy density $\rho e_{\mathrm{tot}}$, and tracer densities
$\rho q_i$.

The vertical **contravariant** velocity $u^3$ (a `Contravariant3Vector`,
stored as `ᶠu³` in precomputed quantities) is a derived diagnostic:

$$
u^3 = u_h^3 + g^{33}\, u_3
$$

where $u_h^3 = g^{3i}\, u_i$ ($i \in \{1,2\}$) captures the metric
cross-terms from topography and $g^{33}$ is the contravariant metric
component that raises the covariant index.

### 2.1 Continuity Equation

$$
\frac{\partial \rho}{\partial t} = -\nabla_h \cdot (\rho \mathbf{u})
                                   - \frac{1}{J}\frac{\partial}{\partial \xi^3}
                                     \left(\frac{\rho J}{J_f} u^3\right)
$$

**Current code locations:**

| Term | Treatment | File | Code |
|------|-----------|------|------|
| Horizontal divergence $-\nabla_h \cdot (\rho \mathbf{u})$ | **Explicit** | `advection.jl:48` | `Yₜ.c.ρ -= split_divₕ(Y.c.ρ * ᶜu, 1)` |
| Vertical divergence | **Implicit** | `implicit_tendency.jl:120` | `Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠu³)` (uses derived contravariant $u^3$) |

### 2.2 Horizontal Momentum Equation

The pressure gradient force on $\mathbf{u}_h$ uses the split-form Exner
function formulation. Define the Exner function
$\Pi = (p / p_0)^{\kappa}$, virtual potential temperature
$\theta_v$, and reference-state quantities $\theta_{v,r}$,
$\Phi_r$. Then the tendency is:

$$
\frac{\partial \mathbf{u}_h}{\partial t} = \cdots
  - \nabla_h(K + \Phi - \Phi_r)
  - \frac{c_{p,d}}{2}\left[
      (\theta_v - \theta_{v,r})\,\nabla_h \Pi
    + \nabla_h\!\left((\theta_v - \theta_{v,r})\Pi\right)
    - \Pi\,\nabla_h(\theta_v - \theta_{v,r})
  \right]
$$

where $K = \tfrac{1}{2}|\mathbf{u}|^2$ is the kinetic energy and $\Phi$ the
geopotential.

**Current code location** (`advection.jl:82–88`, **explicit**):

```julia
@. Yₜ.c.uₕ -= C12(
    gradₕ(ᶜK + ᶜΦ - ᶜΦ_r) +
    cp_d * (
        ᶜθ_v_diff * gradₕ(ᶜΠ) +
        gradₕ(ᶜθ_v_diff * ᶜΠ) -
        ᶜΠ * gradₕ(ᶜθ_v_diff)
    ) / 2,
)
```

### 2.3 Vertical Momentum Equation

The vertical pressure gradient force acts on the prognostic covariant velocity
$u_3$. Using the Exner formulation:

$$
\frac{\partial u_3}{\partial t} = \cdots
  - \frac{\partial \Phi}{\partial \xi^3} + \frac{\partial \Phi_r}{\partial \xi^3}
  - c_{p,d}\,\overline{(\theta_v - \theta_{v,r})}^{\xi^3}
    \frac{\partial \Pi}{\partial \xi^3}
$$

**Current code location** (`implicit_tendency.jl:204–205`, **implicit**).
Note that `Yₜ.f.u₃` is the tendency of the covariant component $u_3$;
the contravariant velocity $u^3$ used in the continuity equation is derived
from $u_3$ via the metric tensor.

```julia
@. Yₜ.f.u₃ -= ᶠgradᵥ_ᶜΦ - ᶠgradᵥ(ᶜΦ_r) +
              cp_d * (ᶠinterp(ᶜθ_v - ᶜθ_vr)) * ᶠgradᵥ(ᶜΠ)
```

### 2.4 Energy Equation

The total energy tendency includes horizontal and vertical advective fluxes:

$$
\frac{\partial (\rho e_{\mathrm{tot}})}{\partial t} = \cdots
  - \nabla_h \cdot (\rho \mathbf{u}\, h_{\mathrm{tot}})
  - \frac{1}{J}\frac{\partial}{\partial \xi^3}
    \left(\frac{\rho J}{J_f} u^3 h_{\mathrm{tot}}\right)
$$

where $h_{\mathrm{tot}}$ is total enthalpy.

| Term | Treatment | File |
|------|-----------|------|
| Horizontal flux divergence | Explicit | `advection.jl:59` |
| Vertical transport (central) | Implicit | `implicit_tendency.jl:123–124` |

### 2.5 Divergence Damping

A fourth-order hyperdiffusion with enhanced divergence damping is applied
explicitly to stabilize grid-scale oscillations. Given a damping factor $d$
(default 5) and biharmonic viscosity $\nu_4$:

$$
\nabla^4 \mathbf{u} = d\,\nabla_h(\nabla_h \cdot \nabla^2 \mathbf{u})
                     - \nabla_h \times (\nabla_h \times \nabla^2 \mathbf{u})
$$

**Current code location** (`hyperdiffusion.jl:164–166`, **explicit**):

```julia
ᶜ∇⁴u = @. ᶜ∇²u =
    divergence_damping_factor * C123(wgradₕ(divₕ(ᶜ∇²u))) -
    C123(wcurlₕ(C123(curlₕ(ᶜ∇²u))))
```

---

## 3. Current IMEX Splitting

ClimaAtmos.jl uses an IMEX (Implicit-Explicit) timestepping scheme provided by
`ClimaTimeSteppers.jl`. The state $Y$ is advanced as:

$$
Y^{n+1} = Y^n + \Delta t\, T_{\mathrm{exp}}(Y^n) + \Delta t\, T_{\mathrm{imp}}(Y^{n+1})
$$

where the implicit residual $R(Y) = Y^n + \Delta t\, T_{\mathrm{imp}}(Y) - Y = 0$
is solved with Newton's method and a Jacobian $\partial R / \partial Y =
\Delta t\, \partial T_{\mathrm{imp}} / \partial Y - I$.

### 3.1 Explicit Tendency: `remaining_tendency!`

Defined in `remaining_tendency.jl:64`. Calls, in order:

1. `horizontal_tracer_advection_tendency!` — horizontal advection of tracers
2. `horizontal_dynamics_tendency!` — **horizontal mass divergence**, horizontal
   energy advection, and **horizontal pressure gradient force on** $\mathbf{u}_h$
3. `hyperdiffusion_tendency!` — biharmonic diffusion with divergence damping
4. `explicit_vertical_advection_tendency!` — vertical advection (upwinding
   corrections, Coriolis/vorticity terms)
5. `additional_tendency!` — all parameterizations (radiation, microphysics,
   surface fluxes, sponges, SGS models, etc.)

### 3.2 Implicit Tendency: `implicit_tendency!`

Defined in `implicit_tendency.jl:8`. Calls:

1. `implicit_vertical_advection_tendency!` — **vertical mass divergence**,
   central vertical advection of energy and moisture, sedimentation,
   **vertical pressure gradient force on** $u^3$, Rayleigh sponge on $u^3$
2. Conditional EDMFX and diffusion terms (gated by mode flags)
3. `pressure_work_tendency!`

### 3.3 Jacobian Structure

The manual sparse Jacobian (`manual_sparse_jacobian.jl`) is **column-wise**,
coupling only vertical neighbors. Key blocks:

| Jacobian block | Physical meaning |
|---|---|
| $\partial R_\rho / \partial u_3$ | Vertical mass flux response to $u_3$ (via $u^3 = g^{33} u_3 + u_h^3$) |
| $\partial R_{u_3} / \partial \rho$ | Vertical PGF response to density (via EOS) |
| $\partial R_{u_3} / \partial (\rho e_{\mathrm{tot}})$ | Vertical PGF response to energy (via EOS) |
| $\partial R_{u_3} / \partial (\rho q_{\mathrm{tot}})$ | Vertical PGF response to moisture (via EOS) |
| $\partial R_{u_3} / \partial u_3$ | Vertical PGF response to $u_3$ (via kinetic energy) |
| $\partial R_{u_3} / \partial \mathbf{u}_h$ | Topography coupling (metric cross-terms in $u^3$) |

The vertical PGF uses the pressure matrix:

```julia
ᶠp_grad_matrix = DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix()
```

and the continuity uses the advection matrix:

```julia
ᶜadvection_matrix = -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ)
```

Both only involve **vertical** finite-difference stencils (bidiagonal and
tridiagonal `MatrixRow` types), enabling efficient column-by-column LU or
Krylov solves.

---

## 4. Acoustic Wave Identification

The linearized compressible Euler equations around a resting reference state
$(\bar\rho, \bar p, \mathbf{u}=0)$ yield the acoustic subsystem:

$$
\frac{\partial \rho'}{\partial t} = -\bar\rho\, \nabla \cdot \mathbf{u}
$$
$$
\frac{\partial \mathbf{u}}{\partial t} = -\frac{c_s^2}{\bar\rho}\, \nabla \rho'
  + \text{(buoyancy, gravity)}
$$

where $c_s^2 = \gamma \bar p / \bar\rho$ is the squared sound speed. The terms
responsible for acoustic wave propagation are:

- **Mass divergence**: $\nabla \cdot (\rho \mathbf{u})$ — both horizontal and vertical
- **Pressure gradient force**: $\nabla p / \rho$ — both horizontal and vertical

Currently, the **vertical** components of both are implicit, while the
**horizontal** components are explicit. This removes the vertical acoustic CFL
but retains the horizontal one.

---

## 5. Proposed Changes for Horizontal Implicit Acoustics

### 5.1 Summary of Approach

Move the horizontal acoustic terms from `T_exp` to `T_imp`, giving a fully
implicit acoustic subsystem. The key challenge is that horizontal operators
couple all degrees of freedom within a spectral-element layer, breaking the
column-wise Jacobian structure. We propose a **reference-state linearization**
that reduces the horizontal implicit problem to a 2D Helmholtz equation per
vertical level, solvable with ClimaCore's spectral-element machinery.

### 5.2 Terms to Move from Explicit to Implicit

| Term | Variable | From | To |
|------|----------|------|----|
| Horizontal mass divergence $-\nabla_h \cdot (\rho \mathbf{u})$ | `Yₜ.c.ρ` | `horizontal_dynamics_tendency!` | `implicit_tendency!` |
| Horizontal PGF (Exner form) | `Yₜ.c.uₕ` | `horizontal_dynamics_tendency!` | `implicit_tendency!` |
| Horizontal energy flux div. $-\nabla_h \cdot (\rho \mathbf{u}\, h_{\mathrm{tot}})$ | `Yₜ.c.ρe_tot` | `horizontal_dynamics_tendency!` | `implicit_tendency!` |

The kinetic energy gradient $\nabla_h K$ is generally treated explicitly since
it is a nonlinear advective term, not an acoustic term. However, its coupling to
$\mathbf{u}_h$ through the Jacobian block
$\partial R_{u^3}/\partial \mathbf{u}_h$ is already present for topography.

### 5.3 Reference-State Linearization

The implicit solve requires inverting a Jacobian that now includes horizontal
operators. To avoid a full 3D global sparse solve, linearize the acoustic terms
around a horizontally-uniform reference state
$(\bar\rho(z), \bar p(z), \bar\theta_v(z))$:

**Linearized continuity:**

$$
\frac{\partial \rho'}{\partial t}\bigg|_{\mathrm{imp}}
  = -\bar\rho\, \nabla_h \cdot \mathbf{u}_h
  - \frac{\partial}{\partial z}(\bar\rho\, w)
$$

**Linearized horizontal momentum (PGF only):**

$$
\frac{\partial \mathbf{u}_h}{\partial t}\bigg|_{\mathrm{imp}}
  = -c_{p,d}\,\bar\theta_v\, \nabla_h \Pi'
$$

**Linearized vertical momentum (PGF only):**

In physical space (using vertical velocity $w$ for clarity in the
linearization; in ClimaAtmos the prognostic is the covariant $u_3$, related
by the metric):

$$
\frac{\partial w}{\partial t}\bigg|_{\mathrm{imp}}
  = -c_{p,d}\,\bar\theta_v\, \frac{\partial \Pi'}{\partial z}
  - g\frac{\rho'}{\bar\rho}
$$

Eliminating $\mathbf{u}_h$ and $w$ yields a 3D Helmholtz equation for
$\Pi'$. With horizontally-uniform coefficients, this separates into:

$$
\left(\frac{1}{c_s^2 \Delta t^2} - \nabla_h^2 - \mathcal{L}_v \right) \Pi'
  = \text{RHS}
$$

where $\mathcal{L}_v$ is the vertical part of the elliptic operator. For each
horizontal spectral mode $k$ with eigenvalue $\lambda_k$ of
$-\nabla_h^2$, this becomes a **tridiagonal vertical problem**:

$$
\left(\frac{1}{c_s^2 \Delta t^2} + \lambda_k - \mathcal{L}_v \right) \hat\Pi'_k
  = \widehat{\text{RHS}}_k
$$

This restores column-wise efficiency within spectral space.

### 5.4 Implementation Steps

#### Step 1: Configuration (`src/solver/types.jl`)

Add a mode flag to `AtmosNumerics`:

```julia
horizontal_acoustic_mode::AbstractTimesteppingMode  # Explicit() or Implicit()
```

Default to `Explicit()` for backward compatibility.

#### Step 2: Gate the Explicit Terms (`src/prognostic_equations/advection.jl`)

In `horizontal_dynamics_tendency!`, conditionally skip the acoustic terms:

```julia
if p.atmos.horizontal_acoustic_mode == Explicit()
    # Horizontal mass divergence
    @. Yₜ.c.ρ -= split_divₕ(Y.c.ρ * ᶜu, 1)
    # Horizontal energy flux
    @. Yₜ.c.ρe_tot -= split_divₕ(Y.c.ρ * ᶜu, ᶜh_tot)
    # Horizontal PGF (split-form Exner)
    @. Yₜ.c.uₕ -= C12(gradₕ(ᶜK + ᶜΦ - ᶜΦ_r) + cp_d * (...) / 2)
end
```

Non-acoustic horizontal terms (e.g., EDMFX horizontal advection, tracer
advection) remain in the explicit tendency.

#### Step 3: Add to Implicit Tendency (`src/prognostic_equations/implicit/implicit_tendency.jl`)

In `implicit_tendency!`, after `implicit_vertical_advection_tendency!`, call a
new function:

```julia
if p.atmos.horizontal_acoustic_mode == Implicit()
    implicit_horizontal_acoustic_tendency!(Yₜ, Y, p, t)
end
```

where `implicit_horizontal_acoustic_tendency!` computes the same horizontal
mass divergence, energy flux divergence, and PGF as in the explicit path, but
now evaluated at the implicit state.

#### Step 4: Horizontal Helmholtz Solver (new file)

Create `src/prognostic_equations/implicit/horizontal_helmholtz.jl`:

1. **Compute reference-state profiles** $\bar\rho(z)$, $\bar p(z)$,
   $\bar\theta_v(z)$ — horizontal averages or prescribed hydrostatically-balanced profiles.
2. **Transform** the horizontal residual to spectral space using the DSS/spectral
   infrastructure in `ClimaCore.Spaces`.
3. **Solve per-mode tridiagonal systems** for $\hat\Pi'_k$ at each horizontal
   eigenvalue $\lambda_k$.
4. **Transform back** to physical space.
5. **Update** velocity and density corrections.

#### Step 5: Jacobian Extension (`src/prognostic_equations/implicit/manual_sparse_jacobian.jl`)

Two approaches:

**Option A — Schur complement / operator-splitting preconditioner:**

Keep the existing column-wise Jacobian as a preconditioner for a global Krylov
solve (GMRES). The horizontal acoustic Jacobian blocks serve as the outer
correction. This avoids modifying the `MatrixFields` infrastructure.

New Jacobian blocks needed (conceptually):

| Block | Operator | Type |
|---|---|---|
| $\partial R_\rho / \partial \mathbf{u}_h$ | $-\Delta t\, \bar\rho\, \nabla_h \cdot (\cdot)$ | Horizontal divergence |
| $\partial R_{\mathbf{u}_h} / \partial \rho$ | $-\Delta t\, (c_s^2/\bar\rho)\, \nabla_h(\cdot)$ | Horizontal gradient |
| $\partial R_{\mathbf{u}_h} / \partial (\rho e_{\mathrm{tot}})$ | $-\Delta t\, c_{p,d}\,\bar\theta_v\, \nabla_h \partial\Pi/\partial(\rho e_{\mathrm{tot}})$ | Horizontal gradient × EOS |

These are not column-local `MatrixRow` objects but global horizontal operators.

**Option B — Direct Helmholtz solve (recommended):**

Replace the Newton iteration for the horizontal acoustic part with a direct
solve of the linearized Helmholtz equation (Step 4). This avoids extending the
`MatrixFields` column-wise Jacobian to include horizontal operators. The
vertical implicit solve proceeds as before. The two are coupled via an outer
predictor-corrector or operator-splitting iteration:

```
1. Predict: Solve vertical implicit system (column-wise, as currently done)
2. Correct: Solve horizontal Helmholtz equation (level-wise, spectral)
3. Repeat if needed (typically 1–2 iterations suffice)
```

#### Step 6: Precomputed Quantities (`src/cache/precomputed_quantities.jl`)

Extend `set_implicit_precomputed_quantities!` to additionally update:

- `ᶜu` (full 3D velocity, currently only computed in explicit precompute)
- Reference-state profiles $\bar\rho$, $\bar p$, $\bar\theta_v$ (can be
  computed once at initialization and stored in `p.core`)
- Horizontal eigenvalues $\lambda_k$ of the Laplacian on the mesh (compute
  once at initialization)

#### Step 7: Solver Wiring (`src/solver/type_getters.jl`)

Update `args_integrator` to pass the horizontal acoustic mode through to the
`ManualSparseJacobian` constructor, and (if using the Helmholtz approach)
register the horizontal Helmholtz solver as an additional callback within the
implicit solve stage.

---

## 6. Summary: Files Requiring Modification

| File | Change |
|------|--------|
| `src/solver/types.jl` | Add `horizontal_acoustic_mode` field to `AtmosNumerics` |
| `src/solver/type_getters.jl` | Parse config, pass mode to Jacobian and solver setup |
| `src/prognostic_equations/advection.jl` | Gate horizontal PGF, mass div, and energy flux div behind mode flag |
| `src/prognostic_equations/implicit/implicit_tendency.jl` | Call new `implicit_horizontal_acoustic_tendency!` when mode is `Implicit()` |
| `src/prognostic_equations/implicit/manual_sparse_jacobian.jl` | Extend Jacobian (Option A) or leave as preconditioner (Option B) |
| `src/prognostic_equations/implicit/jacobian.jl` | Support Helmholtz solve or global Krylov iteration |
| `src/cache/precomputed_quantities.jl` | Extend implicit precomputed quantities |
| **New:** `src/prognostic_equations/implicit/horizontal_helmholtz.jl` | Reference-state Helmholtz solver (Option B) |

---

## 7. Expected Impact

- **Timestep**: No longer limited by horizontal acoustic CFL. The timestep is
  instead limited by the advective CFL,
  $\Delta t \leq \Delta x / |\mathbf{u}|$, which is typically 10–30× larger
  than the acoustic CFL.
- **Cost per step**: Each implicit solve now includes a horizontal Helmholtz
  solve per level (or a small number of global Krylov iterations). For spectral
  element discretizations, the per-level Helmholtz solve is $O(N_{\mathrm{dof}})$
  using the spectral Laplacian eigenmodes.
- **Net speedup**: For climate-scale simulations at $O(10\text{ km})$
  resolution where $c_s / |\mathbf{u}| \approx 10\text{–}30$, the larger
  timestep more than compensates for the additional solve cost.

---

## 8. References

- Smolarkiewicz, P. K., Kühnlein, C., & Wedi, N. P. (2019). Semi-implicit
  integrations of perturbation equations for all-scale atmospheric dynamics.
  *J. Comput. Phys.*, 376, 145–159.
- Wood, N., et al. (2014). An inherently mass-conserving semi-implicit
  semi-Lagrangian discretization of the deep-atmosphere global non-hydrostatic
  equations. *Q. J. R. Meteorol. Soc.*, 140, 1505–1520.
- Wedi, N. P., & Smolarkiewicz, P. K. (2009). A framework for testing global
  non-hydrostatic models. *Q. J. R. Meteorol. Soc.*, 135, 469–484.
- Ullrich, P. A., Jablonowski, C., Kent, J., Lauritzen, P. H., et al. (2017).
  DCMIP2016: a review of non-hydrostatic dynamical core design and intercomparison
  of participating models. *Geosci. Model Dev.*, 10, 4477–4509.
