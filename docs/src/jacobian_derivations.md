# Jacobian Derivations

### Background

On every **implicit step** of duration $Δt$, we must find a state $Y$ that satisfies the equation $Y = \hat{Y} + Δt * \mathcal{T}Y$, where $\mathcal{T}Y$ is the implicit tendency of $Y$, and where $\hat{Y}$ is the state at the beginning of the implicit step, which does not depend on $Y$. We do this by using Newton's method to find the root of the **implicit step error function**
$$
E = \hat{Y} + Δt * \mathcal{T}Y - Y
$$
After initializing $Y$ to $\hat{Y}$, Newton's method iteratively computes a perturbation $δY$ that eliminates some of the error introduced by the implicit step. After finding each perturbation $δY$, it sets $Y$ to $Y + δY$ and brings $||E||$ closer to 0, and the process continues until $||E||$ is sufficiently small. The value of $δY$ is obtained by computing some perturbation $δE$ (this is typically set to $-E$, though it can be larger/smaller if acceleration/relaxation is used) and then solving the linearized equation
$$
\frac{∂E}{∂Y}⋅δY = δE
$$
So, we can achieve good performance in the implicit solver by finding an easily invertible approximation of
$$
\frac{∂E}{∂Y} = Δt*\dfrac{∂\mathcal{T}Y}{∂Y} - I
$$
We refer to this derivative as the **Jacobian** of the implicit solver.

### Outline

In the following, we will derive approximations of the Jacobian for the `ClimaAtmos` dycore with 3 different turbulence parametrizations:
- Simple diffusion
- Diagnostic EDMF
- Prognostic EDMF

The goal of this document is to precisely outline which assumptions are required for our approximations and where these assumptions are used.

In addition, the expressions in this document can be directly matched up with the code in `ClimaAtmos`, so this can serve as an unambiguous reference for our implementation.

### Notation
| Symbol           | Meaning                                                             |
|------------------|---------------------------------------------------------------------|
| $\mathcal{T}$    | denotes the implicit tendency of a prognostic value                 |
| $δ$              | denotes the erroneous perturbation of a value by the implicit step  |
| $\mathcal{M}$    | denotes the `MatrixField` of an operator's first derivative         |
| $⋅$              | multiplies a `MatrixField` by a `Field` or by another `MatrixField` |
| $\textrm{CT3}$   | converts a `Field` of vectors (of any type) to a `Field` of `CT3`s  |
| $\textrm{CT12}$  | converts a `Field` of vectors (of any type) to a `Field` of `CT12`s |
| $\textrm{adj}$   | converts a `Field` of vectors to a `Field` of co-vectors            |
| $\textrm{Diag}$  | converts a `Field` to a diagonal `MatrixField`                      |
| $\textrm{int}ᶜ$  | face-to-center interpolation operator                               |
| $\textrm{int}ᶠ$  | center-to-face interpolation operator                               |
| $\textrm{wint}ᶠ$ | center-to-face weighted interpolation operator                      |
| $\textrm{div}ᶜ$  | face-to-center divergence operator                                  |
| $\textrm{grad}ᶠ$ | center-to-face gradient operator                                    |

For any values $X₁$ and $X₂$, $δX₁$ and $δX₂$ are treated as "infinitesimals", which means that $δX₁*δX₂ = 0$.

If $f₁$ and $f₂$ are `Field`s of vectors, then $\textrm{adj}(f₁)*f₂ = \textrm{adj}(f₂)*f₁$ (i.e., inner products of vectors are commutative).

If $f₁$ and $f₂$ are any `Field`s, then $f₁*f₂ = \textrm{Diag}(f₁)⋅f₂$.

If $\textrm{opᴬ}$ is a one-argument operator, then $\textrm{opᴬ}(f) = \mathcal{M}\textrm{opᴬ}⋅f$. If $\textrm{opᴮ}$ is a two-argument operator, then $\textrm{opᴮ}(f₁, f₂) = \mathcal{M}\textrm{opᴮ}(f₁)⋅f₂$. This means that
$$
\mathcal{M}\textrm{opᴬ} = \frac{\partial\textrm{opᴬ}}{\partial f} \quad \textrm{and} \quad \mathcal{M}\textrm{opᴮ}(f₁) = \frac{\partial\textrm{opᴮ}}{\partial f₂}
$$

Note that $\textrm{op}(f₁*f₂) = \mathcal{M}\textrm{op}⋅(f₁*f₂) = \mathcal{M}\textrm{op}⋅\textrm{Diag}(f₁)⋅f₂$, but $\mathcal{M}\textrm{op}⋅f₁*f₂$ is ambiguous because $\mathcal{M}\textrm{op}⋅(f₁*f₂) ≠ (\mathcal{M}\textrm{op}⋅f₁)*f₂$. Similarly, $f₁*\textrm{op}(f₂) = f₁*(\mathcal{M}\textrm{op}⋅f₂) = \textrm{Diag}(f₁)⋅\mathcal{M}\textrm{op}⋅f₂$, but $f₁*\mathcal{M}\textrm{op}⋅f₂$ is ambiguous. The same also applies when $\textrm{op}$ is a two-argument operator.

## Simple diffusion

### Constants
| Symbol    | Type                                |
|-----------|-------------------------------------|
| $I$       | identity matrix (`LinearAlgebra.I`) |
| $R_d$     | scalar                              |
| $c_{v,d}$ | scalar                              |
| $T_{tri}$ | scalar                              |
| $Φ$       | center `Field` of scalars           |
| $ρ_{ref}$ | center `Field` of scalars           |
| $p_{ref}$ | center `Field` of scalars           |
| $K_h$     | center `Field` of scalars           |
| $K_u$     | center `Field` of scalars           |
| $g³³$     | face `Field` of `CT3xACT3`s         |
| $gʰʰ$     | center `Field` of `CT12xACT12`s     |
| $g³ʰ$     | center `Field` of `CT3xACT12`s      |

If $f$ is a `Field` of `C3`s, then $\textrm{CT3}(f) = g³³*f$.

Similarly, if $f$ is a `Field` of `C12`s, then $\textrm{CT12}(f) = gʰʰ*f$ and $\textrm{CT3}(f) = g³ʰ*f$.

Also, $g³³$ and $gʰʰ$ are `Field`s of symmetric tensors, so $\textrm{adj}(g³³) = g³³$ and $\textrm{adj}(gʰʰ) = gʰʰ$.

Although $K_h$ and $K_u$ depend on $Y$, their derivatives with respect to $Y$ are very small in comparison to the other derivatives considered below, so they are treated as constants by the implicit solver. (In other words, we are assuming that $δKₕ$ and $δKᵤ$ are both negligibly small.)

### Perturbations of prognostic values
| Symbol     | Type                                                      |
|------------|-----------------------------------------------------------|
| $ρ$        | center `Field` of scalars                                 |
| $ρe_{tot}$ | center `Field` of scalars                                 |
| $ρχ$       | one or more center `Field`s of scalars (i.e., $ρq_{tot}$) |
| $uₕ$       | center `Field` of `C12`s                                  |
| $u₃$       | face `Field` of `C3`s                                     |
$$
Y = \begin{bmatrix}ρ \\ ρe_{tot} \\ ρχ \\ uₕ \\ u₃\end{bmatrix}
$$
$$
δY = \begin{bmatrix}δρ \\ δρe_{tot} \\ δρχ \\ δuₕ \\ δu₃\end{bmatrix}
$$
We will be making use of the following assumptions:
- $\textbf{A}1: \quad \mathcal{O}(e_{tot})*δρ \ll δρe_{tot} \quad \text{and} \quad \mathcal{O}(χ)*δρ \ll δρχ$

  That is, the implicit step adds a relatively smaller perturbation to $ρ$ than to $ρe_{tot}$ and $ρχ$.
  
  This is essentially the same as assuming that the implicit solver adds a relatively smaller perturbation to $ρ$ than to $e_{tot}$ and $χ$, which can be expressed as
  $$
  \frac{δρ}{ρ} \ll \frac{δe_{tot}}{e_{tot}} \quad \text{and} \quad \frac{δρ}{ρ} \ll \frac{δχ}{χ}
  $$
- $\textbf{A}2: \quad \mathcal{O}\biggl(\dfrac{δρ}{ρ}\biggr)*\tilde{u} \ll δ\tilde{u}, \quad \mathcal{O}\biggl(\dfrac{δh_{tot}}{h_{tot}}\biggr)*\tilde{u} \ll δ\tilde{u}, \quad \text{and} \quad \mathcal{O}\biggl(\dfrac{δχ}{χ}\biggr)*\tilde{u} \ll δ\tilde{u}$

  That is, the implicit step adds relatively smaller perturbations to $ρ$, $h_{tot}$, and $χ$ than to $\tilde{u} ∈ \{uₕ, u₃, u³\}$.
- $\textbf{A}3: \quad \mathcal{O}(ρ*u)*δ\tilde{u} \ll δρe_{tot}$

  In other words, the implicit step adds a smaller perturbation to $κ$ than to $e_{tot}$. If we let $c$ denote the speed of sound, this assumption will hold when
  $$
  \frac{δκ}{δe_{tot}} \sim \frac{κ}{e_{tot}} ≈ \biggl(\frac{u}{c}\biggr)^2 \ll 1
  $$
  This is somewhat at odds with $\textbf{A}2$, which implies that $e_{tot}$ has a relatively smaller perturbation than $κ$, and therefore that the expression above underestimates the ratio of $δκ$ to $δe_{tot}$. However, since $u$ tends to be significantly smaller than $c$, $δκ$ is still smaller than $δe_{tot}$ in absolute terms.
- $\textbf{A}4: \quad \mathcal{O}(q_{tot}) \ll 1$

  Specifically, $q_{tot}$ tends to be on the order of $10^{-3}$.

*TODO: What is the justification for $\textbf{A}1$ and $\textbf{A}2$? Is there a physics-based explanation for why some variables (like $ρ$) are "more linear" than others (like $uₕ$ and $u₃$)?*

### Perturbations of diagnostic values
$$
\begin{align*}
&χ = \frac{ρχ}{ρ} \\
&\begin{alignat*}{2}
&χ + δχ& &= \frac{1}{ρ + δρ}*(ρχ + δρχ) \\
&      & &= \frac{1}{ρ}*\biggl(1 - \frac{δρ}{ρ} + \mathcal{O}(δρ*δρ)\biggr)*(ρχ + δρχ) \\
&      & &= \frac{1}{ρ}*\bigl(ρχ + δρχ - χ*δρ + \mathcal{O}(δρ*δρχ)\bigr) \\
&      & &= \frac{1}{ρ}*\bigl(ρχ + δρχ - χ*δρ\bigr) \\
&\scriptsize(\textbf{A}1)& &≈ \frac{1}{ρ}*(ρχ + δρχ)
\end{alignat*} \\
&δχ ≈ \textrm{Diag}\biggl(\frac{1}{ρ}\biggr)⋅δρχ = \frac{∂χ}{∂ρχ}⋅δρχ
\end{align*}
$$
---
$$
\begin{align*}
&\begin{align*}
u³ &= \textrm{wint}ᶠ\bigl(J*ρ, \textrm{CT3}(uₕ)\bigr) + \textrm{CT3}(u₃) \\
   &= \mathcal{M}\textrm{wint}ᶠ(J*ρ)⋅\textrm{Diag}(g³ʰ)⋅uₕ + \textrm{Diag}(g³³)⋅u₃
\end{align*} \\
&\begin{alignat*}{2}
&u³ + δu³&
  {}={} &\mathcal{M}\textrm{wint}ᶠ\bigl(J*(ρ + δρ)\bigr)⋅\textrm{Diag}(g³ʰ)⋅(uₕ + δuₕ) +{} \\
&&      &\textrm{Diag}(g³³)⋅(u₃ + δu₃) \\
&&  ={} &\Biggl(\mathcal{M}\textrm{wint}ᶠ(J*ρ) + \mathcal{O}\biggl(\frac{δρ}{ρ}\biggr)\Biggr)⋅
         \textrm{Diag}(g³ʰ)⋅(uₕ + δuₕ) +{} \\
&&      &\textrm{Diag}(g³³)⋅(u₃ + δu₃) \\
&\scriptsize(\textbf{A}2)&
    ≈{} &\mathcal{M}\textrm{wint}ᶠ(J*ρ)⋅\textrm{Diag}(g³ʰ)⋅(uₕ + δuₕ) +{} \\
&&      &\textrm{Diag}(g³³)⋅(u₃ + δu₃)
\end{alignat*} \\
&\begin{align*}
δu³ &≈ \mathcal{M}\textrm{wint}ᶠ(J*ρ)⋅\textrm{Diag}(g³ʰ)⋅δuₕ + \textrm{Diag}(g³³)⋅δu₃ \\
    &= \frac{∂u³}{∂uₕ}⋅δuₕ + \frac{∂u³}{∂u₃}⋅δu₃
\end{align*}
\end{align*}
$$
---
$$
\begin{align*}
&\begin{align*}
κ ={} &\frac{1}{2}*\textrm{adj}\bigl(\textrm{CT12}(uₕ)\bigr)*uₕ +
       \frac{1}{2}*\textrm{int}ᶜ\Bigl(\textrm{adj}\bigl(\textrm{CT3}(u₃)\bigr)*u₃\Bigr) +{} \\
      &\textrm{adj}\bigl(\textrm{CT3}(uₕ)\bigr)*\textrm{int}ᶜ(u₃) \\
  ={} &\frac{1}{2}*\textrm{Diag}\bigl(\textrm{adj}(gʰʰ*uₕ)\bigr)⋅uₕ +
       \frac{1}{2}*\mathcal{M}\textrm{int}ᶜ⋅\textrm{Diag}\bigl(\textrm{adj}(g³³*u₃)\bigr)⋅u₃ +{} \\
      &\textrm{Diag}\bigl(\textrm{adj}(g³ʰ*uₕ)\bigr)⋅\mathcal{M}\textrm{int}ᶜ⋅u₃
\end{align*} \\
&\begin{align*}
κ + δκ
    ={} &\frac{1}{2}*\textrm{Diag}\Bigl(\textrm{adj}\bigl(gʰʰ*(uₕ + δuₕ)\bigr)\Bigr)⋅(uₕ + δuₕ) +{} \\
        &\frac{1}{2}*\mathcal{M}\textrm{int}ᶜ⋅
         \textrm{Diag}\Bigl(\textrm{adj}\bigl(g³³*(u₃ + δu₃)\bigr)\Bigr)⋅(u₃ + δu₃) +{} \\
        &\textrm{Diag}\Bigl(\textrm{adj}\bigl(g³ʰ*(uₕ + δuₕ)\bigr)\Bigr)⋅\mathcal{M}\textrm{int}ᶜ⋅(u₃ + δu₃) \\
    ={} &\textrm{Diag}\bigl(\textrm{adj}(gʰʰ*uₕ)\bigr)⋅\biggl(\frac{1}{2}*uₕ + δuₕ\biggr) +
         \mathcal{O}(δuₕ*δuₕ) +{} \\
        &\mathcal{M}\textrm{int}ᶜ⋅
         \textrm{Diag}\bigl(\textrm{adj}(g³³*u₃)\bigr)⋅\biggl(\frac{1}{2}*u₃ + δu₃\biggr) +
         \mathcal{O}(δu₃*δu₃) +{} \\
        &\textrm{Diag}\bigl(\textrm{adj}(g³ʰ*uₕ)\bigr)⋅\mathcal{M}\textrm{int}ᶜ⋅(u₃ + δu₃) +{} \\
        &\textrm{Diag}\bigl(\textrm{adj}(\mathcal{M}\textrm{int}ᶜ⋅u₃)*g³ʰ\bigr)⋅δuₕ + \mathcal{O}(δuₕ*δu₃) \\
    ={} &\textrm{Diag}\bigl(\textrm{adj}(gʰʰ*uₕ)\bigr)⋅\biggl(\frac{1}{2}*uₕ + δuₕ\biggr) +{} \\
        &\mathcal{M}\textrm{int}ᶜ⋅
         \textrm{Diag}\bigl(\textrm{adj}(g³³*u₃)\bigr)⋅\biggl(\frac{1}{2}*u₃ + δu₃\biggr) +{} \\
        &\textrm{Diag}\bigl(\textrm{adj}(g³ʰ*uₕ)\bigr)⋅\mathcal{M}\textrm{int}ᶜ⋅(u₃ + δu₃) +{} \\
        &\textrm{Diag}\bigl(\textrm{adj}(\mathcal{M}\textrm{int}ᶜ⋅u₃)*g³ʰ\bigr)⋅δuₕ
\end{align*} \\
&\begin{align*}
δκ ={} &\textrm{Diag}\Bigl(
            \textrm{adj}\bigl(\textrm{CT12}(uₕ)\bigr) +
            \textrm{adj}\bigl(\textrm{int}ᶜ(u₃)\bigr)*g³ʰ
        \Bigr)⋅δuₕ +{} \\
       &\biggl(
            \textrm{Diag}\Bigl(\textrm{adj}\bigl(\textrm{CT3}(uₕ)\bigr)\Bigr)⋅\mathcal{M}\textrm{int}ᶜ +
            \mathcal{M}\textrm{int}ᶜ⋅\textrm{Diag}\Bigl(\textrm{adj}\bigl(\textrm{CT3}(u₃)\bigr)\Bigr)
        \biggr)⋅δu₃ \\
   ={} &\frac{∂κ}{∂uₕ}⋅δuₕ + \frac{∂κ}{∂u₃}⋅δu₃
\end{align*}
\end{align*}
$$
---
$$
\begin{align*}
&p_d = R_d*ρ*\biggl(T_{tri} - \frac{1}{c_{v,d}}*(κ + Φ)\biggr) + \frac{R_d}{c_{v,d}}*ρe_{tot} \\
&\begin{align*}
p_d + δp_d
    ={} &R_d*(ρ + δρ)*\biggl(T_{tri} - \frac{1}{c_{v,d}}*(κ + δκ + Φ)\biggr) +
         \frac{R_d}{c_{v,d}}*(ρe_{tot} + δρe_{tot}) \\
    ={} &R_d*(ρ + δρ)*\biggl(T_{tri} - \frac{1}{c_{v,d}}*(κ + Φ)\biggr) +
         \frac{R_d}{c_{v,d}}*(ρe_{tot} + δρe_{tot}) -{} \\
        &\frac{R_d}{c_{v,d}}*ρ*δκ + \mathcal{O}(δρ*δκ) \\
    ={} &R_d*(ρ + δρ)*\biggl(T_{tri} - \frac{1}{c_{v,d}}*(κ + Φ)\biggr) +
         \frac{R_d}{c_{v,d}}*(ρe_{tot} + δρe_{tot}) -{} \\
        &\frac{R_d}{c_{v,d}}*ρ*\biggl(\frac{∂κ}{∂uₕ}⋅δuₕ + \frac{∂κ}{∂u₃}⋅δu₃\biggr)
\end{align*} \\
&\begin{align*}
δp_d ={} &R_d*\textrm{Diag}\biggl(T_{tri} - \frac{1}{c_{v,d}}*(κ + Φ)\biggr)⋅δρ +
          \frac{R_d}{c_{v,d}}*I⋅δρe_{tot} -{} \\
         &\frac{R_d}{c_{v,d}}*\textrm{Diag}(ρ)⋅\biggl(\frac{∂κ}{∂uₕ}⋅δuₕ + \frac{∂κ}{∂u₃}⋅δu₃\biggr) \\
     ={} &\frac{∂p_d}{∂ρ}⋅δρ + \frac{∂p_d}{∂ρe_{tot}}⋅δρe_{tot} + \frac{∂p_d}{∂uₕ}⋅δuₕ + \frac{∂p_d}{∂u₃}⋅δu₃
\end{align*}
\end{align*}
$$
The derivatives in this last expression have the following asymptotic behavior:
$$
\frac{∂p_d}{∂ρ} = \mathcal{O}(e_{tot}) \qquad \frac{∂p_d}{∂ρe_{tot}} = \mathcal{O}(1) \qquad
\frac{∂p_d}{∂uₕ} = \mathcal{O}(ρ*u) \qquad \frac{∂p_d}{∂u₃} = \mathcal{O}(ρ*u)
$$
This means that we can use $\textbf{A}1$ and $\textbf{A}3$ to approximate
$$
δp_d ≈ \frac{∂p_d}{∂ρe_{tot}}⋅δρe_{tot}
$$
However, the contributions to $δp_d$ from $δρ$, $δuₕ$, and $δu₃$ have a non-negligible effect on $δ\mathcal{T}u₃$, so we will only make this approximation when computing $δh_{tot}$.

---
$$
\begin{align*}
&p = p_d + \mathcal{O}(q_{tot}) \\
&\begin{alignat*}{2}
&p + δp&                   &= p + δp_d*\bigl(1 + \mathcal{O}(q_{tot})\bigr) \\
&\scriptsize(\textbf{A}4)& &≈ p + δp_d
\end{alignat*} \\
&δp ≈ δp_d
\end{align*}
$$
---
$$
\begin{align*}
&h_{tot} = \frac{1}{ρ}*(ρe_{tot} + p) \\
&\begin{alignat*}{2}
&h_{tot} + δh_{tot}&
   &= \frac{1}{ρ + δρ}*(ρe_{tot} + δρe_{tot} + p + δp) \\
&& &= \frac{1}{ρ}*\biggl(1 - \frac{δρ}{ρ}\biggr)*(ρe_{tot} + δρe_{tot} + p + δp) \\
&& &= \frac{1}{ρ}*\biggl(ρe_{tot} + δρe_{tot} + p + δp - h_{tot}*δρ\biggr) \\
&\scriptsize(\textbf{A}1)&
   &≈ \frac{1}{ρ}*(ρe_{tot} + δρe_{tot} + p + δp) \\
&\scriptsize(\textbf{A}1\&\textbf{A}3)&
   &≈ \frac{1}{ρ}*\Biggl(ρe_{tot} + p + \biggl(I + \frac{∂p_d}{∂ρe_{tot}}\biggr)⋅δρe_{tot}\Biggr)
\end{alignat*} \\
&δh_{tot}
    ≈ \textrm{Diag}\biggl(\frac{1}{ρ}\biggr)⋅\biggl(I + \frac{∂p_d}{∂ρe_{tot}}\biggr)⋅δρe_{tot} =
      \frac{∂h_{tot,d}}{∂ρe_{tot}}⋅δρe_{tot}
\end{align*}
$$

### Perturbations of implicit tendencies
$$
\begin{align*}
&\begin{align*}
\mathcal{T}ρ
    &= -\textrm{div}ᶜ\bigl(\text{wint}ᶠ(J, ρ)*u³\bigr) \\
    &= -\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)\bigr)⋅u³
\end{align*} \\
&\begin{alignat*}{2}
&\mathcal{T}ρ + δ\mathcal{T}ρ&
  {}={} &{-}\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ + δρ)\bigr)⋅(u³ + δu³) \\
&&  ={} &{-}\mathcal{M}\textrm{div}ᶜ⋅
         \textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ) + \text{wint}ᶠ(J, δρ)\bigr)⋅(u³ + δu³) \\
&&  ={} &{-}\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)\bigr)⋅
         \Biggl(I + \mathcal{O}\biggl(\frac{δρ}{ρ}\biggr)\Biggr)⋅{} \\
&&      &\biggl(u³ + \frac{∂u³}{∂uₕ}⋅δuₕ + \frac{∂u³}{∂u₃}⋅δu₃\biggr) \\
&\scriptsize(\textbf{A}2)&
    ≈{} &{-}\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\mathcal{M}\text{wint}ᶠ(J)⋅ρ\bigr)⋅{} \\
&&      &\biggl(u³ + \frac{∂u³}{∂uₕ}⋅δuₕ + \frac{∂u³}{∂u₃}⋅δu₃\biggr)
\end{alignat*} \\
&\begin{align*}
δ\mathcal{T}ρ
    &≈ -\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)\bigr)⋅
       \biggl(\frac{∂u³}{∂uₕ}⋅δuₕ + \frac{∂u³}{∂u₃}⋅δu₃\biggr) \\
    &= \frac{∂\mathcal{T}ρ}{∂uₕ}⋅δuₕ + \frac{∂\mathcal{T}ρ}{∂u₃}⋅δu₃
\end{align*}
\end{align*}
$$
---
$$
\begin{align*}
&\begin{align*}
\mathcal{T}ρe_{tot}
    ={} &{-}\textrm{div}ᶜ\bigl(\text{wint}ᶠ(J, ρ)*\textrm{int}ᶠ(h_{tot})*u³\bigr) +{} \\
        &\textrm{div}ᶜ\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kₕ)*\textrm{grad}ᶠ(h_{tot})\bigr) \\
    ={} &{-}\mathcal{M}\textrm{div}ᶜ⋅
         \textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)*\textrm{int}ᶠ(h_{tot})\bigr)⋅u³ +{} \\
        &\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kₕ)\bigr)⋅
         \mathcal{M}\textrm{grad}ᶠ⋅h_{tot}
\end{align*} \\
&\begin{alignat*}{2}
&\mathcal{T}ρe_{tot} + δ\mathcal{T}ρe_{tot}&
  {}={} &{-}\mathcal{M}\textrm{div}ᶜ⋅
         \textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ + δρ)*\textrm{int}ᶠ(h_{tot} + δh_{tot})\bigr)⋅ \\
&&      &(u³ + δu³) +{} \\
&&      &\mathcal{M}\textrm{div}ᶜ⋅
         \textrm{Diag}\bigl(\textrm{int}ᶠ(ρ + δρ)*\textrm{int}ᶠ(Kₕ)\bigr)⋅ \\
&&      &\mathcal{M}\textrm{grad}ᶠ⋅(h_{tot} + δh_{tot}) \\
&&  ={} &{-}\mathcal{M}\textrm{div}ᶜ⋅
         \textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)*\textrm{int}ᶠ(h_{tot})\bigr)⋅ \\
&&      &\Biggl(I + \mathcal{O}\biggl(\frac{δρ}{ρ}\biggr)\Biggr)⋅
         \Biggl(I + \mathcal{O}\biggl(\frac{δh_{tot}}{h_{tot}}\biggr)\Biggr)⋅(u³ + δu³) +{} \\
&&      &\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kₕ)\bigr)⋅
         \Biggl(I + \mathcal{O}\biggl(\frac{δρ}{ρ}\biggr)\Biggr)⋅ \\
&&      &\mathcal{M}\textrm{grad}ᶠ⋅\biggl(h_{tot} + \frac{∂h_{tot,d}}{∂ρe_{tot}}⋅δρe_{tot}\biggr) \\
&\scriptsize(\textbf{A}1\&\textbf{A}2)&
    ≈{} &{-}\mathcal{M}\textrm{div}ᶜ⋅
         \textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)*\textrm{int}ᶠ(h_{tot})\bigr)⋅ \\
&&      &\biggl(u³ + \frac{∂u³}{∂uₕ}⋅δuₕ + \frac{∂u³}{∂u₃}⋅δu₃\biggr) +{} \\
&&      &\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kₕ)\bigr)⋅ \\
&&      &\mathcal{M}\textrm{grad}ᶠ⋅\biggl(h_{tot} + \frac{∂h_{tot,d}}{∂ρe_{tot}}⋅δρe_{tot}\biggr) \\
\end{alignat*} \\
&\begin{align*}
δ\mathcal{T}ρe_{tot}
    ≈{} &{-}\mathcal{M}\textrm{div}ᶜ⋅
         \textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)*\textrm{int}ᶠ(h_{tot})\bigr)⋅
         \biggl(\frac{∂u³}{∂uₕ}⋅δuₕ + \frac{∂u³}{∂u₃}⋅δu₃\biggr) +{} \\
        &\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kₕ)\bigr)⋅
         \mathcal{M}\textrm{grad}ᶠ⋅\frac{∂h_{tot,d}}{∂ρe_{tot}}⋅δρe_{tot} \\
    ={} &\frac{∂\mathcal{T}_{d,dif}ρe_{tot}}{∂ρe_{tot}}⋅δρe_{tot} +
         \frac{∂\mathcal{T}ρe_{tot}}{∂uₕ}⋅δuₕ + \frac{∂\mathcal{T}ρe_{tot}}{∂u₃}⋅δu₃
\end{align*}
\end{align*}
$$
---
This is basically the same as $\mathcal{T}ρe_{tot}$, so the full derivation won't be repeated.
$$
\begin{align*}
&\begin{align*}
\mathcal{T}ρχ
    ={} &{-}\textrm{div}ᶜ\bigl(\text{wint}ᶠ(J, ρ)*\textrm{int}ᶠ(χ)*u³\bigr) +{} \\
        &\textrm{div}ᶜ\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kₕ)*\textrm{grad}ᶠ(χ)\bigr)
\end{align*} \\
&\begin{alignat*}{2}
&\mathcal{T}ρχ + δ\mathcal{T}ρχ&
  {}≈{} &{-}\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)*\textrm{int}ᶠ(χ)\bigr)⋅ \\
&&      &\biggl(u³ + \frac{∂u³}{∂uₕ}⋅δuₕ + \frac{∂u³}{∂u₃}⋅δu₃\biggr) +{} \\
&\scriptsize(\textbf{A}1\&\textbf{A}2)&
        &\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kₕ)\bigr)⋅ \\
&&      &\mathcal{M}\textrm{grad}ᶠ⋅\biggl(χ + \frac{∂χ}{∂ρχ}⋅δρχ\biggr) \\
\end{alignat*} \\
&\begin{align*}
δ\mathcal{T}ρχ
    ≈{} &{-}\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)*\textrm{int}ᶠ(χ)\bigr)⋅
         \biggl(\frac{∂u³}{∂uₕ}⋅δuₕ + \frac{∂u³}{∂u₃}⋅δu₃\biggr) +{} \\
        &\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kₕ)\bigr)⋅
         \mathcal{M}\textrm{grad}ᶠ⋅\frac{∂χ}{∂ρχ}⋅δρχ \\
    ={} &\frac{∂\mathcal{T}_{dif}ρχ}{∂ρχ}⋅δρχ +
         \frac{∂\mathcal{T}ρχ}{∂uₕ}⋅δuₕ + \frac{∂\mathcal{T}ρχ}{∂u₃}⋅δu₃
\end{align*}
\end{align*}
$$
---
This is also similar to $\mathcal{T}ρe_{tot}$ and $\mathcal{T}ρχ$.
$$
\begin{align*}
&\mathcal{T}uₕ
    = \frac{1}{ρ}*\textrm{div}ᶜ\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kᵤ)*\textrm{grad}ᶠ(uₕ)\bigr) \\
&\begin{alignat*}{2}
&\mathcal{T}uₕ + δ\mathcal{T}uₕ&
  {}≈{} &\textrm{Diag}\biggl(\frac{1}{ρ}\biggr)⋅\mathcal{M}\textrm{div}ᶜ⋅
         \textrm{Diag}\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kᵤ)\bigr)⋅ \\
&\scriptsize(\textbf{A}2)&
        &\mathcal{M}\textrm{grad}ᶠ⋅(uₕ + δuₕ) \\
\end{alignat*} \\
&\begin{align*}
δ\mathcal{T}uₕ
   &≈ \textrm{Diag}\biggl(\frac{1}{ρ}\biggr)⋅\mathcal{M}\textrm{div}ᶜ⋅
      \textrm{Diag}\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kᵤ)\bigr)⋅\mathcal{M}\textrm{grad}ᶠ⋅δuₕ \\
   &= \frac{∂\mathcal{T}uₕ}{∂uₕ}⋅δuₕ
\end{align*}
\end{align*}
$$
---
$$
\begin{align*}
&\begin{align*}
\mathcal{T}u₃
   &= -\frac{1}{\textrm{int}ᶠ(ρ)}*
      \bigl(\textrm{grad}ᶠ(p - p_{ref}) + \textrm{int}ᶠ(ρ - ρ_{ref})*\textrm{grad}ᶠ(Φ)\bigr) \\
   &= -\frac{1}{\textrm{int}ᶠ(ρ)}*
      \Bigl(
          \mathcal{M}\textrm{grad}ᶠ⋅(p - p_{ref}) +
          \textrm{Diag}\bigl(\textrm{grad}ᶠ(Φ)\bigr)⋅\mathcal{M}\textrm{int}ᶠ⋅(ρ - ρ_{ref})
      \Bigr)
\end{align*} \\
&\begin{align*}
\mathcal{T}u₃ + δ\mathcal{T}u₃
    ={} &{-}\frac{1}{\textrm{int}ᶠ(ρ + δρ)}*\\
        &\Bigl(
             \mathcal{M}\textrm{grad}ᶠ⋅(p + δp - p_{ref}) +
             \textrm{Diag}\bigl(\textrm{grad}ᶠ(Φ)\bigr)⋅\mathcal{M}\textrm{int}ᶠ⋅(ρ + δρ - ρ_{ref})
         \Bigr) \\
    ={} &{-}\frac{1}{\textrm{int}ᶠ(ρ)}*
         \biggl(1 - \frac{\textrm{int}ᶠ(δρ)}{\textrm{int}ᶠ(ρ)} + \mathcal{O}(δρ*δρ)\biggr)*\\
        &\Bigl(
             \mathcal{M}\textrm{grad}ᶠ⋅(p + δp - p_{ref}) +
             \textrm{Diag}\bigl(\textrm{grad}ᶠ(Φ)\bigr)⋅\mathcal{M}\textrm{int}ᶠ⋅(ρ + δρ - ρ_{ref})
         \Bigr) \\
    ={} &\Biggl(
             -\frac{1}{\textrm{int}ᶠ(ρ)} +
             \textrm{Diag}\biggl(\frac{1}{\textrm{int}ᶠ(ρ)^2}\biggr)⋅\mathcal{M}\textrm{int}ᶠ⋅δρ
         \Biggr)*\\
        &\Bigl(
             \mathcal{M}\textrm{grad}ᶠ⋅(p + δp - p_{ref}) +
             \textrm{Diag}\bigl(\textrm{grad}ᶠ(Φ)\bigr)⋅\mathcal{M}\textrm{int}ᶠ⋅(ρ + δρ - ρ_{ref})
         \Bigr) \\
    ={} &{-}\frac{1}{\textrm{int}ᶠ(ρ)}*\\
        &\Bigl(
             \mathcal{M}\textrm{grad}ᶠ⋅(p + δp - p_{ref}) +
             \textrm{Diag}\bigl(\textrm{grad}ᶠ(Φ)\bigr)⋅\mathcal{M}\textrm{int}ᶠ⋅(ρ - ρ_{ref})
         \Bigr) +{} \\
        &\textrm{Diag}\biggl(
             \frac{1}{\textrm{int}ᶠ(ρ)^2}*
             \bigl(\textrm{grad}ᶠ(p - p_{ref}) - \textrm{int}ᶠ(ρ_{ref})*\textrm{grad}ᶠ(Φ)\bigr)
         \biggr)⋅\mathcal{M}\textrm{int}ᶠ⋅δρ +{} \\
        &\mathcal{O}(δρ*δp) + \mathcal{O}(δρ*δρ) \\
    ={} &{-}\textrm{Diag}\biggl(\frac{1}{\textrm{int}ᶠ(ρ)}\biggr)⋅\mathcal{M}\textrm{grad}ᶠ⋅ \\
        &\biggl(
             p + \frac{∂p_d}{∂ρ}⋅δρ + \frac{∂p_d}{∂ρe_{tot}}⋅δρe_{tot} +
             \frac{∂p_d}{∂uₕ}⋅δuₕ + \frac{∂p_d}{∂u₃}⋅δu₃ - p_{ref}
         \biggr) -{} \\
        &\textrm{Diag}\biggl(\frac{\textrm{grad}ᶠ(Φ)}{\textrm{int}ᶠ(ρ)}\biggr)⋅
         \mathcal{M}\textrm{int}ᶠ⋅(ρ - ρ_{ref}) +{} \\
        &\textrm{Diag}\biggl(
             \frac{1}{\textrm{int}ᶠ(ρ)^2}*
             \bigl(\textrm{grad}ᶠ(p - p_{ref}) - \textrm{int}ᶠ(ρ_{ref})*\textrm{grad}ᶠ(Φ)\bigr)
         \biggr)⋅\mathcal{M}\textrm{int}ᶠ⋅δρ
\end{align*} \\
&\begin{align*}
δ\mathcal{T}u₃
    ={} &{-}\textrm{Diag}\biggl(\frac{1}{\textrm{int}ᶠ(ρ)}\biggr)⋅\mathcal{M}\textrm{grad}ᶠ⋅ \\
        &\biggl(
             \frac{∂p_d}{∂ρ}⋅δρ + \frac{∂p_d}{∂ρe_{tot}}⋅δρe_{tot} +
             \frac{∂p_d}{∂uₕ}⋅δuₕ + \frac{∂p_d}{∂u₃}⋅δu₃
         \biggr) -{} \\
        &\textrm{Diag}\biggl(
             \frac{1}{\textrm{int}ᶠ(ρ)^2}*\bigl(\textrm{grad}ᶠ(p - p_{ref}) -
             \textrm{int}ᶠ(ρ_{ref})*\textrm{grad}ᶠ(Φ)\bigr)
         \biggr)⋅\mathcal{M}\textrm{int}ᶠ⋅δρ \\
    ={} &\frac{∂\mathcal{T}_du₃}{∂ρ}⋅δρ + \frac{∂\mathcal{T}_du₃}{∂ρe_{tot}}⋅δρe_{tot} +
         \frac{∂\mathcal{T}_du₃}{∂uₕ}⋅δuₕ + \frac{∂\mathcal{T}_du₃}{∂u₃}⋅δu₃
\end{align*}
\end{align*}
$$
*TODO: What is the justification for not using $\textbf{A}1$ and $\textbf{A}3$ here to ignore $\dfrac{∂p_d}{∂ρ}$, $\dfrac{∂p_d}{∂uₕ}$, and $\dfrac{∂p_d}{∂u₃}$?*

### Summary of nonzero derivatives
| Symbol | Definition | Type |
|-|-|-|
| $\dfrac{∂χ}{∂ρχ}$ | $\textrm{Diag}\biggl(\dfrac{1}{ρ}\biggr)$ | Diagonal |
| $\dfrac{∂u³}{∂uₕ}$ | $\mathcal{M}\textrm{wint}ᶠ(J*ρ)⋅\textrm{Diag}(g³ʰ)$ | Bidiagonal |
| $\dfrac{∂u³}{∂u₃}$ | $\textrm{Diag}(g³³)$ | Diagonal |
| $\dfrac{∂κ}{∂uₕ}$ | $\textrm{Diag}\Bigl(\textrm{adj}\bigl(\textrm{CT12}(uₕ)\bigr) + \textrm{adj}\bigl(\textrm{int}ᶜ(u₃)\bigr)*g³ʰ\Bigr)$ | Diagonal |
| $\dfrac{∂κ}{∂u₃}$ | $\textrm{Diag}\Bigl(\textrm{adj}\bigl(\textrm{CT3}(uₕ)\bigr)\Bigr)⋅\mathcal{M}\textrm{int}ᶜ + \mathcal{M}\textrm{int}ᶜ⋅\textrm{Diag}\Bigl(\textrm{adj}\bigl(\textrm{CT3}(u₃)\bigr)\Bigr)$ | Bidiagonal |
| $\dfrac{∂p_d}{∂ρ}$ | $R_d*\textrm{Diag}\biggl(T_{tri} - \dfrac{1}{c_{v,d}}*(κ + Φ)\biggr)$ | Diagonal |
| $\dfrac{∂p_d}{∂ρe_{tot}}$ | $\dfrac{R_d}{c_{v,d}}*I$ | Diagonal |
| $\dfrac{∂p_d}{∂uₕ}$ | $\dfrac{R_d}{c_{v,d}}*\textrm{Diag}(ρ)⋅\dfrac{∂κ}{∂uₕ}$ | Diagonal |
| $\dfrac{∂p_d}{∂u₃}$ | $\dfrac{R_d}{c_{v,d}}*\textrm{Diag}(ρ)⋅\dfrac{∂κ}{∂u₃}$ | Bidiagonal |
| $\dfrac{∂h_{tot,d}}{∂ρe_{tot}}$ | $\textrm{Diag}\biggl(\dfrac{1}{ρ}\biggr)⋅\biggl(I + \dfrac{∂p_d}{∂ρe_{tot}}\biggr)$ | Diagonal |
| $\dfrac{∂\mathcal{T}ρ}{∂uₕ}$ | $-\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)\bigr)⋅\dfrac{∂u³}{∂uₕ}$ | Tridiagonal |
| $\dfrac{∂\mathcal{T}ρ}{∂u₃}$ | $-\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)\bigr)⋅\dfrac{∂u³}{∂u₃}$ | Bidiagonal |
| $\dfrac{∂\mathcal{T}_{d,dif}ρe_{tot}}{∂ρe_{tot}}$ | $\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kₕ)\bigr)⋅\mathcal{M}\textrm{grad}ᶠ⋅\dfrac{∂h_{tot,d}}{∂ρe_{tot}}$ | Tridiagonal |
| $\dfrac{∂\mathcal{T}ρe_{tot}}{∂uₕ}$ | $-\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)*\textrm{int}ᶠ(h_{tot})\bigr)⋅\dfrac{∂u³}{∂uₕ}$ | Tridiagonal |
| $\dfrac{∂\mathcal{T}ρe_{tot}}{∂u₃}$ | $-\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)*\textrm{int}ᶠ(h_{tot})\bigr)⋅\dfrac{∂u³}{∂u₃}$ | Bidiagonal |
| $\dfrac{∂\mathcal{T}_{dif}ρχ}{∂ρχ}$ | $\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kₕ)\bigr)⋅\mathcal{M}\textrm{grad}ᶠ⋅\dfrac{∂χ}{∂ρχ}$ | Tridiagonal |
| $\dfrac{∂\mathcal{T}ρχ}{∂uₕ}$ | $-\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)*\textrm{int}ᶠ(χ)\bigr)⋅\dfrac{∂u³}{∂uₕ}$ | Tridiagonal |
| $\dfrac{∂\mathcal{T}ρχ}{∂u₃}$ | $-\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\text{wint}ᶠ(J, ρ)*\textrm{int}ᶠ(χ)\bigr)⋅\dfrac{∂u³}{∂u₃}$ | Bidiagonal |
| $\dfrac{∂\mathcal{T}uₕ}{∂uₕ}$ | $\textrm{Diag}\biggl(\dfrac{1}{ρ}\biggr)⋅\mathcal{M}\textrm{div}ᶜ⋅\textrm{Diag}\bigl(\textrm{int}ᶠ(ρ)*\textrm{int}ᶠ(Kᵤ)\bigr)⋅\mathcal{M}\textrm{grad}ᶠ$ | Tridiagonal |
| $\dfrac{∂\mathcal{T}_du₃}{∂ρ}$ | $-\textrm{Diag}\biggl(\dfrac{1}{\textrm{int}ᶠ(ρ)}\biggr)⋅\mathcal{M}\textrm{grad}ᶠ⋅\dfrac{∂p_d}{∂ρ} - \textrm{Diag}\biggl(\dfrac{1}{\textrm{int}ᶠ(ρ)^2}*\bigl(\textrm{grad}ᶠ(p - p_{ref}) - \textrm{int}ᶠ(ρ_{ref})*\textrm{grad}ᶠ(Φ)\bigr)\biggr)⋅\mathcal{M}\textrm{int}ᶠ$ | Bidiagonal |
| $\dfrac{∂\mathcal{T}_du₃}{∂ρe_{tot}}$ | $-\textrm{Diag}\biggl(\dfrac{1}{\textrm{int}ᶠ(ρ)}\biggr)⋅\mathcal{M}\textrm{grad}ᶠ⋅\dfrac{∂p_d}{∂ρe_{tot}}$ | Bidiagonal |
| $\dfrac{∂\mathcal{T}_du₃}{∂uₕ}$ | $-\textrm{Diag}\biggl(\dfrac{1}{\textrm{int}ᶠ(ρ)}\biggr)⋅\mathcal{M}\textrm{grad}ᶠ⋅\dfrac{∂p_d}{∂uₕ}$ | Bidiagonal |
| $\dfrac{∂\mathcal{T}_du₃}{∂u₃}$ | $-\textrm{Diag}\biggl(\dfrac{1}{\textrm{int}ᶠ(ρ)}\biggr)⋅\mathcal{M}\textrm{grad}ᶠ⋅\dfrac{∂p_d}{∂u₃}$ | Tridiagonal |

The approximations above can be summarized as
$$
\frac{∂\mathcal{T}Y}{∂Y} ≈ \begin{bmatrix}
\mathbf{0} & \mathbf{0} & \mathbf{0} & \dfrac{∂\mathcal{T}ρ}{∂uₕ} & \dfrac{∂\mathcal{T}ρ}{∂u₃} \\[1em]
\mathbf{0} & \dfrac{∂\mathcal{T}_{d,dif}ρe_{tot}}{∂ρe_{tot}} & \mathbf{0} & \dfrac{∂\mathcal{T}ρe_{tot}}{∂uₕ} & \dfrac{∂\mathcal{T}ρe_{tot}}{∂u₃} \\[1em]
\mathbf{0} & \mathbf{0} & \dfrac{∂\mathcal{T}_{dif}ρχ}{∂ρχ} & \dfrac{∂\mathcal{T}ρχ}{∂uₕ} & \dfrac{∂\mathcal{T}ρχ}{∂u₃} \\[1em]
\mathbf{0} & \mathbf{0} & \mathbf{0} & \dfrac{∂\mathcal{T}uₕ}{∂uₕ} & \mathbf{0} \\[1em]
\dfrac{∂\mathcal{T}_du₃}{∂ρ} & \dfrac{∂\mathcal{T}_du₃}{∂ρe_{tot}} & \mathbf{0} & \dfrac{∂\mathcal{T}_du₃}{∂uₕ} & \dfrac{∂\mathcal{T}u₃}{∂u₃}
\end{bmatrix}
$$
This gives us a Jacobian approximation with the following internal structure:
$$
\frac{∂E}{∂Y} = Δt*\dfrac{∂\mathcal{T}Y}{∂Y} - I ≈ \begin{bmatrix}
    \mathbb{D} & \mathbf{0} & \mathbf{0} & \mathbb{T} & \mathbb{B} \\[0.5em]
    \mathbf{0} & \mathbb{T} & \mathbf{0} & \mathbb{T} & \mathbb{B} \\[0.5em]
    \mathbf{0} & \mathbf{0} & \mathbb{T} & \mathbb{T} & \mathbb{B} \\[0.5em]
    \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbb{T} & \mathbf{0} \\[0.5em]
    \mathbb{B} & \mathbb{B} & \mathbf{0} & \mathbb{B} & \mathbb{T}
\end{bmatrix}
$$

### Description of linear solver
The Jacobian approximation can be subdivided into a 2×2 block matrix:
$$
\begin{bmatrix}A₁₁ & A₁₂ \\[1em] A₂₁ & A₂₂\end{bmatrix} = \begin{bmatrix}
    \begin{bmatrix}
        \mathbb{D} & \mathbf{0} & \mathbf{0} \\[0.5em]
        \mathbf{0} & \mathbb{T} & \mathbf{0} \\[0.5em]
        \mathbf{0} & \mathbf{0} & \mathbb{T} \\[0.5em]
    \end{bmatrix} &
    \begin{bmatrix}
        \mathbb{T} & \mathbb{B} \\[0.5em]
        \mathbb{T} & \mathbb{B} \\[0.5em]
        \mathbb{T} & \mathbb{B} \\[0.5em]
    \end{bmatrix} \\[3em]
    \begin{bmatrix}
        \mathbf{0} & \mathbf{0} & \mathbf{0} \\[0.5em]
        \mathbb{B} & \mathbb{B} & \mathbf{0}
    \end{bmatrix} &
    \begin{bmatrix}
        \mathbb{T} & \mathbf{0} \\[0.5em]
        \mathbb{B} & \mathbb{T}
    \end{bmatrix}
\end{bmatrix}
$$
Given the value of $δE$, we use this approximation to solve for $δY$ with a **Schur complement reduction solve**. This involves solving a "reduced" equation of the form
$$
\bigl(A₂₂ - A₂₁ * \textrm{inv}(A₁₁) * A₁₂\bigr) * δY₂ = δE₂
$$
We solve this linear system with a **stationary iterative solver**. After initializing $δY₂$ to $0$, the solver iteratively updates $δY₂$ based on the solution to yet another equation,
$$
\bigl(A₂₂ - A₂₁ * \textrm{inv}(P₁₁) * A₁₂\bigr) * δY₂ = δE₂
$$
The matrix $P₁₁$ is called a **preconditioner** of $A₁₁$. In order to solve the preconditioner equation quickly, we need $P₁₁$ to be a diagonal matrix. When that is the case, $\textrm{inv}(P₁₁)$ is also a diagonal matrix, which allows us to use an efficient block lower triangular solver that solves two tridiagonal equations in sequence.

In order for the higher-level iterative solver to converge in a small number of iterations, $\textrm{inv}(P₁₁)$ should be a close approximation of $\textrm{inv}(A₁₁)$. We are currently setting $P₁₁$ to the main diagonal of $A₁₁$ (also known as **Jacobi preconditioning**), which seems to work well for simple diffusion.

### Reasons for assumptions
- $\textbf{A}1$ allows us to ignore $\dfrac{∂\mathcal{T}ρe_{tot}}{∂ρ}$ and $\dfrac{∂\mathcal{T}ρχ}{∂ρ}$.

  If we had to use either of these derivatives, $A₁₁$ would not be a block diagonal matrix, and we would not be able to approximate $\textrm{inv}(A₁₁)$ using a diagonal matrix $P₁₁$.
- $\textbf{A}2$ allows us to ignore $\dfrac{∂\mathcal{T}uₕ}{∂ρ}$, and it also allows us to ignore $\dfrac{∂\mathcal{T}ρe_{tot}}{∂ρ}$, $\dfrac{∂\mathcal{T}ρχ}{∂ρ}$, and $\dfrac{∂u³}{∂ρ}$.

  If we had to use the first derivative, we would not be able to solve the preconditioner equation with a block lower triangular solver. If we had to use any of the other three derivatives, we would not be able to approximate $\textrm{inv}(A₁₁)$ using a diagonal matrix $P₁₁$.
- $\textbf{A}3$ allows us to ignore $\dfrac{∂h_{tot}}{∂u₃}$.

  If we had to use this derivative, one of the blocks in $A₁₂$ would change from a bidiagonal to a quaddiagonal matrix, which would require us to replace one of the tridiagonal solves with a slightly slower pentadiagonal solve.
- $\textbf{A}4$ allows us to ignore saturation adjustment, which is not continuously differentiable.