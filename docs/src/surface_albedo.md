# Ocean Surface Albedo

The ocean surface albedo is the fraction of solar radiation reflected by the ocean surface. It is a function of the solar zenith angle, the sea surface roughness (which depends on wind speed), and the wavelength of the incoming radiation.

Three methods are available to specify the ocean albedo, selected by the
`albedo_model` configuration argument:

## 1) `ConstantAlbedo`

A constant albedo, used for idealized experiments. In YAML configurations the
value comes from the ClimaParams key `idealized_ocean_albedo`, default 0.38
(following [OGorman2008](@cite)); the `AtmosModel` script-API default is 0.07.

## 2) `RegressionFunctionAlbedo`

This is an empirical parameterization of the direct and diffuse surface albedo of Jin et al. (2011) [Jin2011](@cite) (reflectivity of the inner ocean is ignored). The direct reflection is calculated using the Fresnel reflection at the air-sea interface due to the difference in refractive index between air and water. The current implementation uses the broadband representation (the relative refractive index $n$ is independent of wavelength). Its formulation is:

```math
α_{dir}(λ, μ, u) = r_{f}(n, μ) - \frac{r_{f}(n, μ)}{r_{f}(n_{0}, μ)} f(μ, σ(u))
```

where:

  - $λ$ is the wavelength (currently unused).
  - $μ$ is the cosine of the solar zenith angle.
  - $u$ is the surface wind speed.
  - $σ(u)$ is the mean wave slope distribution width following [CoxMunk1954](@cite), with $\sigma^2 = 0.003 + 0.00512u$.
  - $r_{f}(n, μ)$ is the Fresnel reflectance (e.g., see [Warren2019](@cite)):

```math
    r_{f_p}(n, θ) = \left(\frac{n^2 \cos(θ) - \sqrt{n^2 - \sin^2(θ)}}{n^2 \cos(θ) + \sqrt{n^2 - \sin^2(θ)}}\right)^2
```

```math
    r_{f_s}(n, θ) = \left(\frac{\cos(θ) - \sqrt{n^2 - \sin^2(θ)}}{\cos(θ) + \sqrt{n^2 - \sin^2(θ)}}\right)^2
```

where we assume an equal contribution from the p and s polarizations, so that $r_{f}(n, θ) = 0.5(r_{f_p}(n, θ) + r_{f_s}(n, θ))$, and the perfect dielectric medium approximation.

  - $n_0=1.34$ is the refractive index of water for visible light.
  - $n$ is the relative refractive index of water and air ($n = n_w/n_a$), and is assumed to be equal to $n_0$ for the broadband representation.
  - $f(μ, σ)$ is the regression function, defined as:

```math
f(μ, σ) = (p_1 + p_2μ + p_3μ^2 + p_4μ^3 + p_5σ + p_6σμ)  \exp(p_7 + p_8μ + p_9μ^2 + p_{10}σ + p_{11}σμ)
```

where the coefficients are given in the table below:

| Coefficient | Value   |
|:----------- |:------- |
| $p_1$       | 0.0152  |
| $p_2$       | -1.7873 |
| $p_3$       | 6.8972  |
| $p_4$       | -8.5778 |
| $p_5$       | 4.071   |
| $p_6$       | -7.6446 |
| $p_7$       | 0.1643  |
| $p_8$       | -7.8409 |
| $p_9$       | -3.5639 |
| $p_{10}$    | -2.3588 |
| $p_{11}$    | 10.0538 |

Diffuse reflection depends on atmospheric conditions: for clear sky it follows from nearly isotropic Rayleigh scattering, and for cloudy sky the albedo is adjusted for the presence of clouds. The formulations are:

```math
α_{diff}(λ, μ, u) = -0.1482 - 0.012σ(u) + 0.1608n - 0.0244nσ(u)
```

for clear sky, and

```math
α_{diff}(λ, μ, u) = -0.1479 + 0.1502n - 0.016nσ(u)
```

for cloudy sky. In the current implementation we assume clear skies everywhere.

In the code, both the direct and the diffuse albedo are clamped to the
interval ``[0, 1]``, since the diffuse regression can otherwise go negative,
and both return zero at night (``μ \le 0``).

## 3) `CouplerAlbedo`

This informs ClimaAtmos that the coupler sets the albedo.

## Comparison of `RegressionFunctionAlbedo` with Jin et al. (2011)

```@example
include("surface_albedo_jin11_plots.jl")
```

  - direct albedo (compare with Fig. 2)
    ![](assets/direct_albedo_fig2.png)

  - diffuse albedo (compare with Fig. 4)
    ![](assets/diffuse_albedo_fig4.png)
