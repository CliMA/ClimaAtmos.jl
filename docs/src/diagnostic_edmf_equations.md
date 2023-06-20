# Diagnostic EDMF equations

This describes the diagnostic EDMF scheme equations and its discretizations. Where possible, we use a coordinate invariant form: the ClimaCore operators generally handle the conversions between bases internally.


## Grid-scale variables

* ``\rho``: _density_ in kg/m³, discretized at cell centers.
* ``u^3``: the contravariant 3 component of velocity, discretized at cell faces.
* ``h_{\mathrm{tot}}``: total enthalpy in J/kg, discretized at cell centers.
* ``q_t``: total specific humidity in kg/kg, discretized at cell centers.
* ``\Phi = g z``: geopotential in m²/s², where ``g`` is the gravitational acceleration rate and ``z`` is altitude above the mean sea level, discretized at cell centers.
* ``(\nabla \Phi)^3``: the contravariant 3 component of the gradient of geopotential, reconstructed at cell centers.
* ``p``: air pressure, reconstructed at cell centers.

## Subgrid-scale variables
* ``\hat{\rho}^j``: _effective density_ in kg/m³. Superscript ``j`` represents the sub-domain. ``\hat{\rho}^j = \rho^j a^j`` where ``\rho^j`` is the sub-domain density and ``a^j`` is the sub-domain area fraction. This is discretized at cell centers.
* ``\rho^j``: _density_ in kg/m³, derived from the thermodynamic state, reconstructed at cell centers.
* ``u^{j,3}``: the contravariant 3 component of velocity, discretized at cell faces.
* ``h_{\mathrm{tot}}^j``: total enthalpy, discretized at cell centers.
* ``q_t^j``: total specific humidity of the sub-domain j, discretized at cell centers.
* ``u^{0,3}``: the contravariant 3 component of the environmental velocity, obtained as the residual:
  ```math
  \rho u^{0, 3} = \rho u^3 -  \sum_{j\ne 0} \hat\rho^j u^{j, 3}.
  ```

## Equations and discretizations

### Mass

```math
\frac{1}{J} \frac{\partial}{\partial \xi^3} \bigl( \hat\rho^j J u^{j, 3} \bigr)
= (E^{j0} - \Delta^{j0}) \hat\rho^j
```

This is descritized using the following
```math
\frac{1}{J[i-1]} \left( J[i-\frac{1}{2}] \hat\rho^j[i] u^{(j), 3}[i-\frac{1}{2}] -J[i-\frac{3}{2}] \hat\rho^j[i-1] u^{j, 3}[i-\frac{3}{2}] \right)
= (E^{j0}[i-1] - \Delta^{j0}[i-1]) \hat\rho^j[i-1]
```

### Momentum
```math
\frac{1}{J^2} \frac{\partial}{\partial \xi^3}  \bigl(\frac{1}{2} J^2 (u^{j, 3})^2 \bigr)
= - g^{3l} \left( \frac{\rho^j-\rho}{\rho} \frac{\partial}{\partial \xi^l}  \Phi\right) + E^{j0}(u^{0,3} - u^{j,3})
```

This is descritized using the following
```math
\frac{1}{2} \frac{1}{J[i-1]^2} \left( J[i-\frac{1}{2}]^2 u^{j, 3}[i-\frac{1}{2}]^2 -J[i-\frac{3}{2}] u^{j, 3}[i-\frac{3}{2}]^2 \right)
= - \frac{\rho^{j}[i-1]-\rho[i-1]}{\rho[i-1]} \nabla^3 \Phi + E^{j0}[i-1](u^{0, 3}[i-\frac{3}{2}] - u^{j, 3}[i-\frac{3}{2}])
```
    
### Total energy
```math
\frac{1}{J} \frac{\partial}{\partial \xi^3} ( \hat\rho^j J h_{\mathrm{tot}}^j u^{j, 3} )
= \hat\rho^j \left(E^{j0} h_{\mathrm{tot}} - \Delta^{j0} h_{\mathrm{tot}}^j\right)
```

This is descritized using the following
```math
\frac{1}{J[i-1]} \left( J[i-\frac{1}{2}] \hat\rho^j[i] u^{j, 3}[i-\frac{1}{2}] h_{\mathrm{tot}}^j[i] -J[i-\frac{3}{2}] \hat\rho^j[i-1] u^{j, 3}[i-\frac{3}{2}] h_{\mathrm{tot}}^j[i-1] \right)
= \hat\rho^j[i-1] (E^{(j0)}[i-1] h_{\mathrm{tot}}[i-1]  - \Delta^{(j0)}[i-1] h_{\mathrm{tot}}^j[i-1])
```

### Total water
```math
\frac{1}{J} \frac{\partial}{\partial \xi^3} \bigl(\hat\rho^j J q_t^j (u^{j, 3} - W_t^j \hat k^3) \bigr)
= \hat\rho^j \left(E^{j0} q_t - \Delta^{j0} q_t^j\right)
```

This is descritized using the following
```math
\frac{1}{J[i-1]} \left( J[i-\frac{1}{2}] \hat\rho^j[i] u^{j, 3}[i-\frac{1}{2}] q_t^j[i] -J[i-\frac{3}{2}] \hat\rho^j[i-1] u^{j, 3}[i-\frac{3}{2}] q_t^j[i-1] \right)
= \hat\rho^j[i-1] (E^{j0}[i-1] q_t[i-1]  - \Delta^{j0}[i-1] q_t^j[i-1])
```