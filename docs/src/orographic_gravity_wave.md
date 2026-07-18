# Orographic Gravity Wave Parameterization

Gravity waves have a great impact on the atmospheric circulation. They are usually generated from topography or convection, propagate upward and alter temperature and winds in the middle atmosphere, and influence tropospheric circulation through 'downward control' of the residual circulation by stratospheric wave forcing. The horizontal wavelength for gravity waves ranges from tens to thousands of kilometers, and the shorter end of this range up to a few hundred kilometers is unresolved at typical GCM resolution and must be parameterized.

The gravity wave drag on the wind velocities ($\overline{\vec{v}}=(u,v)$) are

```math
\frac{\partial \overline{\vec{v}}}{ \partial t} = ... + \underbrace{\left(-\frac{\partial \overline{\vec{v}'w'}}{\partial z}\Big|_{GW}\right)}_{\vec{X}}
```

with $\vec{X} = (X_\lambda, X_\phi)$ representing the subgrid-scale zonal and meridional components of the gravity wave drag and is calculated with the parameterization.

Throughout this document, an overbar (e.g. $\overline{\vec{v}}$, $\overline{\rho}$) denotes a grid-mean, resolved-scale variable in the sense of a Reynolds decomposition, and primes denote the corresponding subgrid fluctuations. $\epsilon_0$ denotes a small floating-point regularizer (`eps(FT)` in the code) used to prevent division-by-zero and degenerate clamps. It has no physical meaning.

## Orographic gravity wave

The orographic gravity wave drag parameterization follows the methods described in [garner2005](@cite). The momentum drag from subgrid-scale mountains is divided into a non-propagating component and a propagating component. The non-propagating (blocked/deflected flow) component deposits momentum drag in a low-level layer extending from the planetary boundary layer top up to a reference level $z_{\mathrm{ref}}$ (with pressure weighting that concentrates the drag near the surface), while the propagating component generates a stationary ($c = 0$, zero phase speed) gravity wave which propagates upwards and deposits momentum flux to the layers where it breaks.

### Planetary Boundary Layer (PBL) top

There are many ways to determine the PBL top. We implement the following simple criteria to find the PBL top level $k$ as the highest level that satisfies

```math
{}^cp[k] \ge 0.5 ∗ {}^cp[1]
```

and

```math
{}^cT[1] + T_{\mathrm{boost}} - {}^cT[k] > g/c_{pd} * ( {}^cz[k] - {}^cz[1] )
```

where the superscript $c$ represents cell centers in the vertical stencils, and $[1]$ denotes the first (lowest) cell center level. $T_{\mathrm{boost}} = 1.5\,\mathrm{K}$ is the surface temperature boost to improve PBL height estimate.

The first condition restricts the search to the lower atmosphere (below roughly 500 hPa). The second condition compares the observed temperature decrease with height against the dry adiabatic lapse rate $g/c_{pd}$: within the well-mixed boundary layer, turbulent mixing keeps the temperature profile close to (or steeper than) the dry adiabat, so the inequality holds. Above the PBL top, the free atmosphere is stably stratified (temperature decreases more slowly than the dry adiabat) and the inequality fails, marking the transition from the convectively mixed layer to the stable free atmosphere above.

### Orographic information

The orographic information is what is required in generating the base momentum flux for low-level flow encountering the subgrid-scale mountains. We compute the tensor $\textbf{T}$ and the scalar $h_{\mathrm{max}}$ from the Earth elevation data (GFDL code [here](https://github.com/NOAA-GFDL/atmos_phys/blob/main/atmos_param/topo_drag/topo_drag.F90), which requires [Caltech Box access](https://caltech.box.com/s/w4szffattzofarpwyv9rmm5s77jgo3o4)).

#### Velocity potential $\chi$

The velocity potential $\chi$ captures the far-field influence of surrounding topography on the low-level flow at each grid point. It is computed as a 2D Hilbert transform of the surface elevation, smoothed by a Blackman window to suppress contributions from distant terrain beyond the smoothing scale:

```math
\chi(\mathbf{x}) = \frac{R_{\mathrm{earth}}}{2\pi} \int \int \frac{\cos\phi' \; h(\mathbf{x}')}{d(\mathbf{x}, \mathbf{x}')} \; W(d) \; d\lambda' \; d\phi',
```

where $h$ is the surface elevation, $d(\mathbf{x}, \mathbf{x}')$ is the great-circle arc distance, $\phi'$ is latitude, and $W(d)$ is a Blackman window taper that goes to zero at the smoothing scale. The $1/d$ kernel means nearby terrain dominates, while the window prevents spurious contributions from the far field. A singularity correction is added for the grid cell containing $\mathbf{x}$ itself. This follows the GFDL Fortran implementation (`get_velpot.f90`).

#### Ororgraphic tensor $\textbf{T}$

The tensor $\textbf{T}$, which contains all relevant information including amplitude, variance, orientation, and anisotropy about topography, is computed as

```math
\textbf{T} = -\nabla \chi (\nabla h)^T,
```

where $h$ is the earth elevation and $\chi$ is the velocity potential defined above. The leading minus reproduces Garner's sign convention so that the resulting drag opposes the low-level flow (cf. the velocity potential in [garner2005](@cite) Eq. 6).

!!! note "Indexing convention used in the code"

    The four tensor components stored in the code are

    ```math
    t_{ij} \;:=\; -\frac{\partial \chi}{\partial x_j}\,\frac{\partial h}{\partial x_i}, \qquad (x_1, x_2) = (x, y),
    ```

    so the stored $t_{ij}$ equals the $(j,i)$ entry of $-\nabla\chi(\nabla h)^T$. The transpose $\langle\textbf{T}\rangle^T$ in the base-flux formula below accounts for this convention. The explicit $\tau_x$/$\tau_y$ formulas in the next section use the stored components directly.

#### Effective maximum height $h_{\mathrm{max}}$

$h_{\mathrm{max}}$ represents the effective maximum height of the orographic features within a grid cell relative to the mean surface. Using the 4th moment emphasizes the tallest peaks most relevant for generating gravity waves while still averaging over the subgrid terrain.

The computation proceeds in two steps. First, a raw height statistic $h_0$ is computed from the distance-weighted 4th moment of subgrid elevation deviations,

```math
h_0 = \left( \frac{\sum_i w_i (h_i - \bar{h})^4}{\sum_i w_i} \right)^{1/4},
```

where $\bar{h}$ is the distance-weighted local mean elevation, and the weights use a Lorentzian kernel $w_i = 1/(1 + d_i^2/s^2)$ with $d_i$ the angular arc distance and $s$ a latitude-dependent smoothing scale.

Second, $h_0$ is rescaled using the shape parameter $\gamma$ and the height fraction $h_{\mathrm{frac}}$ to obtain the final $h_{\mathrm{max}}$ and $h_{\mathrm{min}}$:

```math
h_{\mathrm{max}} = \left( h_0^{2-\gamma} \cdot \frac{\gamma+2}{2\gamma} \cdot \frac{1 - h_{\mathrm{frac}}^{2\gamma}}{1 - h_{\mathrm{frac}}^{\gamma+2}} \right)^{1/(2-\gamma)}, \quad h_{\mathrm{min}} = h_{\mathrm{frac}} \cdot h_{\mathrm{max}}.
```

This rescaling adjusts the raw statistic to account for the assumed power-law distribution of mountain heights within the grid cell (cf. the Appendix in [garner2005](@cite)). Note that $h_{\mathrm{frac}}$ is a ClimaParams tuning parameter (`ogw_critical_height_threshold`).

### Base flux

The base momentum flux generated is computed and divided into the propagating and non-propagating components.

Let the subscript $(\cdot)_{\text{pbl}}$ denote a property of the low-level flow evaluated at the source level: in the code, this is the value at the highest cell center whose height satisfies ${}^c z[k] \le z_{\text{pbl}}$ (i.e., the cell center at or just below the PBL top). Let $V_{\text{pbl}} = (u_{\text{pbl}}, v_{\text{pbl}})$, $N_{\text{pbl}}$, and $\rho_{\text{pbl}}$ denote the horizontal wind, buoyancy frequency, and density at this source level. $N$ is computed pointwise at every cell center as

```math
N^2 = \frac{g}{T} \left( \frac{dT}{dz} + \frac{g}{c_{pm}} \right),
```

where $c_{pm}$ is the moist isobaric specific heat. $N_{\text{pbl}}$ is then $N$ evaluated at the source level.

The base flux is computed as the linear drag following

```math
\tau = \rho_{\text{pbl}}\, N_{\text{pbl}}\, \langle \textbf{T} \rangle ^T V_{\text{pbl}},
```

where $\langle \textbf{T} \rangle = [t_{11}, t_{12}; t_{21}, t_{22}]$ is the tensor that contains orographic information. In the code, we compute the zonal and meridional components separately as

```math
\tau_x = \rho_{\text{pbl}}\, N_{\text{pbl}}\, (t_{11}\, u_{\text{pbl}} + t_{21}\, v_{\text{pbl}}),
```

```math
\tau_y = \rho_{\text{pbl}}\, N_{\text{pbl}}\, (t_{12}\, u_{\text{pbl}} + t_{22}\, v_{\text{pbl}}).
```

The base flux is then corrected using Froude number and saturation velocity. Let

```math
V_{\tau} = \max\!\left(\epsilon_0, \; - V_{\text{pbl}} \cdot \frac{\tau}{\max(\epsilon_0, |\tau|)}\right),
```

and given the orographic information $h_{\mathrm{max}}$ and $h_{\mathrm{min}}$, the max and min Froude numbers are computed as

```math
\mathrm{Fr}_{\mathrm{max}} = \max(0, h_{\mathrm{max}}) \,\frac{N_{\text{pbl}}}{V_{\tau}},
```

```math
\mathrm{Fr}_{\mathrm{min}} = \max(0, h_{\mathrm{min}}) \,\frac{N_{\text{pbl}}}{V_{\tau}}.
```

The clamps to $\max(0, \cdot)$ on $h_{\mathrm{max}}$ and $h_{\mathrm{min}}$ treat negative height as zero. Here, $\mathrm{Fr}_{\mathrm{crit}} = 0.7$ is the critical Froude number for nonlinear flow. $\mathrm{Fr}_{\mathrm{crit}}$ acts as the primary tuning lever for partitioning drag between the low-level blocked flow and the upper-level wave breaking.

The saturation velocity is computed as

```math
U_{\mathrm{sat}} = \sqrt{\frac{\rho_{\text{pbl}}}{\rho_0} \frac{V_{\pmb{\tau}}^3}{N_{\text{pbl}}\, L_0}},
```

where $\rho_0 = 1.2\,\mathrm{kg/m^3}$ is the arbitrary density scale, and $L_0 = 80\,000\,\mathrm{m}$ is the arbitrary horizontal length scale.

$U_{\mathrm{sat}}$ roughly measures the biggest wave the flow can carry before it breaks. A stronger wind ($V_\tau$) makes much bigger waves (note the cube), while stronger stratification ($N$) or a wider mountain ($L_0$) makes them break sooner and shrinks it.

The following set of intermediate variables ($\mathrm{Fr}U$'s) are computed to correct the linear base flux:

```math
\mathrm{Fr}U_{\mathrm{sat}} = \mathrm{Fr}_{\mathrm{crit}} * U_{\mathrm{sat}},
```

```math
\mathrm{Fr}U_{\mathrm{min}} = \mathrm{Fr}_{\mathrm{min}} * U_{\mathrm{sat}},
```

```math
\mathrm{Fr}U_{\mathrm{max}} = \max(\mathrm{Fr}_{\mathrm{max}} * U_{\mathrm{sat}}, \mathrm{Fr}U_{\mathrm{min}} + \epsilon_0),
```

```math
\mathrm{Fr}U_{\mathrm{clp}} = \min(\mathrm{Fr}U_{\mathrm{max}}, \max(\mathrm{Fr}U_{\mathrm{min}}, \mathrm{Fr}U_{\mathrm{sat}})).
```

Now the correct linear drag is computed as

```math
\tau_l = \frac{\mathrm{Fr}U_{\mathrm{max}}^{2+\gamma-\epsilon} - \mathrm{Fr}U_{\mathrm{min}}^{2+\gamma-\epsilon}}{2+\gamma-\epsilon},
```

and the propagating and non-propagating parts of the drag are computed as

```math
\tau_p = a_0 \left[ \frac{\mathrm{Fr}U_{\mathrm{clp}}^{2+\gamma-\epsilon} - \mathrm{Fr}U_{\mathrm{min}}^{2+\gamma-\epsilon}}{2+\gamma-\epsilon} + \mathrm{Fr}U_{\mathrm{sat}}^{\beta+2} \frac{\mathrm{Fr}U_{\mathrm{max}}^{\gamma-\epsilon-\beta} - \mathrm{Fr}U_{\mathrm{clp}}^{\gamma-\epsilon-\beta}}{\gamma-\epsilon-\beta} \right],
```

```math
\tau_{np} = a_1 \frac{U_{\mathrm{sat}}}{1+\beta} \left[ \frac{\mathrm{Fr}U_{\mathrm{max}}^{1+\gamma-\epsilon} - \mathrm{Fr}U_{\mathrm{clp}}^{1+\gamma-\epsilon}}{1+\gamma-\epsilon} - \mathrm{Fr}U_{\mathrm{sat}}^{\beta+1} \frac{\mathrm{Fr}U_{\mathrm{max}}^{\gamma-\epsilon-\beta} - \mathrm{Fr}U_{\mathrm{clp}}^{\gamma-\epsilon-\beta}}{\gamma-\epsilon-\beta} \right].
```

These two expressions are slices of the same integral over the assumed power-law distribution of subgrid obstacle heights, partitioned at the saturation point $\mathrm{Fr}U_{\mathrm{clp}}$. Obstacles below saturation launch linear waves whose drag grows as $\mathrm{Fr}^2$ (the propagating $\tau_p$). For taller obstacles, the wave amplitude saturates (capped at $\mathrm{Fr}U_{\mathrm{sat}}$) and the excess blocks the low-level flow, which is forced around the obstacle as form drag (the non-propagating $\tau_{np}$).

The non-propagating drag is then scaled by the Froude number:

```math
\tau_{np} = \frac{\tau_{np}}{\max(\mathrm{Fr}_{\mathrm{crit}}, \mathrm{Fr}_{\mathrm{max}})}.
```

Taller mountains (larger $\mathrm{Fr}_{\mathrm{max}}$) block a deeper layer of air, so the same blocking drag is spread over more depth: dividing by $\mathrm{Fr}_{\mathrm{max}}$ gives the drag per unit depth. The $\mathrm{Fr}_{\mathrm{crit}}$ floor just stops the division from blowing up for short mountains, which do no blocking anyway ($\tau_{np}=0$ there).

Here, $(\gamma, \epsilon, \beta) = (0.4, 0.0, 0.5)$ are empirical shape parameters constrained by observations [garner2005](@cite), set by the ClimaParams keys `ogw_mountain_height_width_exponent`, `ogw_number_density_exponent`, and `ogw_mountain_shape_parameter` respectively. Specifically, $\gamma=0.4$ is derived from the observed scaling of mountain width versus height. The parameters $a_0=0.9$ and $a_1=3.0$ by default and are ClimaParams `ogw_linear_drag_coefficient` and `ogw_nonlinear_drag_coefficient` respectively.

### Saturation profiles for the propagating component

Only the propagating component requires a saturation profile, as it carries a vertically propagating wave whose flux can saturate aloft. The non-propagating component remains the scalar $\tau_{np}$ from the base-flux calculation and is distributed in $z$ by pressure weighting across the blocking layer $[z_{\text{pbl}}, z_{\mathrm{ref}})$ (see [Non-propagating component](#non-propagating-component) below).

The vertical profiles of saturated momentum flux $\tau_{\mathrm{sat}}$ is computed such that momentum forcing can be obtained for $d\overline{V}/dt = -\overline{\rho}^{-1}d\tau_{\mathrm{sat}}/dz$. This only applies to the propagating part.

Similar to the base flux calculation but for the 3D fields, we compute $V_\tau$ at cell centers as

```math
{}^c V_{\tau}[k] = \max\!\left(\epsilon_0, \; - V[k] \cdot \frac{\tau}{\max(\epsilon_0, |\tau|)}\right).
```

Let $L_1 = L_0 * \max(0.5, \min(2.0, 1.0 - 2 V_\tau \cdot d^2V_{\tau} / N^2))$ where the factor of 2 is a correction for coarse sampling of $d^2V/dz^2$, and

```math
{}^c d^2V_{\tau}[k] = \max\!\left(\epsilon_0, \; - \frac{d^2 V}{dz^2}[k] \cdot \frac{\tau}{\max(\epsilon_0, |\tau|)}\right).
```

The saturated velocity $U_{\mathrm{sat}}$ is computed level-by-level via a cumulative column-accumulator (initialized at the lowest level, i.e., the surface cell center, with the base-flux $U_{\mathrm{sat}}$):

```math
U_{\mathrm{sat}}[k] = \min\!\left(U_{\mathrm{sat}}[k-1], \; \sqrt{\frac{{}^c \rho}{\rho_0} \frac{{}^c V_{\pmb{\tau}}^3}{{}^c N \cdot L_1} }\right),
```

so $U_{\mathrm{sat}}$ is monotonically non-increasing with height.

Intuitively, these three equations build up the saturation ceiling level by level. First, $V_\tau$ is the wind component that actually pushes against the wave (the flow projected onto the drag direction), so it measures how hard the flow drives the wave at each height. Second, $L_1$ is an effective mountain width: the local flow curvature ($d^2 V_\tau$) stretches or shrinks it, which tunes how readily the wave saturates. Third, $U_{\mathrm{sat}}$ is the largest wave amplitude the background flow can hold before the wave gets too steep and breaks: the local conditions ($\rho$, $V_\tau$, $N$, $L_1$) set a ceiling at each level, and the $\min$ with the level below enforces that once the wave has been clipped, it stays clipped going up (a wave can lose flux to breaking but never regain it).

The $\mathrm{Fr}U_{\mathrm{min}}$ and $\mathrm{Fr}U_{\mathrm{max}}$ are inherited from the base flux calculation. The source level $\mathrm{Fr}U_{\mathrm{sat}}$ and $\mathrm{Fr}U_{\mathrm{clp}}$ are then

```math
\mathrm{Fr}U_{\mathrm{sat0}} = \mathrm{Fr}U_{\mathrm{sat}},
```

```math
\mathrm{Fr}U_{\mathrm{clp0}} = \mathrm{Fr}U_{\mathrm{clp}},
```

and they may be updated as

```math
\mathrm{Fr}U_{\mathrm{sat}} = \mathrm{Fr}_{\mathrm{crit}} * U_{\mathrm{sat}},
```

```math
\mathrm{Fr}U_{\mathrm{clp}} = \min(\mathrm{Fr}U_{\mathrm{max}}, \max(\mathrm{Fr}U_{\mathrm{min}}, \mathrm{Fr}U_{\mathrm{sat}})).
```

$\mathrm{Fr}U_{\mathrm{sat}}$ is the "breaking line" at the current level: any part of the wave above it is too steep and gets shaved off. Because $U_{\mathrm{sat}}$ only decreases with height, this line decreases as the wave climbs, so more of the wave is removed the higher it goes. $\mathrm{Fr}U_{\mathrm{clp}}$ is just that breaking line clamped to stay within the obstacle range $[\mathrm{Fr}U_{\mathrm{min}}, \mathrm{Fr}U_{\mathrm{max}}]$, and the $(\cdot)_0$ versions remember these values at the launch level.

Then, the saturated profile of propagating component of the momentum flux **above the source level** is

```math
\tau_{\mathrm{sat}}[k] = a_0 \left[ \frac{\mathrm{Fr}U_{\mathrm{clp}}^{2+\gamma-\epsilon} - \mathrm{Fr}U_{\mathrm{min}}^{2+\gamma-\epsilon}}{2+\gamma-\epsilon} + \mathrm{Fr}U_{\mathrm{sat}}^2 \mathrm{Fr}U_{\mathrm{sat0}}^\beta \frac{\mathrm{Fr}U_{\mathrm{max}}^{\gamma-\epsilon-\beta} - \mathrm{Fr}U_{\mathrm{clp0}}^{\gamma-\epsilon-\beta}}{\gamma-\epsilon-\beta} + \mathrm{Fr}U_{\mathrm{sat}}^2 \frac{\mathrm{Fr}U_{\mathrm{clp0}}^{\gamma-\epsilon} - \mathrm{Fr}U_{\mathrm{clp}}^{\gamma-\epsilon}}{\gamma-\epsilon} \right].
```

This is the same obstacle-distribution integral as the base-level $\tau_p$, but re-evaluated with the lowered breaking line at level $k$: it encodes how much wave flux still survives at that height. As the wave climbs the breaking line drops, so $\tau_{\mathrm{sat}}$ shrinks with height, and that loss of flux (its vertical gradient $d\tau_{\mathrm{sat}}/dz$) is exactly the momentum the breaking wave dumps back into the flow.

**Below the source level** (${}^c z[k] \le z_{\text{pbl}}$), the saturated flux is set to the source-level propagating drag, $\tau_{\mathrm{sat}}[k] = \tau_p$ (so its vertical derivative vanishes inside the PBL and only the values above $z_{\text{pbl}}$ produce a tendency).

If the wave does not break and propagates all the way up to the model top (diagnosed by $\tau_{\mathrm{sat}}[\mathrm{end}] > 0$), the residual momentum carried by this part is redistributed throughout the column weighted by pressure to conserve momentum:

```math
\tau_{\mathrm{sat}}[k] = \tau_{\mathrm{sat}}[k] - \tau_{\mathrm{sat}}[\mathrm{end}] \, \frac{{}^cp[1] - {}^cp[k]}{{}^cp[1] - {}^cp[\mathrm{end}]}.
```

The redistribution is skipped when $\tau_{\mathrm{sat}}[\mathrm{end}] \le 0$.

### Velocity tendencies due to the orographic drag

#### Propagating component

The forcing from the propagating part on the zonal and meridional wind are

```math
{}^c \left( \frac{du}{dt} \right) _{p} = - \frac{1}{{}^c \rho} \frac{\tau_x}{\tau_l} \frac{d\tau_{\mathrm{sat}}}{dz},
```

```math
{}^c \left( \frac{dv}{dt} \right) _{p} = - \frac{1}{{}^c \rho} \frac{\tau_y}{\tau_l} \frac{d\tau_{\mathrm{sat}}}{dz}.
```

Here, $(\tau_x, \tau_y, \tau_l)$ is computed in the base flux calculation, and $\tau_{\mathrm{sat}}$ is calculated in the saturation flux profile. The propagating tendency vanishes below the source level (since $\tau_{\mathrm{sat}} = \tau_p$ is constant there, so $d\tau_{\mathrm{sat}}/dz = 0$) and contributes only at and above $z_{\text{pbl}}$.

#### Non-propagating component

The non-propagating drag is confined to a finite layer above the PBL top, bounded above by a reference level $z_{\mathrm{ref}}$. To locate $z_{\mathrm{ref}}$, we iterate over face levels above $z_{\text{pbl}}$ and accumulate a wave phase

```math
\mathrm{phase} \mathrel{+}= ({}^fz[k] - z_{\text{pbl}}) \cdot \frac{\max(N_{\mathrm{min}}, \min(N_{\mathrm{max}}, {}^fN[k]))}{\max(vvmin, {}^fV_{\tau}[k])},
```

setting $z_{\mathrm{ref}} = {}^fz[k]$ at the first face where $\mathrm{phase} > \pi$. Here $N_{\mathrm{min}} = 0.7 \times 10^{-2}\,\mathrm{s^{-1}}$, $N_{\mathrm{max}} = 1.7 \times 10^{-2}\,\mathrm{s^{-1}}$, $vvmin = 1.0\,\mathrm{m/s}$, and $({}^fN, {}^fV_{\tau})$ are obtained from the saturation-profile calculation. If $\mathrm{phase}$ never exceeds $\pi$ in the column, $z_{\mathrm{ref}}$ falls back to the model top. Note that the pressure weighting below still concentrates most of the drag near the surface.

Here, $N / V_\tau$ is the vertical wavenumber of a stationary hydrostatic gravity wave, so summing it over height accumulates the wave's vertical phase. Reaching $\pi$ means the flow has turned through half a vertical wavelength. This marks the top of the blocked/deflected low-level layer, which sets $z_{\mathrm{ref}}$. Half a wavelength is the natural cutoff because the wave wiggles up and down as $\sin(mz)$: over the first half-wavelength it makes one single bump, which is the air piling up and overturning right against the mountain. Past that point the wave flips to the other sign and starts behaving like the free wave traveling upward, not the air trapped and blocked near the surface, so that is where the blocked layer ends. The clamps on $N$ and $V_\tau$ just keep the wavenumber physical so the layer depth stays sensible.

The drag forcing due to non-propagating component is confined within the PBL top to $z_{\mathrm{ref}}$ and is weighted by pressure. The weights at each cell center are computed by interpolating face-level pressure differences:

```math
\mathrm{weight}[k] = \overline{{}^f p - {}^f p_{\mathrm{ref}}}^c,
```

where ${}^f p_{\mathrm{ref}}$ is the face pressure at $z_{\mathrm{ref}}$, and the overbar denotes interpolation to cell centers. The pressure layer thickness is similarly interpolated:

```math
\mathrm{diff}[k] = \overline{{}^f p[k-1] - {}^f p[k]}^c,
```

and the sum of the weights is

```math
wtsum = \sum_{k \in \mathrm{mask}} \frac{\mathrm{diff}[k]}{\mathrm{weight}[k]}.
```

The mask selects cells that overlap with the interval $[z_{\text{pbl}}, z_{\mathrm{ref}})$ and have nonzero weights.

For masked levels, the forcing due to non-propagating component is

```math
{}^c \left( \frac{du}{dt} [k] \right)_{np} = g \frac{\tau_{np}}{\tau_l} \frac{\mathrm{weight}[k]}{wtsum} \tau_x,
```

```math
{}^c \left( \frac{dv}{dt} [k] \right)_{np} = g \frac{\tau_{np}}{\tau_l} \frac{\mathrm{weight}[k]}{wtsum} \tau_y,
```

where $(\tau_x, \tau_y, \tau_l, \tau_{np})$ is computed in the base flux calculation. When the mask is empty ($wtsum = 0$, i.e. no cells overlap $[z_{\text{pbl}}, z_{\mathrm{ref}})$), the non-propagating forcing is set to zero in that column to avoid division by zero.

### Constrain the forcings

Total drag from both components are

```math
{}^c \left( \frac{du}{dt} [k] \right)_{\tau} = {}^c \left( \frac{du}{dt} [k] \right)_{p} + {}^c \left( \frac{du}{dt} [k] \right)_{np},
```

```math
{}^c \left( \frac{dv}{dt} [k] \right)_{\tau} = {}^c \left( \frac{dv}{dt} [k] \right)_{p} + {}^c \left( \frac{dv}{dt} [k] \right)_{np}.
```

To avoid instability due to large tendencies from the forcing, we constrain the forcing magnitude with $\epsilon_V = 3e-3$, and let

```math
{}^c \left( \frac{du}{dt} [k] \right)_{\tau} = \max(-\epsilon_V, \min(\epsilon_V, {}^c \left( \frac{du}{dt} [k] \right)_{\pmb{\tau}})),
```

```math
{}^c \left( \frac{dv}{dt} [k] \right)_{\tau} = \max(-\epsilon_V, \min(\epsilon_V, {}^c \left( \frac{dv}{dt} [k] \right)_{\pmb{\tau}})).
```

The tendencies above act on the physical horizontal wind components.

## Implementation Summary

The parameterization splits into an offline preprocessing step (Earth topography only) that builds an HDF5 artifact of $(h_{\mathrm{max}}, h_{\mathrm{min}}, t_{11}, t_{12}, t_{21}, t_{22})$ on the spectral element grid, and a runtime step that consumes the artifact via a `dt_ogw` callback and applies the cached forcing every integrator step.

```
Offline (Earth topography only):
  compute_OGW_info
    ├─ calc_hpoz_latlon         → h₀ (raw 4th-moment statistic; rescaled to hmax, hmin in compute_OGW_info)
    ├─ calc_velocity_potential  → χ  (2D Hilbert transform)
    └─ calc_orographic_tensor   → t11, t12, t21, t22
  regrid_OGW_info → SpaceVaryingInput to spectral element
  write_computed_drag! → HDF5 artifact (common configs loadable via ClimaArtifacts)

Every dt_ogw seconds:
  ogw_model_callback!
    └─ orographic_gravity_wave_compute_tendency!
        ├─ Compute N(z), find z_pbl
        ├─ calc_base_flux!              → τ_x, τ_y, τ_l, τ_p, τ_np
        ├─ calc_saturation_profile!     → τ_sat(z)
        ├─ calc_propagate_forcing!      → propagating tendency
        ├─ calc_nonpropagating_forcing! → blocked-flow tendency
        └─ Clamp forcing to ±3e-3 m/s²

Every dt (integrator step):
  orographic_gravity_wave_apply_tendency!
    └─ Yₜ.c.uₕ += Covariant12Vector(ᶜuforcing, ᶜvforcing)
```

For analytical topographies (DCMIP200, Hughes2023, Agnesi, Schar, Cosine2d, Cosine3d), the tensor is computed on-the-fly at startup using ClimaCore horizontal gradient operators in place of the offline pipeline.
