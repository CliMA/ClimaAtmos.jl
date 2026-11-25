# Orographic Gravity Wave Parameterization

Gravity waves shape the atmospheric circulation. They are usually generated from topography or convection, propagate upward and alter temperature and winds in the middle atmosphere, and influence tropospheric circulation through 'downward control' of the residual circulation by stratospheric wave forcing. The horizontal wavelength for gravity waves ranges from tens to thousands of kilometers, and the shorter end of this range up to a few hundred kilometers is unresolved at typical GCM resolution and must be parameterized.

The gravity wave drag on the wind velocities ($\overline{\vec{v}}=(u,v)$) is

```math
\frac{\partial \overline{\vec{v}}}{ \partial t} = ... + \underbrace{\left(-\frac{\partial \overline{\vec{v}'w'}}{\partial z}\Big|_{GW}\right)}_{\vec{X}}
```

where $\vec{X} = (X_\lambda, X_\phi)$ contains the subgrid-scale zonal and meridional components of the gravity wave drag, computed by the parameterization.

Throughout this document, an overbar (e.g. $\overline{\vec{v}}$, $\overline{\rho}$) denotes a grid-mean, resolved-scale variable in the sense of a Reynolds decomposition, and primes denote the corresponding subgrid fluctuations. $\epsilon_0$ denotes a small floating-point regularizer (`eps(FT)` in the code) used to prevent division-by-zero and degenerate clamps. It has no physical meaning.

## Orographic gravity wave

The orographic gravity wave drag parameterization follows the methods described in [garner2005](@cite). The momentum drag from subgrid-scale mountains is divided into a non-propagating component and a propagating component. The non-propagating (blocked/deflected flow) component deposits momentum drag in a low-level layer extending from the planetary boundary layer top up to a reference level $z_{\mathrm{ref}}$ (with pressure weighting that concentrates the drag near the surface), while the propagating component generates a stationary ($c = 0$, zero phase speed) gravity wave which propagates upwards and deposits momentum flux to the layers where it breaks.

### Planetary boundary layer (PBL) top

We implement the following criteria to find the PBL-top cell $k$ as the highest cell center that satisfies

```math
{}^cp[k] \ge 0.5 \cdot {}^fp[\mathrm{bottom}]
```

and

```math
{}^cT[1] + T_{\mathrm{boost}} - {}^cT[k] > g/c_{pd} * ( {}^cz[k] - {}^cz[1] )
```

where the superscripts $c$ and $f$ represent cell centers and faces in the vertical stencils, $[1]$ denotes the first (lowest) cell center level, and ${}^fp[\mathrm{bottom}]$ is the pressure at the bottom face. $T_{\mathrm{boost}} = 1.5\,\mathrm{K}$ is the surface temperature boost to improve the PBL height estimate.

The first condition restricts the search to the lower atmosphere (below roughly 500 hPa). The second condition compares the observed temperature decrease with height against the dry adiabatic lapse rate $g/c_{pd}$: within the well-mixed boundary layer, turbulent mixing keeps the temperature profile close to (or steeper than) the dry adiabat, so the inequality holds. Above the PBL top, the free atmosphere is stably stratified (temperature decreases more slowly than the dry adiabat) and the inequality fails, marking the transition from the convectively mixed layer to the stable free atmosphere above.

### Orographic information

Generating the base momentum flux for low-level flow encountering the subgrid-scale mountains requires orographic information. We compute the tensor $\textbf{T}$ and the scalar $h_{\mathrm{max}}$ from the Earth elevation data (ETOPO 2022 60-arcsecond, obtained through the `earth_orography_60arcseconds` ClimaArtifact; GFDL reference code [here](https://github.com/NOAA-GFDL/atmos_phys/blob/main/atmos_param/topo_drag/topo_drag.F90)). The preprocessed drag fields are distributed as the `ogw_computed_drag_h*` ClimaArtifacts.

#### Velocity potential $\chi$

The velocity potential $\chi$ captures the far-field influence of surrounding topography on the low-level flow at each grid point. It is computed as a 2D Hilbert transform of the surface elevation, smoothed by a Blackman window to suppress contributions from distant terrain beyond the smoothing scale:

```math
\chi(\mathbf{x}) = \frac{R_{\mathrm{earth}}}{2\pi} \int \int \frac{\cos\phi' \; h(\mathbf{x}')}{d(\mathbf{x}, \mathbf{x}')} \; W(d) \; d\lambda' \; d\phi',
```

where $h$ is the surface elevation, $d(\mathbf{x}, \mathbf{x}')$ is the great-circle arc distance, $\phi'$ is latitude, and $W(d)$ is a Blackman window taper that goes to zero at the smoothing scale. The $1/d$ kernel means nearby terrain dominates, while the window prevents spurious contributions from the far field. A singularity correction is added for the grid cell containing $\mathbf{x}$ itself. This follows the GFDL Fortran implementation (`get_velpot.f90`).

#### Orographic tensor $\textbf{T}$

The tensor $\textbf{T}$, which encodes the topographic amplitude, variance, orientation, and anisotropy, is computed as

```math
\textbf{T} = -\nabla \chi (\nabla h)^T,
```

where $h$ is the earth elevation and $\chi$ is the velocity potential defined above. The leading minus reproduces Garner's sign convention so that the resulting drag opposes the low-level flow (cf. the velocity potential in [garner2005](@cite) Eq. 6).

Both gradients are centered differences on the latitude-longitude grid, periodic in longitude, with the east-west spacing $R_{\mathrm{earth}}\,\Delta\lambda\cos\phi$. The tensor is set to zero poleward of 88° here, and again on the model grid after the interpolation. The tensor is then averaged over square blocks at most one model grid spacing wide north-south (`latlon_block_size`), similar to a conservative regridding. The east-west side of a block narrows towards the poles. Each model grid point then takes its value by linear interpolation between the nearest block centers. This averaging is important, as a pointwise tensor $-\nabla\chi(\nabla h)^T$ gives no drag for some wind directions, most notably along coastal terrain, while the block mean gives drag for every wind direction on 99.9% of the land area at `h_elem = 16`.

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

where $\bar{h}$ is the weighted local mean elevation, with weights $w_i = 1/(1 + d_i^2/s^2)$ for a point at angular distance $d_i$. The smoothing scale $s$ is $\alpha_s$ times the model grid spacing, rescaled with latitude, where $\alpha_s$ is the smoothing fraction (`ogw_smoothing_scale_fraction`, default 1.0). The window wraps around in longitude. The elevation is sampled every 7.5 km, or finer when the window is shorter than 30 km (`ogw_topography_stride`), so a wider window averages over more points, and the sample spacing is never coarser than about 7.5 km.

$\mathrm{hpos} = \max(0, h_0)^{2-\gamma}$ is averaged over the same blocks as the tensor, interpolated to the model grid points, and only then converted to $h_{\mathrm{max}}$ and $h_{\mathrm{min}}$ (`regrid_OGW_latlon`). The order here matters because the conversion is concave: converting first would give a lower $h_{\mathrm{max}}$ for a model grid point that lies between a mountain and a flat terrain block.

The interpolated $\mathrm{hpos}$ is rescaled using the mountain height–width exponent $\gamma$ and the height fraction $h_{\mathrm{frac}}$ to obtain the final $h_{\mathrm{max}}$ and $h_{\mathrm{min}}$:

```math
h_{\mathrm{max}} = \left( \mathrm{hpos} \cdot \frac{\gamma+2}{2\gamma} \cdot \frac{1 - h_{\mathrm{frac}}^{2\gamma}}{1 - h_{\mathrm{frac}}^{\gamma+2}} \right)^{1/(2-\gamma)}, \quad h_{\mathrm{min}} = h_{\mathrm{frac}} \cdot h_{\mathrm{max}}.
```

This rescaling adjusts the raw statistic to account for the assumed power-law distribution of mountain heights within the grid cell (cf. the Appendix in [garner2005](@cite)). $h_{\mathrm{frac}}$ is a ClimaParams tuning parameter (`ogw_critical_height_threshold`).

### Base flux

The base momentum flux is computed and divided into the propagating and non-propagating components.

The subscript $(\cdot)_{\text{pbl}}$ denotes a value in the **source cell**: the cell just above the PBL-top cell, which is also the lowest cell of the blocked layer (for now). $V_{\text{pbl}} = (u_{\text{pbl}}, v_{\text{pbl}})$, $N_{\text{pbl}}$, and $\rho_{\text{pbl}}$ are the horizontal wind, buoyancy frequency, and density in the source cell. $N$ is computed pointwise at every cell center as

```math
N^2 = \frac{g}{T} \left( \frac{dT}{dz} + \frac{g}{c_{pm}} \right),
```

where $c_{pm}$ is the moist isobaric specific heat. $dT/dz$ is the difference of the interpolated face temperatures across each cell, except in the bottom and top cells, which take the one-sided difference to their only neighbouring cell. $N_{\text{pbl}}$ is then $N$ evaluated in the source cell.

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

The base flux is then corrected using the Froude number, $\mathrm{Fr} = hN/V$ (obstacle height times $N$ over wind speed), which determines whether air goes over or around the mountain, and the saturation velocity. Let

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

The clamps to $\max(0, \cdot)$ on $h_{\mathrm{max}}$ and $h_{\mathrm{min}}$ treat negative height as zero. Here, $\mathrm{Fr}_{\mathrm{crit}} = 0.7$ (the parameter `ogw_critical_froude_number`) is the critical Froude number for nonlinear flow. $\mathrm{Fr}_{\mathrm{crit}}$ is the main parameter controlling how the drag is partitioned between the low-level blocked flow and the upper-level wave breaking.

The saturation velocity is computed as

```math
U_{\mathrm{sat}} = \sqrt{\frac{\rho_{\text{pbl}}}{\rho_0} \frac{V_{\tau}^3}{N_{\text{pbl}}\, L_0}},
```

where $\rho_0 = 1.2\,\mathrm{kg/m^3}$ is the arbitrary density scale (`ogw_density_scale_factor`), and $L_0 = 80\,000\,\mathrm{m}$ is the arbitrary horizontal length scale (`ogw_reference_mountain_width`).

$U_{\mathrm{sat}}$ roughly measures the biggest wave the flow can carry before it breaks. A stronger wind ($V_\tau$) makes much bigger waves (note the cube), while stronger stratification ($N$) or a wider mountain ($L_0$) makes them break sooner and shrinks it.

The following intermediate variables ($\mathrm{Fr}U$'s) are computed to correct the linear base flux:

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

The drag integrals below use the empirical shape parameters $(\gamma, \epsilon, \beta) = (0.4, 0.0, 0.5)$ [garner2005](@cite) (`ogw_mountain_height_width_exponent`, `ogw_number_density_exponent`, `ogw_mountain_shape_parameter`), where $\gamma$ comes from the observed scaling of mountain width with height and $\epsilon$ is unrelated to $\epsilon_0$, and the coefficients $a_0 = 0.9$ and $a_1 = 3.0$ (`ogw_linear_drag_coefficient`, `ogw_nonlinear_drag_coefficient`).

The corrected linear drag is computed as

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

The $\tau_{np}$ from the integral above scales with the mountain height $h_{\mathrm{max}}$, but the blocked layer is only about $V/N$ deep, the height the flow can climb against the stratification. Since $\mathrm{Fr}_{\mathrm{max}} = h_{\mathrm{max}}/(V/N)$, dividing by it swaps $h_{\mathrm{max}}$ for $V/N$. The floor $\mathrm{Fr}_{\mathrm{crit}}$ keeps the division from blowing up for short mountains, which does not contribute to blocking anyway ($\tau_{np} = 0$ there).

### Saturation profiles for the propagating component

Only the propagating component requires a saturation profile, as it carries a vertically propagating wave whose flux can saturate aloft. The non-propagating component remains the scalar $\tau_{np}$ from the base-flux calculation and is distributed in $z$ by pressure weighting across the blocking layer, from the first cell above the PBL-top cell up to $z_{\mathrm{ref}}$ (see [Non-propagating component](#Non-propagating-component) below).

The vertical profile of the saturated flux $\tau_{\mathrm{sat}}$ is computed such that the momentum forcing follows from $d\overline{V}/dt = -\overline{\rho}^{-1}d\tau_{\mathrm{sat}}/dz$.

The profile is built on cell faces, and the drag in each cell is computed as the change in flux between its bottom and top faces. The wind along the drag direction is computed at cell centers, interpolated to faces, and then floored:

```math
{}^f V_{\tau} = \max\!\left(\epsilon_0, \; {}^f\overline{\left(- V \cdot \frac{\tau}{\max(\epsilon_0, |\tau|)}\right)}\right).
```

Flooring before averaging would overestimate the face wind next to a wind reversal.

At each face, let $L_1 = L_0 \cdot \max(0.5, \min(2.0, 1.0 - 2 \, {}^f V_\tau \cdot {}^f d^2V_{\tau} / {}^f N^2))$, where ${}^f N$ is the interpolated cell-center $N$, and ${}^f d^2V_\tau$ is the face average of the cell-center curvature

```math
{}^c d^2V_{\tau}[k] = - \frac{d^2 V}{dz^2}[k] \cdot \frac{\tau}{\max(\epsilon_0, |\tau|)}
```

is the curvature of the wind projected onto the drag direction. Positive curvature (for example near a wind minimum) narrows the effective width, and negative curvature (for example near a jet core) widens it. The factor 2 on the curvature term is GFDL's coarse-sampling factor `samp`, as set in the AM4 production namelist. (This will be moved to `ClimaParams.jl`.) The curvature is zero from the surface up to the PBL-top cell and in the top cell, so it acts from the source cell upward.

The saturation velocity $U_{\mathrm{sat}}$ is computed face by face from the surface upward, with the face density ${}^f \rho$ interpolated from cell centers. It starts from the base-flux $U_{\mathrm{sat}}$ at the bottom face and can only shrink going up; the top face is skipped:

```math
U_{\mathrm{sat}}[k] = \min\!\left(U_{\mathrm{sat}}[k-1], \; \sqrt{\frac{{}^f \rho}{\rho_0} \frac{{}^f V_{\tau}^3}{{}^f N \cdot L_1} }\right),
```

so $U_{\mathrm{sat}}$ is monotonically non-increasing with height.

Intuitively, these three equations build up the saturation ceiling level by level. First, $V_\tau$ is the wind component that pushes against the wave (the flow projected onto the drag direction), so it measures how hard the flow drives the wave at each height. Second, $L_1$ is an effective mountain width: the local flow curvature ($d^2 V_\tau$) stretches or shrinks it, which tunes how readily the wave saturates. Third, $U_{\mathrm{sat}}$ is the largest wave amplitude the background flow can hold before the wave gets too steep and breaks: the local conditions ($\rho$, $V_\tau$, $N$, $L_1$) set a ceiling at each level, and the $\min$ with the level below enforces that once the wave has been clipped, it stays clipped going up (a wave can lose flux to breaking but never regain it).

The $\mathrm{Fr}U_{\mathrm{min}}$ and $\mathrm{Fr}U_{\mathrm{max}}$ are inherited from the base flux calculation. The source-cell $\mathrm{Fr}U_{\mathrm{sat}}$ and $\mathrm{Fr}U_{\mathrm{clp}}$ are then

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

$\mathrm{Fr}U_{\mathrm{sat}}$ is the "breaking line" at the current level: any part of the wave above it is too steep and gets shaved off. Because $U_{\mathrm{sat}}$ only decreases with height, this line decreases as the wave climbs, so more of the wave is removed the higher it goes. $\mathrm{Fr}U_{\mathrm{clp}}$ is that breaking line clamped to stay within the obstacle range $[\mathrm{Fr}U_{\mathrm{min}}, \mathrm{Fr}U_{\mathrm{max}}]$, and the $(\cdot)_0$ versions remember these values in the source cell.

Then, the saturated flux of the propagating component **from the top face of the source cell upward** is

```math
\tau_{\mathrm{sat}}[k] = a_0 \left[ \frac{\mathrm{Fr}U_{\mathrm{clp}}^{2+\gamma-\epsilon} - \mathrm{Fr}U_{\mathrm{min}}^{2+\gamma-\epsilon}}{2+\gamma-\epsilon} + \mathrm{Fr}U_{\mathrm{sat}}^2 \mathrm{Fr}U_{\mathrm{sat0}}^\beta \frac{\mathrm{Fr}U_{\mathrm{max}}^{\gamma-\epsilon-\beta} - \mathrm{Fr}U_{\mathrm{clp0}}^{\gamma-\epsilon-\beta}}{\gamma-\epsilon-\beta} + \mathrm{Fr}U_{\mathrm{sat}}^2 \frac{\mathrm{Fr}U_{\mathrm{clp0}}^{\gamma-\epsilon} - \mathrm{Fr}U_{\mathrm{clp}}^{\gamma-\epsilon}}{\gamma-\epsilon} \right].
```

This is the same obstacle-distribution integral as the base-level $\tau_p$, but re-evaluated with the lowered breaking line at level $k$: it encodes how much wave flux still survives at that height. As the wave climbs the breaking line drops, so $\tau_{\mathrm{sat}}$ shrinks with height, and that loss of flux (its vertical gradient $d\tau_{\mathrm{sat}}/dz$) is exactly the momentum the breaking wave dumps back into the flow.

**From the surface up to the bottom face of the source cell**, the saturated flux is held at the launched flux, $\tau_{\mathrm{sat}} = \tau_p$, so breaking drag starts in the source cell. The model-top face copies the face below it, such that the top layer has no breaking drag.

The flux left at the model top (waves that did not fully break) is then spread over the whole column as drag. Each face loses a share of it, from none at the surface to all of it at the top, in proportion to pressure. The column-integrated propagating drag equals $\tau_p$, i.e.,

```math
{}^f\tau_{\mathrm{sat}} = {}^f\tau_{\mathrm{sat}} - {}^f\tau_{\mathrm{sat}}[\mathrm{top}] \, \frac{{}^fp[\mathrm{bottom}] - {}^fp}{{}^fp[\mathrm{bottom}] - {}^fp[\mathrm{top}]}.
```

### Velocity tendencies due to the orographic drag

#### Propagating component

The forcing from the propagating part on the zonal and meridional wind are

```math
{}^c \left( \frac{du}{dt} \right) _{p} = - \frac{1}{{}^c \rho} \frac{\tau_x}{\tau_l} \frac{d\tau_{\mathrm{sat}}}{dz},
```

```math
{}^c \left( \frac{dv}{dt} \right) _{p} = - \frac{1}{{}^c \rho} \frac{\tau_y}{\tau_l} \frac{d\tau_{\mathrm{sat}}}{dz}.
```

Here, $(\tau_x, \tau_y, \tau_l)$ are computed from the base flux (`calc_base_flux!`) and $\tau_{\mathrm{sat}}$ from the saturation profile (`calc_saturation_profile!`). Breaking drag starts in the source cell, since $\tau_{\mathrm{sat}} = \tau_p$ up to the bottom face of the source cell. The PBL cells nevertheless receive a small redistributed share of the flux left at the model top.

#### Non-propagating component

The non-propagating (blocked) drag goes into a layer from the source cell (the cell just above the PBL-top cell) up to a reference level $z_{\mathrm{ref}}$, such that no PBL cell gets blocked drag. This follows GFDL `topo_drag`. If no level above the surface cell passes the PBL test, the blocked layer starts in the second cell.

Note that most other schemes, such as Lott and Miller (1997) in the ECMWF IFS, start the blocked layer at the ground, and GFDL gives no reason for starting it above the PBL (its code has the ground start commented out). Near the ground over hills, the main orographic drag is the turbulent orographic form drag (TOFD, below). Future work may consider starting the blocked layer from the surface.

To locate $z_{\mathrm{ref}}$, we start at $z_{\mathrm{start}}$, the top face of the lowest blocked cell (GFDL `zhalf(kpbl)`), iterate over the face levels above it, and accumulate a wave phase

```math
\mathrm{phase} \mathrel{+}= ({}^fz[k] - {}^fz[k-1]) \cdot \frac{\max(N_{\mathrm{min}}, \min(N_{\mathrm{max}}, {}^fN[k]))}{\max(vvmin, {}^fV_{\tau}[k])},
```

with ${}^fz[k-1] = z_{\mathrm{start}}$ for the first face above $z_{\mathrm{start}}$, and
setting $z_{\mathrm{ref}} = {}^fz[k]$ at the first face where $\mathrm{phase} > \pi$. Each layer uses ${}^fN$ and ${}^fV_{\tau}$ at its upper face. Here $N_{\mathrm{min}} = 0.7 \times 10^{-2}\,\mathrm{s^{-1}}$, $N_{\mathrm{max}} = 1.7 \times 10^{-2}\,\mathrm{s^{-1}}$, $vvmin = 1.0\,\mathrm{m/s}$, ${}^fV_{\tau}$ is the face wind from the saturation profile, and ${}^fN$ is the cell-center $N$ interpolated to faces. If $\mathrm{phase}$ never exceeds $\pi$ in the column, or $z_{\mathrm{start}}$ is the model top face, $z_{\mathrm{ref}}$ is the model top face. If the PBL-top cell is the top cell, there is no cell above it and the column gets no blocked drag; this cannot happen in practice, because the PBL search only accepts levels with at least half the surface pressure.

$N / V_\tau$ is the vertical wavenumber of a stationary hydrostatic gravity wave, so summing it over height accumulates the wave's vertical phase. Reaching $\pi$ means the flow has turned through half a vertical wavelength. This marks the top of the blocked/deflected low-level layer, which sets $z_{\mathrm{ref}}$. Half a wavelength is the natural cutoff because the wave oscillates up and down as $\sin(mz)$: over the first half-wavelength the displacement completes a single half-cycle (it keeps one sign), which is the air piling up and overturning right against the mountain. Past that point, the wave flips to the other sign and starts behaving like the free wave traveling upward, not the air trapped and blocked near the surface, so that is where the blocked layer ends. The clamps on $N$ and $V_\tau$ keep the wavenumber physical so the layer depth stays sensible.

The phase sum here differs slightly from GFDL `topo_drag`, so the top of the blocked layer can differ from GFDL's by up to one cell (see "Comparison with GFDL" at the end of this section).

The drag forcing due to the non-propagating component is confined to the blocked layer and is weighted by pressure. The weight of each cell is its center pressure minus the pressure at $z_{\mathrm{ref}}$:

```math
\mathrm{weight}[k] = {}^c p[k] - {}^f p_{\mathrm{ref}},
```

where ${}^f p_{\mathrm{ref}}$ is the face pressure at $z_{\mathrm{ref}}$. The pressure thickness of each cell is the pressure difference across its own two faces:

```math
\mathrm{diff}[k] = {}^f p_{\mathrm{lower}}[k] - {}^f p_{\mathrm{upper}}[k],
```

where the subscripts denote the lower and upper faces of cell $k$. Interior face pressures are the mean of the two adjacent cell-center pressures. The bottom and top face pressures are integrated hydrostatically over the half cell from the adjacent center, ${}^f p_{\mathrm{bottom}} = p[1] + \rho[1]\, g\, (z[1] - {}^f z_{\mathrm{bottom}})$ and ${}^f p_{\mathrm{top}} = p[k_{\mathrm{top}}] - \rho[k_{\mathrm{top}}]\, g\, ({}^f z_{\mathrm{top}} - z[k_{\mathrm{top}}])$, because copying the nearest center pressure onto a boundary face would give the lowest and highest cells only about half their pressure thickness. The model keeps no surface pressure, so this half-cell integral is used instead.

The column stress sum is

```math
wtsum = \sum_{k \in \mathrm{mask}} \mathrm{diff}[k] \cdot \mathrm{weight}[k].
```

Dividing the forcing by this sum conserves the column-integrated blocked stress:
$\sum_k (\partial u/\partial t)_k \,\mathrm{diff}[k]/g = \tau_x\,\tau_{np}/\tau_l$.

The mask selects the cells from the lowest blocked cell (top face at $z_{\mathrm{start}}$) up to the cell whose top face is $z_{\mathrm{ref}}$, and drops other cells with zero weight. The lowest blocked cell is always selected, so the mask is empty only when there is no cell above the PBL-top cell.

For masked levels, the forcing due to the non-propagating component is

```math
{}^c \left( \frac{du}{dt} [k] \right)_{np} = g \frac{\tau_{np}}{\tau_l} \frac{\mathrm{weight}[k]}{wtsum} \tau_x,
```

```math
{}^c \left( \frac{dv}{dt} [k] \right)_{np} = g \frac{\tau_{np}}{\tau_l} \frac{\mathrm{weight}[k]}{wtsum} \tau_y,
```

where $(\tau_x, \tau_y, \tau_l, \tau_{np})$ is computed in the base flux calculation. When the mask is empty ($wtsum = 0$, i.e. no cell above the PBL-top cell), the non-propagating forcing is set to zero in that column to avoid division by zero.

!!! details "Comparison with GFDL"

    A large part of the OGW parameterization here is based on GFDL's `topo_drag` (NOAA-GFDL/atmos_phys) with the AM4 production
    settings: `frcrit = 0.7`, `alin = 0.9`, `anonlin = 3`, `use_mg_scaling`, `tboost = 1.5`,
    `samp = 2`, and `h_frac = 0.1`; `keep_residual_flux` is on by default and AM4 keeps it.
    GFDL names for the terms used above: "full levels" are cell centers and "half levels" are
    faces; `kpbl` is the source cell (the cell just above the PBL-top cell, where the base
    flux is read); `zhalf(kpbl)` is $z_{\mathrm{start}}$; and `kref` is the top of the blocked
    layer.

    **The same as GFDL:**

      - PBL top: the highest cell center that passes the temperature test with
        $T_{\mathrm{boost}}$, among those with at least half the bottom-face pressure (`get_pbl`,
        `ppbl`); if none passes, the blocked layer starts in the second cell
        (`kpbl = kdim - 1`).
      - The base flux is read in the source cell, which is also the lowest blocked cell.
      - The saturated flux is built on faces and held at $\tau_p$ from the surface up to the bottom face of the source cell, so drag can only start breaking in the source cell; the top cell gets no breaking drag; the flux left at the top is spread over the column by pressure (keep_residual_flux).
      - The face wind along the drag is the face average of the signed wind, floored after.
      - The curvature keeps its sign and carries the factor `samp = 2` (GFDL's correction for
        coarse vertical sampling of the curvature); it is zero from the surface to the PBL-top cell, and the top cell.
      - Blocked drag: the phase sum starts at `zhalf(kpbl)`; each cell's weight is its center
        pressure minus the pressure at the top (`pfull - pref`), normalized by
        `wtsum = Σ dp·weight`.
      - Preprocessing: $\mathrm{hpos}$ is averaged onto the model grid and only then converted
        to $h_{\mathrm{max}}$ (GFDL: a conservative remap; ClimaAtmos: block area means, then
        linear interpolation); the east-west spacing uses the true $\cos\phi$.

    **Different from GFDL, on purpose:**

      - Top of the blocked layer: the phase sum steps on faces, so the top can be one cell
        higher than GFDL's (table and cases below). It uses the moist, face-interpolated $N$;
        GFDL uses a dry $N$.
      - Source-cell values: ClimaAtmos uses the cell-center density and the cell-center $N$
        with the moist heat capacity; GFDL takes the density as $\Delta p/(g\,\Delta z)$ of the
        cell and $N$ from the dry lapse rate across the cells above and below.
      - Face values: ClimaAtmos interpolates $N$, density and curvature from cell centers to
        faces; GFDL takes $N$ (dry) and density from the two cell centers on either side.
      - PBL search: ClimaAtmos also tests the second-lowest cell center; GFDL starts at the
        third-lowest (`do k=kdim-2,2,-1`). This is the only case where this counting results in a difference: when the second-lowest is the highest level that
        passes, GFDL's source cell is the second-lowest and ClimaAtmos's the third-lowest. The latter is consistent with the blocked layer starting just above the highest cell that passes the PBL test.
      - Propagating drag: ClimaAtmos divides by the model density, $-\rho^{-1}\,d\tau/dz$, so
        momentum is conserved in the model's own mass; GFDL uses $g\,\Delta\tau/\Delta p$.
      - Drag limit: the size of the drag vector is capped at $3\times10^{-3}\sqrt{2}$ m/s², which
        keeps its direction; GFDL clips each component to $\pm 3\times10^{-3}$ m/s² (`max_udt`).
      - Time stepping: ClimaAtmos recomputes the drag every `dt_ogw`, reuses the last computed value in between,
        and applies it semi-implicitly so one step cannot reverse the wind; GFDL applies its
        tendency explicitly at every call.
      - Heating: AM4 adds the frictional heating explicitly (`do_conserve_energy`); in
        ClimaAtmos the kinetic energy removed by the drag becomes internal energy through the
        total-energy equation.
      - Near-surface drag over hills: ClimaAtmos has TOFD (below); AM4 instead raises the
        surface roughness (`topo_rough_nml`).
      - Face pressures: the bottom and top faces are a half-cell hydrostatic step from the
        nearest center; GFDL's model has face pressures (`phalf`).
      - Preprocessing: the smoothing scale is determined by the model grid ($\alpha_s$ times the grid
        spacing; Garner's default `scale0` is a fixed 200 km, while both codes rescale it with latitude), the
        elevation is sampled every 7.5 km or less (Garner uses the full, finer 1-arcminute grid), the
        tensor is zero poleward of 88° in both hemispheres (Garner: south only), and the
        half-width of the $\mathrm{hpos}$ window is capped at 1/8 of the latitude circle
        (Garner's `get_hmax.f90`: 1/16).
      - AM4's `use_pbl_from_lock` and `use_uref_4stable` have no ClimaAtmos counterpart and are
        off.

    **Top of the blocked layer.** Both codes add up the phase from the same face; they differ
    in where each step lands and in how the top is picked. For example, with 250 m cells, a
    PBL-top cell at 750–1000 m, and the first blocked cell at 1000–1250 m:

    |                          | ClimaAtmos                                   | GFDL                                                                                               |
    |:------------------------ |:-------------------------------------------- |:-------------------------------------------------------------------------------------------------- |
    | start                    | 1250 m (a face)                              | 1250 m (the same face)                                                                             |
    | steps land on            | faces: 1500, 1750, 2000 m, …                 | cell centers: 1625, 1875, 2125 m, …                                                                |
    | first step               | 1250 → 1500 m (one cell)                     | 1250 → 1625 m (one and a half cells, past the center at 1375 m)                                    |
    | $N$ and $V_\tau$         | at the top face of each step                 | from the two cell centers of each step                                                             |
    | top of the blocked layer | the face where the phase first exceeds $\pi$ | the top face of the cell around the lower of the two centers where the phase passes $\pi$ (`kref`) |

    We take the last row as an example. Higher up in the same column, and assuming both codes
    add up the same phase, the faces near 4.4 km are at 4250, 4500 and 4750 m, and the cell
    centers at 4375 and 4625 m. ClimaAtmos picks the first face above the point where the
    phase passes $\pi$. GFDL finds the two cell centers that point lies between, and picks the
    top face of the cell around the lower one. Specifically,

     1. The phase passes $\pi$ at 4392 m. ClimaAtmos: the next face up, 4500 m. GFDL: 4392 m
        lies between the centers 4375 and 4625 m; the cell around 4375 m has its top face at
        4500 m. Both pick 4500 m.
     2. The phase passes $\pi$ at 4550 m. ClimaAtmos: 4750 m. GFDL: still between 4375 and
        4625 m, so still 4500 m, one cell lower than ClimaAtmos.

### Turbulent orographic form drag (TOFD)

The propagating and blocked components both act from the source cell upward, so they
supply no drag in the layer right next to the surface, apart from the small
redistributed leftover at the model top. Small-scale orography
(scales too short to launch resolved gravity waves) nonetheless exerts a
form drag there. This turbulent orographic form drag (TOFD) of [beljaars2004](@cite)
is added as a low-level term, in the form and with the constants of the IFS
(IFS Documentation CY47R1, Part IV, Section 3.4):

```math
{}^c \left( \frac{d\mathbf{u}}{dt}[k] \right)_{\mathrm{TOFD}} = - C_d(z)\, |\mathbf{u}|\, \mathbf{u},
\qquad
C_d(z) = a_{\mathrm{tofd}}\, \alpha\, \beta\, C_{md}\, C_{\mathrm{corr}}\, 2.109\,
e^{-(z/1500)^{1.5}}\, a_2\, z^{-1.2},
\qquad
a_2 = \frac{\sigma_{\mathrm{flt}}^2\, k_1^{\,n_1-n_2}}{I_H\, k_{\mathrm{flt}}^{\,n_1}},
```

where $z$ is the height above the local surface (floored at 1 m so $z^{-1.2}$ stays
finite); $a_{\mathrm{tofd}}$, $\sigma_{\mathrm{flt}}$, $I_H$, and $k_{\mathrm{flt}}$ are defined below. The vertical profile is an analytic fit of the drag integrated over a
power-law orography spectrum with slopes $n_1 = -1.9$ and $n_2 = -2.8$ and a break at
$k_1 = 3\times10^{-3}\,\mathrm{m^{-1}}$; it concentrates the drag in the lowest
$\sim 1\,\mathrm{km}$. The other constants are $\beta = 1$, $C_{md} = 0.005$,
$C_{\mathrm{corr}} = 0.6$, and $\alpha = 35$: [beljaars2004](@cite) obtained 12 from
large-eddy simulations, and the IFS multiplies it by $1.7^2$ after tuning.

The only geographical input is $\sigma_{\mathrm{flt}}$, the standard deviation of the
filtered orography, computed at initialization from the 60 arc-second ETOPO 2022
elevation as in the IFS (IFS Documentation CY47R1, Part IV, Section 11.3.3):

 1. the orography, with ocean bathymetry set to zero, is smoothed twice with the IFS
    operator (a top hat of width $\Delta$ with cosine edges of width $\delta = 1$ km),
    with $\Delta_1 = 2$ km and $\Delta_2 = 20$ km;
 2. the squared difference of the two smoothed fields, which keeps horizontal scales
    of roughly 3 to 22 km (half wavelengths), is averaged over blocks of about the
    model's horizontal grid spacing and regridded to the model surface, giving
    $\sigma_{\mathrm{flt}}^2$.

On the ETOPO 1.85 km elevation data currently used, the 2 km pass removes little, so the
data spacing sets the small-scale end of the band.

$I_H = 1.02\times10^{-3}\,\mathrm{m^{-1}}$ and
$k_{\mathrm{flt}} = 3.5\times10^{-4}\,\mathrm{m^{-1}}$ are the integral and the
representative wavenumber of this band-pass filter, so they are tied to this
definition of $\sigma_{\mathrm{flt}}$. TOFD is computed only when the model surface
is Earth's orography (`topography` is `Earth` and `orographic_gravity_wave` is
`gfdl_restart`, `raw_topo`, or `raw_topo_online`).

The amplitude $a_{\mathrm{tofd}}$ is the parameter `ogw_tofd_coefficient`:
$a_{\mathrm{tofd}} = 1$ reproduces the IFS, the default $a_{\mathrm{tofd}} = 2.7$
is tuned to 0M WeatherBench forecasts, and
$a_{\mathrm{tofd}} = 0$ disables TOFD.

Near steep orography the damping time $(C_d |\mathbf{u}|)^{-1}$ in the lowest
model levels is minutes, far shorter than both `dt_ogw` and the integrator step,
so an explicit, cached tendency would overshoot and reverse the wind. TOFD is
therefore not part of the cached `dt_ogw` forcing: $C_d(z)$ is precomputed once
at initialization, and every integrator step applies the backward-Euler form

```math
{}^c \left( \frac{d\mathbf{u}}{dt}[k] \right)_{\mathrm{TOFD}} = - \frac{r}{1 + r\,\Delta t}\, \mathbf{u},
\qquad
r = C_d(z)\, |\mathbf{u}|,
```

evaluated with the current wind. The factor is a scalar, so the drag stays
antiparallel to the flow, and its magnitude saturates at $|\mathbf{u}|/\Delta t$,
so one forward-Euler step can at most bring the wind to rest. TOFD is not subject to the
magnitude constraint below.

### Constrain the forcings

The cached gravity-wave drag combines the propagating and blocked components

```math
{}^c \left( \frac{d\mathbf{u}}{dt} [k] \right)_{\tau} = {}^c \left( \frac{d\mathbf{u}}{dt} [k] \right)_{p} + {}^c \left( \frac{d\mathbf{u}}{dt} [k] \right)_{np}.
```

As a safety check on the cached forcing, we limit the
forcing *magnitude* to $\epsilon_V = 3\times10^{-3}\sqrt{2}\,\mathrm{m/s^2}$ while preserving
its direction, by scaling the horizontal vector by a single factor rather than clipping each
component independently (which would rotate the drag away from the flow). The cap is the
diagonal of the $\pm 3\times10^{-3}\,\mathrm{m/s^2}$ per-component box used by GFDL
`topo_drag` (`max_udt`), so the limiter is isotropic and no tighter than that box in any
direction:

```math
{}^c \left( \frac{d\mathbf{u}}{dt} [k] \right)_{\tau} \leftarrow
f[k]\, {}^c \left( \frac{d\mathbf{u}}{dt} [k] \right)_{\tau},
\qquad
f[k] = \min\!\left(1, \frac{\epsilon_V}{\left| {}^c \left( \frac{d\mathbf{u}}{dt} [k] \right)_{\tau} \right|}\right).
```

This is a numerical safety rail, not a physical bound: it conserves the drag direction but,
where it binds, reduces the column-integrated stress below the parameterized value. The
tendencies above act on the physical horizontal wind components.

### Applying the drag at each step

When `orographic_gravity_wave_apply_tendency!` applies the cached drag vector
$\boldsymbol{D} = ({}^c\mathrm{uforcing}, {}^c\mathrm{vforcing})$ to the
horizontal momentum tendency at each step, its decelerating component parallel
to the horizontal wind $\boldsymbol{u}_h = (u, v)$ is damped by a
backward-Euler factor. With
$\hat{\boldsymbol{u}} = \boldsymbol{u}_h / |\boldsymbol{u}_h|$, the deceleration along the
wind $d = \max(0, -\boldsymbol{D} \cdot \hat{\boldsymbol{u}})$, and
$\boldsymbol{D}_{\perp}$ the part of $\boldsymbol{D}$ across the wind,

```math
\boldsymbol{D}_{\mathrm{eff}} = \boldsymbol{D}_{\perp} + \frac{\boldsymbol{D}_{\parallel}}{1 + \Delta t \, d / |\boldsymbol{u}_h|}
= \boldsymbol{D}_{\perp} - \hat{\boldsymbol{u}} \, \frac{d \, |\boldsymbol{u}_h|}{|\boldsymbol{u}_h| + \Delta t \, d},
\qquad
\boldsymbol{D}_{\parallel} = -d \, \hat{\boldsymbol{u}}.
```

The damped deceleration along the wind is always less than $|\boldsymbol{u}_h| / \Delta t$, so one
forward-Euler step of size $\Delta t$ cannot reverse the wind, at any wind speed. This holds
for this drag on its own; TOFD (above) is added separately and bounded separately. With a
multi-stage IMEX scheme, the bound holds for each stage, and the stages are then combined with
their $b_{\mathrm{exp}}$ weights.

  - Only the part that slows the wind is damped; the part that turns the wind, and any part
    that speeds it up, are applied unchanged.
  - There is no tunable parameter.
  - As the wind goes to zero, the slowing part goes to zero too, so only the turning part
    is left; with no wind at all, the drag is applied as is.

## Implementation summary

The parameterization has two parts. Preprocessing (Earth topography only) builds $(h_{\mathrm{max}}, h_{\mathrm{min}}, t_{11}, t_{12}, t_{21}, t_{22})$ on the model grid, either offline into an HDF5 artifact (`raw_topo`) or at start-up (`raw_topo_online`). At run time, the drag is recomputed every `dt_ogw` and applied every step. Preprocessing computes in `Float64` on the latitude-longitude grid. However, the results take the model float type when they are regridded, and a stored artifact has the float type of the run that built it (`Float32` for the shipped ones).

The scheme and its topographic-information source are selected with the
`orographic_gravity_wave` configuration key: `gfdl_restart` regrids the GFDL
`topo_drag.res.nc` artifact, `raw_topo` loads a preprocessed HDF5 artifact built
by the pipeline below, `raw_topo_online` runs that pipeline at initialization so
the drag input matches the run's resolution and the `γ`, `h_frac`, and
$\alpha_s$ (`ogw_smoothing_scale_fraction`, `α_smoothing` in the code) parameters.

!!! note "raw_topo artifacts are tied to their generation parameters"

    The `raw_topo` artifacts fix $(h_{\mathrm{max}}, h_{\mathrm{min}}, t_{11}, t_{12}, t_{21}, t_{22})$
    at the values used when they were generated (the ClimaParams defaults
    `γ = 0.4`, `h_frac = 0.1`, `α_smoothing = 1.0`). The runtime drag physics still
    uses the current parameters, so overriding `γ`, `h_frac`, or `α_smoothing` while
    keeping `raw_topo` leaves the drag input inconsistent with the physics (a warning
    is emitted). Use `raw_topo_online` to rebuild the drag input from the current
    parameters.

!!! note "Drag files carry a build record"

    A build record is a set of HDF5 attributes that `write_computed_drag!` stores in every
    drag file, saying how the file was built. It lets a `raw_topo` run refuse a file that
    does not match the current code or grid, instead of silently using it.

    Recorded: the preprocessing version (`OGW_PREPROCESSING_VERSION`), `h_elem`, `γ`,
    `h_frac`, `α_smoothing`, the float type of the stored fields and of the computation
    (`Float64`), and the elevation file name.

    Loading (`check_ogw_drag_stamp`) throws an error if the record is missing, or its version or `h_elem` differs; different `γ`, `h_frac`, or `α_smoothing` logs a
    warning. Bump `OGW_PREPROCESSING_VERSION` whenever a change to the preprocessing method or algorithm
    changes its output, and rebuild the `ogw_computed_drag_h*` artifacts.

```
Preprocessing (Earth topography only); offline for raw_topo, or at
initialization for raw_topo_online:
  compute_OGW_info
    ├─ compute_OGW_latlon
    │    ├─ calc_hpoz_latlon        → h₀ (raw 4th-moment statistic)
    │    ├─ calc_velocity_potential → χ  (2D Hilbert transform)
    │    ├─ calc_orographic_tensor  → t11, t21, t12, t22
    │    └─ block_average_latlon    → area means of hpos = h₀^(2-γ) and the tensor over
    │                                 ~model grid spacing (latlon_block_size)
    └─ regrid_OGW_latlon
         ├─ SpaceVaryingInput        → hpos and the tensor on the model grid points
         ├─ ogw_hmax_from_hpos       → hmax, hmin on the model grid points
         └─ tensor set to zero poleward of 88° on the model grid points
  write_computed_drag! → HDF5 artifact (raw_topo; loadable via ClimaArtifacts)

TOFD input, at initialization when a_tofd > 0 (Earth orography only):
  compute_filtered_orography_variance
    ├─ calc_filtered_orography_variance_latlon → (S_2km h − S_20km h)² on the ETOPO grid
    ├─ block_average_latlon                    → average over ~model grid spacing
    └─ SpaceVaryingInput                       → σ_flt² on the model surface
  tofd_drag_coefficient!                       → ᶜtofd_Cd

Every dt_ogw seconds:
  ogw_model_callback!
    └─ orographic_gravity_wave_compute_tendency!
        ├─ Compute N(z) and face pressure
        └─ orographic_gravity_wave_forcing!
            ├─ get_pbl_z!                   → z_pbl, center of the PBL-top cell
            ├─ blocked_layer_start_height!  → z_start, top face of the cell above it
            ├─ calc_base_flux!              → τ_x, τ_y, τ_l, τ_p, τ_np
            ├─ calc_saturation_profile!     → τ_sat(z)
            ├─ calc_propagate_forcing!      → propagating tendency
            ├─ calc_nonpropagating_forcing! → blocked-flow tendency
            └─ Limit forcing magnitude to 3e-3·√2 m/s² (direction preserved)

Every dt (integrator step):
  orographic_gravity_wave_apply_tendency!
    ├─ Yₜ.c.uₕ += C12(UVVector(ᶜuforcing, ᶜvforcing))
    └─ Yₜ.c.uₕ -= tofd_implicit_rate(ᶜtofd_Cd, |uₕ|, dt) · uₕ   (if a_tofd > 0)
```

For analytical topographies (DCMIP200, Hughes2023, Agnesi, Schar, Cosine2d, Cosine3d), the tensor is computed on-the-fly at startup using ClimaCore horizontal gradient operators in place of the offline pipeline. In test cases using analytical topography, the tensor is set to zero only south of 88°S, and TOFD is off, since it needs Earth's elevation data.

## Where this is implemented

| Concept                                                                       | Source                                                                                                                                                                                                                       |
|:----------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Base flux, saturation profile, propagating, blocked-flow, and TOFD tendencies | [src/parameterized_tendencies/gravity_wave_drag/orographic_gravity_wave.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/gravity_wave_drag/orographic_gravity_wave.jl)                      |
| Orographic statistics and tensor helpers                                      | [orographic_gravity_wave_helper.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/gravity_wave_drag/orographic_gravity_wave_helper.jl)                                                       |
| Offline topography preprocessing                                              | [preprocess_topography.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/gravity_wave_drag/preprocess_topography.jl)                                                                         |
| Callback scheduling at `dt_ogw`                                               | [src/callbacks/get_callbacks.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/callbacks/get_callbacks.jl), [src/callbacks/callbacks.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/callbacks/callbacks.jl) |
| Model types                                                                   | [`ClimaAtmos.OrographicGravityWave`](@ref), [`ClimaAtmos.FullOrographicGravityWave`](@ref)                                                                                                                                   |

The scheme is selected with the `orographic_gravity_wave` key
(`nothing`, `gfdl_restart`, `raw_topo`, or `raw_topo_online`) and its update interval
with `dt_ogw`. The surface field it acts on comes from
[Topography Representation](topography.md); the scheme needs a surface with topography,
so it raises an error with `topography: "NoWarp"`. See
[Configuration Options](configuration_options.md) and
[Running Global Simulations](global_simulations.md).
