# Running a 2.5D Squall Line

This page walks through the idealized squall-line benchmark of
[Gabersek2012](@cite) as a worked example of a moist, convection-resolving
box simulation, and uses it as a scientific demonstration of switching
between the 0-moment and 1-moment microphysics schemes (see
[Microphysics](microphysics.md)).

A squall line is a line of thunderstorms maintained by the interplay of
low-level wind shear and the cold pool produced by evaporating
precipitation. The case is a stringent test of moist dynamics: it exercises
deep saturated updrafts, precipitation formation and fallout, and the
resulting density currents, all in a domain small enough to run on a single
node.

## Case description

The computational domain is ``\Omega = 150 \times 12 \times 24\;\mathrm{km}^3``,
discretized with a single spectral element in the ``y`` direction ("2.5D"):
the flow is free to develop weak three-dimensional structure across the four
quadrature points in ``y``, but the case is effectively a vertical slice.
The lateral boundaries are periodic, the bottom boundary is free-slip and
thermally insulating, and a Rayleigh sponge above 15 km damps vertically
propagating gravity waves before they reflect off the model top.

The initial state is a tabulated mid-latitude sounding (from the appendix of
Tissaoui et al., 2023, following [Gabersek2012](@cite)): a saturated boundary
layer with ``q_v = 14\;\mathrm{g\,kg^{-1}}``, a weakly stable troposphere
(``N \approx 0.01\;\mathrm{s^{-1}}``) below the 12 km tropopause, and a more
stable stratosphere (``N \approx 0.02\;\mathrm{s^{-1}}``) above. A low-level
wind shear in ``x`` (``12 \to 0\;\mathrm{m\,s^{-1}}`` over the lowest
2.4 km) breaks the storm's symmetry and sustains its propagation. Convection
is triggered by a ``y``-independent warm bubble,

```math
\Delta\theta = \theta_c \cos^2\!\left(\frac{\pi r}{2}\right), \qquad
r = \sqrt{\left(\frac{x - x_c}{10\,\mathrm{km}}\right)^2
        + \left(\frac{z - z_c}{1.5\,\mathrm{km}}\right)^2} \le 1,
```

with ``\theta_c = 3\;\mathrm{K}`` centered at ``x_c = 75\;\mathrm{km}``,
``z_c = 2\;\mathrm{km}``. The cloud begins to form around ``t \approx 500``
s, and rain reaches the surface at ``t \approx 900`` s.

## The initial sounding

The case is packaged as the [`ClimaAtmos.Setups.Gabersek2012`](@ref) setup,
which precomputes the sounding profiles (with pressure re-integrated to
discrete hydrostatic balance) at construction time:

```@example squall_line
using Logging # hide
Logging.disable_logging(Logging.Info) # hide
import ClimaAtmos as CA

setup = CA.Setups.Gabersek2012()
propertynames(setup.profiles)
```

Each profile is a function of height, which makes the sounding easy to
inspect. The Brunt–Väisälä frequency computed from the potential temperature
profile shows the two-layer stratification that defines the case:

```@example squall_line
import CairoMakie as MK

z = 0.0f0:50.0f0:24_000.0f0
(; θ, q_tot, u) = setup.profiles
g = 9.81f0
N = [sqrt(g * (θ(zi + 25) - θ(zi - 25)) / 50 / θ(zi)) for zi in z[2:(end - 1)]]

fig = MK.Figure(; size = (900, 350))
ax1 = MK.Axis(fig[1, 1]; xlabel = "θ [K]", ylabel = "z [km]")
MK.lines!(ax1, θ.(z), z ./ 1000)
ax2 = MK.Axis(fig[1, 2]; xlabel = "q_tot [g/kg]")
MK.lines!(ax2, q_tot.(z) .* 1000, z ./ 1000)
ax3 = MK.Axis(fig[1, 3]; xlabel = "u [m/s]")
MK.lines!(ax3, u.(z), z ./ 1000)
ax4 = MK.Axis(fig[1, 4]; xlabel = "N [1/s]")
MK.lines!(ax4, N, z[2:(end - 1)] ./ 1000)
MK.vlines!(ax4, [0.01, 0.02]; color = :gray, linestyle = :dash)
MK.hideydecorations!.((ax2, ax3, ax4); grid = false)
MK.save("squall_line_sounding.png", fig)
nothing # hide
```

![Squall-line sounding](squall_line_sounding.png)

The warm-bubble trigger is applied to this sounding by
`center_initial_condition`; evaluating the bubble formula shows the
perturbation that seeds the storm:

```@example squall_line
xs = 50_000.0f0:500.0f0:100_000.0f0
zs = 0.0f0:100.0f0:5_000.0f0
Δθ = [
    (r = sqrt(((x - 75_000) / 10_000)^2 + ((z - 2_000) / 1_500)^2);
    r < 1 ? 3 * cospi(r / 2)^2 : 0.0f0) for x in xs, z in zs
]

fig = MK.Figure(; size = (700, 300))
ax = MK.Axis(fig[1, 1]; xlabel = "x [km]", ylabel = "z [km]")
hm = MK.heatmap!(ax, xs ./ 1000, zs ./ 1000, Δθ; colormap = :Reds)
MK.Colorbar(fig[1, 2], hm; label = "Δθ [K]")
MK.save("squall_line_bubble.png", fig)
nothing # hide
```

![Squall-line trigger bubble](squall_line_bubble.png)

## Running the case

Two ready-made configurations ship with the repository, identical except for
the microphysics scheme:

  - [`config/model_configs/gabersek_squall_line_2p5d_0M.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/model_configs/gabersek_squall_line_2p5d_0M.yml)
  - [`config/model_configs/gabersek_squall_line_2p5d_1M.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/model_configs/gabersek_squall_line_2p5d_1M.yml)

Both use Smagorinsky–Lilly subgrid diffusion with ``C_s = 0.18`` (and no
hyperdiffusion), a ``\sim 1`` km effective horizontal resolution, a 300 m
uniform vertical grid, and run for 9000 s. Start Julia in the project root
(`julia --project`) and execute:

```julia
import ClimaAtmos as CA

config = CA.AtmosConfig(
    "config/model_configs/gabersek_squall_line_2p5d_0M.yml";
    job_id = "gabersek_squall_0M",
)
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation)
```

The full run takes a few node-hours on CPU; reduce `t_end`, `x_elem`, and
`z_elem` for a quick look.

## Switching between 0M and 1M microphysics

The two configurations differ in a single physics choice, visible in their
YAML diff:

```yaml
# 0M configuration
microphysics_model: "0M"

# 1M configuration
microphysics_model: "1M"
cloud_model: "grid_scale"
```

The same switch works from a script by overriding the parsed configuration
before building the simulation. Constructing the configuration is cheap, so
the resolved model types can be compared directly:

```@example squall_line
case_yml = joinpath(
    pkgdir(CA), "config", "model_configs", "gabersek_squall_line_2p5d_0M.yml",
)
config_0M = CA.AtmosConfig(case_yml; job_id = "gabersek_squall_0M")

config_1M = CA.AtmosConfig(case_yml; job_id = "gabersek_squall_1M")
config_1M.parsed_args["microphysics_model"] = "1M"
config_1M.parsed_args["cloud_model"] = "grid_scale"

(
    typeof(CA.get_microphysics_model(config_0M.parsed_args)).name.name,
    typeof(CA.get_microphysics_model(config_1M.parsed_args)).name.name,
)
```

Physically, the two schemes bracket the treatment of precipitation:

  - **0-moment** (`EquilibriumMicrophysics0M`): cloud condensate is diagnosed
    by saturation adjustment, and precipitation is removed *instantaneously*
    wherever condensate exceeds a threshold. There are no falling
    hydrometeors, so there is no melting level and — most importantly for a
    squall line — no sub-cloud rain evaporation.
  - **1-moment** (`NonEquilibriumMicrophysics1M`): cloud liquid, cloud ice,
    rain, and snow are prognostic (`ρq_lcl`, `ρq_icl`, `ρq_rai`, `ρq_sno`),
    with autoconversion, accretion, sedimentation, rain evaporation, and
    snow melt. Rain falling out of the storm evaporates into the sub-cloud
    layer, and this evaporative cooling is the primary driver of the cold
    pool that organizes and maintains the line.

Consequently the 1M run develops a stronger, colder outflow, a delayed but
more realistic onset of surface precipitation, and a trailing region of
falling snow aloft, while the 0M run rains out immediately and produces a
weaker density current.

## Diagnostics

Both configurations write instantaneous output every 300 s. The shared list
covers the storm dynamics and thermodynamics — winds (`ua`, `wa`),
temperature and potential temperature (`ta`, `thetaa`), cloud condensate
(`clw`, `cli`), precipitation reaching the surface (`pr`), column water
(`lwp`, `iwp`, `prw`), and instability (`cape`). Two are worth calling out
for this case:

  - `thetaap`, the potential-temperature anomaly from the level-wise
    horizontal mean, is the natural field for visualizing the cold pool
    (a ``-2`` to ``-6`` K surface anomaly spreading from ``x = 75`` km) and
    the warm anomaly of the convective core.
  - The 1M configuration additionally outputs the hydrometeor fields
    (`husra`, `hussn`, `rwp`, `swp`) and the microphysical process rates
    `mp1m_S_acnv_lcl_rai` (autoconversion), `mp1m_S_accr_lcl_rai`
    (accretion), `mp1m_S_phase_change_vap_rai` (rain
    condensation/evaporation), and `mp1m_S_melt_sno_rai` (snow melt), which
    together budget the rain production and the evaporative cooling feeding
    the cold pool. None of these exist under 0M, which is the scientific
    content of the scheme switch made measurable.

A minimal comparison of the two schemes after both runs finish:

```julia
import ClimaAnalysis

sim_0M = ClimaAnalysis.SimDir("output/gabersek_squall_0M/output_active")
sim_1M = ClimaAnalysis.SimDir("output/gabersek_squall_1M/output_active")

# Surface precipitation onset and magnitude
pr_0M = ClimaAnalysis.get(sim_0M; short_name = "pr")
pr_1M = ClimaAnalysis.get(sim_1M; short_name = "pr")

# Cold-pool strength at the lowest model level
thetaap_1M = ClimaAnalysis.get(sim_1M; short_name = "thetaap")
```

See [Loading and Visualizing Output](visualizing_output.md) for working with
the output files.

## Expected evolution

At the shipped (coarse) resolution the storm follows the reference timeline:
cloud condensate appears between 300 and 600 s, surface precipitation begins
at ``t \approx 900`` s, and by 1500 s the updraft reaches
``\sim 15``–``20\;\mathrm{m\,s^{-1}}`` with a developing cold pool of several
kelvin in `thetaap`. Refining toward the resolutions of
[Gabersek2012](@cite) (``\Delta x \approx 250`` m) sharpens the gust front
and strengthens the simulated line.
