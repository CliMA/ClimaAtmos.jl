# ClimaAtmos.jl workshop talks

Two sessions live in this folder.

| File | Slot | Kind |
|------|------|------|
| `main.tex` | ~45 min | Overview talk (science framing, ecosystem, LES / squall hook) |
| `tutorial_1h.tex` | 60 min | Hands-on: clone, instantiate, first runs, plots |

---

## Tutorial — 1 hour hands-on

Running and exploring variables. Grids and durations are coarse/short on
purpose; this is **not** a session on physically relevant scales.

### Participant files

| File | Use |
|------|-----|
| `tutorial_1h.jl` | Copy-paste into `julia +1.11 --project` (column + sphere) |
| `tutorial_1h_plots.jl` | Plots, in a **default** Julia session |
| `workshop_configs/` | Optional YAML overlays if you prefer the config API |

### Timing

| min | Block | If instantiate is pre-done |
|-----|--------|----------------------------|
| 0–5 | Welcome, `Y` / diagnostics | same |
| 5–15 | `git clone`, tree tour | shrink; start REPL early |
| 15–25 | `Pkg.instantiate()` + first `import` | skip; inspect `Y` instead |
| 25–45 | BOMEX column + dry baroclinic wave | more time on `propertynames` / `step!` |
| 45–55 | ClimaAnalysis plots | extra slices |
| 55–60 | Cheat sheet, Q&A | same |

**Start instantiate as soon as the clone exists.** First precompile and the
first `solve_atmos!` are the long poles. Do not restart the REPL.

### Instructor homework (the day before)

1. Julia 1.11 via [juliaup](https://julialang.org/install/).
2. Clone, `julia +1.11 --project -e 'using Pkg; Pkg.instantiate()'`.
3. In a default env: `Pkg.add(["ClimaAnalysis","CairoMakie"])`.
4. Run `tutorial_1h.jl` once so the compile cache is warm and you have
   sample output.
5. If the room has poor wifi, share a pre-warmed `JULIA_DEPOT_PATH`.

Ask participants to do steps 1–3 before they arrive if you can. Instantiation
can take tens of minutes on a cold machine.

### Live commands (canonical)

```bash
git clone https://github.com/CliMA/ClimaAtmos.jl.git
cd ClimaAtmos.jl
julia +1.11 --project
```

```julia
using Pkg
Pkg.instantiate()
import ClimaAtmos as CA
include("talks/tutorial_1h.jl")   # or paste section by section
```

Plotting (new terminal, **no** `--project`):

```bash
julia +1.11
```

```julia
import Pkg; Pkg.add(["ClimaAnalysis", "CairoMakie"])  # once
include("talks/tutorial_1h_plots.jl")
```

Do not `Pkg.add` plotting packages into the ClimaAtmos project.

### What we run, and what we skip

- **Column:** `CA.Presets.bomex` — moist 0-moment, no EDMF, 1 hour.
  The published PROPHET case is
  `config/model_configs/prognostic_edmfx_bomex_column.yml`; name it, do not
  run it live.
- **Sphere:** `CA.Presets.baroclinic_wave` — dry, `h_elem = 6`, 1 day.
  A developed wave is an 8–10 day integration; we are looking at fields.
- Default diagnostics write nothing below 1 hour, and only a time average
  at 1 hour / 1 day. The scripts request instantaneous snapshots instead.
  Do not request `hus` on the dry wave.

### Build the slides

```
cd talks
latexmk -pdf tutorial_1h.tex     # or: pdflatex twice
```

Speaker notes are in `\note{}`. Enable with `\setbeameroption{show notes}`.

The theme block at the top of the `.tex` matches `main.tex`.

---

## Talk 1 — overview ("Next-generation earth-system modelling")

Opening talk of the how-to series. Mixed/general-science audience; framing is
*come collaborate and use it* for extreme-event / extreme-rainfall science, with LES and the
Gaberšek squall line (`as/squall`) as the group hook. Calibration is a forward pointer only
(PI's talk the next day). No WIP branches featured except the requested squall case.

Full slide outline & source map: `~/.claude/plans/parallel-questing-dahl.md`.

### Sources

Content is grounded in the ClimaAtmos & ClimaCore docs (`docs/src/`, `docs/bibliography.bib`,
`docs/refs.bib`) and the **CliMA blog** (`clima.caltech.edu/blog`) — notably *An Entirely New
Earth System Model* (Schneider & Ferrari, 2026), *CliMA 0.1* (2020), the *ClimateMachine LES*
post (2022), and the per-package posts (Thermodynamics/Titan, RRTMGP, Insolation, CloudMicrophysics).
The expanded introduction (legacy-model framing, the 2018 founding question, the compute/data/ML
opportunity, "physics as far as it goes, learn from data beyond", and "change parameters, not
code") comes from those posts.

**Note on length:** the added context slides push the talk to ~42–43 min of content. For a 45-min
slot with Q&A, trim the numerics (§5) and physics (§7) light-touch sections first — they are
intentionally shallow in talk 1.

### Build

```
latexmk -pdf main.tex     # or: pdflatex twice
```

Compiles as-is with a stock TeX Live (no external images required — see below).

### Applying your own theme

The theme is isolated in one block at the top of the `.tex`, marked
`>>> THEME BLOCK <<<`. Replace that block (currently `\usetheme{Madrid}` + color tweaks)
with your academic theme. Content slides are theme-independent; the two custom colors
`climablue` / `climateal` are used by the TikZ diagrams — redefine them to match your palette.

### Remaining figure placeholders

Two schematics are already drawn natively in TikZ (ecosystem stack; scale-ladder), so they need
nothing. The dashed `\figbox{...}` boxes are the ones still to fill — each is labelled with its
intended source. Replace with `\includegraphics[...]{path}`:

| Slide | Placeholder | Suggested asset |
|-------|-------------|-----------------|
| CliMA's answer | box-to-globe montage | `ClimaCore.jl/docs/src/assets/dry_baroclinic_wave.png` |
| The grid | cubed-sphere + column | `ClimaCore.jl/docs/src/assets/cubed_sphere_extruded.png` (or `APIobjects.png`) |
| Performance | scaling curves | `ClimaCore.jl/docs/src/assets/{weak_scaling,strong_scaling}.png` |
| LES | representative LES field | group plots / Sridhar et al. 2022 figures |
| Squall line | precip / reflectivity / w | your `as/squall` run outputs |

### Speaker notes

Per-slide notes live in `\note{}`. To show them, add near the top:
`\setbeameroption{show notes}` (or `show notes on second screen` for a presenter view).
