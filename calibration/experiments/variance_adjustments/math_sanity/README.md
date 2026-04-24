# math_sanity

Development harness for scalar quadrature checks and mosaic figures (Gaussian SGS baseline, subcell `(1/12)Δz²`-style geometry). It is **not** a second specification: the authoritative write-up for this branch is [`../README.md`](../README.md).

## Drivers

- `run_all.jl` — summary figures plus optional Gaussian mosaics (see inline comments for panel counts).
- `run_sweep.jl` — mosaics-focused driver.

Use these scripts only when iterating on visuals or offline checks; regression coverage for vertical-profile SGS belongs in `test/` plus ClimaAtmos smoke workflows.
