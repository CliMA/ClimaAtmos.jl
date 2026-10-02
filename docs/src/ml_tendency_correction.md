# ML Tendency Correction

The ML tendency correction adds a learned, state-dependent forcing to the
temperature and humidity equations. An offline-trained network predicts, from
the instantaneous model state, the rate at which a short forecast drifts away
from ERA5 (CliMA − ERA5, per hour); the model then adds `-gain` times that
prediction as a tendency, nudging itself against its own systematic drift
without seeing any reanalysis at run time.

The correction is off by default. It is switched on by setting the
`ml_correction` configuration key to the path of an exported network, and it
requires a moist model (`microphysics_model` other than `dry`):

| Key                       | Default    | Meaning                                                                   |
|:------------------------- |:---------- |:------------------------------------------------------------------------- |
| `ml_correction`           | `~` (off)  | Path of the exported network (NetCDF)                                     |
| `ml_correction_gain`      | `0.5`      | The applied tendency is `-gain` times the predicted drift rate            |
| `ml_correction_dt`        | `"1hours"` | Interval between recomputations; the correction is held fixed in between  |
| `ml_correction_variables` | `"t"`      | Corrected variables: `t`, `q`, or `tq`                                    |
| `ml_correction_t_cap`     | `0.5`      | Maximum magnitude of the temperature correction [K per hour]              |
| `ml_correction_q_cap`     | `2.0e-4`   | Maximum magnitude of the humidity correction [kg/kg per hour]             |
| `ml_correction_p_full`    | `10000.0`  | Full strength at `p ≥ p_full` [Pa]                                        |
| `ml_correction_p_zero`    | `5000.0`   | Zero at `p ≤ p_zero`, half-cosine taper in between [Pa]                   |
| `ml_correction_smoothing` | `0.0`      | Gaussian smoothing of the prediction on the network grid [cells]; 0 = off |
| `ml_correction_start`     | `"0secs"`  | Simulation time at which the correction switches on                       |
| `ml_correction_ramp`      | `"0secs"`  | Linear ramp from 0 to full strength after `ml_correction_start`           |

For example, to correct temperature only at half gain, refreshed hourly:

```yaml
microphysics_model: "0M"
ml_correction: "/path/to/col_w3h_ctx_julia.nc"
ml_correction_variables: "t"
ml_correction_gain: 0.5
```

## How the correction is computed

The network was trained on CliMA output written by the pressure-coordinate
diagnostics and regridded to a 1° latitude-longitude grid, so the correction
reproduces that path rather than evaluating the network on model columns.
Every `ml_correction_dt` (and once at initialization), [`ml_correction_update!`](@ref):

 1. interpolates `T`, the eastward and northward wind, `q_tot`, and `p` to the
    network's pressure levels column by column, as the pressure diagnostics do
    (`ClimaCore.Remapping.PressureInterpolator`), and bilinearly remaps them to
    the network's grid, gathered on the root process;
 2. on the root process, evaluates the network ensemble
    ([`ml_correction_on_grid`](@ref)) and broadcasts the result;
 3. on every process, regrids each level back to the model's horizontal space
    and interpolates in `log p` to the model levels.

Between refreshes the cached correction is added every step through the same
temperature-and-humidity forcing used by single-column cases
(`apply_Tq_forcing!`), which converts it to total-energy and total-water
tendencies. The correction is explicit.

The network sees what it saw in training:

  - The reference surface pressure is `p` at the highest-pressure level, i.e. the
    lowest model level pressure capped at that level, and below-ground
    temperatures are extrapolated with a 6.5 K/km lapse rate from it.
  - Levels below the reference surface pressure, and columns within two cells of
    a terrain edge, are masked; the correction there is filled with that of the
    lowest valid level above.
  - The cosine of the solar zenith angle is given at the start, middle, and end
    of a training-length window centered on the coming refresh interval.

The correction is limited in three ways. Temperature and humidity are clipped to
`±ml_correction_t_cap` and `±ml_correction_q_cap` after the gain. Both are
multiplied by a half-cosine weight in pressure that is 1 at `p ≥ p_full`
(100 hPa) and 0 at `p ≤ p_zero` (50 hPa), so the correction acts in the
troposphere and lower stratosphere only: it does not fight the sponge layer, and
it vanishes above the network's top level, where the vertical interpolation only
extrapolates. Finally, the humidity correction, applied to `q_tot`, cannot remove
more than the local `q_tot` within one refresh interval.

## The network file

The file is written by `tendency_corr/export_julia.py` in the training
repository. It holds the weights and normalization of every ensemble member,
the pressure levels, the latitude-longitude grid, the static orography and land
fraction on that grid, and a reference case (inputs and torch predictions) for
checking the Julia forward pass. [`load_column_net`](@ref) reads it into a
[`ColumnNet`](@ref); the forward pass ([`column_net_predict`](@ref)) is written
as dense matrix products and shifted array views, so it runs unchanged on CPU
arrays and on the GPU.

The parity test in `test/parameterized_tendencies/ml_correction.jl` compares
the Julia prediction with the exported torch reference; it runs when the
environment variable `CLIMAATMOS_ML_CORRECTION_NET` points to a network file.

## Cost and parallelism

Only the root process loads and evaluates the network, on its device. On a CPU
an evaluation takes tens of seconds per ensemble member for the 1° grid, so CPU
runs are useful for testing only.
The remapping gathers one global 3D array per refresh to the root process, and
the broadcast sends the `(lon, lat, level, 2)` correction back to every process.
