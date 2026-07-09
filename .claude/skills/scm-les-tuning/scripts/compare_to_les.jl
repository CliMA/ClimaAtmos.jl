#!/usr/bin/env julia
# Compare SCM column-sweep timeseries diagnostics against an LES Stats.*.nc file.
#
# Usage:
#   julia --project=.buildkite compare_to_les.jl <les_nc> <sweep_basedir> <var1,var2,...> <label1=dir1> [<label2=dir2> ...]
#
# <dir> is the directory (relative to <sweep_basedir>) that nc_files.tar was
# extracted into; <var> are diagnostic short names whose *.nc files are named
# "<var>_1h_average.nc" and whose LES counterpart lives in the LES file's
# timeseries group (e.g. iwp, swp, lwp, rwp).
#
# Variable name mapping: use "model_var:les_var" to handle different names
# between model output and LES. e.g. "cl:cloud_fraction".
#
# Profile variables: if the model variable is 2-D (z × time), the column
# maximum is used as the scalar summary (maximum-overlap cloud fraction).
#
# Example:
#   julia --project=.buildkite compare_to_les.jl \
#     /Users/zshen/Work/clima_repo/les_data/Stats.TRMM_LBA.nc \
#     /tmp/trmm_sweep iwp,swp,cl:cloud_fraction \
#     baseline=extracted_baseline detr01=extracted_detrvertdiv_01

using NCDatasets
using Printf

length(ARGS) >= 4 || error("usage: compare_to_les.jl <les_nc> <sweep_basedir> <var1,var2,...> <label1=dir1> [...]")

les_nc, basedir, varlist = ARGS[1], ARGS[2], ARGS[3]
labelpairs = ARGS[4:end]

# Parse var specs: "model_var" or "model_var:les_var"
var_specs = map(split(varlist, ",")) do spec
    parts = split(spec, ":")
    length(parts) == 2 ? (String(parts[1]), String(parts[2])) : (String(parts[1]), String(parts[1]))
end
model_vars = [s[1] for s in var_specs]
les_vars   = [s[2] for s in var_specs]

les = NCDataset(les_nc)
les_t = les.group["timeseries"]["t"][:]
# Compare against the LES timestep nearest the column run's end time.
# 21600s = 6h matches this repo's standard TRMM/DYCOMS column run length;
# pass a different ARGS entry or edit here if the run end time differs.
target_t = 21600.0
idx = argmin(abs.(les_t .- target_t))

les_vals = Dict(lv => les.group["timeseries"][lv][idx] for lv in les_vars)
print("LES @ t=", les_t[idx], "s: ")
for (mv, lv) in var_specs
    @printf("%s=%.4f  ", mv, les_vals[lv])
end
println("\n")

for pair in labelpairs
    label, dir = split(pair, "=", limit = 2)
    print(rpad(label, 24))
    for (mv, lv) in var_specs
        ds = NCDataset(joinpath(basedir, dir, "$(mv)_1h_average.nc"))
        arr = ds[mv][axes(ds[mv])...]  # works for both 1-D (time,) and 2-D (time, z)
        # For profile variables (time, z): take column max at last time step
        val = ndims(arr) == 2 ? maximum(arr[end, :]) : arr[end]
        les_val = les_vals[lv]
        @printf(" %s=%.4f (%4.0f%%)", mv, val, 100 * val / les_val)
    end
    println()
end
