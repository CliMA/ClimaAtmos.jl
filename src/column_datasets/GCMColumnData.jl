"""
    GCMColumnData

Reader for GCM-driven single-column (cfsite) forcing files. Its data lives in a
per-site subgroup and is steady: the case is driven by time-mean profiles and
tendencies. [`read_cfsite`](@ref) builds an [`InMemoryColumnData`](@ref) of those
time-mean profiles, so the case runs through the same `ExternalDrivenTVForcing`
per-term composition and `ForcingFromFile` setup as any other column source.

Same pattern as the VARANAL and ERA5 converters (one reader per source), except
the GCM data never touches disk in the ClimaColumn schema: it stays in memory
because the cfsite file is a read-only artifact and the forcing is steady.
"""
module GCMColumnData

import NCDatasets as NC
import Statistics: mean
import Thermodynamics.Parameters as TDP

import ..ColumnDatasets: InMemoryColumnData

"""
    REQUIRED_VARS

The cfsite subgroup variables [`read_cfsite`](@ref) reads. Checked up front so a
file missing any of them errors at setup, naming all that are absent, rather
than on first access.
"""
const REQUIRED_VARS = (
    "zg",       # geopotential height [m]
    "ta",       # air temperature [K]
    "ua",       # eastward wind [m s-1]
    "va",       # northward wind [m s-1]
    "hus",      # specific humidity [kg kg-1]
    "tntha",    # temperature horizontal-advection tendency [K s-1]
    "tnhusha",  # humidity horizontal-advection tendency [kg kg-1 s-1]
    "tntva",    # temperature vertical-advection tendency [K s-1]
    "tnhusva",  # humidity vertical-advection tendency [kg kg-1 s-1]
    "alpha",    # specific volume [m3 kg-1]
    "wap",      # pressure velocity [Pa s-1]
    "ts",       # surface (skin) temperature [K]
    "coszen",   # cosine of the solar zenith angle [1]
    "rsdt",     # TOA incoming shortwave radiation [W m-2]
)

"""
    read_cfsite(path, cfsite_number; thermo_params)

Read the cfsite subgroup `cfsite_number` from the GCM forcing file `path` and
return a steady [`InMemoryColumnData`](@ref).

The column variables are time means: `ta`, `ua`, `va`, `hus`, and the
horizontal-advection tendencies `tntha`/`tnhusha`. Subsidence is
`wa = w̄ₐₚ · (−ᾱ) / g` (mean pressure velocity and specific volume), density is
`mean(1/α)`, and the vertical-fluctuation tendencies `tntva`/`tnhusva` are the
eddy part of the GCM's total vertical advective tendency, `tntva + w̄ · ∂χ̄/∂z`,
with the vertical gradient differenced on the GCM grid (Shen et al. 2022).

Differencing `∂χ̄/∂z` on the GCM grid is a behavior change: the previous
`GCMForcing` cache interpolated to the model grid first and differenced there.
Taking the gradient where the data lives is more accurate, but it shifts the
forcing wherever the two grids differ, most at sharp gradients.

Surface variables reproduce the previous `GCMDrivenInsolation`: the skin
temperature `ts` (time mean), and a constant `coszen`/`rsdt` chosen so that
`coszen = coszen[1]` and `rsdt / coszen = mean(rsdt / coszen)`.
"""
function read_cfsite(path, cfsite_number; thermo_params)
    isnothing(path) && error(
        "initial_condition `GCM` requires `external_forcing_file` to point at a \
         GCM cfsite forcing file",
    )
    g = TDP.grav(thermo_params)
    NC.NCDataset(path, "r") do ds
        groups = keys(ds.group)
        cfsite_number in groups || error(
            "GCM forcing file $(path) has no cfsite group `$(cfsite_number)`; \
             it contains $(join(groups, ", ")).",
        )
        site = ds.group[cfsite_number]
        missing_vars = [v for v in REQUIRED_VARS if !haskey(site, v)]
        isempty(missing_vars) || error(
            "cfsite group `$(cfsite_number)` of $(path) is missing the \
             required variables $(join(missing_vars, ", ")).",
        )
        tmean(v) = vec(mean(site[v][:, :], dims = 2))

        z = tmean("zg")
        order = sortperm(z)

        ta = tmean("ta")
        ua = tmean("ua")
        va = tmean("va")
        hus = tmean("hus")
        tntha = tmean("tntha")
        tnhusha = tmean("tnhusha")
        alpha = tmean("alpha")
        wap = tmean("wap")
        wa = wap .* (.-alpha) ./ g
        # Density from the time mean of the specific volume (matches the former
        # GCMDriven initial condition), not the reciprocal of the mean.
        rho = vec(mean(inv, site["alpha"][:, :], dims = 2))

        # Eddy vertical fluctuation on the GCM grid: the GCM's total vertical
        # advective tendency with the mean advection removed.
        tntva = tmean("tntva") .+ wa .* _ddz(ta, z)
        tnhusva = tmean("tnhusva") .+ wa .* _ddz(hus, z)

        # Constant insolation matching the former GCMDrivenInsolation.
        ts = mean(site["ts"][:])
        coszen = Float64(site["coszen"][1])
        rsdt = mean(site["rsdt"][:] ./ site["coszen"][:]) * coszen

        col(v) = Float64.(v[order])
        return InMemoryColumnData(;
            z = z[order],
            column = (;
                ta = col(ta),
                ua = col(ua),
                va = col(va),
                hus = col(hus),
                wa = col(wa),
                rho = col(rho),
                tntha = col(tntha),
                tnhusha = col(tnhusha),
                tntva = col(tntva),
                tnhusva = col(tnhusva),
            ),
            surface = (; ts, coszen, rsdt),
            source = "GCM cfsite `$(cfsite_number)` from $(basename(path))",
        )
    end
end

# Vertical derivative `df/dz` by central differences (one-sided at the ends) on
# a non-uniform grid `z`. Works for ascending or descending `z`.
function _ddz(f, z)
    n = length(z)
    d = similar(f)
    d[1] = (f[2] - f[1]) / (z[2] - z[1])
    d[n] = (f[n] - f[n - 1]) / (z[n] - z[n - 1])
    for i in 2:(n - 1)
        d[i] = (f[i + 1] - f[i - 1]) / (z[i + 1] - z[i - 1])
    end
    return d
end

end # module
