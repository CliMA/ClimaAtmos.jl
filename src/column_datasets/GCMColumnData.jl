"""
    GCMColumnData

Reader for GCM-driven single-column (cfsite) forcing files. Its data lives in a
per-site subgroup and is steady: the case is driven by time-mean profiles and
tendencies. [`read`](@ref) builds an [`InMemoryColumnData`](@ref) of those
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
    read(path, cfsite_number; thermo_params)

Read the cfsite subgroup `cfsite_number` from the GCM forcing file `path` and
return a steady [`InMemoryColumnData`](@ref).

The column variables are time means: `ta`, `ua`, `va`, `hus`, and the
horizontal-advection tendencies `tntha`/`tnhusha`. Subsidence is
`wa = w̄ₐₚ · (−ᾱ) / g` (mean pressure velocity and specific volume), density is
`mean(1/α)`, and the vertical-fluctuation tendencies `tntva`/`tnhusva` are the
eddy part of the GCM's total vertical advective tendency, `tntva + w̄ · ∂χ̄/∂z`,
with the vertical gradient differenced on the GCM grid (Shen et al. 2022).

Surface variables reproduce the previous `GCMDrivenInsolation`: the skin
temperature `ts` (time mean), and a constant `coszen`/`rsdt` chosen so that
`coszen = coszen[1]` and `rsdt / coszen = mean(rsdt / coszen)`.
"""
function read(path, cfsite_number; thermo_params)
    g = TDP.grav(thermo_params)
    NC.NCDataset(path, "r") do ds
        site = ds.group[cfsite_number]
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
