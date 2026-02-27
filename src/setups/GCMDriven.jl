"""
    GCMDriven — pointwise column IC from GCM forcing files.

Reads time-averaged vertical profiles from a GCM forcing NetCDF file and
builds 1D interpolators. The `center_initial_condition` evaluates these
profiles at each grid height.

Surface conditions are read from the same file (mean surface temperature).
"""

"""
    GCMDriven{P, FT}

Pointwise column IC driven by GCM forcing NetCDF data.

## Fields
- `external_forcing_file`: Path to the GCM forcing NetCDF file.
- `cfsite_number`: Site identifier within the NetCDF file (e.g., `"site23"`).
- `profiles`: NamedTuple of 1D interpolators `(T, u, v, q_tot, ρ)`.
- `T_sfc`: Mean surface temperature from the forcing file.

## Example
```julia
setup = GCMDriven("path/to/HadGEM2-A_amip.2004-2008.07.nc", "site23")
```
"""
struct GCMDriven{P, FT}
    external_forcing_file::String
    cfsite_number::String
    profiles::P
    T_sfc::FT
end

"""
    GCMDriven(external_forcing_file, cfsite_number)

Construct a GCMDriven setup by reading time-averaged profiles from a
GCM forcing file and building 1D vertical interpolators.
"""
function GCMDriven(external_forcing_file::String, cfsite_number::String)
    z_gcm, vars, T_sfc = NC.NCDataset(external_forcing_file) do ds
        site = ds.group[cfsite_number]
        z = vec(gcm_height(site))
        profiles = (
            gcm_driven_profile_tmean(site, "ta"),
            gcm_driven_profile_tmean(site, "ua"),
            gcm_driven_profile_tmean(site, "va"),
            gcm_driven_profile_tmean(site, "hus"),
            # Convert specific volume (alpha) to density: ρ = 1/α
            vec(mean(1 ./ site["alpha"][:, :], dims = 2)),
        )
        sfc_T = mean(gcm_driven_timeseries(site, "ts"))
        (z, profiles, sfc_T)
    end

    T_prof, u_prof, v_prof, q_tot_prof, ρ_prof = map(vars) do value
        Intp.extrapolate(
            Intp.interpolate((z_gcm,), value, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
    end

    profiles = (; T = T_prof, u = u_prof, v = v_prof, q_tot = q_tot_prof, ρ = ρ_prof)
    return GCMDriven(external_forcing_file, cfsite_number, profiles, T_sfc)
end

function center_initial_condition(setup::GCMDriven, local_geometry, params)
    (; profiles) = setup
    (; z) = local_geometry.coordinates
    FT = typeof(z)

    T = FT(profiles.T(z))
    q_tot = FT(profiles.q_tot(z))
    ρ = FT(profiles.ρ(z))

    return physical_state(;
        T,
        ρ,
        q_tot,
        u = FT(profiles.u(z)),
        v = FT(profiles.v(z)),
        tke = FT(0),
    )
end

function surface_condition(setup::GCMDriven, params)
    FT = eltype(params)
    return (; T = FT(setup.T_sfc), z0 = FT(1e-4))
end
