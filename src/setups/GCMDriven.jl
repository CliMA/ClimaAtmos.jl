"""
    GCMDriven — pointwise column IC from GCM forcing files.

Reads time-averaged vertical profiles from a GCM forcing NetCDF file and
builds 1D interpolators via `ColumnProfiles`. The
`center_initial_condition` evaluates these profiles at each grid height.

Surface conditions are read from the same file (mean surface temperature).
"""

"""
    GCMDriven{FT}

Pointwise column IC driven by GCM forcing NetCDF data.

## Fields
- `external_forcing_file`: Path to the GCM forcing NetCDF file.
- `cfsite_number`: Site identifier within the NetCDF file (e.g., `"site23"`).
- `profiles`: `ColumnProfiles` of 1D interpolators `(T, u, v, q_tot, ρ)`.
- `T_sfc`: Mean surface temperature from the forcing file.

## Example
```julia
setup = GCMDriven("path/to/HadGEM2-A_amip.2004-2008.07.nc", "site23")
```
"""
struct GCMDriven{P <: ColumnProfiles, FT}
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
    z_gcm, T, u, v, q_tot, ρ, T_sfc = NC.NCDataset(external_forcing_file) do ds
        site = ds.group[cfsite_number]
        z = vec(gcm_height(site))
        T = gcm_driven_profile_tmean(site, "ta")
        u = gcm_driven_profile_tmean(site, "ua")
        v = gcm_driven_profile_tmean(site, "va")
        q_tot = gcm_driven_profile_tmean(site, "hus")
        ρ = vec(mean(inv, site["alpha"][:, :], dims = 2))
        sfc_T = mean(gcm_driven_timeseries(site, "ts"))
        (z, T, u, v, q_tot, ρ, sfc_T)
    end

    return GCMDriven(
        external_forcing_file,
        cfsite_number,
        ColumnProfiles(z_gcm, T, u, v, q_tot, ρ),
        T_sfc,
    )
end

center_initial_condition(setup::GCMDriven, local_geometry, params) =
    column_profiles_ic(setup.profiles, local_geometry)

function surface_condition(setup::GCMDriven, params)
    FT = eltype(params)
    parameterization = MoninObukhov(; z0 = FT(1e-4))
    return SurfaceState(; parameterization, T = FT(setup.T_sfc))
end

external_forcing(setup::GCMDriven, ::Type{FT}) where {FT} =
    GCMForcing{FT}(setup.external_forcing_file, setup.cfsite_number)

insolation_model(::GCMDriven) = GCMDrivenInsolation()
