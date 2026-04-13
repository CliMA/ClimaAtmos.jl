"""
    MAGIC — pointwise column IC from DEPHY-SCM format MAGIC field campaign data.

Reads initial condition profiles from a DEPHY-SCM format NetCDF file and
builds 1D interpolators via `ColumnProfiles`. The
`center_initial_condition` evaluates these profiles at each grid height.

Surface conditions are read from the same file (initial surface temperature).
"""

"""
    MAGIC{FT}

Pointwise column IC driven by MAGIC field campaign NetCDF data (DEPHY-SCM format).

## Fields
- `external_forcing_file`: Path to the MAGIC forcing NetCDF file.
- `profiles`: `ColumnProfiles` of 1D interpolators `(T, u, v, q_tot, ρ)`.
- `T_sfc`: Initial surface temperature from the forcing file.

## Example
```julia
setup = MAGIC("path/to/MAGIC_LEG04A_DEF_driver.nc")
```
"""
struct MAGIC{P <: ColumnProfiles, FT}
    external_forcing_file::String
    profiles::P
    T_sfc::FT
end

"""
    mixing_ratio_to_specific_humidity(rv)

Convert mixing ratio `rv = q_v / q_d` to specific humidity `q = q_v / (q_d + q_v)`.
"""
mixing_ratio_to_specific_humidity(rv) = rv / (1 + rv)

"""
    MAGIC(external_forcing_file::String)

Construct a MAGIC setup by reading initial condition profiles from a
DEPHY-SCM format forcing file and building 1D vertical interpolators.
"""
function MAGIC(external_forcing_file::String)
    z_magic, T, u, v, q_tot, ρ, T_sfc = NC.NCDataset(external_forcing_file) do ds
        # Height coordinates for each variable (different dimensions in DEPHY format)
        # Convert to Float64 for consistency
        z_ta = Float64.(vec(ds["lev_ta"][:]))
        z_ua = Float64.(vec(ds["lev_ua"][:]))
        z_va = Float64.(vec(ds["lev_va"][:]))
        z_rv = Float64.(vec(ds["lev_rv"][:]))
        z_pa = Float64.(vec(ds["lev_pa"][:]))
        
        # Read initial condition profiles (t0 dimension = 1)
        # NCDatasets reverses dimension order from ncdump, so ta(t0, lev_ta) -> ta[lev_ta, t0]
        # Convert to Float64 for consistency
        T_raw = Float64.(vec(ds["ta"][:, 1]))
        u_raw = Float64.(vec(ds["ua"][:, 1]))
        v_raw = Float64.(vec(ds["va"][:, 1]))
        rv_raw = Float64.(vec(ds["rv"][:, 1]))
        pa_raw = Float64.(vec(ds["pa"][:, 1]))
        
        # Convert mixing ratio to specific humidity
        q_tot_raw = mixing_ratio_to_specific_humidity.(rv_raw)
        
        # Surface temperature
        sfc_T = Float64(ds["ts"][1])
        
        # Compute density from pressure and temperature using ideal gas law
        # ρ = p / (R_d * T * (1 + (R_v/R_d - 1)*q))
        R_d = 287.0  # J/(kg·K)
        R_v = 461.5  # J/(kg·K)
        
        # Use temperature height coordinate as reference
        # Interpolate pressure to temperature grid
        interp_p = Intp.extrapolate(
            Intp.interpolate((z_pa,), pa_raw, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
        pa_at_ta = interp_p.(z_ta)
        
        # Interpolate specific humidity to temperature grid
        interp_q = Intp.extrapolate(
            Intp.interpolate((z_rv,), q_tot_raw, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
        q_at_ta = interp_q.(z_ta)
        
        # Compute density
        ρ_raw = pa_at_ta ./ (R_d .* T_raw .* (1 .+ (R_v / R_d - 1) .* q_at_ta))
        
        # Interpolate u, v to the temperature height grid (reference)
        interp_u = Intp.extrapolate(
            Intp.interpolate((z_ua,), u_raw, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
        interp_v = Intp.extrapolate(
            Intp.interpolate((z_va,), v_raw, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
        
        u_at_ta = interp_u.(z_ta)
        v_at_ta = interp_v.(z_ta)
        
        (z_ta, T_raw, u_at_ta, v_at_ta, q_at_ta, ρ_raw, sfc_T)
    end

    return MAGIC(
        external_forcing_file,
        ColumnProfiles(z_magic, T, u, v, q_tot, ρ),
        T_sfc,
    )
end

center_initial_condition(setup::MAGIC, local_geometry, params) =
    column_profiles_ic(setup.profiles, local_geometry)

function surface_condition(setup::MAGIC, params)
    FT = eltype(params)
    parameterization = MoninObukhov(; z0 = FT(1e-4))
    return SurfaceState(; parameterization, T = FT(setup.T_sfc))
end

external_forcing(setup::MAGIC, ::Type{FT}) where {FT} =
    MAGICForcing{FT}(setup.external_forcing_file)

surface_temperature_model(::MAGIC) = MAGICTimeVaryingSST()
