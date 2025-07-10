import Thermodynamics as TD

"""
    total_specific_humidity_from_RH(thermo_params, T, p, relative_humidity)

Compute total specific humidity from relative humidity, temperature and pressure.

# Theory
The calculation is based on the following relationships:

1. Relative humidity is defined as the ratio of vapor pressure to saturation vapor pressure:
   ```
   RH = e_v/e_sat
   ```
   where:
   - e_v is the actual vapor pressure
   - e_sat is the saturation vapor pressure at temperature T

2. Specific humidity is defined as:
   ```
   q = (ε * e_v) / (p - e_v + ε * e_v)
   ```
   where:
   - ε is the ratio of gas constants (R_d/R_v)
   - p is total pressure
   - e_v is vapor pressure

3. For a given relative humidity:
   ```
   e_v = e_sat * RH
   ```

# Arguments
- `thermo_params`: Thermodynamic parameters
- `T`: Temperature in K
- `p`: Pressure in Pa
- `relative_humidity`: Relative humidity (0-1)

# Returns
- Total specific humidity in kg/kg

# References
- Pressel et al. (2015), equation 37
- Rogers, R. R., & Yau, M. K. (1989). A Short Course in Cloud Physics (3rd ed.). 
  Butterworth-Heinemann. ISBN: 9780750632157
"""
function total_specific_humidity_from_RH(thermo_params, T, p, relative_humidity)
    # Get molecular mass ratio (ε = R_d/R_v)
    molmass_ratio = TD.Parameters.molmass_ratio(thermo_params)

    # Calculate saturation vapor pressure (e_sat)
    p_v_sat = TD.saturation_vapor_pressure(thermo_params, T, TD.Liquid())

    # Calculate denominator with RH term
    # p - e_v + ε * e_v where e_v = e_sat * RH
    denominator =
        p - p_v_sat + (1 / molmass_ratio) * p_v_sat * relative_humidity

    # Calculate q_v_sat and scale by RH
    q_v_sat = p_v_sat * (1 / molmass_ratio) / denominator
    q_tot = q_v_sat * relative_humidity

    return q_tot
end
