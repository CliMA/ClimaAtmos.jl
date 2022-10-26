#####
##### Reference state
#####

import Thermodynamics as TD
import OrdinaryDiffEq as ODE
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import ClimaCore.Topologies as Topologies

"""
    compute_ref_pressure!(p::Fields.ColumnField, logpressure_fun)

Computes the hydrostatically balanced reference pressure, given
 - `logpressure_fun` a callable object by `logpressure_fun(z)`,
    which returns the log of the pressure
 - `p` the air pressure field (output)
"""
function compute_ref_pressure!(p::Fields.ColumnField, logpressure_fun)
    z = Fields.coordinate_field(axes(p)).z
    @. p .= exp(logpressure_fun(z))
    return nothing
end

"""
    compute_ref_density!(
        ρ::Fields.ColumnField,
        p::Fields.ColumnField,
        thermo_params::TD.Parameters.ThermodynamicsParameters,
        ts_g::TD.ThermodynamicState
    )

Computes density from the hydrostatically balanced air pressure, given
 - `ρ` the air density field (output)
 - `p` the air pressure field
 - `thermo_params` thermodynamic parameters
 - `ts_g` the surface (ground) reference state (a thermodynamic state)
"""
function compute_ref_density!(
    ρ::Fields.ColumnField,
    p::Fields.ColumnField,
    thermo_params::TD.Parameters.ThermodynamicsParameters,
    ts_g::TD.ThermodynamicState,
)
    grav = TD.Parameters.grav(thermo_params)
    vertical_domain = Topologies.domain(Spaces.vertical_topology(axes(ρ)))
    ᶠz_surf = vertical_domain.coord_min.z
    Φ_g = grav * ᶠz_surf
    q_tot_g = TD.total_specific_humidity(thermo_params, ts_g)
    mse_g = TD.moist_static_energy(thermo_params, ts_g, Φ_g)
    z = Fields.coordinate_field(axes(ρ)).z
    # Compute reference state thermodynamic profiles
    @. ρ = TD.air_density(
        thermo_params,
        TD.PhaseEquil_phq(
            thermo_params,
            p,
            mse_g - grav * z, # (mse - Φ) = enthalpy
            q_tot_g,
        ),
    )
    return nothing
end

"""
    log_pressure_profile(
        ᶠz::Spaces.AbstractSpace,
        thermo_params::TD.Parameters.ThermodynamicsParameters,
        ts_g::TD.ThermodynamicState,
    )

A hydrostatically balanced reference state (log of) pressure profile,
which can be interpolated by calling `sol(z)` on the result.
"""
function log_pressure_profile(
    ᶠz_space::Spaces.AbstractSpace,
    thermo_params::TD.Parameters.ThermodynamicsParameters,
    ts_g::TD.ThermodynamicState,
)
    q_tot_g = TD.total_specific_humidity(thermo_params, ts_g)
    vertical_domain = Topologies.domain(Spaces.vertical_topology(ᶠz_space))
    z_span = (vertical_domain.coord_min.z, vertical_domain.coord_max.z)
    ᶠz_surf = z_span[1]
    grav = TD.Parameters.grav(thermo_params)
    Φ_g = grav * ᶠz_surf
    mse_g = TD.moist_static_energy(thermo_params, ts_g, Φ_g)
    Pg = TD.air_pressure(thermo_params, ts_g)

    # We are integrating the log pressure so need to take the log of the
    # surface pressure
    logp = log(Pg)

    # Form a right hand side for integrating the hydrostatic equation to
    # determine the reference pressure
    function minus_inv_scale_height(logp, u, z)
        p_ = exp(logp)
        Φ = grav * z
        h = mse_g - Φ
        ts = TD.PhaseEquil_phq(thermo_params, p_, h, q_tot_g)
        R_m = TD.gas_constant_air(thermo_params, ts)
        T = TD.air_temperature(thermo_params, ts)
        return -grav / (T * R_m)
    end

    # Perform the integration
    prob = ODE.ODEProblem(minus_inv_scale_height, logp, z_span)
    sol = ODE.solve(prob, ODE.Tsit5(), reltol = 1e-12, abstol = 1e-12)
    return sol
end
