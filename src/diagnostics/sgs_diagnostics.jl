compute_kinematic_viscosity!(out, state, cache, time) =
    compute_kinematic_viscosity!(out, state, cache, time, cache.atmos.smagorinsky_lilly)
compute_kinematic_viscosity!(_, _, _, _, sgs_model::SGS) where {SGS} =
    error_diagnostic_variable("kinematic_viscosity", sgs_model)

function compute_kinematic_viscosity!(
    out,
    state,
    cache,
    time,
    sgs_model::SGS,
) where {SGS <: SmagorinskyLilly}
    if isnothing(out)
        return cache.smagorinsky_lilly.v_t
    else
        out .= cache.smagorinsky_lilly.v_t
    end
end
add_diagnostic_variable!(
    short_name = "kvis",
    long_name = "Kinematic Viscosity",
    standard_name = "kinematic_viscosity",
    units = "m^2 s^-1",
    comments = "Kinematic viscosity",
    compute! = compute_kinematic_viscosity!
)
