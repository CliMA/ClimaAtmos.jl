"""
    default_initial_conditions(model)

Construct the initial conditions for `model`.
"""
function default_initial_conditions(model::AbstractModel) end

"""
    make_ode_function(model)

Construct the ordinary differential equations for `model`.
"""
function make_ode_function(model::AbstractModel) end

"""
    get_velocities

Get velocity components from model
"""
function get_velocities(u,model::AbstractModel) end

"""
    get_boundary_flux

Construct the boundary fluxes.
"""
function get_boundary_flux end
