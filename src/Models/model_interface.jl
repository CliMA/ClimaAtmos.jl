"""
    make_initial_conditions
"""
function default_initial_conditions(model::AbstractModel) end

"""
    make_ode_function
"""
function make_ode_function(model::AbstractModel) end

"""
    get_boundary_flux
"""
function get_boundary_flux end
