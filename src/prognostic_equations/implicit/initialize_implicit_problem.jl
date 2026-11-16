#####
##### Initialize implicit problem
#####

import ClimaCore
import ClimaCore: Fields, Spaces

"""
    initialize_implicit_stage_problem!(Y, p, dtγ)

Prepare the state `Y` for an IMEX implicit stage with step fraction `dtγ`,
registered as the `initialize_imp!` hook of `ClimaODEFunction`. Records the
post-DSS stage-entry specific internal-plus-kinetic energy for the post-ARK
transport step and returns `nothing`.
"""
function initialize_implicit_stage_problem!(Y, p, dtγ)
    record_implicit_stage_entry_energy!(Y, p)
    return nothing
end
