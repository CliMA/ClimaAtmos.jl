"""
function add_climate_machine()

# Description
Grabs the particular ClimateMachine branch used in ClimaAtmos
"""
function add_climate_machine()
       Pkg.add(url = "https://github.com/CliMA/ClimateMachine.jl.git#tb/refactoring_ans_sphere")
end

"""
function add_climate_machine()

# Description
Grabs the particular ClimaCore branch used in ClimaAtmos
"""
function add_clima_core()
       Pkg.add(url = "https://github.com/CliMA/ClimaCore.jl.git")
       Pkg.add("DiffEqBase")
end