using Test
import ClimaAtmos.Parameters as CAP

include(joinpath(@__DIR__, "..", "src", "DGDycore.jl"))
using .DGDycore
const DG = DGDycore

@testset "DGDycore" begin
    include("test_constants_parity.jl")
    include("test_pressure.jl")
    include("test_ic_parity.jl")
    include("test_hs_equivalence.jl")
    include("test_vi_core.jl")
    include("test_vi_kep_budget.jl")
    include("test_es2.jl")
    include("test_hydrostatic_composition.jl")
    include("test_wb_gravity_fddg.jl")
    include("test_mountain_wave.jl")
    include("test_no_dss.jl")
end
