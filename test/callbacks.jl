using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import SciMLBase as SMB
import ClimaTimeSteppers.Callbacks as CB

testfun!() = π
test_nsteps = 999
test_dt = 1
test_tend = 999.0
cb_1 = CB.EveryXSimulationSteps(
    CA.AtmosCallback(testfun!, CA.EveryNSteps(test_nsteps)),
    test_nsteps,
    atinit = true,
    call_at_end = false,
)
cb_2 = CB.EveryXSimulationTime(
    CA.AtmosCallback(testfun!, CA.EveryΔt(test_dt)),
    test_dt;
    atinit = true,
    call_at_end = false,
)
cb_3 = CA.callback_from_affect(cb_2.affect!)
cb_4 = CB.EveryXSimulationSteps(
    CA.AtmosCallback(testfun!, CA.EveryNSteps(3)),
    3,
    atinit = true,
    call_at_end = false,
)
cb_set = SMB.CallbackSet(cb_1, cb_2, cb_4)

@testset "atmos callbacks and callback sets" begin
    # atmoscallbacks from discrete callbacks
    @test cb_3.f!() == π
    atmos_cbs = CA.atmos_callbacks(cb_set)
    # test against expected callback outcomes
    tspan = [0, test_tend]
    @test CA.n_expected_calls(cb_set, test_dt, tspan)[2] == test_tend
    @test CA.n_steps_per_cycle_per_cb(cb_set, test_dt) ==
          [test_nsteps; test_dt; 3]
end

# query functions
