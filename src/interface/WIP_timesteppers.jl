abstract type AbstractTimestepper end

struct TimeStepper
    method
    dt
    
    progress
end

# set up the simulation
dydt = similar(y0)
rhs!(dydt, y0, nothing, 0.0)
prob = ODEProblem(rhs!, y0, (0.0, 200.0))

# run simulation
sol = solve(
    prob,
    SSPRK33(),
    dt = 0.02,
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)