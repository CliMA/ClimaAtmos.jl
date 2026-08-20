#####
##### Pressure work tendencies in moist static energy equation
#####

# We ignore the small pressure term in the advective MSE equation in EDMFX for now
"""
    pressure_work_tendency!(Yₜ, Y, p, t, turbconv_model)

Add the pressure-work tendency in the moist-static-energy equation.

Currently a no-op for every turbulence-convection model: the pressure term in the
advective `mse` equation of `PrognosticEDMFX` is small and is neglected, and no
other model carries a prognostic `mse`. Kept as a dispatch point so the term can
be reinstated without changing the call site in `additional_tendency!`, which
invokes it after all `ρa` tendencies have been applied. Returns `nothing`.
"""
pressure_work_tendency!(Yₜ, Y, p, t, ::PrognosticEDMFX) = nothing

pressure_work_tendency!(Yₜ, Y, p, t, ::Any) = nothing
