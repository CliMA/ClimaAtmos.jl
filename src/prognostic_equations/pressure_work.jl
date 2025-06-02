#####
##### Pressure work tendencies in moist static energy equation
#####

# We ignore the small pressure term in the advective MSE equation in EDMFX for now
pressure_work_tendency!(Yₜ, Y, p, t, ::PrognosticEDMFX) = nothing

pressure_work_tendency!(Yₜ, Y, p, t, ::Any) = nothing
