#####
##### Pressure work
#####

# We ignore the small pressure term in advective EDMFX for now
pressure_work_tendency!(Yₜ, Y, p, t, ::PrognosticEDMFX) = nothing

pressure_work_tendency!(Yₜ, Y, p, t, ::Any) = nothing
