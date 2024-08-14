#####
##### Pressure work
#####

# We ignore the small pressure term in advective EDMF for now
pressure_work_tendency!(Yₜ, Y, p, t, colidx, ::PrognosticEDMFX) = nothing

pressure_work_tendency!(Yₜ, Y, p, t, colidx, ::Any) = nothing
