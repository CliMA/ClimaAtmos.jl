mutable struct TimeStepping{FT}
    dt::FT
    t_max::FT
    t::FT
    nstep::Int
    cfl_limit::FT
    dt_max::FT
    dt_max_edmf::FT
    dt_io::FT
end

function TimeStepping(::Type{FT}, namelist) where {FT}
    dt = TC.parse_namelist(namelist, "time_stepping", "dt_min"; default = FT(1.0))
    t_max = TC.parse_namelist(namelist, "time_stepping", "t_max"; default = FT(7200.0))
    cfl_limit = TC.parse_namelist(namelist, "time_stepping", "cfl_limit"; default = FT(0.5))
    dt_max = TC.parse_namelist(namelist, "time_stepping", "dt_max"; default = FT(10.0))
    dt_max_edmf = FT(0)

    # set time
    t = FT(0)
    dt_io = FT(0)
    nstep = 0

    return TimeStepping{FT}(dt, t_max, t, nstep, cfl_limit, dt_max, dt_max_edmf, dt_io)
end
