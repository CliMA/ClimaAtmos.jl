import ClimaCore
import ClimaUtilities
import ClimaCore: Domains, Spaces, Topologies

# To allow for backwards compatibility of ClimaCore:
if pkgversion(ClimaCore) < v"0.14.18"
    """
        z_max(::ClimaCore.Spaces.AbstractSpace)

    The domain maximum along the z-direction.
    """
    function z_max end

    z_max(domain::Domains.IntervalDomain) = domain.coord_max.z
    function z_max(space::Spaces.AbstractSpace)
        mesh = Topologies.mesh(Spaces.vertical_topology(space))
        domain = Topologies.domain(mesh)
        return z_max(domain)
    end

else
    z_max(s::Spaces.AbstractSpace) = Spaces.z_max(s)
end

if pkgversion(ClimaUtilities) < v"0.1.20"
    # From ClimaUtilities.OnlineLogging.WallTimeInfo
    struct WallTimeInfo
        n_calls::Base.RefValue{Int}
        t_wall_last::Base.RefValue{Float64}
        ∑Δt_wall::Base.RefValue{Float64}
        function WallTimeInfo()
            n_calls = Ref(0)
            t_wall_last = Ref(-1.0)
            ∑Δt_wall = Ref(0.0)
            return new(n_calls, t_wall_last, ∑Δt_wall)
        end
    end
    function _update!(wt::WallTimeInfo)
        if wt.n_calls[] == 0 || wt.n_calls[] == 1
            Δt_wall = 0.0
        else
            Δt_wall = time() - wt.t_wall_last[]
            wt.n_calls[] == 2 && (Δt_wall = 2Δt_wall)
        end
        wt.n_calls[] += 1
        wt.t_wall_last[] = time()
        wt.∑Δt_wall[] += Δt_wall
        return nothing
    end
    function report_walltime(wt, integrator)
        _update!(wt)

        t_start, t_end = integrator.sol.prob.tspan
        dt = integrator.dt
        t = integrator.t

        n_steps_total = ceil(Int, (t_end - t_start) / dt)
        n_steps = ceil(Int, (t - t_start) / dt)

        wall_time_ave_per_step = wt.∑Δt_wall[] / n_steps
        wall_time_ave_per_step_str = time_and_units_str(wall_time_ave_per_step)
        percent_complete = round((t - t_start) / t_end * 100; digits = 1)
        n_steps_remaining = n_steps_total - n_steps
        wall_time_remaining = wall_time_ave_per_step * n_steps_remaining
        wall_time_remaining_str = time_and_units_str(wall_time_remaining)
        wall_time_total =
            time_and_units_str(wall_time_ave_per_step * n_steps_total)
        wall_time_spent = time_and_units_str(wt.∑Δt_wall[])
        simulation_time = time_and_units_str(Float64(t))

        simulated_seconds_per_second = (t - t_start) / wt.∑Δt_wall[]
        simulated_seconds_per_day = simulated_seconds_per_second * 86400
        simulated_days_per_day = simulated_seconds_per_day / 86400
        simulated_years_per_day = simulated_days_per_day / 365.25

        sypd_estimate = string(round(simulated_years_per_day; digits = 3))
        if simulated_years_per_day < 0.01
            sdpd_estimate = round(simulated_days_per_day, digits = 3)
            sypd_estimate *= " (sdpd_estimate = $sdpd_estimate)"
        end

        estimated_finish_date =
            Dates.now() + Dates.Second(ceil(wall_time_remaining))

        @info "Progress" simulation_time = simulation_time n_steps_completed =
            n_steps wall_time_per_step = wall_time_ave_per_step_str wall_time_total =
            wall_time_total wall_time_remaining = wall_time_remaining_str wall_time_spent =
            wall_time_spent percent_complete = "$percent_complete%" estimated_sypd =
            sypd_estimate date_now = Dates.now() estimated_finish_date =
            estimated_finish_date

        return nothing
    end
else
    WallTimeInfo = ClimaUtilities.OnlineLogging.WallTimeInfo
    report_walltime = ClimaUtilities.OnlineLogging.report_walltime
end
