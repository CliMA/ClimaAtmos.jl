"""
Map Atmos diagnostics to G ensemble columns matching LES observation vectors.
"""

using LinearAlgebra: LinearAlgebra
using Statistics: mean

function _simdir_for_case(member_dir, case_name)
    case_dir = joinpath(member_dir, case_name)
    # Atmos writes under output_active inside output_dir
    active = joinpath(case_dir, "output_active")
    isdir(active) && return ClimaAnalysis.SimDir(active)
    # fallback: case_dir itself may be the output root
    return ClimaAnalysis.SimDir(case_dir)
end

function _load_avg_var(simdir, short_name)
    reduction = "average"
    period = first(ClimaAnalysis.available_periods(simdir; short_name, reduction))
    coord_type = first(
        ClimaAnalysis.available_coord_types(simdir; short_name, reduction, period),
    )
    return get(simdir; short_name, reduction, period, coord_type)
end

"""
Extract time-mean vector for one short_name from Atmos output, matching LES window / z_bounds.
"""
function atmos_mean_field(var, case::SOCRATESCase, exp_cfg, t0_abs::Dates.DateTime)
    t0, t1 = score_window_sec(case, exp_cfg)
    z0, z1 = z_bounds(case)
    dates = ClimaAnalysis.dates(var)
    # seconds since LES / Atmos start
    t_sec = [Dates.value(Dates.Second(d - t0_abs)) for d in dates]
    it = findall(t -> t0 <= t <= t1, t_sec)
    isempty(it) && error(
        "No Atmos times in score window for $(case.name) / $(ClimaAnalysis.short_name(var))",
    )

    data = Float64.(var.data)
    while ndims(data) > 2
        data = dropdims(data; dims = 1)
    end

    if !ClimaAnalysis.has_altitude(var) || ndims(data) == 1
        return [mean(vec(data)[it])]
    end

    z = Float64.(var.dims[ClimaAnalysis.altitude_name(var)])
    iz = findall(zk -> z0 <= zk <= z1, z)
    isempty(iz) && error("No Atmos levels in z_bounds for $(case.name)")
    slice = data[iz, it]
    return vec(mean(slice; dims = 2))
end

function g_vector_for_case(member_dir, case::SOCRATESCase, exp_cfg)
    simdir = _simdir_for_case(member_dir, case.name)
    t0_abs = start_datetime(case)
    y_vars = String.(exp_cfg["y_var_names"])
    pieces = Float64[]
    for name in y_vars
        var = _load_avg_var(simdir, name)
        append!(pieces, atmos_mean_field(var, case, exp_cfg, t0_abs))
    end
    return pieces
end

"""
    ClimaCalibrate.observation_map(interface::SOCRATESAtmosModelInterface, iteration)

Fill G_ensemble for the current minibatch of cases.
"""
function ClimaCalibrate.observation_map(interface::SOCRATESAtmosModelInterface, iteration)
    (; output_dir, experiment_config, cases) = interface
    ekp = JLD2.load_object(ClimaCalibrate.ekp_path(output_dir, iteration))
    ensemble_size = EKP.get_N_ens(ekp)

    # Combined observation length = sum over all cases (batch_size must cover all cases)
    obs_series = EKP.get_observation_series(ekp)
    all_obs = EKP.get_observations(obs_series)
    obs_len = sum(length(first(EKP.get_samples(o))) for o in all_obs)
    G = fill(NaN, obs_len, ensemble_size)

    for m in 1:ensemble_size
        try
            member_dir = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, m)
            pieces = Float64[]
            for case in cases
                append!(pieces, g_vector_for_case(member_dir, case, experiment_config))
            end
            length(pieces) == obs_len || error(
                "G length $(length(pieces)) ≠ obs length $obs_len for member $m",
            )
            G[:, m] = pieces
        catch e
            @error "observation_map failed for member $m" exception = (e, catch_backtrace())
        end
    end
    return G
end
