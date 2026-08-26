"""
Build EKP.Observation vector from Atlas LES outputs (one Observation per case).
"""

using LinearAlgebra: LinearAlgebra
using Statistics: Statistics

# LES SAM short names → Atmos diagnostic short names
const LES_PROFILE_MAP = Dict(
    "clw" => "QCL",
    "cli" => "QCI",
    "husra" => "QPL",
    "hussn" => "QPI",
)
const LES_PATH_MAP = Dict(
    "lwp" => "LWP",
    "iwp" => "IWP",
    "rwp" => "RWP",
    "swp" => "SWP",
)


obs_var_additional_uncertainty_factor = FT(0.1) # Given no known uncertainty, apply 10% relative uncertainty to all variables
obs_var_additional_uncertainty_factor = Dict( # Per-variable uncertainty factors
    "ql_mean"          => obs_var_additional_uncertainty_factor,
    "qi_mean"          => obs_var_additional_uncertainty_factor,
    "qr_mean"          => obs_var_additional_uncertainty_factor,
    "qs_mean"          => obs_var_additional_uncertainty_factor,
    "qip_mean"         => obs_var_additional_uncertainty_factor,
    "ql_all_mean"      => obs_var_additional_uncertainty_factor,
    "qi_all_mean"      => obs_var_additional_uncertainty_factor,
    "qt_mean"          => obs_var_additional_uncertainty_factor,
    "lwp_mean"         => obs_var_additional_uncertainty_factor,
    "iwp_mean"         => obs_var_additional_uncertainty_factor,
)

# Characteristic values for normalization (default if true profile is all 0's)
calibration_vars_characteristic_values = Dict(
    "ql_mean"          => FT(1e-4), #  kg/kg
    "qi_mean"          => FT(1e-8), #  kg/kg
    "qr_mean"          => FT(1e-6), # kg/kg
    # "qs_mean"          => FT(1e-3), # kg/kg
    "qs_mean"          => FT(1e-7), # kg/kg # reducing this.. not sure why it was 1e-3. peak value is 1e-5 in RF09 so this prolly is ok? we get 1e-8 in RF12, 13
    # "qip_mean"         => FT(1e-3), # 1e-3 kg/kg
    "ql_all_mean"      => FT(1e-3), # kg/kg
    "qi_all_mean"      => FT(1e-8), # kg/kg
    "qt_mean"          => FT(1e-3), # kg/kg
    "lwp_mean"         => FT(1e-1), # 1e-2 kg/m^2 max
    "iwp_mean"         => FT(1e-5), # 1e-4 kg/m^2 max
)
calibration_vars_characteristic_values["qip_mean"] = calibration_vars_characteristic_values["qs_mean"]

# Boost variable importance in loss by scaling variance (and thus normalization) by this factor
default_obs_var_scaling = Dict{String, FT}(
    "ql_mean" => FT((1.0/1.0)^2), # boost by factor of 1
    "qi_mean" => FT((1.0/2.5)^2), # boost by factor of 5
    "qr_mean" => FT(1.0),
    "qs_mean" => FT((1.0/2.0)^2), # boost by factor of 2
    "qip_mean" => FT(1.0),
    "ql_all_mean" => FT(1.0), # leave equal
    "qi_all_mean" => FT(1.0), # leave equal
    "qt_mean" => FT(1.0), # leave equal
    # CAN WE TIE THISE TO dt/dz SOMEHOW? LIKE 1/(C * nz)
    "lwp_mean" => FT((1.0/0.5)^2), # leave equal [scaled up in config_calibrate_template_body() to account for a whole profile]) [make slightly less important so it doesn't lead to slowing down liquid to match structural error]
    "iwp_mean" => FT((1.0/3.0)^2), # boost by factor of 2 [scaled up in config_calibrate_template_body() to account for a whole profile])
    ) # scale down so ice becomes more important (factor of 5 rn), maybe will help calibrations...


"""Convert LES mixing ratio / path from g/kg or g/m² to kg/kg or kg/m²."""
g_to_kg(x::FT) where {FT} = x / FT(1000)
g_to_kg(x::AbstractArray{FT}) where {FT} = g_to_kg.(x)

function les_time_seconds(les_data)
    t_days = Float64.(vec(Array(les_data["time"])))
    return (t_days .- first(t_days)) .* 86400
end

function time_mask(t_sec, t0, t1)
    return findall(t -> t0 <= t <= t1, t_sec)
end

function z_mask(z, z0, z1)
    return findall(zk -> z0 <= zk <= z1, z)
end

"""
Time-mean profile (or scalar path) from LES for one Atmos short_name.
"""
function les_mean_field(case::SOCRATESCase, short_name::AbstractString, t0, t1, z0, z1)
    ft = sscf_forcing(case.forcing_type)
    les = SSCF.open_atlas_les_output(case.flight_number, ft)
    data = les.data
    t_sec = les_time_seconds(data)
    it = time_mask(t_sec, t0, t1)
    isempty(it) && error("No LES times in [$t0, $t1] for $(case.name)")

    if haskey(LES_PROFILE_MAP, short_name)
        raw = Float64.(Array(data[LES_PROFILE_MAP[short_name]])) # (z, time)
        z = Float64.(vec(Array(data["z"])))
        iz = z_mask(z, z0, z1)
        isempty(iz) && error("No LES levels in [$z0, $z1] for $(case.name)")
        field = g_to_kg(raw[iz, it])
        return Statistics.mean(field; dims = 2)[:, 1], z[iz]
    elseif haskey(LES_PATH_MAP, short_name)
        raw = Float64.(vec(Array(data[LES_PATH_MAP[short_name]]))) # (time,)
        field = g_to_kg(raw[it])
        return [Statistics.mean(field)], Float64[]
    else
        error("No LES mapping for Atmos short_name `$short_name`")
    end
end

"""
    build_case_observation(case, exp_cfg) -> EKP.Observation

Concatenate configured y variables into one observation vector with diagonal noise.
"""
function build_case_observation(case::SOCRATESCase, exp_cfg)
    t0, t1 = score_window_sec(case, exp_cfg)
    z0, z1 = z_bounds(case)
    y_vars = String.(exp_cfg["y_var_names"])
    noise = exp_cfg["const_noise_by_var"]

    pieces = Float64[]
    diag = Float64[]
    for name in y_vars
        y, _ = les_mean_field(case, name, t0, t1, z0, z1)
        append!(pieces, y)
        σ2 = Float64(noise[name])
        append!(diag, fill(σ2, length(y)))
    end
    Γ = LinearAlgebra.Diagonal(diag)
    return EKP.Observation(
        Dict("samples" => pieces, "covariances" => Γ, "names" => case.name),
    )
end

function build_observation_vector(interface::SOCRATESAtmosModelInterface)
    return [build_case_observation(case, interface.experiment_config) for case in interface.cases]
end

function save_observations!(interface::SOCRATESAtmosModelInterface; overwrite = false)
    path = joinpath(interface.output_dir, "observation_vec.jld2")
    if !overwrite && isfile(path)
        @info "Reusing observations at $path"
        return path
    end
    obs_vec = build_observation_vector(interface)
    JLD2.save_object(path, obs_vec)
    @info "Wrote observations" path length(obs_vec)
    return path
end
