module NameList
# See Table 2 of Cohen et al, 2020
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_factor"] = 0.13
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["detrainment_factor"] = 0.51
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_massflux_div_factor"] = 0.0
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["turbulent_entrainment_factor"] = 0.075
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_smin_tke_coeff"] = 0.3
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["updraft_mixing_frac"] = 0.25
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_scale"] = 0.0004
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["sorting_power"] = 2.0

# See Table 1 of Lopez Gomez et al, 2020
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["tke_ed_coeff"] = 0.14
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["tke_diss_coeff"] = 0.22
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["static_stab_coeff"] = 0.4 # Square of value in the paper
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["tke_surf_scale"] = 3.75 # Square of value in the paper
# namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["Prandtl_number_0"] = 0.74

# See Table ? of He et al, 2021
# namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_buoy_coeff1"] = 0.12 ==> alpha_b (scaling constant for virtual mass term)
# namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_buoy_coeff2"] = 0.0 ==> alpha_b (scaling constant for virtual mass term)
# namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_adv_coeff"] = 0.1 alpha_a (scaling constant for advection term)
# namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_drag_coeff"] = 10.0 ==> alpha_d (scaling constant for drag term)
# namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_plume_spacing"] ==> r_d (horizontal length scale of plume spacing)

#NB: except for Bomex and life_cycle_Tan2018 cases, the parameters listed have not been thoroughly tuned/tested
# and should be regarded as placeholders only. Optimal parameters may also depend on namelist options, such as
# entrainment/detrainment rate formulation, diagnostic vs. prognostic updrafts, and vertical resolution
export default_namelist

using ArgParse

import StaticArrays
const SA = StaticArrays

include(joinpath(@__DIR__, "..", "integration_tests", "artifact_funcs.jl"))

import Random

import JSON

function parse_commandline()
    s = ArgParseSettings(; description = "namelist Generator")

    @add_arg_table! s begin
        "case_name"
        help = "The case name"
        arg_type = String
        required = true
    end

    return parse_args(s)
end

function default_namelist(::Nothing)

    args = parse_commandline()
    case_name = args["case_name"]
    return default_namelist(case_name)
end

function default_namelist(
    case_name::String;
    root::String = ".",
    write::Bool = true,
    set_seed::Bool = true,
    seed::Int = 2022,
    truncate_stack_trace::Bool = false,
)

    if set_seed
        Random.seed!(seed)
    end

    namelist_defaults = Dict()
    namelist_defaults["meta"] = Dict()
    namelist_defaults["meta"]["uuid"] = basename(tempname())

    namelist_defaults["config"] = "column"
    namelist_defaults["set_src_seed"] = false
    namelist_defaults["test_duals"] = false

    namelist_defaults["logging"] = Dict()
    namelist_defaults["logging"]["truncate_stack_trace"] = truncate_stack_trace

    namelist_defaults["float_type"] = "Float64"

    namelist_defaults["turbulence"] = Dict()

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"] = Dict()
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["surface_area"] = 0.1
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["max_area"] = 0.9
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["min_area"] = 1e-5

    # mixing_length
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["tke_ed_coeff"] = 0.14
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["tke_diss_coeff"] = 0.22
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["static_stab_coeff"] = 0.4
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["tke_surf_scale"] = 3.75
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["Prandtl_number_scale"] = 53.0 / 13.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["Prandtl_number_0"] = 0.74
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["Ri_crit"] = 0.25
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["smin_ub"] = 0.1
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["smin_rm"] = 1.5
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["l_max"] = 1.0e6
    # entrainment
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_factor"] = 0.13
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["detrainment_factor"] = 0.51

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_massflux_div_factor"] = 0.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["turbulent_entrainment_factor"] = 0.075
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_smin_tke_coeff"] = 0.3
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["updraft_mixing_frac"] = 0.25
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["area_limiter_scale"] = 10.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["area_limiter_power"] = 4.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_scale"] = 0.0004
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["sorting_power"] = 2.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["min_upd_velocity"] = 0.001
    # pressure
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["min_updraft_top"] = 500.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_buoy_coeff1"] = 0.12
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_buoy_coeff2"] = 0.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_adv_coeff"] = 0.1
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_drag_coeff"] = 10.0

    # From namelist
    namelist_defaults["grid"] = Dict()
    namelist_defaults["grid"]["dims"] = 1
    namelist_defaults["grid"]["stretch"] = Dict()
    namelist_defaults["grid"]["stretch"]["flag"] = false
    namelist_defaults["grid"]["stretch"]["nz"] = 55
    namelist_defaults["grid"]["stretch"]["dz_surf"] = 30.0
    namelist_defaults["grid"]["stretch"]["dz_toa"] = 8000.0
    namelist_defaults["grid"]["stretch"]["z_toa"] = 45000.0

    namelist_defaults["forcing"] = Dict()
    namelist_defaults["forcing"]["coriolis"] = 0.0

    namelist_defaults["thermodynamics"] = Dict()
    namelist_defaults["thermodynamics"]["moisture_model"] = "equilibrium" #"nonequilibrium"
    namelist_defaults["thermodynamics"]["thermo_covariance_model"] = "diagnostic" #"prognostic" or "diagnostic"
    namelist_defaults["thermodynamics"]["diagnostic_covar_limiter"] = 1e-3 # this controls the magnitude of the spike in covariance
    namelist_defaults["thermodynamics"]["sgs"] = "mean" # "quadrature" or "mean"
    namelist_defaults["thermodynamics"]["quadrature_order"] = 3
    namelist_defaults["thermodynamics"]["quadrature_type"] = "log-normal" #"gaussian" or "log-normal"

    namelist_defaults["time_stepping"] = Dict()
    namelist_defaults["time_stepping"]["dt_max"] = 12.0
    namelist_defaults["time_stepping"]["dt_min"] = 1.0
    namelist_defaults["time_stepping"]["adapt_dt"] = true
    namelist_defaults["time_stepping"]["cfl_limit"] = 0.5

    namelist_defaults["microphysics"] = Dict()
    namelist_defaults["microphysics"]["precipitation_model"] = "None"

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["updraft_number"] = 1
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment"] = "moisture_deficit"  # {"moisture_deficit", "NN", "NN_nonlocal", "Linear", "FNO", "RF"}
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entr_dim_scale"] = "buoy_vel" # {"buoy_vel", "inv_z", "none"}
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["detr_dim_scale"] = "buoy_vel" # {"buoy_vel", "inv_z", "none"}
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entr_pi_subset"] = ntuple(i -> i, 6) # or, e.g., (1, 3, 6)
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pi_norm_consts"] = [478.298, 1.0, 1.0, 1.0, 1.0, 1.0] # normalization constants for Pi groups
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["stochastic_entrainment"] = "deterministic"  # {"deterministic", "noisy_relaxation_process", "lognormal_scaling", "prognostic_noisy_relaxation_process"}

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_closure_buoy"] = "normalmode"
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_closure_drag"] = "normalmode"

    namelist_defaults["output"] = Dict()
    namelist_defaults["output"]["output_root"] = "./"

    namelist_defaults["stats_io"] = Dict()
    namelist_defaults["stats_io"]["stats_dir"] = "stats"
    namelist_defaults["stats_io"]["frequency"] = 60.0
    namelist_defaults["stats_io"]["skip"] = false
    namelist_defaults["stats_io"]["calibrate_io"] = false # limit io for calibration when `true`

    # nn parameters
    #! format: off
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["nn_arc"] = (6, 5, 4, 2) # [#inputs, #neurons in L1, #neurons in L2, ...., #outputs]

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["nn_ent_params"] =
        [-1.2791038,
        -1.1564382,
        1.2115442,
        0.3241886,
        0.4968626,
        -1.1304279,
        0.1335658,
        -1.5348452,
        -0.661436,
        -1.7114845,
        0.5861781,
        -0.0008509,
        1.8417425,
        -0.0692783,
        -0.1800616,
        0.4822386,
        0.1193081,
        0.4175962,
        -1.7491715,
        0.0331383,
        -1.1470715,
        0.1166774,
        -0.5425564,
        -0.3332106,
        -1.3615489,
        -0.2109712,
        -0.1836716,
        -1.7316646,
        -0.5677241,
        1.5717871,
        -0.2886485,
        0.2341374,
        -1.1329331,
        -0.5656705,
        -0.7084822,
        0.0904796,
        -0.2447465,
        -0.8655097,
        0.1487832,
        1.737286,
        0.3335405,
        1.2600883,
        0.1105,
        -0.9467368,
        -0.1319687,
        -0.3314595,
        1.2015113,
        -0.5082856,
        -1.3685998,
        -1.6377497,
        0.0398038,
        0.6748809,
        1.1398398,
        -1.1689684,
        -0.9930153,
        0.8116707,
        -0.006826,
        0.0822948,
        -0.7905997,
        -0.3131524,
        -0.4061444,
        1.6651008,
        1.555821,
        0.1941005,
        1.0823169,
        -0.3608865,
        0.7003173,
        0.2197592,
        -0.1367713]

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["nn_ent_biases"] = true

    # m=100 random features, d=6 input Pi groups
    # RF: parameters to optimize, 2 x (m + 1 + d)
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["rf_opt_ent_params"] =
        vec(cat(randn(2,100), # vec(cat(randn(2, m),
                    ones(2,7), dims=2)) # ones(2, d + 1), dims=2))

    # RF: fixed realizations of random variables, 2 x m x (1 + d)
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["rf_fix_ent_params"] =
        vec(cat(2*pi*rand(2,100,1), # vec(cat(2*pi*rand(2, m, 1),
                    randn(2,100,6), dims=3)) # randn(2, m, d), dims=3))

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["linear_ent_params"] =
        SA.SVector{14}(rand(14))

    # General stochastic entrainment/detrainment parameters
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["general_stochastic_ent_params"] =
    SA.SVector(
        0.1, 0.1,   # ε_σ², δ_σ²
        0.05, 0.05  # ε_λ, δ_λ
    )

    # For FNO add here
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["fno_ent_width"] = 2
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["fno_ent_n_modes"] = 2
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["fno_ent_params"] =
        SA.SVector{50}(rand(50))

    #! format: on

    if case_name == "Soares"
        namelist = Soares(namelist_defaults)
    elseif case_name == "Nieuwstadt"
        namelist = Nieuwstadt(namelist_defaults)
    elseif case_name == "Bomex"
        namelist = Bomex(namelist_defaults)
    elseif case_name == "life_cycle_Tan2018"
        namelist = life_cycle_Tan2018(namelist_defaults)
    elseif case_name == "Rico"
        namelist = Rico(namelist_defaults)
    elseif case_name == "TRMM_LBA"
        namelist = TRMM_LBA(namelist_defaults)
    elseif case_name == "ARM_SGP"
        namelist = ARM_SGP(namelist_defaults)
    elseif case_name == "GATE_III"
        namelist = GATE_III(namelist_defaults)
    elseif case_name == "DYCOMS_RF01"
        namelist = DYCOMS_RF01(namelist_defaults)
    elseif case_name == "DYCOMS_RF02"
        namelist = DYCOMS_RF02(namelist_defaults)
    elseif case_name == "GABLS"
        namelist = GABLS(namelist_defaults)
    elseif case_name == "DryBubble"
        namelist = DryBubble(namelist_defaults)
    elseif case_name == "LES_driven_SCM"
        namelist = LES_driven_SCM(namelist_defaults)
    else
        error("Not a valid case name")
    end

    if write
        write_file(namelist, root)
    end
    return namelist
end
function Soares(namelist_defaults)

    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "Soares"

    namelist["grid"]["nz"] = 75
    namelist["grid"]["dz"] = 50.0

    namelist["time_stepping"]["t_max"] = 8 * 3600.0
    namelist["time_stepping"]["dt_min"] = 1.0

    namelist["meta"]["simname"] = "Soares"
    namelist["meta"]["casename"] = "Soares"

    return namelist
end
function Nieuwstadt(namelist_defaults)

    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "Nieuwstadt"

    namelist["grid"]["nz"] = 75
    namelist["grid"]["dz"] = 50.0

    namelist["time_stepping"]["t_max"] = 8 * 3600.0
    namelist["time_stepping"]["dt_min"] = 1.2

    namelist["meta"]["simname"] = "Nieuwstadt"
    namelist["meta"]["casename"] = "Nieuwstadt"

    return namelist
end
function Bomex(namelist_defaults)

    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "Bomex"

    namelist["grid"]["nz"] = 60
    namelist["grid"]["dz"] = 50.0

    namelist["forcing"]["coriolis"] = 0.376e-4

    namelist["time_stepping"]["t_max"] = 21600.0
    namelist["time_stepping"]["dt_min"] = 6.0

    namelist["meta"]["simname"] = "Bomex"
    namelist["meta"]["casename"] = "Bomex"

    return namelist
end
function life_cycle_Tan2018(namelist_defaults)

    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "life_cycle_Tan2018"

    namelist["grid"]["nz"] = 75
    namelist["grid"]["dz"] = 40.0

    namelist["forcing"]["coriolis"] = 0.376e-4

    namelist["time_stepping"]["t_max"] = 6 * 3600.0
    namelist["time_stepping"]["dt_min"] = 10.0
    namelist["meta"]["simname"] = "life_cycle_Tan2018"
    namelist["meta"]["casename"] = "life_cycle_Tan2018"

    return namelist
end
function Rico(namelist_defaults)

    namelist = deepcopy(namelist_defaults)

    namelist["meta"]["casename"] = "Rico"

    namelist["grid"]["nz"] = 80
    namelist["grid"]["dz"] = 50.0

    namelist["time_stepping"]["adapt_dt"] = false
    namelist["time_stepping"]["t_max"] = 86400.0
    #namelist["time_stepping"]["dt_max"] = 5.0
    namelist["time_stepping"]["dt_min"] = 1.5

    namelist["forcing"]["latitude"] = 18.0
    namelist["forcing"]["coriolis"] = 4.5e-5

    namelist["microphysics"]["precipitation_model"] = "clima_1m"
    namelist["microphysics"]["precip_fraction_model"] = "prescribed" # "prescribed" or "cloud_cover"
    namelist["microphysics"]["prescribed_precip_frac_value"] = 1.0
    namelist["microphysics"]["precip_fraction_limiter"] = 0.3
    namelist["microphysics"]["τ_acnv_rai"] = 2500.0
    namelist["microphysics"]["τ_acnv_sno"] = 100.0
    namelist["microphysics"]["q_liq_threshold"] = 0.5e-3
    namelist["microphysics"]["q_ice_threshold"] = 1e-6
    namelist["microphysics"]["microph_scaling"] = 1.0
    namelist["microphysics"]["microph_scaling_dep_sub"] = 1.0
    namelist["microphysics"]["microph_scaling_melt"] = 1.0
    namelist["microphysics"]["E_liq_rai"] = 0.8
    namelist["microphysics"]["E_liq_sno"] = 0.1
    namelist["microphysics"]["E_ice_rai"] = 1.0
    namelist["microphysics"]["E_ice_sno"] = 0.1
    namelist["microphysics"]["E_rai_sno"] = 1.0

    namelist["meta"]["simname"] = "Rico"
    namelist["meta"]["casename"] = "Rico"
    return namelist
end
function TRMM_LBA(namelist_defaults)

    namelist = deepcopy(namelist_defaults)

    namelist["meta"]["casename"] = "TRMM_LBA"

    namelist["grid"]["nz"] = 82
    namelist["grid"]["dz"] = 200

    namelist["time_stepping"]["adapt_dt"] = true
    namelist["time_stepping"]["t_max"] = 60 * 60 * 6.0
    namelist["time_stepping"]["dt_max"] = 5.0
    namelist["time_stepping"]["dt_min"] = 1.0

    namelist["microphysics"]["precipitation_model"] = "clima_1m" # "cutoff"
    namelist["microphysics"]["precip_fraction_model"] = "prescribed" # "prescribed" or "cloud_cover"
    namelist["microphysics"]["prescribed_precip_frac_value"] = 1.0
    namelist["microphysics"]["precip_fraction_limiter"] = 0.3
    namelist["microphysics"]["τ_acnv_rai"] = 2500.0
    namelist["microphysics"]["τ_acnv_sno"] = 100.0
    namelist["microphysics"]["q_liq_threshold"] = 0.5e-3
    namelist["microphysics"]["q_ice_threshold"] = 1e-6
    namelist["microphysics"]["microph_scaling"] = 1.0
    namelist["microphysics"]["microph_scaling_dep_sub"] = 1.0
    namelist["microphysics"]["microph_scaling_melt"] = 1.0
    namelist["microphysics"]["E_liq_rai"] = 0.8
    namelist["microphysics"]["E_liq_sno"] = 0.1
    namelist["microphysics"]["E_ice_rai"] = 1.0
    namelist["microphysics"]["E_ice_sno"] = 0.1
    namelist["microphysics"]["E_rai_sno"] = 1.0

    namelist["meta"]["simname"] = "TRMM_LBA"
    namelist["meta"]["casename"] = "TRMM_LBA"

    return namelist
end
function ARM_SGP(namelist_defaults)

    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "ARM_SGP"

    namelist["grid"]["nz"] = 88
    namelist["grid"]["dz"] = 50.0

    namelist["forcing"]["coriolis"] = 8.5e-5

    namelist["time_stepping"]["t_max"] = 3600.0 * 14.5
    namelist["time_stepping"]["dt_min"] = 2.0

    namelist["meta"]["simname"] = "ARM_SGP"
    namelist["meta"]["casename"] = "ARM_SGP"

    return namelist
end
function GATE_III(namelist_defaults)

    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "GATE_III"

    namelist["grid"]["nz"] = 200 # 1700
    namelist["grid"]["dz"] = 85  # 10

    namelist["time_stepping"]["adapt_dt"] = false
    namelist["time_stepping"]["t_max"] = 3600.0 * 24.0
    namelist["time_stepping"]["dt_max"] = 5.0
    namelist["time_stepping"]["dt_min"] = 2.0

    namelist["microphysics"]["precipitation_model"] = "clima_1m" #"cutoff"

    namelist["meta"]["simname"] = "GATE_III"
    namelist["meta"]["casename"] = "GATE_III"

    return namelist
end
function DYCOMS_RF01(namelist_defaults)

    namelist = deepcopy(namelist_defaults)

    namelist["meta"]["casename"] = "DYCOMS_RF01"

    namelist["grid"]["nz"] = 30
    namelist["grid"]["dz"] = 50

    namelist["time_stepping"]["t_max"] = 60 * 60 * 16.0
    namelist["time_stepping"]["dt_min"] = 6.0

    namelist["meta"]["simname"] = "DYCOMS_RF01"
    namelist["meta"]["casename"] = "DYCOMS_RF01"

    return namelist
end
function DYCOMS_RF02(namelist_defaults)

    namelist = deepcopy(namelist_defaults)

    namelist["meta"]["casename"] = "DYCOMS_RF02"

    namelist["grid"]["nz"] = 30
    namelist["grid"]["dz"] = 50

    namelist["time_stepping"]["adapt_dt"] = true
    namelist["time_stepping"]["t_max"] = 60 * 60 * 6.0
    namelist["time_stepping"]["dt_max"] = 4.0
    namelist["time_stepping"]["dt_min"] = 1.0

    namelist["microphysics"]["precipitation_model"] = "clima_1m" #"cutoff"
    namelist["microphysics"]["precip_fraction_model"] = "prescribed" # "prescribed" or "cloud_cover"
    namelist["microphysics"]["prescribed_precip_frac_value"] = 1.0
    namelist["microphysics"]["precip_fraction_limiter"] = 0.3
    namelist["microphysics"]["τ_acnv_rai"] = 2500.0
    namelist["microphysics"]["τ_acnv_sno"] = 100.0
    namelist["microphysics"]["q_liq_threshold"] = 0.5e-3
    namelist["microphysics"]["q_ice_threshold"] = 1e-6
    namelist["microphysics"]["microph_scaling"] = 1.0
    namelist["microphysics"]["microph_scaling_dep_sub"] = 1.0
    namelist["microphysics"]["microph_scaling_melt"] = 1.0
    namelist["microphysics"]["E_liq_rai"] = 0.8
    namelist["microphysics"]["E_liq_sno"] = 0.1
    namelist["microphysics"]["E_ice_rai"] = 1.0
    namelist["microphysics"]["E_ice_sno"] = 0.1
    namelist["microphysics"]["E_rai_sno"] = 1.0

    namelist["meta"]["simname"] = "DYCOMS_RF02"
    namelist["meta"]["casename"] = "DYCOMS_RF02"

    return namelist
end
function GABLS(namelist_defaults)

    namelist = deepcopy(namelist_defaults)

    namelist["meta"]["casename"] = "GABLS"

    namelist["grid"]["nz"] = 8
    namelist["grid"]["dz"] = 50.0

    namelist["forcing"]["coriolis"] = 1.39e-4

    namelist["time_stepping"]["t_max"] = 9 * 3600.0
    namelist["time_stepping"]["dt_min"] = 4.0
    namelist["time_stepping"]["dt_max"] = 8.0
    namelist["meta"]["simname"] = "GABLS"
    namelist["meta"]["casename"] = "GABLS"

    return namelist
end

function DryBubble(namelist_defaults)
    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "DryBubble"
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["surface_area"] = 0.0

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_buoy_coeff1"] = 0.12
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_adv_coeff"] = 0.25
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_drag_coeff"] = 0.1

    namelist["grid"]["nz"] = 200
    namelist["grid"]["dz"] = 50.0

    namelist["stats_io"]["frequency"] = 10.0
    namelist["time_stepping"]["t_max"] = 1000.0
    namelist["time_stepping"]["dt_min"] = 0.5

    namelist["meta"]["simname"] = "DryBubble"
    namelist["meta"]["casename"] = "DryBubble"

    namelist["turbulence"]["EDMF_PrognosticTKE"]["entrainment_massflux_div_factor"] = 0.4

    return namelist
end

function LES_driven_SCM(namelist_defaults)
    namelist = deepcopy(namelist_defaults)
    # Only one can be defined by user
    # namelist["grid"]["dz"] = 50.0
    namelist["grid"]["nz"] = 80

    namelist["stats_io"]["frequency"] = 10.0
    namelist["time_stepping"]["t_max"] = 3600.0 * 6
    namelist["time_stepping"]["dt_min"] = 1.0

    # use last 6 hours of LES simulation to drive LES
    namelist["t_interval_from_end_s"] = 3600.0 * 6
    # average in 1 hour interval around `t_interval_from_end_s`
    namelist["initial_condition_averaging_window_s"] = 3600.0

    # LES filename should follow pattern:
    # Stats.cfsite<SITE-NUMBER>_<FORCING-MODEL>_<EXPERIMENT>_2004-2008.<MONTH>.nc
    namelist["meta"]["lesfile"] =
        joinpath(les_driven_scm_data_folder(), "Stats.cfsite23_HadGEM2-A_amip_2004-2008.07.nc")
    namelist["meta"]["simname"] = "LES_driven_SCM"
    namelist["meta"]["casename"] = "LES_driven_SCM"

    return namelist
end

function write_file(namelist, root::String = ".")
    mkpath(root)

    @assert haskey(namelist, "meta")
    @assert haskey(namelist["meta"], "simname")

    casename = namelist["meta"]["casename"]
    open(joinpath(root, "namelist_$casename.in"), "w") do io
        JSON.print(io, namelist, 4)
    end

    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    default_namelist(nothing)
end

end
