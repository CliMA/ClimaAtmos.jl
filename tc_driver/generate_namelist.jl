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
export default_namelist, namelist_to_toml_file

using ArgParse

import Random

import JSON
import TOML

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
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["tke_diss_coeff"] =
        0.22
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["static_stab_coeff"] =
        0.4
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["tke_surf_scale"] =
        3.75
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["Prandtl_number_scale"] =
        53.0 / 13.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["Prandtl_number_0"] =
        0.74
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["Ri_crit"] = 0.25
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["smin_ub"] = 0.1
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["smin_rm"] = 1.5
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["l_max"] = 1.0e6
    # entrainment
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_factor"] =
        0.13
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["detrainment_factor"] =
        0.51

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_massflux_div_factor"] =
        0.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["turbulent_entrainment_factor"] =
        0.075
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_smin_tke_coeff"] =
        0.3
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["updraft_mixing_frac"] =
        0.25
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["area_limiter_scale"] =
        10.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["area_limiter_power"] =
        4.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_scale"] =
        0.0004
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["sorting_power"] = 2.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["min_upd_velocity"] =
        0.001
    # pressure
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["min_updraft_top"] =
        500.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_buoy_coeff1"] =
        0.12
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_buoy_coeff2"] =
        0.0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_adv_coeff"] =
        0.1
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_drag_coeff"] =
        10.0

    # From namelist
    namelist_defaults["grid"] = Dict()
    namelist_defaults["grid"]["dims"] = 1
    namelist_defaults["grid"]["stretch"] = Dict()
    namelist_defaults["grid"]["stretch"]["flag"] = false
    namelist_defaults["grid"]["stretch"]["nz"] = 55
    namelist_defaults["grid"]["stretch"]["dz_surf"] = 30.0
    namelist_defaults["grid"]["stretch"]["dz_toa"] = 8000.0
    namelist_defaults["grid"]["stretch"]["z_toa"] = 45000.0

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
    # namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment"] = "moisture_deficit"  # not currently used

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_closure_buoy"] = "normalmode"
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_closure_drag"] = "normalmode"

    namelist_defaults["output"] = Dict()
    namelist_defaults["output"]["output_root"] = "./"

    namelist_defaults["stats_io"] = Dict()
    namelist_defaults["stats_io"]["stats_dir"] = "stats"
    namelist_defaults["stats_io"]["frequency"] = 60.0
    namelist_defaults["stats_io"]["skip"] = false
    namelist_defaults["stats_io"]["calibrate_io"] = false # limit io for calibration when `true`

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

    namelist["time_stepping"]["t_max"] = 9 * 3600.0
    namelist["time_stepping"]["dt_min"] = 4.0
    namelist["time_stepping"]["dt_max"] = 8.0
    namelist["meta"]["simname"] = "GABLS"
    namelist["meta"]["casename"] = "GABLS"

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


"""
Flattens a namelist Dict, prepending keys of formerly nested Dictionaries
"""
function flatten_namelist!(namelist::Dict)
    for (key, value) in namelist
        if value isa Dict
            temp_dict = flatten_namelist(value)
            for (newkey, newvalue) in temp_dict
                namelist[string("$key", "-", "$newkey")] = newvalue
            end
            delete!(namelist, key)
        else
            namelist[key] = value
        end
    end
end

"""
Helper function for flatten_namelist!, doesn't flatten in-place
"""
function flatten_namelist(namelist::Dict, prependkey = "")
    out_namelist = Dict()
    for (key, value) in namelist
        fullkey = prependkey == "" ? key : string("$prependkey", "-", "$key")
        if typeof(value) <: Dict
            temp_dict = flatten_namelist(value, fullkey)
            out_namelist = merge(out_namelist, temp_dict)
        else
            out_namelist[fullkey] = value
        end
    end
    return out_namelist
end

flatten_namelist(obj, prepend) = Dict(prepend => obj)

"""
Takes a namelist Dict and parses it into a flat TOML file.
"""
function namelist_to_toml_file(namelist::Dict, fname::String)
    open(fname, "w") do io
        TOML.print(io, namelist_to_toml_dict(namelist))
    end
end

function namelist_to_toml_dict(namelist::Dict)
    flatten_namelist!(namelist)
    toml_dict = Dict()
    for (key, val) in namelist
        (type, value) = parse_namelist_entry(key, val)
        alias = last(split(key, "-"))
        toml_dict[key] =
            Dict("alias" => "$alias", "value" => value, "type" => "$type")
    end
    return toml_dict
end

"""
Helper functions that parse namelist entries of different types
for namelist_to_toml functions
"""
function parse_namelist_entry(key::String, value::Tuple)
    # Changes type from tuple to array{float}
    return ("array{float}", [i for i in value])
end

function parse_namelist_entry(key::String, value::AbstractArray)
    return ("array{float}", value)
end

function parse_namelist_entry(key::String, value::String)
    return ("string", "$value")
end

function parse_namelist_entry(key::String, value::Bool)
    return ("bool", value)
end

function parse_namelist_entry(key::String, value::Real)
    return ("float", value)
end

if abspath(PROGRAM_FILE) == @__FILE__
    default_namelist(nothing)
end

end
