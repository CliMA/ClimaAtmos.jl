module NameList

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
)

    if set_seed
        Random.seed!(seed)
    end

    namelist_defaults = Dict()
    namelist_defaults["meta"] = Dict()
    namelist_defaults["meta"]["uuid"] = basename(tempname())

    namelist_defaults["config"] = "column"
    namelist_defaults["test_duals"] = false

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
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_adv_coeff"] =
        0.1
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_drag_coeff"] =
        10.0

    # From namelist
    namelist_defaults["thermodynamics"] = Dict()
    namelist_defaults["thermodynamics"]["moisture_model"] = "equilibrium" #"nonequilibrium"
    namelist_defaults["thermodynamics"]["thermo_covariance_model"] = "diagnostic" #"prognostic" or "diagnostic"
    namelist_defaults["thermodynamics"]["diagnostic_covar_limiter"] = 1e-3 # this controls the magnitude of the spike in covariance
    namelist_defaults["thermodynamics"]["sgs"] = "mean" # "quadrature" or "mean"
    namelist_defaults["thermodynamics"]["quadrature_order"] = 3
    namelist_defaults["thermodynamics"]["quadrature_type"] = "log-normal" #"gaussian" or "log-normal"

    namelist_defaults["microphysics"] = Dict()

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["updraft_number"] = 1
    # namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment"] = "moisture_deficit"  # not currently used

    #! format: on

    if case_name == "Soares"
        namelist = Soares(namelist_defaults)
    elseif case_name == "Nieuwstadt"
        namelist = Nieuwstadt(namelist_defaults)
    elseif case_name == "Bomex"
        namelist = Bomex(namelist_defaults)
    elseif case_name == "LifeCycleTan2018"
        namelist = LifeCycleTan2018(namelist_defaults)
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
    return namelist
end
function Nieuwstadt(namelist_defaults)
    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "Nieuwstadt"
    return namelist
end
function Bomex(namelist_defaults)
    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "Bomex"
    return namelist
end
function LifeCycleTan2018(namelist_defaults)
    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "LifeCycleTan2018"
    return namelist
end
function Rico(namelist_defaults)
    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "Rico"

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
    return namelist
end
function TRMM_LBA(namelist_defaults)
    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "TRMM_LBA"
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

    return namelist
end
function ARM_SGP(namelist_defaults)
    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "ARM_SGP"
    return namelist
end
function GATE_III(namelist_defaults)
    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "GATE_III"
    return namelist
end
function DYCOMS_RF01(namelist_defaults)
    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "DYCOMS_RF01"
    return namelist
end
function DYCOMS_RF02(namelist_defaults)
    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "DYCOMS_RF02"

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

    return namelist
end
function GABLS(namelist_defaults)
    namelist = deepcopy(namelist_defaults)
    namelist["meta"]["casename"] = "GABLS"
    return namelist
end

function write_file(namelist, root::String = ".")
    mkpath(root)

    @assert haskey(namelist, "meta")

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
