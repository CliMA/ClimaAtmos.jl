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

Base.@kwdef struct EDMFParameters{FT}
    surface_area::FT
    max_area::FT
    min_area::FT
    tke_ed_coeff::FT
    tke_diss_coeff::FT
    static_stab_coeff::FT
    tke_surf_scale::FT
    Prandtl_number_scale::FT
    Prandtl_number_0::FT
    Ri_crit::FT
    smin_ub::FT
    smin_rm::FT
    l_max::FT
    entrainment_factor::FT
    detrainment_factor::FT
    turbulent_entrainment_factor::FT
    entrainment_smin_tke_coeff::FT
    updraft_mixing_frac::FT
    area_limiter_scale::FT
    area_limiter_power::FT
    entrainment_scale::FT
    sorting_power::FT
    min_upd_velocity::FT
    min_updraft_top::FT
    pressure_normalmode_buoy_coeff1::FT
    pressure_normalmode_adv_coeff::FT
    pressure_normalmode_drag_coeff::FT
    moisture_model::String
    thermo_covariance_model::String
    diagnostic_covar_limiter::FT
    sgs::String
    quadrature_order::Integer
    quadrature_type::String
    updraft_number::Integer
    precip_fraction_model::String
    prescribed_precip_frac::FT
    precip_fraction_limiter::FT
    τ_acnv_rai::FT
    τ_acnv_sno::FT
    q_liq_threshold::FT
    q_ice_threshold::FT
    microph_scaling::FT
    microph_scaling_dep_sub::FT
    microph_scaling_melt::FT
    E_liq_rai::FT
    E_liq_sno::FT
    E_ice_rai::FT
    E_ice_sno::FT
    E_rai_sno::FT
end

function default_namelist(::Nothing)

    args = parse_commandline()
    case_name = args["case_name"]
    return default_namelist(case_name)
end

function edmf_paramset(
    EDMF_pairs::NamedTuple,
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
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["surface_area"] = EDMF_pairs.surface_area
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["max_area"] = EDMF_pairs.max_area
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["min_area"] = EDMF_pairs.min_area

    # mixing_length
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["tke_ed_coeff"] = EDMF_pairs.tke_ed_coeff
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["tke_diss_coeff"] = EDMF_pairs.tke_diss_coeff
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["static_stab_coeff"] = EDMF_pairs.static_stab_coeff
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["tke_surf_scale"] = EDMF_pairs.tke_surf_scale
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["Prandtl_number_scale"] = EDMF_pairs.Prandtl_number_scale
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["Prandtl_number_0"] = EDMF_pairs.Prandtl_number_0
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["Ri_crit"] = EDMF_pairs.Ri_crit
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["smin_ub"] = EDMF_pairs.smin_ub
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["smin_rm"] = EDMF_pairs.smin_rm
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["l_max"] = EDMF_pairs.l_max
    # entrainment
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_factor"] = EDMF_pairs.entrainment_factor
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["detrainment_factor"] = EDMF_pairs.detrainment_factor

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["turbulent_entrainment_factor"] = EDMF_pairs.turbulent_entrainment_factor
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_smin_tke_coeff"] = EDMF_pairs.entrainment_smin_tke_coeff
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["updraft_mixing_frac"] = EDMF_pairs.updraft_mixing_frac
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["area_limiter_scale"] = EDMF_pairs.area_limiter_scale
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["area_limiter_power"] = EDMF_pairs.area_limiter_power
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment_scale"] = EDMF_pairs.entrainment_scale
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["sorting_power"] = EDMF_pairs.sorting_power
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["min_upd_velocity"] = EDMF_pairs.min_upd_velocity
    # pressure
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["min_updraft_top"] = EDMF_pairs.min_updraft_top
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_buoy_coeff1"] = EDMF_pairs.pressure_normalmode_buoy_coeff1
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_adv_coeff"] = EDMF_pairs.pressure_normalmode_adv_coeff
    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_drag_coeff"] = EDMF_pairs.pressure_normalmode_drag_coeff

    # thermodynamics
    namelist_defaults["thermodynamics"] = Dict()
    namelist_defaults["thermodynamics"]["moisture_model"] = EDMF_pairs.moisture_model
    namelist_defaults["thermodynamics"]["thermo_covariance_model"] = EDMF_pairs.thermo_covariance_model
    namelist_defaults["thermodynamics"]["diagnostic_covar_limiter"] = EDMF_pairs.diagnostic_covar_limiter
    namelist_defaults["thermodynamics"]["sgs"] = EDMF_pairs.sgs
    namelist_defaults["thermodynamics"]["quadrature_order"] = EDMF_pairs.quadrature_order
    namelist_defaults["thermodynamics"]["quadrature_type"] = EDMF_pairs.quadrature_type

    namelist_defaults["microphysics"] = Dict()

    namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["updraft_number"] = EDMF_pairs.updraft_number
    # namelist_defaults["turbulence"]["EDMF_PrognosticTKE"]["entrainment"] = "moisture_deficit"  # not currently used

    #! format: on
    normal_cases = [
        "Soares",
        "Nieuwstadt",
        "Bomex",
        "LifeCycleTan2018",
        "ARM_SGP",
        "GATE_III",
        "DYCOMS_RF01",
        "GABLS"
    ]
    microphys_param_cases = [
        "Rico",
        "TRMM_LBA",
        "DYCOMS_RF02"
    ]
    
    if case_name in normal_cases
        namelist_defaults["meta"]["casename"] = case_name
    elseif case_name in microphys_param_cases
        namelist_defaults["meta"]["casename"] = case_name
        namelist_defaults["microphysics"]["precip_fraction_model"] = EDMF_pairs.precip_fraction_model
        namelist_defaults["microphysics"]["prescribed_precip_frac_value"] = EDMF_pairs.prescribed_precip_frac
        namelist_defaults["microphysics"]["precip_fraction_limiter"] = EDMF_pairs.precip_fraction_limiter
        namelist_defaults["microphysics"]["τ_acnv_rai"] = EDMF_pairs.τ_acnv_rai
        namelist_defaults["microphysics"]["τ_acnv_sno"] = EDMF_pairs.τ_acnv_sno
        namelist_defaults["microphysics"]["q_liq_threshold"] = EDMF_pairs.q_liq_threshold
        namelist_defaults["microphysics"]["q_ice_threshold"] = EDMF_pairs.q_ice_threshold
        namelist_defaults["microphysics"]["microph_scaling"] = EDMF_pairs.microph_scaling
        namelist_defaults["microphysics"]["microph_scaling_dep_sub"] = EDMF_pairs.microph_scaling_dep_sub
        namelist_defaults["microphysics"]["microph_scaling_melt"] = EDMF_pairs.microph_scaling_melt
        namelist_defaults["microphysics"]["E_liq_rai"] = EDMF_pairs.E_liq_rai
        namelist_defaults["microphysics"]["E_liq_sno"] = EDMF_pairs.E_liq_sno
        namelist_defaults["microphysics"]["E_ice_rai"] = EDMF_pairs.E_ice_rai
        namelist_defaults["microphysics"]["E_ice_sno"] = EDMF_pairs.E_ice_sno
        namelist_defaults["microphysics"]["E_rai_sno"] = EDMF_pairs.E_rai_sno
    else
        error("Not a valid case name")
    end

    if write
        write_file(namelist_defaults, root)
    end
    return namelist_defaults
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
