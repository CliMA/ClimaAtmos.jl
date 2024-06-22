using Glob

"""
"""

# function get_les_calibration_library()
#     les_library = get_shallow_LES_library()
#     # AMIP4K data: July, NE Pacific
#     cfsite_numbers = (17, 18, 19, 20, 22, 23)
#     les_kwargs = (forcing_model = "HadGEM2-A", month = 7, experiment = "amip")
#     ref_paths = [get_stats_path(get_cfsite_les_dir(cfsite_number; les_kwargs...)) for cfsite_number in cfsite_numbers]
#     return ref_paths
# end

function get_les_calibration_library()
    return get_all_les_paths()[1:20]
end

"""
    get_LES_library

Hierarchical dictionary of available cfSite LES simulations, following similar
forcing to that described in [Shen2022](@cite), but including additional sites.
The following cfsites are available across listed models, months,
and experiments.
"""
function get_LES_library()
    LES_library = get_shallow_LES_library()
    deep_sites = (collect(30:33)..., collect(66:70)..., 82, 92, 94, 96, 99, 100)

    append!(LES_library["HadGEM2-A"]["07"]["cfsite_numbers"], deep_sites)
    append!(LES_library["HadGEM2-A"]["01"]["cfsite_numbers"], deep_sites)
    sites_04 = deepcopy(setdiff(deep_sites, [32, 92, 94]))
    append!(LES_library["HadGEM2-A"]["04"]["cfsite_numbers"], sites_04)
    sites_10 = deepcopy(setdiff(deep_sites, [94, 100]))
    append!(LES_library["HadGEM2-A"]["10"]["cfsite_numbers"], sites_10)

    LES_library_full = deepcopy(LES_library)
    for model in keys(LES_library_full)
        for month in keys(LES_library_full[model])
            LES_library_full[model][month]["cfsite_numbers"] = Dict()
            for cfsite_number in LES_library[model][month]["cfsite_numbers"]
                cfsite_number_str = string(cfsite_number, pad = 2)
                LES_library_full[model][month]["cfsite_numbers"][cfsite_number_str] = if cfsite_number >= 30
                    "deep"
                else
                    "shallow"
                end
            end
        end
    end
    return LES_library_full
end



"""
    parse_les_path(les_path)
Given path to LES stats file, return cfsite_number, forcing_model, month, and experiment from filename.
    LES filename should follow pattern: `Stats.cfsite<SITE-NUMBER>_<FORCING-MODEL>_<EXPERIMENT>_2004-2008.<MONTH>.nc`
Inputs:
 - les_path - path to les simulation containing stats folder
Outputs:
 - cfsite_number  :: cfsite number
 - forcing_model :: {"HadGEM2-A", "CNRM-CM5", "CNRM-CM6-1", "IPSL-CM6A-LR"} - name of climate model used for forcing. Currently, only "HadGEM2-A" simulations are available reliably.
 - month :: {1, 4, 7, 10} - month of simulation.
 - experiment :: {"amip", "amip4K"} - experiment from which LES was forced.
"""
function parse_les_path(les_path)
    fname = basename(les_path)
    fname_split = split(fname, ('.', '_'))
    forcing_model = fname_split[3]
    experiment = fname_split[4]
    month = parse(Int64, fname_split[6])
    cfsite_number = parse(Int64, replace(fname_split[2], "cfsite" => ""))
    return (cfsite_number, forcing_model, month, experiment)
end

function valid_lespath(les_path)
    cfsite_number, forcing_model, month, experiment = parse_les_path(les_path)
    month = string(month, pad = 2)
    cfsite_number = string(cfsite_number, pad = 2)
    LES_library = get_LES_library()
    @assert forcing_model in keys(LES_library) "Forcing model $(forcing_model) not valid."
    @assert month in keys(LES_library[forcing_model]) "Month $(month) not available for $(forcing_model)."
    @assert cfsite_number in keys(LES_library[forcing_model][month]["cfsite_numbers"]) "cfSite $(cfsite_number) not found for $(forcing_model), month $(month)."
    @assert experiment in LES_library[forcing_model][month]["experiments"]
end



"""
    get_shallow_LES_library

Hierarchical dictionary of available LES simulations described in [Shen2022](@cite).
The following cfsites are available across listed models, months,
and experiments.
"""
function get_shallow_LES_library()
    LES_library = Dict("HadGEM2-A" => Dict(), "CNRM-CM5" => Dict(), "CNRM-CM6-1" => Dict())
    Shen_et_al_sites = collect(4:15)
    append!(Shen_et_al_sites, collect(17:23))

    # HadGEM2-A model (76 AMIP-AMIP4K pairs)
    LES_library["HadGEM2-A"]["10"] = Dict()
    LES_library["HadGEM2-A"]["10"]["cfsite_numbers"] = Shen_et_al_sites
    LES_library["HadGEM2-A"]["07"] = Dict()
    LES_library["HadGEM2-A"]["07"]["cfsite_numbers"] = deepcopy(Shen_et_al_sites)
    LES_library["HadGEM2-A"]["04"] = Dict()
    LES_library["HadGEM2-A"]["04"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [15, 17, 18])
    LES_library["HadGEM2-A"]["01"] = Dict()
    LES_library["HadGEM2-A"]["01"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [15, 17, 18, 19, 20])

    # CNRM-CM5 model (59 AMIP-AMIP4K pairs)
    LES_library["CNRM-CM5"]["10"] = Dict()
    LES_library["CNRM-CM5"]["10"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [15, 22, 23])
    LES_library["CNRM-CM5"]["07"] = Dict()
    LES_library["CNRM-CM5"]["07"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [13, 14, 15, 18])
    LES_library["CNRM-CM5"]["04"] = Dict()
    LES_library["CNRM-CM5"]["04"]["cfsite_numbers"] =
        setdiff(Shen_et_al_sites, [11, 12, 13, 14, 15, 17, 18, 21, 22, 23])
    LES_library["CNRM-CM5"]["01"] = Dict()
    LES_library["CNRM-CM5"]["01"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [14, 15, 17, 18, 19, 20, 21, 22, 23])

    # CNRM-CM6-1 model (69 AMIP-AMIP4K pairs)
    LES_library["CNRM-CM6-1"]["10"] = Dict()
    LES_library["CNRM-CM6-1"]["10"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [22, 23])
    LES_library["CNRM-CM6-1"]["07"] = Dict()
    LES_library["CNRM-CM6-1"]["07"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [12, 13, 14, 15, 17])
    LES_library["CNRM-CM6-1"]["04"] = Dict()
    LES_library["CNRM-CM6-1"]["04"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [13, 14, 15])
    LES_library["CNRM-CM6-1"]["01"] = Dict()
    LES_library["CNRM-CM6-1"]["01"]["cfsite_numbers"] = setdiff(Shen_et_al_sites, [14, 15, 21, 22, 23])

    for month in ["01", "04", "07", "10"]
        LES_library["HadGEM2-A"][month]["experiments"] = ["amip", "amip4K"]
        LES_library["CNRM-CM5"][month]["experiments"] = ["amip", "amip4K"]
        LES_library["CNRM-CM6-1"][month]["experiments"] = ["amip", "amip4K"]
    end
    return LES_library
end






"""
get_cfsite_les_dir(
    cfsite_number::Integer;
    forcing_model::String = "HadGEM2-A",
    month::Integer = 7,
    experiment::String = "amip",)

Given information about an LES run from [Shen2022](@cite),
fetch LES directory on central cluster.

Inputs:

- cfsite_number  :: cfsite number
- forcing_model :: {"HadGEM2-A", "CNRM-CM5", "CNRM-CM6-1", "IPSL-CM6A-LR"} - name of climate model used for forcing. Currently, only "HadGEM2-A" simulations are available reliably.
- month :: {1, 4, 7, 10} - month of simulation.
- experiment :: {"amip", "amip4K"} - experiment from which LES was forced.

Outputs:

- les_dir - path to les simulation containing stats folder
"""
function get_cfsite_les_dir(
    cfsite_number::Integer;
    forcing_model::String = "HadGEM2-A",
    month::Integer = 7,
    experiment::String = "amip",
    )
    month = string(month, pad = 2)
    cfsite_number = string(cfsite_number)
    root_dir = "/central/groups/esm/zhaoyi/GCMForcedLES/cfsite/$month/$forcing_model/$experiment/"
    rel_dir = join(["Output.cfsite$cfsite_number", forcing_model, experiment, "2004-2008.$month.4x"], "_")
    les_dir = joinpath(root_dir, rel_dir)
    # Check lespath is valid
    valid_lespath(les_dir)
    return les_dir
end


"""
    get_stats_path(dir)

Given directory to standard LES or SCM output, fetch path to stats file.
"""
function get_stats_path(dir)
    stats = joinpath(dir, "stats")
    if !ispath(stats)
        nc_path = joinpath(dir, "*.nc")
        stat_files = glob(relpath(abspath(nc_path)))
        @assert length(stat_files) == 1 "$(length(stat_files)) stats files found with paths $nc_path"
        return stat_files[1]
    end
    try
        nc_path = joinpath(stats, "*.nc")
        stat_files = glob(relpath(abspath(nc_path)))
        @assert length(stat_files) == 1 "$(length(stat_files)) stats files found with paths $nc_path"
        return stat_files[1]
    catch e
        if isa(e, AssertionError)
            @warn "No unique stats netCDF file found in $stats. Extending search to other files."
            try
                stat_files = readdir(stats, join = true) # WindowsOS/julia relpath bug
                if length(stat_files) == 1
                    return stat_files[1]
                else
                    @error "No unique stats file found at $dir. The search returned $(length(stat_files)) results."
                end
            catch f
                if isa(f, Base.IOError)
                    @warn "Extended search errored with: $f"
                    return ""
                else
                    throw(f)
                end
            end
        else
            @warn "An error occurred retrieving the stats path at $dir. Throwing..."
            throw(e)
        end
    end
end

function get_all_les_paths()
    NUM_LES_CASES  = 176
    les_library = get_shallow_LES_library()

    ref_dirs = []
    for model in keys(les_library)
        for month in keys(les_library[model])
            cfsite_numbers = Tuple(les_library[model][month]["cfsite_numbers"])
            les_kwargs = (forcing_model = model, month = parse(Int, month), experiment = "amip")
            append!(ref_dirs, [get_stats_path(get_cfsite_les_dir(cfsite_number; les_kwargs...)) for cfsite_number in cfsite_numbers])
        end
    end

    ref_dirs = ref_dirs[1:NUM_LES_CASES]
    return ref_dirs
end

