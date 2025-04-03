
include("../get_les_metadata.jl")


function get_les_calibration_library_runner() # All HADGEM
    les_library = get_LES_library()
    ref_paths = String[]
    cfsite_numbers = Int[]
    models = ["HadGEM2-A"]
    experiment = "amip"
    for model in models
        for month in keys(les_library[model])
            cfsite_numbers_i = [
                parse(Int, key) for
                key in keys(les_library[model][month]["cfsite_numbers"])
            ]
            les_kwargs = (
                forcing_model = model,
                month = parse(Int, month),
                experiment = experiment,
            )
            for cfsite_number in cfsite_numbers_i
                try
                    stats_path = get_stats_path(
                        get_cfsite_les_dir(cfsite_number; les_kwargs...),
                    )
                    push!(ref_paths, stats_path)
                    push!(cfsite_numbers, cfsite_number)
                catch e
                    if isa(e, AssertionError)
                        continue
                    else
                        rethrow(e)
                    end
                end
            end
        end
    end


    return (ref_paths, cfsite_numbers)
end


# function get_les_calibration_library_runner()
#     les_library = get_shallow_LES_library()
#     # AMIP data: July, NE Pacific
#     cfsite_numbers = (17, 18, 22, 23, 30, 94)
#     les_kwargs = (forcing_model = "HadGEM2-A", month = 7, experiment = "amip")
#     ref_paths = [
#         get_stats_path(get_cfsite_les_dir(cfsite_number; les_kwargs...)) for
#         cfsite_number in cfsite_numbers
#     ]
#     return (ref_paths, cfsite_numbers)
# end
