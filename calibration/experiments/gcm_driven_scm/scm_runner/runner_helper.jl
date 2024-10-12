

include("../get_les_metadata.jl")



function get_les_calibration_library_runner()
    les_library = get_shallow_LES_library()
    # AMIP data: July, NE Pacific
    cfsite_numbers = (17, 18, 22, 23, 30, 94)
    les_kwargs = (forcing_model = "HadGEM2-A", month = 7, experiment = "amip")
    ref_paths = [
        get_stats_path(get_cfsite_les_dir(cfsite_number; les_kwargs...)) for
        cfsite_number in cfsite_numbers
    ]
    return (ref_paths, cfsite_numbers)
end
