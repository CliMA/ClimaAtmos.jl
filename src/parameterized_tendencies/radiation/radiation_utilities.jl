import NCDatasets as NC

#####
##### RRTMGP
#####

"""
    data_loader(fn, file_name)

Loads data from an `NCDataset` from the `RRTMGP.jl` artifact stored in
`RRTMGPReferenceData/<file_name>`, and calls a function, `fn` on the
dataset.
"""
function rrtmgp_data_loader(fn, file_name)
    NC.Dataset(file_name, "r") do ds
        fn(ds)
    end
end
