import Pkg
import NCDatasets as NC
import RRTMGP

#####
##### RRTMGP
#####

"""
    rrtmgp_artifact(file_name)

Returns the filename of an artifact stored in
`RRTMGPReferenceData/<file_name>`.
"""
rrtmgp_artifact(file_name) = joinpath(rrtmgp_artifact_path(), file_name)

"""
    data_loader(fn, file_name)

Loads data from an `NCDataset` from the `RRTMGP.jl` artifact stored in
`RRTMGPReferenceData/<file_name>`, and calls a function, `fn` on the
dataset.
"""
function rrtmgp_data_loader(fn, file_name)
    NC.Dataset(rrtmgp_artifact(file_name), "r") do ds
        fn(ds)
    end
end
