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
function rrtmgp_artifact(file_name)
    artifact_name = "RRTMGPReferenceData"
    artifacts_file = joinpath(pkgdir(RRTMGP), "test", "Artifacts.toml")
    data_folder = joinpath(
        Pkg.Artifacts.ensure_artifact_installed(artifact_name, artifacts_file),
        artifact_name,
    )
    return joinpath(data_folder, file_name)
end

"""
    data_loader(fn, file_name)

Loads data from an `NCDataset` from the `RRTMGP.jl` artifact stored in
`RRTMGPReferenceData/<file_name>`, and calls a function, `fn` on the
dataset.
"""
function data_loader(fn, file_name)
    NC.Dataset(rrtmgp_artifact(file_name), "r") do ds
        fn(ds)
    end
end
