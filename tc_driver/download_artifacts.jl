include(joinpath(@__DIR__, "..", "integration_tests", "artifact_funcs.jl"))

# Trigger download if data doesn't exist locally
function trigger_download(lazy_download = true)
    @info "pycles Artifact folder:`$(pycles_output_dataset_folder(lazy_download))`"
    @info "scampy Artifact folder:`$(scampy_output_dataset_folder(lazy_download))`"
    @info "LES_driven_SCM Artifact folder:`$(les_driven_scm_data_folder(lazy_download))`"
    return nothing
end
trigger_download()
