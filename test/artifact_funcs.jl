import Downloads

const _artifact_cache = joinpath(tempdir(), "climaatmos_test_artifacts")

"""
    _download_artifact(name, url, filename)

Download a file from `url` into a cached directory under `name/`.
Returns the full path to the downloaded file.
"""
function _download_artifact(name, url, filename)
    dir = joinpath(_artifact_cache, name)
    filepath = joinpath(dir, filename)
    if !isfile(filepath)
        mkpath(dir)
        Downloads.download(url, filepath)
    end
    return filepath
end

# ERA5 data 1973 Jan: downloaded from Caltech box
function era_global_dataset_path()
    return _download_artifact(
        "era-global",
        "https://caltech.box.com/shared/static/2489yvlwhnxsrl05jvqnch7fcrd0k1vw.nc",
        "box-era5-monthly.nc",
    )
end

# download ERA5 nc data: wind, temperature, geopotential height
function era_single_column_dataset_path()
    return _download_artifact(
        "era-single-column",
        "https://caltech.box.com/shared/static/of5wi39o643a333yy9vbx5pnf0za503g.nc",
        "box-single_column_test.nc",
    )
end

function topo_res_path()
    return _download_artifact(
        "topo-info",
        "https://caltech.box.com/shared/static/isa7l4ow4xvv9vs09bivdwttbnnw5tte.nc",
        "topo_drag.res.nc",
    )
end

function mima_gwf_path()
    return _download_artifact(
        "mima-gwf",
        "https://caltech.box.com/shared/static/6r566rv4631ibfbr5p5vtv4ls7vl5fge.nc",
        "mima_gwf.nc",
    )
end

function gfdl_ogw_data_path()
    return _download_artifact(
        "gfdl-ogw-data",
        "https://caltech.box.com/shared/static/zubipz298q5ar5rpfais8c0ymtfyz2oc.nc",
        "gfdl_ogw.nc",
    )
end
