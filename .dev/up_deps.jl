#=
A simple script for updating the manifest
files in all of our environments.
=#

root = dirname(@__DIR__)
dirs = (
    root,
    joinpath(root, "test"),
    joinpath(root, "perf"),
    joinpath(root, "docs"),
    joinpath(root, "examples"),
)

cd(root) do
    for dir in dirs
        @info "Updating environment `$dir`"
        cmd = `$(Base.julia_cmd()) --project=$dir -e 'import Pkg; Pkg.update()'`
        run(cmd)
    end
end

# https://github.com/JuliaLang/Pkg.jl/issues/3014
for dir in dirs
    cd(dir) do
        rm("LocalPreferences.toml"; force = true)
    end
end
