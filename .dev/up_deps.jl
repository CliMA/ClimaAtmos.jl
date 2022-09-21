#=
A simple script for updating the manifest
files in all of our environments.
=#

root = dirname(@__DIR__)
dirs = (
    root,
    joinpath(root, "test"),
    joinpath(root, ".dev"),
    joinpath(root, "perf"),
    joinpath(root, "docs"),
    joinpath(root, "examples"),
)

cd(root) do
    for dir in dirs
        reldir = relpath(dir, root)
        @info "Updating environment `$reldir`"
        cmd = if dir == root
            `$(Base.julia_cmd()) --project -e """import Pkg; Pkg.update()"""`
        elseif dir == joinpath(root, ".dev")
            `$(Base.julia_cmd()) --project=$reldir -e """import Pkg; Pkg.update()"""`
        else
            `$(Base.julia_cmd()) --project=$reldir -e """import Pkg; Pkg.develop(;path=\".\"); Pkg.update()"""`
        end
        run(cmd)
    end
end

# https://github.com/JuliaLang/Pkg.jl/issues/3014
for dir in dirs
    cd(dir) do
        rm("LocalPreferences.toml"; force = true)
    end
end
