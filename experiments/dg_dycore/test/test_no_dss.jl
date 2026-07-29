# DG invariant: this module must never call weighted_dss! — the DG face
# operators replace DSS entirely.
@testset "no weighted_dss!" begin
    srcdir = joinpath(@__DIR__, "..", "src")
    for f in readdir(srcdir)
        endswith(f, ".jl") || continue
        # match call sites only ("weighted_dss!(") — docs/comments may
        # mention the name when stating this invariant
        @test !occursin("weighted_dss!(", read(joinpath(srcdir, f), String))
    end
end
