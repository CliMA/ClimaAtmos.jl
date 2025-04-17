#!/usr/bin/env julia

println("up_dev.jl has been discontinued in favor of PkgDevTools")
println("To use PkgDevTool, add it to your base environment with: ")
println()
println("julia -e 'using Pkg; Pkg.add(\"PkgDevTools\")'")
println()
println("Then, update the manifests with")
println("using PkgDevTools; PkgDevTools.update_deps(\".\")")
println("in a Julia REPL.")
println("See documentation to read more about this change")
println("This file will be removed in future releases")
