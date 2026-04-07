#!/usr/bin/env julia
# Regenerate `googleles_cases_seed42_10.yaml` (optional). Same logic as committed registry.
using Random
Random.seed!(42)
triples = [(s, m, e) for s in 0:499 for m in (1, 4, 7, 10) for e in ("amip",)]
shuffle!(triples)
for i in 1:10
    s, m, e = triples[i]
    println("- { index: $i, site_id: $s, month: $m, experiment: $e }")
end
