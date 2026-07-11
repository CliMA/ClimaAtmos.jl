#=
Generator for the per-degree hyperdiffusion grid-scale factor table in
`hyperdiffusion_grid_scale_factor` (src/prognostic_equations/hyperdiffusion.jl).

`β_op(p)` is the spectral radius of the assembled scalar horizontal biharmonic
`(wdivₕ ∘ gradₕ)²` on a uniform, periodic, degree-`p` spectral-element grid,
defined by `ρ(∇⁴) = (β_op / h)⁴` with `h` the mean nodal distance. The
operator matrix is formed column by column and its spectral radius taken with
`eigvals`; the higher degrees are too expensive for the test suite, so the values
are tabulated and this script regenerates them. The tabulated values are converged
in element count at the default `x_elem = 4`. Run:

    julia --project=.buildkite test/prognostic_equations/hyperdiffusion_grid_factor.jl
=#

import ClimaCore: Spaces, Fields, Operators
import ClimaCore.CommonSpaces: RectangleXYSpace
import ClimaComms
ClimaComms.@import_required_backends
using LinearAlgebra, Printf

grad = Operators.Gradient()
wdiv = Operators.WeakDivergence()

function grid_scale_factor(degree; x_elem = 4)
    space = RectangleXYSpace(
        Float64;
        x_min = 0,
        x_max = 2720,
        y_min = 0,
        y_max = 2720,
        periodic_x = true,
        periodic_y = true,
        n_quad_points = degree + 1,
        x_elem = x_elem,
        y_elem = x_elem,
    )
    buffer = Spaces.create_dss_buffer(Fields.zeros(space))
    laplacian(s) = (d = wdiv.(grad.(s)); Spaces.weighted_dss!(d, buffer); d)
    n = length(parent(Fields.zeros(space)))
    matrix = zeros(n, n)
    e = Fields.zeros(space)
    for j in 1:n
        fill!(parent(e), 0.0)
        parent(e)[j] = 1.0
        matrix[:, j] .= vec(parent(laplacian(e)))
    end
    biharmonic = maximum(real.(eigvals(matrix * matrix)))
    return biharmonic^(1 / 4) * Spaces.node_horizontal_length_scale(space)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for degree in 2:7
        @printf("degree %d: β_op = %.4f\n", degree, grid_scale_factor(degree))
    end
end
