function create_dg_grid(
    domain::SphericalShell;
    elements = nothing,
    polynomial_order = nothing,
    grid_stretching = nothing,
    FT = Float64,
    mpicomm = MPI.COMM_WORLD,
    array = ClimateMachine.array_type()
)

    new_polynomial_order = convention(polynomial_order, Val(2))
    # new_polynomial_order = new_polynomial_order .+ convention(overintegration_order, Val(2))
    horizontal, vertical = new_polynomial_order

    Rrange = grid1d(
        domain.radius, 
        domain.radius + domain.height, 
        grid_stretching,
        nelem = elements.vertical,
    )

    topl = StackedCubedSphereTopology(
        mpicomm,
        elements.horizontal,
        Rrange,
    )

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = array,
        polynomialorder = (horizontal, vertical),
        meshwarp = equiangular_cubed_sphere_warp,
    )

    return grid

end

"""
    Conventions for polynomial order and overintegration order 
"""
function convention(
    a::NamedTuple{(:vertical, :horizontal), T},
    ::Val{3},
) where {T}
    return (a.horizontal, a.horizontal, a.vertical)
end

function convention(a::Number, ::Val{3})
    return (a, a, a)
end

function convention(
    a::NamedTuple{(:vertical, :horizontal), T},
    ::Val{2},
) where {T}
    return (a.horizontal, a.vertical)
end

function convention(a::Number, ::Val{2})
    return (a, a)
end

function convention(a::Tuple, b)
    return a
end

