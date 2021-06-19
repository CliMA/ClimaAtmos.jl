import ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid

function create_dg_grid(
    domain::ProductDomain;
    elements,
    polynomial_order,
    grid_stretching = nothing,
    FT = Float64,
    mpicomm = MPI.COMM_WORLD,
    array = ClimateMachine.array_type(),
    topology = StackedBrickTopology,
)
    if elements === nothing
        error_message = "Please specify the number of elements as a tuple whose size is commensurate with the domain,"
        error_message *= " e.g., a 3 dimensional domain would need a specification like elements = (10,10,10)."
        error_message *= " or elements = (vertical = 8, horizontal = 5)"

        @error(error_message)
        return nothing
    end

    if polynomial_order === nothing
        error_message = "Please specify the polynomial order as a tuple whose size is commensurate with the domain,"
        error_message *= "e.g., a 3 dimensional domain would need a specification like polynomialorder = (3,3,3)."
        error_message *= " or polynomialorder = (vertical = 8, horizontal = 5)"

        @error(error_message)
        return nothing
    end

    dimension = ndims(domain)
 
    if (dimension < 2) || (dimension > 3)
        error_message = "SpectralElementGrid only works with dimensions 2 or 3. "
        error_message *= "The current dimension is " * string(ndims(domain))

        println("The domain is ", domain)
        @error(error_message)
        return nothing
    end

    elements = convention(elements, Val(dimension))
    if ndims(domain) != length(elements)
        @error("Incorrectly specified elements for the dimension of the domain")
        return nothing
    end

    polynomial_order = convention(polynomial_order, Val(dimension))
    if ndims(domain) != length(polynomial_order)
        @error("Incorrectly specified polynomialorders for the dimension of the domain")
        return nothing
    end

    brickrange = brick_builder(domain, grid_stretching, elements; FT = FT)

    if dimension == 2
        boundary = ((1, 2), (3, 4))
    else
        boundary = ((1, 2), (3, 4), (5, 6))
    end

    periodicity = domain.periodicity
    connectivity = dimension == 2 ? :face : :full

    topl = topology(
        mpicomm,
        brickrange;
        periodicity = periodicity,
        boundary = boundary,
        connectivity = connectivity,
    )

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = array,
        polynomialorder = polynomial_order,
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

"""
    Brick builder
"""
function brick_builder(domain::ProductDomain, ::Nothing, elements; FT = Float64)
    dimension = ndims(domain)

    tuple_ranges = []
    for i in 1:dimension
        push!(
            tuple_ranges,
            range(
                FT(domain[i].min);
                length = elements[i] + 1,
                stop = FT(domain[i].max),
            ),
        )
    end

    brickrange = Tuple(tuple_ranges)
    return brickrange
end

function brick_builder(domain::ProductDomain, grid_stretching::SingleExponentialStretching, elements; FT = Float64)
    dimension = ndims(domain)

    tuple_ranges = []
    for i in 1:dimension-1
        push!(
            tuple_ranges,
            range(
                FT(domain[i].min);
                length = elements[i] + 1,
                stop = FT(domain[i].max),
            ),
        )
    end
    push!(
        tuple_ranges,
        grid1d(
            FT(domain[dimension].min),
            FT(domain[dimension].max),
            grid_stretching,
            nelem = elements[dimension],
        ),
    )

    brickrange = Tuple(tuple_ranges)
    return brickrange
end