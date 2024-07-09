import ClimaCore:
    DataLayouts,
    Geometry,
    Topologies,
    Grids,
    Hypsography,
    Spaces,
    Fields,
    Operators
import ClimaTimeSteppers
import ClimaComms
import LinearAlgebra: adjoint, ldiv!, DenseMatrix, lu, norm

ClimaTimeSteppers.NVTX.@annotate function ClimaTimeSteppers.solve_newton!(
    alg::ClimaTimeSteppers.NewtonsMethod,
    cache,
    x,
    f!,
    j! = nothing,
    pre_iteration! = nothing,
    post_solve! = nothing,
)
    (; max_iters, update_j, krylov_method, convergence_checker, verbose) = alg
    (; krylov_method_cache, convergence_checker_cache) = cache
    (; Δx, f, j) = cache
    if (!isnothing(j)) && ClimaTimeSteppers.needs_update!(
        update_j,
        ClimaTimeSteppers.NewNewtonSolve(),
    )
        j!(j, x)
    end
    for n in 1:max_iters
        # Compute Δx[n].
        if (!isnothing(j)) && ClimaTimeSteppers.needs_update!(
            update_j,
            ClimaTimeSteppers.NewNewtonIteration(),
        )
            j!(j, x)
        end
        f!(f, x)
        if isnothing(krylov_method)
            if j isa DenseMatrix
                ldiv!(Δx, lu(j), f) # Highly inefficient! Only used for testing.
            else
                ldiv!(Δx, j, f)
            end
        else
            ClimaTimeSteppers.solve_krylov!(
                krylov_method,
                krylov_method_cache,
                Δx,
                x,
                f!,
                f,
                n,
                pre_iteration!,
                j,
            )
        end
        ClimaTimeSteppers.is_verbose(verbose) &&
            @info "Newton iteration $n: ‖x‖ = $(norm(x)), ‖Δx‖ = $(norm(Δx))"

        c_dss_buffer = Spaces.create_dss_buffer(Δx.c)
        f_dss_buffer = Spaces.create_dss_buffer(Δx.f)
        Spaces.weighted_dss!(Δx.c => c_dss_buffer, Δx.f => f_dss_buffer)

        x .-= Δx
        # Update x[n] with Δx[n - 1], and exit the loop if Δx[n] is not needed.
        # Check for convergence if necessary.
        if ClimaTimeSteppers.is_converged!(
            convergence_checker,
            convergence_checker_cache,
            x,
            Δx,
            n,
        )
            isnothing(post_solve!) || post_solve!(x)
            break
        elseif n == max_iters
            isnothing(post_solve!) || post_solve!(x)
        else
            isnothing(pre_iteration!) || pre_iteration!(x)
        end
        if ClimaTimeSteppers.is_verbose(verbose) && n == max_iters
            @warn "Newton's method did not converge within $n iterations: ‖x‖ = $(norm(x)), ‖Δx‖ = $(norm(Δx))"
        end
    end
end

function Grids._ExtrudedFiniteDifferenceGrid(
    horizontal_grid::Grids.AbstractGrid,
    vertical_grid::Grids.FiniteDifferenceGrid,
    adaption::Grids.HypsographyAdaption,
    global_geometry::Geometry.AbstractGlobalGeometry,
    face_z::DataLayouts.AbstractData{Geometry.ZPoint{FT}},
) where {FT}
    # construct the "flat" grid
    # avoid cached constructor so that it gets cleaned up automatically
    flat_grid = Grids._ExtrudedFiniteDifferenceGrid(
        horizontal_grid,
        vertical_grid,
        Grids.Flat(),
        global_geometry,
    )
    center_flat_space = Spaces.space(flat_grid, Grids.CellCenter())
    face_flat_space = Spaces.space(flat_grid, Grids.CellFace())

    # compute the "z-only local geometry" based on face z coords
    ArrayType = ClimaComms.array_type(horizontal_grid.topology)
    # currently only works on Arrays
    (center_z_local_geometry, face_z_local_geometry) = Grids.fd_geometry_data(
        Adapt.adapt(Array, face_z),
        Val(Topologies.isperiodic(vertical_grid.topology)),
    )

    center_z_local_geometry = Adapt.adapt(ArrayType, center_z_local_geometry)
    face_z_local_geometry = Adapt.adapt(ArrayType, face_z_local_geometry)

    # compute ∇Z at face and centers
    grad = Operators.Gradient()

    center_∇Z_field =
        grad.(
            Fields.Field(
                center_z_local_geometry,
                center_flat_space,
            ).coordinates.z
        )
    face_∇Z_field =
        grad.(
            Fields.Field(face_z_local_geometry, face_flat_space).coordinates.z
        )
    # buffer = (;
    #     c = Spaces.create_dss_buffer(center_∇Z_field),
    #     f = Spaces.create_dss_buffer(face_∇Z_field),
    # )

    # Spaces.weighted_dss!(center_∇Z_field => buffer.c, face_∇Z_field => buffer.f)

    # construct full local geometry
    center_local_geometry =
        Geometry.product_geometry.(
            horizontal_grid.local_geometry,
            center_z_local_geometry,
            Ref(global_geometry),
            Ref(Geometry.WVector(1)) .*
            adjoint.(Fields.field_values(center_∇Z_field)),
        )
    face_local_geometry =
        Geometry.product_geometry.(
            horizontal_grid.local_geometry,
            face_z_local_geometry,
            Ref(global_geometry),
            Ref(Geometry.WVector(1)) .*
            adjoint.(Fields.field_values(face_∇Z_field)),
        )

    return Grids.ExtrudedFiniteDifferenceGrid(
        horizontal_grid,
        vertical_grid,
        adaption,
        global_geometry,
        center_local_geometry,
        face_local_geometry,
    )
end
