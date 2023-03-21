import ClimaAtmos as CA
using ClimaCore: Fields

# To add additional terms to the explicit part of the tendency, define new
# methods for `additional_cache` and `additional_tendency!`.

get_cache(
    Y,
    parsed_args,
    params,
    spaces,
    atmos,
    numerics,
    simulation,
    comms_ctx,
) = merge(
    CA.default_cache(
        Y,
        parsed_args,
        params,
        atmos,
        spaces,
        numerics,
        simulation,
        comms_ctx,
    ),
    additional_cache(Y, parsed_args, params, atmos, simulation.dt),
)

function implicit_tendency!(Yₜ, Y, p, t)
    p.test_dycore_consistency && CA.fill_with_nans!(p)
    @nvtx "precomputed quantities" color = colorant"orange" begin
        CA.precomputed_quantities!(Y, p, t)
    end
    @nvtx "implicit tendency" color = colorant"yellow" begin
        Fields.bycolumn(axes(Y.c)) do colidx
            CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t, colidx)

            if p.turbconv_model isa CA.TurbulenceConvection.EDMFModel
                parent(Yₜ.c.turbconv[colidx]) .= zero(eltype(Yₜ))
                parent(Yₜ.f.turbconv[colidx]) .= zero(eltype(Yₜ))
                TCU.implicit_sgs_flux_tendency!(
                    Yₜ,
                    Y,
                    p,
                    t,
                    colidx,
                    p.turbconv_model,
                )
            end
        end
    end
end

function remaining_tendency!(Yₜ, Y, p, t)
    p.test_dycore_consistency && CA.fill_with_nans!(p)
    @nvtx "remaining tendency" color = colorant"yellow" begin
        Yₜ .= zero(eltype(Yₜ))
        @nvtx "precomputed quantities" color = colorant"orange" begin
            CA.precomputed_quantities!(Y, p, t)
        end
        @nvtx "horizontal" color = colorant"orange" begin
            CA.horizontal_advection_tendency!(Yₜ, Y, p, t)
        end
        @nvtx "vertical" color = colorant"orange" begin
            CA.explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
        end
        @nvtx "additional_tendency!" color = colorant"orange" begin
            additional_tendency!(Yₜ, Y, p, t)
        end
        @nvtx "dss_remaining_tendency" color = colorant"blue" begin
            CA.dss!(Yₜ, p, t)
        end
    end
    return Yₜ
end
