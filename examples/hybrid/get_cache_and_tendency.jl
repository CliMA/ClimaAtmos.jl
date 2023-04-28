import ClimaAtmos as CA
using ClimaCore: Fields

function get_cache(
    Y,
    parsed_args,
    params,
    spaces,
    atmos,
    numerics,
    simulation,
    comms_ctx,
)
    default_cache = CA.default_cache(
        Y,
        parsed_args,
        params,
        atmos,
        spaces,
        numerics,
        simulation,
        comms_ctx,
    )
    merge(
        default_cache,
        additional_cache(
            Y,
            default_cache,
            parsed_args,
            params,
            atmos,
            simulation.dt,
        ),
    )
end

function implicit_tendency!(Yₜ, Y, p, t)
    p.test_dycore_consistency && CA.fill_with_nans!(p)
    @nvtx "implicit tendency" color = colorant"yellow" begin
        Yₜ .= zero(eltype(Yₜ))
        @nvtx "precomputed quantities" color = colorant"orange" begin
            CA.set_precomputed_quantities!(Y, p, t)
        end
        Fields.bycolumn(axes(Y.c)) do colidx
            CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t, colidx)
            if p.turbconv_model isa CA.TurbulenceConvection.EDMFModel
                CA.implicit_sgs_flux_tendency!(
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
            CA.set_precomputed_quantities!(Y, p, t)
        end
        @nvtx "horizontal" color = colorant"orange" begin
            CA.horizontal_advection_tendency!(Yₜ, Y, p, t)
            @nvtx "hyperdiffusion tendency" color = colorant"yellow" begin
                CA.hyperdiffusion_tendency!(Yₜ, Y, p, t)
            end
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
