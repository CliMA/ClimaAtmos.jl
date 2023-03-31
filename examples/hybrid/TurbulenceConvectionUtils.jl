module TurbulenceConvectionUtils

using LinearAlgebra
import ClimaAtmos
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore as CC
import ClimaCore.Geometry as CCG
import ClimaCore.Operators as CCO
import ClimaCore.Geometry: ⊗
import OrdinaryDiffEq as ODE
import ClimaAtmos.TurbulenceConvection
import ClimaAtmos.TurbulenceConvection as TC
import ClimaAtmos.InitialConditions as ICs

import Logging
import TerminalLoggers

const ca_dir = pkgdir(ClimaAtmos)

include(joinpath(ca_dir, "tc_driver", "dycore.jl"))
include(joinpath(ca_dir, "tc_driver", "Surface.jl"))

#####
##### No TurbulenceConvection scheme
#####

turbconv_cache(
    Y,
    turbconv_model,
    atmos,
    param_set,
    parsed_args,
    initial_condition,
) = (; turbconv_model)

implicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, _) = nothing
explicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, _) = nothing

#####
##### EDMF
#####

function get_aux(atmos, edmf, Y, ::Type{FT}) where {FT}
    fspace = axes(Y.f)
    cspace = axes(Y.c)
    aux_cent_fields =
        TC.FieldFromNamedTuple(cspace, cent_aux_vars, FT, atmos, edmf)
    aux_face_fields =
        TC.FieldFromNamedTuple(fspace, face_aux_vars, FT, atmos, edmf)
    aux = CC.Fields.FieldVector(cent = aux_cent_fields, face = aux_face_fields)
    return aux
end

function turbconv_cache(
    Y,
    turbconv_model::TC.EDMFModel,
    atmos,
    param_set,
    parsed_args,
    initial_condition,
)
    FT = CC.Spaces.undertype(axes(Y.c))
    imex_edmf_turbconv = parsed_args["imex_edmf_turbconv"]
    imex_edmf_gm = parsed_args["imex_edmf_gm"]
    test_consistency = parsed_args["test_edmf_consistency"]
    thermo_params = CAP.thermodynamics_params(param_set)
    surf_params = ICs.surface_params(initial_condition, thermo_params)
    edmf = turbconv_model
    @info "EDMFModel: \n$(summary(edmf))"
    cache = (;
        edmf,
        turbconv_model,
        imex_edmf_turbconv,
        imex_edmf_gm,
        test_consistency,
        surf_params,
        param_set,
        aux = get_aux(atmos, edmf, Y, FT),
        Y_filtered = similar(Y),
    )
    return (; edmf_cache = cache, turbconv_model)
end

# TODO: Split update_aux! and other functions into implicit and explicit parts.

function implicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, ::TC.EDMFModel)
    (; edmf_cache, Δt) = p
    (; edmf, param_set, surf_params, Y_filtered) = edmf_cache
    (; imex_edmf_turbconv, imex_edmf_gm, test_consistency) = edmf_cache
    thermo_params = CAP.thermodynamics_params(param_set)
    tc_params = CAP.turbconv_params(param_set)

    imex_edmf_turbconv || imex_edmf_gm || return nothing

    Y_filtered.c[colidx] .= Y.c[colidx]
    Y_filtered.f[colidx] .= Y.f[colidx]

    state = TC.tc_column_state(Y_filtered, p, Yₜ, colidx)

    grid = TC.Grid(state)
    if test_consistency
        parent(state.aux.face) .= NaN
        parent(state.aux.cent) .= NaN
    end

    assign_thermo_aux!(state, grid, edmf.moisture_model, thermo_params)

    surf = get_surface(
        p.atmos.model_config,
        surf_params,
        grid,
        state,
        t,
        tc_params,
    )

    TC.affect_filter!(edmf, grid, state, tc_params, surf, t)

    TC.update_aux!(edmf, grid, state, surf, tc_params, t, Δt)

    imex_edmf_turbconv &&
        TC.compute_implicit_turbconv_tendencies!(edmf, grid, state)

    imex_edmf_gm &&
        compute_implicit_gm_tendencies!(edmf, grid, state, surf, tc_params)

    # Note: The "filter relaxation tendency" should not be included in the
    # implicit tendency because its derivative with respect to Y is
    # discontinuous, which means that including it would make the linear
    # linear equation being solved by Newton's method ill-conditioned.

    return nothing
end

function explicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, ::TC.EDMFModel)
    (; edmf_cache, Δt) = p
    (; edmf, param_set, surf_params, Y_filtered) = edmf_cache
    (; imex_edmf_turbconv, imex_edmf_gm, test_consistency) = edmf_cache
    thermo_params = CAP.thermodynamics_params(param_set)
    tc_params = CAP.turbconv_params(param_set)

    # Note: We could also do Y_filtered .= Y further upstream if needed.
    Y_filtered.c[colidx] .= Y.c[colidx]
    Y_filtered.f[colidx] .= Y.f[colidx]

    state = TC.tc_column_state(Y_filtered, p, Yₜ, colidx)

    grid = TC.Grid(state)
    if test_consistency
        parent(state.aux.face) .= NaN
        parent(state.aux.cent) .= NaN
    end

    assign_thermo_aux!(state, grid, edmf.moisture_model, thermo_params)

    surf = get_surface(
        p.atmos.model_config,
        surf_params,
        grid,
        state,
        t,
        tc_params,
    )

    TC.affect_filter!(edmf, grid, state, tc_params, surf, t)

    TC.update_aux!(edmf, grid, state, surf, tc_params, t, Δt)

    TC.compute_precipitation_sink_tendencies(
        p.precip_model,
        grid,
        state,
        tc_params,
        Δt,
    )

    # Ensure that, when a tendency is not computed with an IMEX formulation,
    # both its implicit and its explicit components are computed here.

    TC.compute_explicit_turbconv_tendencies!(edmf, grid, state)
    imex_edmf_turbconv ||
        TC.compute_implicit_turbconv_tendencies!(edmf, grid, state)

    # TODO: incrementally disable this and enable proper grid mean terms
    compute_explicit_gm_tendencies!(edmf, grid, state, surf, tc_params)
    imex_edmf_gm ||
        compute_implicit_gm_tendencies!(edmf, grid, state, surf, tc_params)

    # Note: This "filter relaxation tendency" can be scaled down if needed, but
    # it must be present in order to prevent Y and Y_filtered from diverging
    # during each timestep.
    Yₜ_turbconv = CC.Fields.FieldVector(
        c = Yₜ.c.turbconv[colidx],
        f = Yₜ.f.turbconv[colidx],
    )
    Y_filtered_turbconv = CC.Fields.FieldVector(
        c = Y_filtered.c.turbconv[colidx],
        f = Y_filtered.f.turbconv[colidx],
    )
    Y_turbconv = CC.Fields.FieldVector(
        c = Y.c.turbconv[colidx],
        f = Y.f.turbconv[colidx],
    )
    Yₜ_turbconv .+= (Y_filtered_turbconv .- Y_turbconv) ./ Δt
    return nothing
end

end # module
