module TurbulenceConvectionUtils

using LinearAlgebra
using LinearAlgebra: norm_sqr
import ClimaAtmos
import ClimaAtmos.Parameters as CAP
import ClimaCore as CC
import ClimaCore.Geometry as CCG
import ClimaCore.Operators as CCO
import ClimaCore.Geometry: ⊗
import OrdinaryDiffEq as ODE
import ClimaAtmos.TurbulenceConvection
import ClimaAtmos.TurbulenceConvection as TC

import Logging
import TerminalLoggers

const ca_dir = pkgdir(ClimaAtmos)

include(joinpath(ca_dir, "tc_driver", "Cases.jl"))
import .Cases

include(joinpath(ca_dir, "tc_driver", "dycore.jl"))
include(joinpath(ca_dir, "tc_driver", "Surface.jl"))
include(joinpath(ca_dir, "tc_driver", "initial_conditions.jl"))
include(joinpath(ca_dir, "tc_driver", "generate_namelist.jl"))
import .NameList

#####
##### No TurbulenceConvection scheme
#####

turbconv_cache(
    Y,
    turbconv_model::Nothing,
    atmos,
    namelist,
    param_set,
    parsed_args,
) = (; turbconv_model)

implicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing
explicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing

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
    namelist,
    param_set,
    parsed_args,
)
    tc_params = CAP.turbconv_params(param_set)
    FT = CC.Spaces.undertype(axes(Y.c))
    imex_edmf_turbconv = parsed_args["imex_edmf_turbconv"]
    imex_edmf_gm = parsed_args["imex_edmf_gm"]
    test_consistency = parsed_args["test_edmf_consistency"]
    case = Cases.get_case(namelist["meta"]["casename"])
    thermo_params = CAP.thermodynamics_params(param_set)
    surf_ref_thermo_state =
        Cases.surface_reference_thermo_state(case, thermo_params)
    surf_params =
        Cases.surface_params(case, surf_ref_thermo_state, thermo_params)
    edmf = turbconv_model
    #anelastic_column_kwargs = if true # CA.is_anelastic_column(atmos) # TODO: make conditional
    #    ᶠspace_1 = axes(Y.f[CC.Fields.ColumnIndex((1, 1), 1)])
    #    ᶜspace_1 = axes(Y.c[CC.Fields.ColumnIndex((1, 1), 1)])
    #    logpressure_fun = CA.log_pressure_profile(
    #        ᶠspace_1,
    #        thermo_params,
    #        surf_ref_thermo_state,
    #    )
    #    ᶜz = CC.Fields.coordinate_field(ᶜspace_1).z
    #    ᶜp₀ = @. exp(logpressure_fun(ᶜz))
    #    (; ᶜp₀)
    #else
    #    NamedTuple()
    #end
    @info "EDMFModel: \n$(summary(edmf))"
    cache = (;
        edmf,
        turbconv_model,
        #anelastic_column_kwargs...,
        case,
        imex_edmf_turbconv,
        imex_edmf_gm,
        test_consistency,
        surf_params,
        param_set,
        surf_ref_thermo_state,
        aux = get_aux(atmos, edmf, Y, FT),
        Y_filtered = similar(Y),
    )
    return (; edmf_cache = cache, turbconv_model)
end

function init_tc!(Y, p, params)
    CC.Fields.bycolumn(axes(Y.c)) do colidx
        init_tc!(Y, p, params, colidx)
    end
end

function init_tc!(Y, p, params, colidx)

    (; edmf, surf_ref_thermo_state, surf_params, case) = p.edmf_cache
    tc_params = CAP.turbconv_params(params)

    FT = eltype(edmf)
    # `nothing` goes into State because OrdinaryDiffEq.jl owns tendencies.
    state = TC.tc_column_state(Y, p, nothing, colidx)
    thermo_params = CAP.thermodynamics_params(params)

    grid = TC.Grid(state)
    FT = eltype(grid)
    C123 = CCG.Covariant123Vector
    t = FT(0)

    # TODO if CA.is_anelastic_column(p.atmos)
    #@. p.ᶜp[colidx] = p.edmf_cache.ᶜp₀

    #CA.compute_ref_density!(
    #    Y.c.ρ[colidx],
    #    p.ᶜp[colidx],
    #    thermo_params,
    #    surf_ref_thermo_state,
    #)
    #else
    # ...
    #end

    # TODO: convert initialize_profiles to set prognostic state, not aux state
    Cases.initialize_profiles(case, grid, thermo_params, state)
    set_thermo_state_pθq_no_density!(Y, p, colidx)

    # Temporarily, we'll re-populate state based on edmf aux
    q_tot = p.edmf_cache.aux.cent.q_tot[colidx]
    @. Y.c.ρq_tot[colidx] = Y.c.ρ[colidx] * q_tot
    set_grid_mean_from_thermo_state!(thermo_params, state, grid)
    assign_thermo_aux!(state, grid, edmf.moisture_model, thermo_params)
    initialize_edmf(edmf, grid, state, surf_params, tc_params, t)
end

# TODO: Split update_aux! and other functions into implicit and explicit parts.

function implicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, ::TC.EDMFModel)
    (; edmf_cache, Δt) = p
    (; edmf, param_set, surf_params, surf_ref_thermo_state, Y_filtered) =
        edmf_cache
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
    (; edmf, param_set, surf_params, surf_ref_thermo_state, Y_filtered) =
        edmf_cache
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
        edmf.precip_fraction_model,
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
    Yₜ.c.turbconv[colidx] .+=
        (Y_filtered.c.turbconv[colidx] .- Y.c.turbconv[colidx]) ./ Δt
    Yₜ.f.turbconv[colidx] .+=
        (Y_filtered.f.turbconv[colidx] .- Y.f.turbconv[colidx]) ./ Δt

    return nothing
end

end # module
