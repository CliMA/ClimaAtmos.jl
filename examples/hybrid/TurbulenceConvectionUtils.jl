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

import Logging
import TerminalLoggers

const ca_dir = pkgdir(ClimaAtmos)

include(joinpath(ca_dir, "tc_driver", "Cases.jl"))
import .Cases

include(joinpath(ca_dir, "tc_driver", "dycore.jl"))
include(joinpath(ca_dir, "tc_driver", "Surface.jl"))


#####
##### No TurbulenceConvection scheme
#####

turbconv_cache(Y, turbconv_model::Nothing, atmos, param_set, parsed_args) =
    (; turbconv_model)

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
    param_set,
    parsed_args,
)
    tc_params = CAP.turbconv_params(param_set)
    FT = CC.Spaces.undertype(axes(Y.c))
    imex_edmf_turbconv = parsed_args["imex_edmf_turbconv"]
    imex_edmf_gm = parsed_args["imex_edmf_gm"]
    test_consistency = parsed_args["test_edmf_consistency"]
    case = Cases.get_case(parsed_args["turbconv_case"])
    thermo_params = CAP.thermodynamics_params(param_set)
    @assert atmos.moisture_model in (CA.DryModel(), CA.EquilMoistModel())
    surf_params =
        Cases.surface_params(case, thermo_params, atmos.moisture_model)
    edmf = turbconv_model
    @info "EDMFModel: \n$(summary(edmf))"
    cache = (;
        edmf,
        turbconv_model,
        case,
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

function init_tc!(Y, p, params)
    CC.Fields.bycolumn(axes(Y.c)) do colidx
        init_tc!(Y, p, params, colidx)
    end
end

function init_tc!(Y, p, params, colidx)

    (; edmf, surf_params, case) = p.edmf_cache
    tc_params = CAP.turbconv_params(params)
    thermo_params = CAP.thermodynamics_params(params)

    FT = eltype(edmf)
    # `nothing` goes into State because OrdinaryDiffEq.jl owns tendencies.
    state = TC.tc_column_state(Y, p, nothing, colidx)
    grid = TC.Grid(state)

    parent(state.aux.face) .= NaN
    parent(state.aux.cent) .= NaN

    # Initialization of required auxiliary variables:
    (; ᶜp, ᶜts) = p
    ᶜθ_liq_ice = p.edmf_cache.aux.cent.θ_liq_ice
    ᶜq_tot = p.edmf_cache.aux.cent.q_tot
    ᶜtke = p.edmf_cache.aux.cent.tke
    ᶜHvar = p.edmf_cache.aux.cent.Hvar
    ᶜQTvar = p.edmf_cache.aux.cent.QTvar
    ᶜHQTcov = p.edmf_cache.aux.cent.HQTcov
    ᶜq_tot .= 0 # set to 0 because it is only initialized for moist cases
    ᶜHvar .= 0 # set to 0 because it is only initialized for GABLS
    ᶜQTvar .= 0 # set to 0 because it is not initialized for any cases
    ᶜHQTcov .= 0 # set to 0 because it is not initialized for any cases
    Cases.initialize_profiles(case, grid, thermo_params, state)
    if p.atmos.moisture_model isa CA.DryModel
        @. ᶜts[colidx] =
            TD.PhaseDry_pθ(thermo_params, ᶜp[colidx], ᶜθ_liq_ice[colidx])
    else
        @. ᶜts[colidx] = TD.PhaseEquil_pθq(
            thermo_params,
            ᶜp[colidx],
            ᶜθ_liq_ice[colidx],
            ᶜq_tot[colidx],
        )
    end

    # Initialization of grid-mean prognostic variables:
    (; ᶜinterp) = p.operators
    C123 = CCG.Covariant123Vector
    @. Y.c.ρ[colidx] = TD.air_density(thermo_params, ᶜts[colidx])
    @. Y.c.ρe_tot[colidx] =
        Y.c.ρ[colidx] * TD.total_energy(
            thermo_params,
            ᶜts[colidx],
            LA.norm_sqr(C123(Y.c.uₕ[colidx]) + C123(ᶜinterp(Y.f.w[colidx]))) /
            2,
            p.ᶜΦ[colidx],
        )
    if !(p.atmos.moisture_model isa CA.DryModel)
        @. Y.c.ρq_tot[colidx] = Y.c.ρ[colidx] * ᶜq_tot[colidx]
    end

    # Initialization of EDMF prognostic variables:
    N_up = TC.n_updrafts(edmf)
    a_min = edmf.minimum_area
    lg = CC.Fields.local_geometry_field(axes(Y.f))
    @inbounds for i in 1:N_up
        @. Y.c.turbconv.up[i].ρarea[colidx] = Y.c.ρ[colidx] * a_min
        @. Y.c.turbconv.up[i].ρaq_tot[colidx] =
            Y.c.turbconv.up[i].ρarea[colidx] * ᶜq_tot[colidx]
        @. Y.c.turbconv.up[i].ρaθ_liq_ice[colidx] =
            Y.c.turbconv.up[i].ρarea[colidx] * ᶜθ_liq_ice[colidx]
        @. Y.f.turbconv.up[i].w[colidx] =
            CC.Geometry.Covariant3Vector(CC.Geometry.WVector(FT(0)), lg[colidx])
    end
    a_en = (1 - N_up * a_min)
    @. Y.c.turbconv.en.ρatke[colidx] = Y.c.ρ[colidx] * a_en * ᶜtke[colidx]
    if p.atmos.turbconv_model.thermo_covariance_model isa
       TC.PrognosticThermoCovariances
        @. Y.c.turbconv.en.ρaHvar[colidx] = Y.c.ρ[colidx] * a_en * ᶜHvar[colidx]
        @. Y.c.turbconv.en.ρaQTvar[colidx] =
            Y.c.ρ[colidx] * a_en * ᶜQTvar[colidx]
        @. Y.c.turbconv.en.ρaHQTcov[colidx] =
            Y.c.ρ[colidx] * a_en * ᶜHQTcov[colidx]
    end

    # Modification of EDMF prognostic variables with BCs:
    t = FT(0)
    surf = get_surface(
        state.p.atmos.model_config,
        surf_params,
        grid,
        state,
        t,
        tc_params,
    )
    TC.affect_filter!(edmf, grid, state, tc_params, surf, t)

    # Initialization of all other auxiliary variables, some of which are saved
    # for diagnostics in the saving callback:
    assign_thermo_aux!(state, grid, edmf.moisture_model, thermo_params)
    TC.update_aux!(edmf, grid, state, surf, tc_params, t, p.Δt)
    # TODO: Compute diagnostic values in the saving callback, not in update_aux.
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
    Yₜ.c.turbconv[colidx] .+=
        (Y_filtered.c.turbconv[colidx] .- Y.c.turbconv[colidx]) ./ Δt
    Yₜ.f.turbconv[colidx] .+=
        (Y_filtered.f.turbconv[colidx] .- Y.f.turbconv[colidx]) ./ Δt

    return nothing
end

end # module
