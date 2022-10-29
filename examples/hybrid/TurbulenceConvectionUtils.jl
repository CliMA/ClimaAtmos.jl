module TurbulenceConvectionUtils

using LinearAlgebra
import ClimaAtmos
import ClimaAtmos.Parameters as CAP
import ClimaCore as CC
import ClimaCore.Geometry as CCG
import ClimaCore.Operators as CCO
import ClimaCore.Geometry: ⊗
import OrdinaryDiffEq as ODE
import ClimaAtmos.TurbulenceConvection
import ClimaAtmos.TurbulenceConvection as TC

import UnPack
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

function get_aux(edmf, Y, ::Type{FT}) where {FT}
    fspace = axes(Y.f)
    cspace = axes(Y.c)
    aux_cent_fields = TC.FieldFromNamedTuple(cspace, cent_aux_vars, FT, edmf)
    aux_face_fields = TC.FieldFromNamedTuple(fspace, face_aux_vars, FT, edmf)
    aux = CC.Fields.FieldVector(cent = aux_cent_fields, face = aux_face_fields)
    return aux
end

function get_edmf_cache(
    Y,
    turbconv_model,
    precip_model,
    namelist,
    param_set,
    parsed_args,
)
    tc_params = CAP.turbconv_params(param_set)
    Ri_bulk_crit = namelist["turbulence"]["EDMF_PrognosticTKE"]["Ri_crit"]
    FT = CC.Spaces.undertype(axes(Y.c))
    test_consistency = parsed_args["test_edmf_consistency"]
    case = Cases.get_case(namelist)
    thermo_params = CAP.thermodynamics_params(param_set)
    surf_ref_state = Cases.surface_ref_state(case, tc_params, namelist)
    surf_params =
        Cases.surface_params(case, surf_ref_state, tc_params; Ri_bulk_crit)
    edmf = turbconv_model
    ᶠspace_1 = axes(Y.f[CC.Fields.ColumnIndex((1, 1), 1)])
    logpressure_fun =
        CA.log_pressure_profile(ᶠspace_1, thermo_params, surf_ref_state)
    @info "EDMFModel: \n$(summary(edmf))"
    return (;
        edmf,
        logpressure_fun,
        case,
        test_consistency,
        surf_params,
        param_set,
        surf_ref_state,
        aux = get_aux(edmf, Y, FT),
        precip_model,
    )
end

function init_tc!(Y, p, params, namelist)
    (; edmf_cache, Δt) = p
    (; edmf, param_set, surf_ref_state, logpressure_fun, surf_params, case) =
        edmf_cache
    tc_params = CAP.turbconv_params(params)

    FT = eltype(edmf)

    CC.Fields.bycolumn(axes(Y.c)) do colidx
        # `nothing` goes into State because OrdinaryDiffEq.jl owns tendencies.
        state = TC.tc_column_state(Y, p, nothing, colidx)
        thermo_params = CAP.thermodynamics_params(params)

        grid = TC.Grid(state)
        FT = eltype(grid)
        t = FT(0)

        CA.compute_ref_pressure!(p.ᶜp[colidx], logpressure_fun)
        CA.compute_ref_pressure!(
            p.edmf_cache.aux.face.p[colidx],
            logpressure_fun,
        )

        CA.compute_ref_density!(
            Y.c.ρ[colidx],
            p.ᶜp[colidx],
            thermo_params,
            surf_ref_state,
        )
        CA.compute_ref_density!(
            p.edmf_cache.aux.face.ρ[colidx],
            p.edmf_cache.aux.face.p[colidx],
            thermo_params,
            surf_ref_state,
        )

        # TODO: convert initialize_profiles to set prognostic state, not aux state
        Cases.initialize_profiles(case, grid, tc_params, state)

        # Temporarily, we'll re-populate ρq_tot based on initial aux q_tot
        q_tot = edmf_cache.aux.cent.q_tot[colidx]
        @. Y.c.ρq_tot[colidx] = Y.c.ρ[colidx] * q_tot
        set_thermo_state_pθq!(Y, p, colidx)
        set_grid_mean_from_thermo_state!(tc_params, state, grid)
        assign_thermo_aux!(state, grid, edmf.moisture_model, tc_params)
        initialize_edmf(edmf, grid, state, surf_params, tc_params, t)
    end
end


function sgs_flux_tendency!(Yₜ, Y, p, t, colidx)
    (; edmf_cache, Δt, compressibility_model) = p
    (; edmf, param_set, surf_params, surf_ref_state) = edmf_cache
    (; precip_model, test_consistency, logpressure_fun) = edmf_cache
    thermo_params = CAP.thermodynamics_params(param_set)
    tc_params = CAP.turbconv_params(param_set)
    state = TC.tc_column_state(Y, p, Yₜ, colidx)
    grid = TC.Grid(state)
    if test_consistency
        parent(state.aux.face) .= NaN
        parent(state.aux.cent) .= NaN
    end

    if compressibility_model isa CA.AnelasticFluid
        # TODO: how should this be computed if compressible?
        # pressure at cell centers have been populated, we need
        # pressure BCs
        CA.compute_ref_pressure!(
            p.edmf_cache.aux.face.p[colidx],
            logpressure_fun,
        )
        CA.compute_ref_density!(
            p.edmf_cache.aux.face.ρ[colidx],
            p.edmf_cache.aux.face.p[colidx],
            thermo_params,
            surf_ref_state,
        )
    end
    assign_thermo_aux!(state, grid, edmf.moisture_model, tc_params)

    aux_gm = TC.center_aux_grid_mean(state)

    surf = get_surface(surf_params, grid, state, t, tc_params)

    TC.affect_filter!(edmf, grid, state, tc_params, surf, t)

    TC.update_aux!(edmf, grid, state, surf, tc_params, t, Δt)

    TC.compute_precipitation_sink_tendencies(
        precip_model,
        edmf.precip_fraction_model,
        grid,
        state,
        tc_params,
        Δt,
    )
    TC.compute_precipitation_advection_tendencies(
        precip_model,
        edmf.precip_fraction_model,
        grid,
        state,
        tc_params,
    )

    TC.compute_turbconv_tendencies!(edmf, grid, state, tc_params, surf, Δt)

    # TODO: incrementally disable this and enable proper grid mean terms
    compute_gm_tendencies!(edmf, grid, state, surf, tc_params)
    return nothing
end

end # module
