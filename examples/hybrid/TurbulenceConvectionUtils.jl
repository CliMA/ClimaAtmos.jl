module TurbulenceConvectionUtils

using LinearAlgebra, StaticArrays
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

function get_edmf_cache(Y, namelist, param_set, parsed_args)
    tc_params = CAP.turbconv_params(param_set)
    Ri_bulk_crit = namelist["turbulence"]["EDMF_PrognosticTKE"]["Ri_crit"]
    case = Cases.get_case(namelist)
    FT = CC.Spaces.undertype(axes(Y.c))
    test_consistency = parsed_args["test_edmf_consistency"]
    forcing =
        Cases.ForcingBase(case, FT; Cases.forcing_kwargs(case, namelist)...)
    radiation = Cases.RadiationBase(case, FT)
    surf_ref_state = Cases.surface_ref_state(case, tc_params, namelist)
    surf_params =
        Cases.surface_params(case, surf_ref_state, tc_params; Ri_bulk_crit)
    precip_name = TC.parse_namelist(
        namelist,
        "microphysics",
        "precipitation_model";
        default = "None",
        valid_options = ["None", "cutoff", "clima_1m"],
    )
    # TODO: move to grid mean model
    precip_model = if precip_name == "None"
        TC.NoPrecipitation()
    elseif precip_name == "cutoff"
        TC.CutoffPrecipitation()
    elseif precip_name == "clima_1m"
        TC.Clima1M()
    else
        error("Invalid precip_name $(precip_name)")
    end
    edmf = TC.EDMFModel(FT, namelist, precip_model, parsed_args)
    @info "EDMFModel: \n$(summary(edmf))"
    return (;
        edmf,
        case,
        forcing,
        test_consistency,
        radiation,
        surf_params,
        param_set,
        surf_ref_state,
        aux = get_aux(edmf, Y, FT),
        precip_model,
    )
end

function tc_column_state(prog, p, tendencies, colidx)
    prog_cent_column = CC.column(prog.c, colidx)
    prog_face_column = CC.column(prog.f, colidx)
    aux_cent_column = CC.column(p.edmf_cache.aux.cent, colidx)
    aux_face_column = CC.column(p.edmf_cache.aux.face, colidx)
    tends_cent_column = CC.column(tendencies.c, colidx)
    tends_face_column = CC.column(tendencies.f, colidx)
    prog_column =
        CC.Fields.FieldVector(cent = prog_cent_column, face = prog_face_column)
    aux_column =
        CC.Fields.FieldVector(cent = aux_cent_column, face = aux_face_column)
    tends_column = CC.Fields.FieldVector(
        cent = tends_cent_column,
        face = tends_face_column,
    )

    return TC.State(prog_column, aux_column, tends_column, p, colidx)
end

function tc_column_state(prog, p, tendencies::Nothing, colidx)
    prog_cent_column = CC.column(prog.c, colidx)
    prog_face_column = CC.column(prog.f, colidx)
    aux_cent_column = CC.column(p.edmf_cache.aux.cent, colidx)
    aux_face_column = CC.column(p.edmf_cache.aux.face, colidx)
    prog_column =
        CC.Fields.FieldVector(cent = prog_cent_column, face = prog_face_column)
    aux_column =
        CC.Fields.FieldVector(cent = aux_cent_column, face = aux_face_column)
    tends_column = nothing

    return TC.State(prog_column, aux_column, tends_column, p, colidx)
end


function init_tc!(Y, p, param_set, namelist)
    (; edmf_cache, Δt) = p
    (; edmf, param_set, surf_ref_state, surf_params, forcing, radiation, case) =
        edmf_cache
    tc_params = CAP.turbconv_params(param_set)

    FT = eltype(edmf)
    N_up = TC.n_updrafts(edmf)

    CC.Fields.bycolumn(axes(Y.c)) do colidx
        # `nothing` goes into State because OrdinaryDiffEq.jl owns tendencies.
        state = tc_column_state(Y, p, nothing, colidx)

        grid = TC.Grid(state)
        FT = eltype(grid)
        t = FT(0)
        compute_ref_state!(state, grid, tc_params; ts_g = surf_ref_state)

        Cases.initialize_profiles(case, grid, tc_params, state)
        set_thermo_state_pθq!(state, grid, edmf.moisture_model, tc_params)
        set_grid_mean_from_thermo_state!(tc_params, state, grid)
        assign_thermo_aux!(state, grid, edmf.moisture_model, tc_params)
        Cases.initialize_forcing(case, forcing, grid, state, tc_params)
        Cases.initialize_radiation(case, radiation, grid, state, tc_params)
        initialize_edmf(edmf, grid, state, surf_params, tc_params, t, case)
    end
end


function sgs_flux_tendency!(Yₜ, Y, p, t, colidx)
    (; edmf_cache, Δt) = p
    (; edmf, param_set, case, surf_params) = edmf_cache
    (; radiation, forcing, precip_model, test_consistency) = edmf_cache
    tc_params = CAP.turbconv_params(param_set)
    state = tc_column_state(Y, p, Yₜ, colidx)
    grid = TC.Grid(state)
    if test_consistency
        parent(state.aux.face) .= NaN
        parent(state.aux.cent) .= NaN
    end

    set_thermo_state_peq!(
        state,
        grid,
        edmf.moisture_model,
        edmf.compressibility_model,
        tc_params,
    )
    assign_thermo_aux!(state, grid, edmf.moisture_model, tc_params)

    aux_gm = TC.center_aux_grid_mean(state)

    surf = get_surface(surf_params, grid, state, t, tc_params)

    TC.affect_filter!(edmf, grid, state, tc_params, surf, t)

    # Update aux / pre-tendencies filters. TODO: combine these into a function that minimizes traversals
    # Some of these methods should probably live in `compute_tendencies`, when written, but we'll
    # treat them as auxiliary variables for now, until we disentangle the tendency computations.
    Cases.update_forcing(case, grid, state, t, tc_params)
    Cases.update_radiation(radiation, grid, state, t, tc_params)

    TC.update_aux!(edmf, grid, state, surf, tc_params, t, Δt)

    TC.compute_precipitation_sink_tendencies(
        precip_model,
        edmf,
        grid,
        state,
        tc_params,
        Δt,
    )
    TC.compute_precipitation_advection_tendencies(
        precip_model,
        edmf,
        grid,
        state,
        tc_params,
    )

    TC.compute_turbconv_tendencies!(edmf, grid, state, tc_params, surf, Δt)

    # TODO: incrementally disable this and enable proper grid mean terms
    compute_gm_tendencies!(
        edmf,
        grid,
        state,
        surf,
        radiation,
        forcing,
        tc_params,
    )
    return nothing
end

end # module
