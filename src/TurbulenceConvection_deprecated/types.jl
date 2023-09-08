"""
    PrecipFormation

Storage for tendencies due to precipitation formation

$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct PrecipFormation{FT}
    θ_liq_ice_tendency::FT = FT(0)
    e_tot_tendency::FT = FT(0)
    qt_tendency::FT = FT(0)
    ql_tendency::FT = FT(0)
    qi_tendency::FT = FT(0)
    qr_tendency::FT = FT(0)
    qs_tendency::FT = FT(0)
end

"""
    GradBuoy

Environmental buoyancy gradients.

$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct GradBuoy{FT}
    "environmental vertical buoyancy gradient"
    ∂b∂z::FT = FT(0)
    "vertical buoyancy gradient in the unsaturated part of the environment"
    ∂b∂z_unsat::FT = FT(0)
    "vertical buoyancy gradient in the saturated part of the environment"
    ∂b∂z_sat::FT = FT(0)
end

abstract type AbstractEnvBuoyGradClosure end
struct BuoyGradMean <: AbstractEnvBuoyGradClosure end
struct BuoyGradQuadratures <: AbstractEnvBuoyGradClosure end

Base.broadcastable(x::BuoyGradMean) = tuple(x)
Base.broadcastable(x::BuoyGradQuadratures) = tuple(x)

"""
    EnvBuoyGrad

Variables used in the environmental buoyancy gradient computation.

$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct EnvBuoyGrad{FT, EBC <: AbstractEnvBuoyGradClosure}
    "temperature in the saturated part"
    t_sat::FT
    "vapor specific humidity  in the saturated part"
    qv_sat::FT
    "total specific humidity in the saturated part"
    qt_sat::FT
    "potential temperature in the saturated part"
    θ_sat::FT
    "liquid ice potential temperature in the saturated part"
    θ_liq_ice_sat::FT
    "virtual potential temperature gradient in the non saturated part"
    ∂θv∂z_unsat::FT
    "total specific humidity gradient in the saturated part"
    ∂qt∂z_sat::FT
    "liquid ice potential temperature gradient in the saturated part"
    ∂θl∂z_sat::FT
    "reference pressure"
    p::FT
    "cloud fraction"
    en_cld_frac::FT
    "density"
    ρ::FT
end
function EnvBuoyGrad(
    ::EBG,
    t_sat::FT,
    args...,
) where {FT <: Real, EBG <: AbstractEnvBuoyGradClosure}
    return EnvBuoyGrad{FT, EBG}(t_sat, args...)
end

"""
    MinDisspLen

Minimum dissipation model

$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct MinDisspLen{FT}
    "height"
    z::FT
    "obukhov length"
    obukhov_length::FT
    "surface TKE values"
    tke_surf::FT
    "u star - surface velocity scale"
    ustar::FT
    "turbulent Prandtl number"
    Pr::FT
    "reference pressure"
    p::FT
    "vertical buoyancy gradient struct"
    ∇b::GradBuoy{FT}
    "env shear"
    Shear²::FT
    "environment turbulent kinetic energy"
    tke::FT
    "Updraft tke source"
    b_exch::FT
end

const_or_func(s::Function, t::Real) = s(t)
const_or_func(s::Dierckx.Spline1D, t::Real) = s(t)
const_or_func(s, t::Real) = s

struct EDMFModel{N_up, FT, MM, PM, EBGC}
    surface_area::FT
    max_area::FT
    minimum_area::FT
    moisture_model::MM
    precip_model::PM
    bg_closure::EBGC
    zero_uv_fluxes::Bool
end
function EDMFModel(
    ::Type{FT},
    moisture_model,
    precip_model,
    parsed_args,
    turbconv_params,
) where {FT}

    tc_case = parsed_args["turbconv_case"]
    zero_uv_fluxes = any(tcc -> tcc == tc_case, ["TRMM_LBA", "ARM_SGP"])
    # Set the number of updrafts (1)
    n_updrafts = turbconv_params.updraft_number

    surface_area = turbconv_params.surface_area
    max_area = turbconv_params.max_area
    minimum_area = turbconv_params.min_area

    bg_closure = BuoyGradMean()
    if !(moisture_model isa EquilMoistModel)
        @warn string(
            "SGS model only supports equilibrium moisture choice.",
            "EDMF will carry zeros, but avoid saturation adjustment calls.",
        )
    end

    MM = typeof(moisture_model)
    PM = typeof(precip_model)
    EBGC = typeof(bg_closure)
    return EDMFModel{n_updrafts, FT, MM, PM, EBGC}(
        surface_area,
        max_area,
        minimum_area,
        moisture_model,
        precip_model,
        bg_closure,
        zero_uv_fluxes,
    )
end

parameter_set(obj) = obj.param_set
n_updrafts(::EDMFModel{N_up}) where {N_up} = N_up
Base.eltype(::EDMFModel{N_up, FT}) where {N_up, FT} = FT

Base.broadcastable(edmf::EDMFModel) = tuple(edmf)

function Base.summary(io::IO, edmf::EDMFModel)
    pns = string.(propertynames(edmf))
    buf = maximum(length.(pns))
    keys = propertynames(edmf)
    vals = repeat.(" ", map(s -> buf - length(s) + 2, pns))
    bufs = (; zip(keys, vals)...)
    print(io, '\n')
    for pn in propertynames(edmf)
        prop = getproperty(edmf, pn)
        # Skip some data:
        prop isa Bool && continue
        prop isa NTuple && continue
        prop isa Int && continue
        prop isa Float64 && continue
        prop isa Float32 && continue
        s = string(
            "  ", # needed for some reason
            getproperty(bufs, pn),
            '`',
            string(pn),
            '`',
            "::",
            '`',
            typeof(prop),
            '`',
            '\n',
        )
        print(io, s)
    end
end


struct State{P, A, T, CACHE, C, SC}
    prog::P
    aux::A
    tendencies::T
    p::CACHE
    colidx::C
    surface_conditions::SC
end

Grid(state::State) = Grid(axes(state.prog.cent))

float_type(state::State) = eltype(state.prog)
# float_type(field::CC.Fields.Field) = CC.Spaces.undertype(axes(field))
float_type(field::CC.Fields.Field) = eltype(parent(field))

import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces


Base.@propagate_inbounds function field_vector_column(
    fv::Fields.FieldVector{T},
    colidx::Fields.ColumnIndex,
) where {T}
    values = map(x -> x[colidx], Fields._values(fv))
    return Fields.FieldVector{T, typeof(values)}(values)
end

function tc_column_state(prog, p, tendencies, colidx, t)
    prog_column = field_vector_column(prog, colidx)
    aux_column = field_vector_column(p.edmf_cache.aux, colidx)
    tends_column = field_vector_column(tendencies, colidx)
    surface_conditions = CC.column(p.sfc_conditions, colidx)[]
    return State(
        prog_column,
        aux_column,
        tends_column,
        p,
        colidx,
        surface_conditions,
    )
end

function tc_column_state(prog, p, tendencies::Nothing, colidx, t)
    prog_column = field_vector_column(prog, colidx)
    aux_column = field_vector_column(p.edmf_cache.aux, colidx)
    tends_column = nothing
    surface_conditions = CC.column(p.sfc_conditions, colidx)[]
    return State(
        prog_column,
        aux_column,
        tends_column,
        p,
        colidx,
        surface_conditions,
    )
end
