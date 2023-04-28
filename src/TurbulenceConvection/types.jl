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

abstract type AbstractSurfaceParameters{FT <: Real} end

const FloatOrFunc{FT} = Union{FT, Function, Dierckx.Spline1D}

Base.@kwdef struct FixedSurfaceFlux{FT, TS, SHF, LHF} <:
                   AbstractSurfaceParameters{FT}
    zrough::FT
    ts::TS
    shf::SHF
    lhf::LHF
end

Base.@kwdef struct FixedSurfaceFluxAndFrictionVelocity{FT, TS, SHF, LHF} <:
                   AbstractSurfaceParameters{FT}
    zrough::FT
    ts::TS
    shf::SHF
    lhf::LHF
    ustar::FT
end

Base.@kwdef struct FixedSurfaceCoeffs{FT, TS, CH, CM} <:
                   AbstractSurfaceParameters{FT}
    zrough::FT
    ts::TS
    ch::CH
    cm::CM
end

Base.@kwdef struct MoninObukhovSurface{FT, TS} <: AbstractSurfaceParameters{FT}
    zrough::FT
    ts::TS
end

const_or_func(s::Function, t::Real) = s(t)
const_or_func(s::Dierckx.Spline1D, t::Real) = s(t)
const_or_func(s, t::Real) = s

surface_thermo_state(s::AbstractSurfaceParameters, thermo_params, t::Real = 0) =
    const_or_func(s.ts, t)
sensible_heat_flux(s::AbstractSurfaceParameters, t::Real = 0) =
    const_or_func(s.shf, t)
latent_heat_flux(s::AbstractSurfaceParameters, t::Real = 0) =
    const_or_func(s.lhf, t)

fixed_ustar(::FixedSurfaceFluxAndFrictionVelocity) = true
fixed_ustar(::FixedSurfaceFlux) = false

shf(surf) = surf.shf
lhf(surf) = surf.lhf
cm(surf) = surf.Cd
ch(surf) = surf.Ch
bflux(surf) = surf.buoy_flux
get_ustar(surf) = surf.ustar
get_ρe_tot_flux(surf) = shf(surf) + lhf(surf)
get_ρq_tot_flux(surf) = surf.evaporation
get_ρu_flux(surf) = surf.ρτxz
get_ρv_flux(surf) = surf.ρτyz
obukhov_length(surf) = surf.L_MO

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


struct State{P, A, TS, T, CACHE, C}
    prog::P
    aux::A
    ts_sfc::TS
    tendencies::T
    p::CACHE
    colidx::C
end

"""
    column_state(prog, aux, tendencies, colidx)

Create a columnar state given full 3D states
 - `prog` prognostic state
 - `aux` auxiliary state
 - `tendencies` tendencies state
 - `colidx` column index

## Example
```julia
Fields.bycolumn(axes(Y.c)) do colidx
    state = TC.column_state(prog, aux, tendencies, colidx)
    ...
end
"""
function column_state(prog, p, tendencies, colidx)
    prog_cent_column = CC.column(prog.cent, colidx)
    prog_face_column = CC.column(prog.face, colidx)
    aux_cent_column = CC.column(p.edmf_cache.aux.cent, colidx)
    aux_face_column = CC.column(p.edmf_cache.aux.face, colidx)
    tends_cent_column = CC.column(tendencies.cent, colidx)
    tends_face_column = CC.column(tendencies.face, colidx)
    prog_column =
        CC.Fields.FieldVector(cent = prog_cent_column, face = prog_face_column)
    aux_column =
        CC.Fields.FieldVector(cent = aux_cent_column, face = aux_face_column)
    tends_column = CC.Fields.FieldVector(
        cent = tends_cent_column,
        face = tends_face_column,
    )

    return State(prog_column, aux_column, tends_column, p, colidx)
end

function column_prog_aux(prog, p, colidx)
    prog_cent_column = CC.column(prog.cent, colidx)
    prog_face_column = CC.column(prog.face, colidx)
    aux_cent_column = CC.column(p.edmf_cache.aux.cent, colidx)
    aux_face_column = CC.column(p.edmf_cache.aux.face, colidx)
    prog_column =
        CC.Fields.FieldVector(cent = prog_cent_column, face = prog_face_column)
    aux_column =
        CC.Fields.FieldVector(cent = aux_cent_column, face = aux_face_column)

    return State(prog_column, aux_column, nothing, p, colidx)
end

function column_diagnostics(diagnostics, colidx)
    diag_cent_column = CC.column(diagnostics.cent, colidx)
    diag_face_column = CC.column(diagnostics.face, colidx)
    diag_svpc_column = CC.column(diagnostics.svpc, colidx)
    return CC.Fields.FieldVector(
        cent = diag_cent_column,
        face = diag_face_column,
        svpc = diag_svpc_column,
    )
end


Grid(state::State) = Grid(axes(state.prog.cent))

float_type(state::State) = eltype(state.prog)
# float_type(field::CC.Fields.Field) = CC.Spaces.undertype(axes(field))
float_type(field::CC.Fields.Field) = eltype(parent(field))


function tc_column_state(
    prog,
    p,
    tendencies,
    colidx,
    surf_params,
    thermo_params,
    t,
)
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
    ts = surface_thermo_state(surf_params, thermo_params, t)

    return State(prog_column, aux_column, ts, tends_column, p, colidx)
end

function tc_column_state(
    prog,
    p,
    tendencies::Nothing,
    colidx,
    surf_params,
    thermo_params,
    t,
)
    prog_cent_column = CC.column(prog.c, colidx)
    prog_face_column = CC.column(prog.f, colidx)
    aux_cent_column = CC.column(p.edmf_cache.aux.cent, colidx)
    aux_face_column = CC.column(p.edmf_cache.aux.face, colidx)
    prog_column =
        CC.Fields.FieldVector(cent = prog_cent_column, face = prog_face_column)
    aux_column =
        CC.Fields.FieldVector(cent = aux_cent_column, face = aux_face_column)
    tends_column = nothing
    ts_sfc = surface_thermo_state(surf_params, thermo_params, t)

    return State(prog_column, aux_column, ts_sfc, tends_column, p, colidx)
end
