abstract type AbstractPrecipFractionModel end
struct PrescribedPrecipFraction{FT} <: AbstractPrecipFractionModel
    prescribed_precip_frac_value::FT
end
struct DiagnosticPrecipFraction{FT} <: AbstractPrecipFractionModel
    precip_fraction_limiter::FT
end

function PrecipFractionModel(paramset::NamedTuple)
    precip_fraction_model_name = paramset.precip_fraction_model
    if precip_fraction_model_name == "prescribed"
        return PrescribedPrecipFraction(paramset.prescribed_precip_frac)
    elseif precip_fraction_model_name == "cloud_cover"
        return DiagnosticPrecipFraction(paramset.precip_fraction_limiter)
    else
        error(
            "Something went wrong. Invalid `precip_fraction` model: `$precip_fraction_model_name`",
        )
    end
end

abstract type AbstractQuadratureType end
struct LogNormalQuad <: AbstractQuadratureType end
struct GaussianQuad <: AbstractQuadratureType end

function QuadratureType(s::String)
    if s == "log-normal"
        LogNormalQuad()
    elseif s == "gaussian"
        GaussianQuad()
    end
end

abstract type AbstractEnvThermo end
struct SGSMean <: AbstractEnvThermo end
struct SGSQuadrature{N, QT, A, W} <: AbstractEnvThermo
    quadrature_type::QT
    a::A
    w::W
end

function EnvThermo(paramset::NamedTuple)
    if paramset.sgs == "mean"
        SGSMean()
    elseif paramset.sgs == "quadrature"
        SGSQuadrature(FT, paramset)
    else
        error("Something went wrong. Invalid environmental sgs type '$(typeof(paramset.sgs))'")
    end
end

function SGSQuadrature(::Type{FT}, paramset) where {FT}
    N = paramset.quadrature_order
    quadrature_type = paramset.quadrature_type
    # TODO: double check this python-> julia translation
    # a, w = np.polynomial.hermite.hermgauss(N)
    a, w = FastGaussQuadrature.gausshermite(N)
    a, w = SA.SVector{N, FT}(a), SA.SVector{N, FT}(w)
    QT = typeof(quadrature_type)
    return SGSQuadrature{N, QT, typeof(a), typeof(w)}(quadrature_type, a, w)
end

quadrature_order(::SGSQuadrature{N}) where {N} = N
quad_type(::SGSQuadrature{N}) where {N} = N

abstract type AbstractCovarianceModel end
struct PrognosticThermoCovariances <: AbstractCovarianceModel end
struct DiagnosticThermoCovariances{FT} <: AbstractCovarianceModel
    covar_lim::FT
end

function CovarianceModel(paramset::NamedTuple)
    thermo_covariance_model_name = paramset.thermo_covariance_model
    if thermo_covariance_model_name == "prognostic"
        return PrognosticThermoCovariances()
    elseif thermo_covariance_model_name == "diagnostic"
        covar_lim = paramset.diagnostic_covar_limiter
        return DiagnosticThermoCovariances(covar_lim)
    else
        error(
            "Something went wrong. Invalid thermo_covariance model: '$thermo_covariance_model_name'",
        )
    end
end

"""
    PrecipFormation

Storage for tendencies due to precipitation formation

$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct PrecipFormation{FT}
    θ_liq_ice_tendency::FT
    e_tot_tendency::FT
    qt_tendency::FT
    ql_tendency::FT
    qi_tendency::FT
    qr_tendency::FT
    qs_tendency::FT
end

"""
    NoneqMoistureSources

Storage for tendencies due to nonequilibrium moisture formation

$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct NoneqMoistureSources{FT}
    ql_tendency::FT
    qi_tendency::FT
end

"""
    EntrDetr

$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct EntrDetr{FT}
    "Fractional dynamical entrainment [1/m]"
    ε_dyn::FT
    "Fractional dynamical detrainment [1/m]"
    δ_dyn::FT
    "Turbulent entrainment"
    ε_turb::FT
end

Base.@kwdef struct εδModelParams{FT}
    w_min::FT # minimum updraft velocity to avoid zero division in b/w²
    c_ε::FT # factor multiplier for dry term in entrainment/detrainment
    μ_0::FT # dimensional scale logistic function in the dry term in entrainment/detrainment
    β::FT # sorting power for ad-hoc moisture detrainment function
    χ::FT # fraction of updraft air for buoyancy mixing in entrainment/detrainment (0≤χ≤1)
    c_λ::FT # scaling factor for TKE in entrainment scale calculations
    γ_lim::FT
    β_lim::FT
    c_γ::FT # scaling factor for turbulent entrainment rate
    c_δ::FT # factor multiplier for moist term in entrainment/detrainment
end

abstract type AbstractEntrDetrModel end
struct ConstantEntrDetrModel <: AbstractEntrDetrModel end
Base.@kwdef struct MDEntr{P} <: AbstractEntrDetrModel
    params::P
end  # existing model (moisture deficit closure)
εδ_params(m::AbstractEntrDetrModel) = m.params

abstract type EntrDimScale end
struct BuoyVelEntrDimScale <: EntrDimScale end
struct InvZEntrDimScale <: EntrDimScale end
struct InvMeterEntrDimScale <: EntrDimScale end

"""
    GradBuoy

Environmental buoyancy gradients.

$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct GradBuoy{FT}
    "environmental vertical buoyancy gradient"
    ∂b∂z::FT
    "vertical buoyancy gradient in the unsaturated part of the environment"
    ∂b∂z_unsat::FT
    "vertical buoyancy gradient in the saturated part of the environment"
    ∂b∂z_sat::FT
end

abstract type AbstractEnvBuoyGradClosure end
struct BuoyGradMean <: AbstractEnvBuoyGradClosure end
struct BuoyGradQuadratures <: AbstractEnvBuoyGradClosure end

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
    ::EBG;
    t_sat::FT,
    bg_kwargs...,
) where {FT <: Real, EBG <: AbstractEnvBuoyGradClosure}
    return EnvBuoyGrad{FT, EBG}(; t_sat, bg_kwargs...)
end

Base.@kwdef struct MixingLengthParams{FT}
    ω_pr::FT # cospectral budget factor for turbulent Prandtl number
    c_m::FT # tke diffusivity coefficient
    c_d::FT # tke dissipation coefficient
    c_b::FT # static stability coefficient
    κ_star²::FT # Ratio of TKE to squared friction velocity in surface layer
    Pr_n::FT # turbulent Prandtl number in neutral conditions
    Ri_c::FT # critical Richardson number
    smin_ub::FT # lower limit for smin function
    smin_rm::FT # upper ratio limit for smin function
    l_max::FT
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

Base.@kwdef struct PressureModelParams{FT}
    α_b::FT # factor multiplier for pressure buoyancy terms (effective buoyancy is (1-α_b))
    α_a::FT # factor multiplier for pressure advection
    α_d::FT # factor multiplier for pressure drag
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
get_ρe_tot_flux(surf, thermo_params, ts_in) = shf(surf) + lhf(surf)
get_ρq_tot_flux(surf, thermo_params, ts_in) =
    lhf(surf) / TD.latent_heat_vapor(thermo_params, ts_in)
get_ρu_flux(surf) = surf.ρτxz
get_ρv_flux(surf) = surf.ρτyz
obukhov_length(surf) = surf.L_MO


struct EDMFModel{N_up, FT, MM, TCM, PM, PFM, ENT, EBGC, MLP, PMP, EC}
    surface_area::FT
    max_area::FT
    minimum_area::FT
    moisture_model::MM
    thermo_covariance_model::TCM
    precip_model::PM
    precip_fraction_model::PFM
    en_thermo::ENT
    bg_closure::EBGC
    mixing_length_params::MLP
    pressure_model_params::PMP
    entr_closure::EC
    H_up_min::FT # minimum updraft top to avoid zero division in pressure drag and turb-entr
    zero_uv_fluxes::Bool
end
function EDMFModel(
    ::Type{FT},
    turbconv_params,
    moisture_model,
    precip_model,
    parsed_args,
    config_params
) where {FT}

    tc_case = parsed_args["turbconv_case"]
    zero_uv_fluxes = any(tcc -> tcc == tc_case, ["TRMM_LBA", "ARM_SGP"])
    # Set the number of updrafts (1)
    n_updrafts = turbconv_params.updraft_number

    surface_area = turbconv_params.surface_area
    max_area = turbconv_params.max_area
    minimum_area = turbconv_params.min_area

    precip_fraction_model = PrecipFractionModel(config_params)
    thermo_covariance_model = CovarianceModel(config_params)

    # Create the environment variable class (major diagnostic and prognostic variables)
    en_thermo = EnvThermo(config_params)
    # Create the class for environment thermodynamics and buoyancy gradient computation
    bg_closure = 
    if en_thermo isa SGSMean
        BuoyGradMean()
    elseif en_thermo isa SGSQuadrature
        BuoyGradQuadratures()
    else
        error(
            "Something went wrong. Invalid environmental buoyancy gradient closure type '$(typeof(turbconv_params.sgs))'",
        )
    end
    if !(moisture_model isa EquilMoistModel)
        @warn string(
            "SGS model only supports equilibrium moisture choice.",
            "EDMF will carry zeros, but avoid saturation adjustment calls.",
        )
    end

    entr_closure_name = parsed_args["edmf_entr_closure"]
    if entr_closure_name == "MoistureDeficit"
        w_min = turbconv_params.min_upd_velocity
        c_ε = turbconv_params.entrainment_factor
        μ_0 = turbconv_params.entrainment_scale
        β = turbconv_params.sorting_power
        χ = turbconv_params.updraft_mixing_frac
        c_λ = turbconv_params.entrainment_smin_tke_coeff
        γ_lim = turbconv_params.area_limiter_scale
        β_lim = turbconv_params.area_limiter_power
        c_γ = turbconv_params.turbulent_entrainment_factor
        c_δ = turbconv_params.detrainment_factor

        εδ_params = εδModelParams{FT}(;
            w_min,
            c_ε,
            μ_0,
            β,
            χ,
            c_λ,
            γ_lim,
            β_lim,
            c_γ,
            c_δ,
        )

        entr_closure = MDEntr(; params = εδ_params)
    elseif entr_closure_name == "Constant"
        entr_closure = ConstantEntrDetrModel()
    else
        error(
            "Something went wrong. Invalid entrainment closure type '$entr_closure_name'",
        )
    end
    # entr closure
    # entr_type = parse_namelist(
    #     namelist,
    #     "turbulence",
    #     "EDMF_PrognosticTKE",
    #     "entrainment";
    #     default = "moisture_deficit",
    #     valid_options = ["moisture_deficit"],
    # )

    # minimum updraft top to avoid zero division in pressure drag and turb-entr
    H_up_min = turbconv_params.min_updraft_top

    pressure_model_params = PressureModelParams{FT}(;
        α_b = turbconv_params.pressure_normalmode_buoy_coeff1,
        α_a = turbconv_params.pressure_normalmode_adv_coeff,
        α_d = turbconv_params.pressure_normalmode_drag_coeff,
    )

    mixing_length_params = MixingLengthParams{FT}(;
        ω_pr = turbconv_params.Prandtl_number_scale,
        c_m = turbconv_params.tke_ed_coeff,
        c_d = turbconv_params.tke_diss_coeff,
        c_b = turbconv_params.static_stab_coeff, # this is here due to a value error in CliMAParmameters.j,
        κ_star² = turbconv_params.tke_surf_scale,
        Pr_n = turbconv_params.Prandtl_number_0,
        Ri_c = turbconv_params.Ri_crit,
        smin_ub = turbconv_params.smin_ub,
        smin_rm = turbconv_params.smin_rm,
        l_max = turbconv_params.l_max,
    )

    EC = typeof(entr_closure)
    MM = typeof(moisture_model)
    TCM = typeof(thermo_covariance_model)
    PM = typeof(precip_model)
    PFM = typeof(precip_fraction_model)
    EBGC = typeof(bg_closure)
    ENT = typeof(en_thermo)
    MLP = typeof(mixing_length_params)
    PMP = typeof(pressure_model_params)
    return EDMFModel{n_updrafts, FT, MM, TCM, PM, PFM, ENT, EBGC, MLP, PMP, EC}(
        surface_area,
        max_area,
        minimum_area,
        moisture_model,
        thermo_covariance_model,
        precip_model,
        precip_fraction_model,
        en_thermo,
        bg_closure,
        mixing_length_params,
        pressure_model_params,
        entr_closure,
        H_up_min,
        zero_uv_fluxes,
    )
end

parameter_set(obj) = obj.param_set
n_updrafts(::EDMFModel{N_up}) where {N_up} = N_up
Base.eltype(::EDMFModel{N_up, FT}) where {N_up, FT} = FT
pressure_model_params(m::EDMFModel) = m.pressure_model_params
mixing_length_params(m::EDMFModel) = m.mixing_length_params

Base.broadcastable(edmf::EDMFModel) = Ref(edmf)

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


struct State{P, A, T, CACHE, C}
    prog::P
    aux::A
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

    return State(prog_column, aux_column, tends_column, p, colidx)
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

    return State(prog_column, aux_column, tends_column, p, colidx)
end
