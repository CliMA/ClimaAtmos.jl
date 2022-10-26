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
    c_div::FT
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
Base.@kwdef struct MDEntr{P} <: AbstractEntrDetrModel
    params::P
end  # existing model

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

abstract type AbstractCovarianceModel end
struct PrognosticThermoCovariances <: AbstractCovarianceModel end
struct DiagnosticThermoCovariances{FT} <: AbstractCovarianceModel
    covar_lim::FT
end

abstract type AbstractPrecipitationModel end
struct NoPrecipitation <: AbstractPrecipitationModel end
struct Clima0M <: AbstractPrecipitationModel end
struct Clima1M <: AbstractPrecipitationModel end

abstract type AbstractPrecipFractionModel end
struct PrescribedPrecipFraction{FT} <: AbstractPrecipFractionModel
    prescribed_precip_frac_value::FT
end
struct DiagnosticPrecipFraction{FT} <: AbstractPrecipFractionModel
    precip_fraction_limiter::FT
end

abstract type AbstractQuadratureType end
struct LogNormalQuad <: AbstractQuadratureType end
struct GaussianQuad <: AbstractQuadratureType end

abstract type AbstractEnvThermo end
struct SGSMean <: AbstractEnvThermo end
struct SGSQuadrature{N, QT, A, W} <: AbstractEnvThermo
    quadrature_type::QT
    a::A
    w::W
    function SGSQuadrature(::Type{FT}, namelist) where {FT}
        N = parse_namelist(
            namelist,
            "thermodynamics",
            "quadrature_order";
            default = 3,
        )
        quadrature_name = parse_namelist(
            namelist,
            "thermodynamics",
            "quadrature_type";
            default = "log-normal",
        )
        quadrature_type = if quadrature_name == "log-normal"
            LogNormalQuad()
        elseif quadrature_name == "gaussian"
            GaussianQuad()
        else
            error("Invalid thermodynamics quadrature $(quadrature_name)")
        end
        # TODO: double check this python-> julia translation
        # a, w = np.polynomial.hermite.hermgauss(N)
        a, w = FastGaussQuadrature.gausshermite(N)
        a, w = SA.SVector{N, FT}(a), SA.SVector{N, FT}(w)
        QT = typeof(quadrature_type)
        return new{N, QT, typeof(a), typeof(w)}(quadrature_type, a, w)
    end
end
quadrature_order(::SGSQuadrature{N}) where {N} = N
quad_type(::SGSQuadrature{N}) where {N} = N

abstract type FrictionVelocityType end
struct FixedFrictionVelocity <: FrictionVelocityType end
struct VariableFrictionVelocity <: FrictionVelocityType end

abstract type AbstractSurfaceParameters{FT <: Real} end

const FloatOrFunc{FT} = Union{FT, Function, Dierckx.Spline1D}

Base.@kwdef struct FixedSurfaceFlux{
    FT,
    FVT <: FrictionVelocityType,
    TS,
    QS,
    SHF,
    LHF,
} <: AbstractSurfaceParameters{FT}
    zrough::FT = FT(0)
    Tsurface::TS = FT(0)
    qsurface::QS = FT(0)
    shf::SHF = FT(0)
    lhf::LHF = FT(0)
    cq::FT = FT(0)
    Ri_bulk_crit::FT = FT(0)
    ustar::FT = FT(0)
    zero_uv_fluxes::Bool = false
end

function FixedSurfaceFlux(
    ::Type{FT},
    ::Type{FVT};
    Tsurface::FloatOrFunc{FT},
    qsurface::FloatOrFunc{FT},
    shf::FloatOrFunc{FT},
    lhf::FloatOrFunc{FT},
    kwargs...,
) where {FT, FVT}
    TS = typeof(Tsurface)
    QS = typeof(qsurface)
    SHF = typeof(shf)
    LHF = typeof(lhf)
    return FixedSurfaceFlux{FT, FVT, TS, QS, SHF, LHF}(;
        Tsurface,
        qsurface,
        shf,
        lhf,
        kwargs...,
    )
end

Base.@kwdef struct FixedSurfaceCoeffs{FT, TS, QS, CH, CM} <:
                   AbstractSurfaceParameters{FT}
    zrough::FT = FT(0)
    Tsurface::TS = FT(0)
    qsurface::QS = FT(0)
    ch::CH = FT(0)
    cm::CM = FT(0)
    Ri_bulk_crit::FT = FT(0)
end

function FixedSurfaceCoeffs(
    ::Type{FT};
    Tsurface::FloatOrFunc{FT},
    qsurface::FloatOrFunc{FT},
    ch::FloatOrFunc{FT},
    cm::FloatOrFunc{FT},
    kwargs...,
) where {FT}
    TS = typeof(Tsurface)
    QS = typeof(qsurface)
    CH = typeof(ch)
    CM = typeof(cm)
    return FixedSurfaceCoeffs{FT, TS, QS, CH, CM}(;
        Tsurface,
        qsurface,
        ch,
        cm,
        kwargs...,
    )
end

Base.@kwdef struct MoninObukhovSurface{FT, TS, QS} <:
                   AbstractSurfaceParameters{FT}
    zrough::FT = FT(0)
    Tsurface::TS = FT(0)
    qsurface::QS = FT(0)
    Ri_bulk_crit::FT = FT(0)
end

function MoninObukhovSurface(
    ::Type{FT};
    Tsurface::FloatOrFunc{FT},
    qsurface::FloatOrFunc{FT},
    kwargs...,
) where {FT}
    TS = typeof(Tsurface)
    QS = typeof(qsurface)
    return MoninObukhovSurface{FT, TS, QS}(; Tsurface, qsurface, kwargs...)
end

float_or_func(s::Function, t::Real) = s(t)
float_or_func(s::Dierckx.Spline1D, t::Real) = s(t)
float_or_func(s::Real, t::Real) = s

surface_temperature(s::AbstractSurfaceParameters, t::Real = 0) =
    float_or_func(s.Tsurface, t)
surface_q_tot(s::AbstractSurfaceParameters, t::Real = 0) =
    float_or_func(s.qsurface, t)
sensible_heat_flux(s::AbstractSurfaceParameters, t::Real = 0) =
    float_or_func(s.shf, t)
latent_heat_flux(s::AbstractSurfaceParameters, t::Real = 0) =
    float_or_func(s.lhf, t)

fixed_ustar(::FixedSurfaceFlux{FT, FixedFrictionVelocity}) where {FT} = true
fixed_ustar(::FixedSurfaceFlux{FT, VariableFrictionVelocity}) where {FT} = false

Base.@kwdef struct SurfaceBase{FT}
    shf::FT = 0
    lhf::FT = 0
    cm::FT = 0
    ch::FT = 0
    bflux::FT = 0
    ustar::FT = 0
    ρq_tot_flux::FT = 0
    ρq_liq_flux::FT = 0
    ρq_ice_flux::FT = 0
    ρe_tot_flux::FT = 0
    ρu_flux::FT = 0
    ρv_flux::FT = 0
    obukhov_length::FT = 0
    wstar::FT = 0
end

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
end
function EDMFModel(
    ::Type{FT},
    namelist,
    moisture_model,
    precip_model,
    parsed_args,
) where {FT}

    # Set the number of updrafts (1)
    n_updrafts = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "updraft_number";
        default = 1,
    )

    pressure_func_drag_str = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "pressure_closure_drag";
        default = "normalmode",
        valid_options = ["normalmode", "normalmode_signdf"],
    )

    surface_area = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "surface_area";
        default = 0.1,
    )
    max_area = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "max_area";
        default = 0.9,
    )
    minimum_area = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "min_area";
        default = 1e-5,
    )

    thermo_covariance_model_name = parse_namelist(
        namelist,
        "thermodynamics",
        "thermo_covariance_model";
        default = "prognostic",
    )

    thermo_covariance_model = if thermo_covariance_model_name == "prognostic"
        PrognosticThermoCovariances()
    elseif thermo_covariance_model_name == "diagnostic"
        covar_lim = parse_namelist(
            namelist,
            "thermodynamics",
            "diagnostic_covar_limiter",
        )
        DiagnosticThermoCovariances(covar_lim)
    else
        error(
            "Something went wrong. Invalid thermo_covariance model: '$thermo_covariance_model_name'",
        )
    end

    precip_fraction_model_name = parse_namelist(
        namelist,
        "microphysics",
        "precip_fraction_model";
        default = "prescribed",
    )

    precip_fraction_model = if precip_fraction_model_name == "prescribed"
        prescribed_precip_frac_value = parse_namelist(
            namelist,
            "microphysics",
            "prescribed_precip_frac_value";
            default = 1.0,
        )
        PrescribedPrecipFraction(prescribed_precip_frac_value)
    elseif precip_fraction_model_name == "cloud_cover"
        precip_fraction_limiter = parse_namelist(
            namelist,
            "microphysics",
            "precip_fraction_limiter";
            default = 0.3,
        )
        DiagnosticPrecipFraction(precip_fraction_limiter)
    else
        error(
            "Something went wrong. Invalid `precip_fraction` model: `$precip_fraction_model_name`",
        )
    end

    # Create the environment variable class (major diagnostic and prognostic variables)

    # Create the class for environment thermodynamics and buoyancy gradient computation
    en_sgs_name = parse_namelist(
        namelist,
        "thermodynamics",
        "sgs";
        default = "mean",
        valid_options = ["mean", "quadrature"],
    )
    en_thermo = if en_sgs_name == "mean"
        SGSMean()
    elseif en_sgs_name == "quadrature"
        SGSQuadrature(FT, namelist)
    else
        error("Something went wrong. Invalid environmental sgs type '$en_sgs_name'")
    end
    bg_closure = if en_sgs_name == "mean"
        BuoyGradMean()
    elseif en_sgs_name == "quadrature"
        BuoyGradQuadratures()
    else
        error(
            "Something went wrong. Invalid environmental buoyancy gradient closure type '$en_sgs_name'",
        )
    end
    if moisture_model isa NonEquilMoistModel && en_thermo == "quadrature"
        error(
            "SGS quadratures are not yet implemented for non-equilibrium moisture. Please use the option: mean.",
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

    c_div = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "entrainment_massflux_div_factor";
        default = 0.0,
    )
    w_min = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "min_upd_velocity",
    )
    c_ε = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "entrainment_factor",
    )
    μ_0 = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "entrainment_scale",
    )
    β = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "sorting_power",
    )
    χ = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "updraft_mixing_frac",
    )
    c_λ = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "entrainment_smin_tke_coeff",
    )
    γ_lim = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "area_limiter_scale",
    )
    β_lim = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "area_limiter_power",
    )
    c_γ = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "turbulent_entrainment_factor",
    )
    c_δ = parse_namelist(
        namelist,
        "turbulence",
        "EDMF_PrognosticTKE",
        "detrainment_factor",
    )

    εδ_params = εδModelParams{FT}(;
        c_div,
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

    # minimum updraft top to avoid zero division in pressure drag and turb-entr
    H_up_min = FT(
        parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "min_updraft_top",
        ),
    )

    pressure_model_params = PressureModelParams{FT}(;
        α_b = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "pressure_normalmode_buoy_coeff1",
        ),
        α_a = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "pressure_normalmode_adv_coeff",
        ),
        α_d = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "pressure_normalmode_drag_coeff",
        ),
    )

    mixing_length_params = MixingLengthParams{FT}(;
        ω_pr = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "Prandtl_number_scale",
        ),
        c_m = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "tke_ed_coeff",
        ),
        c_d = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "tke_diss_coeff",
        ),
        c_b = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "static_stab_coeff";
            default = 0.4,
        ), # this is here due to a value error in CliMAParmameters.j,
        κ_star² = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "tke_surf_scale",
        ),
        Pr_n = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "Prandtl_number_0",
        ),
        Ri_c = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "Ri_crit",
        ),
        smin_ub = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "smin_ub",
        ),
        smin_rm = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "smin_rm",
        ),
        l_max = parse_namelist(
            namelist,
            "turbulence",
            "EDMF_PrognosticTKE",
            "l_max";
            default = 1.0e6,
        ),
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
