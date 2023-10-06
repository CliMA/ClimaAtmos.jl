import FastGaussQuadrature
import StaticArrays as SA

abstract type AbstractMoistureModel end
struct DryModel <: AbstractMoistureModel end
struct EquilMoistModel <: AbstractMoistureModel end
struct NonEquilMoistModel <: AbstractMoistureModel end

abstract type AbstractEnergyFormulation end
struct PotentialTemperature <: AbstractEnergyFormulation end
struct TotalEnergy <: AbstractEnergyFormulation end

abstract type AbstractPrecipitationModel end
struct NoPrecipitation <: AbstractPrecipitationModel end
struct Microphysics0Moment <: AbstractPrecipitationModel end
struct Microphysics1Moment <: AbstractPrecipitationModel end

abstract type AbstractModelConfig end
struct SingleColumnModel <: AbstractModelConfig end
struct SphericalModel <: AbstractModelConfig end
struct BoxModel <: AbstractModelConfig end
struct PlaneModel <: AbstractModelConfig end

abstract type AbstractSST end
struct ZonallySymmetricSST <: AbstractSST end
struct ZonallyAsymmetricSST <: AbstractSST end

abstract type AbstractSurfaceTemperature end
struct PrescribedSurfaceTemperature <: AbstractSurfaceTemperature end
struct PrognosticSurfaceTemperature <: AbstractSurfaceTemperature end

abstract type AbstractHyperdiffusion end
Base.@kwdef struct ClimaHyperdiffusion{FT} <: AbstractHyperdiffusion
    κ₄::FT
    divergence_damping_factor::FT
end

abstract type AbstractVerticalDiffusion end
Base.@kwdef struct VerticalDiffusion{DM, FT} <: AbstractVerticalDiffusion
    C_E::FT
end
diffuse_momentum(::VerticalDiffusion{DM}) where {DM} = DM
diffuse_momentum(::Nothing) = false

abstract type AbstractSponge end
Base.@kwdef struct ViscousSponge{FT} <: AbstractSponge
    zd::FT
    κ₂::FT
end

Base.@kwdef struct RayleighSponge{FT} <: AbstractSponge
    zd::FT
    α_uₕ::FT
    α_w::FT
end

abstract type AbstractGravityWave end
Base.@kwdef struct NonOrographyGravityWave{FT} <: AbstractGravityWave
    source_pressure::FT = 31500
    damp_pressure::FT = 85
    source_height::FT = 15000
    Bw::FT = 1.0
    Bn::FT = 1.0
    dc::FT = 0.6
    cmax::FT = 99.6
    c0::FT = 0
    nk::FT = 1
    cw::FT = 40.0
    cw_tropics::FT = 40.0
    cn::FT = 40.0
    Bt_0::FT = 0.0003
    Bt_n::FT = 0.0003
    Bt_s::FT = 0.0003
    Bt_eq::FT = 0.0003
    ϕ0_n::FT = 30
    ϕ0_s::FT = -30
    dϕ_n::FT = 5
    dϕ_s::FT = -5
end

Base.@kwdef struct OrographicGravityWave{FT, S} <: AbstractGravityWave
    γ::FT = 0.4
    ϵ::FT = 0.0
    β::FT = 0.5
    h_frac::FT = 0.1
    ρscale::FT = 1.2
    L0::FT = 80e3
    a0::FT = 0.9
    a1::FT = 3.0
    Fr_crit::FT = 0.7
    topo_info::S = "gfdl_restart"
end

abstract type AbstractForcing end
struct HeldSuarezForcing <: AbstractForcing end
struct Subsidence{T} <: AbstractForcing
    prof::T
end
# TODO: is this a forcing?
struct LargeScaleAdvection{PT, PQ}
    prof_dTdt::PT # Set large-scale cooling
    prof_dqtdt::PQ # Set large-scale drying
end

struct EDMFCoriolis{U, V, FT}
    prof_ug::U
    prof_vg::V
    coriolis_param::FT
end

abstract type AbstractEnvBuoyGradClosure end
struct BuoyGradMean <: AbstractEnvBuoyGradClosure end

Base.broadcastable(x::BuoyGradMean) = tuple(x)

"""
    EnvBuoyGrad

Variables used in the environmental buoyancy gradient computation.
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

abstract type AbstractEDMF end

struct EDMFX{N, TKE, FT} <: AbstractEDMF
    a_half::FT # WARNING: this should never be used outside of divide_by_ρa
end
EDMFX{N, TKE}(a_half::FT) where {N, TKE, FT} = EDMFX{N, TKE, FT}(a_half)

struct DiagnosticEDMFX{N, TKE, FT} <: AbstractEDMF
    a_int::FT # area fraction of the first interior cell above the surface
    a_half::FT # WARNING: this should never be used outside of divide_by_ρa
end
DiagnosticEDMFX{N, TKE}(a_int::FT, a_half::FT) where {N, TKE, FT} =
    DiagnosticEDMFX{N, TKE, FT}(a_int, a_half)

n_mass_flux_subdomains(::EDMFX{N}) where {N} = N
n_mass_flux_subdomains(::DiagnosticEDMFX{N}) where {N} = N
n_mass_flux_subdomains(::Any) = 0

n_prognostic_mass_flux_subdomains(::EDMFX{N}) where {N} = N
n_prognostic_mass_flux_subdomains(::Any) = 0

use_prognostic_tke(::EDMFX{N, TKE}) where {N, TKE} = TKE
use_prognostic_tke(::DiagnosticEDMFX{N, TKE}) where {N, TKE} = TKE
use_prognostic_tke(::Any) = false

abstract type AbstractEntrainmentModel end
struct NoEntrainment <: AbstractEntrainmentModel end
struct PiGroupsEntrainment <: AbstractEntrainmentModel end
struct ConstantCoefficientEntrainment <: AbstractEntrainmentModel end
struct ConstantCoefficientHarmonicsEntrainment <: AbstractEntrainmentModel end
struct ConstantTimescaleEntrainment <: AbstractEntrainmentModel end

abstract type AbstractDetrainmentModel end

struct NoDetrainment <: AbstractDetrainmentModel end
struct PiGroupsDetrainment <: AbstractDetrainmentModel end
struct ConstantCoefficientDetrainment <: AbstractDetrainmentModel end
struct ConstantCoefficientHarmonicsDetrainment <: AbstractDetrainmentModel end

abstract type AbstractQuadratureType end
struct LogNormalQuad <: AbstractQuadratureType end
struct GaussianQuad <: AbstractQuadratureType end

abstract type AbstractEnvThermo end
struct SGSMean <: AbstractEnvThermo end
struct SGSQuadrature{N, QT, A, W} <: AbstractEnvThermo
    quadrature_type::QT
    a::A
    w::W
    function SGSQuadrature(
        ::Type{FT};
        quadrature_name = "gaussian",
        quadrature_order = 3,
    ) where {FT}
        quadrature_type = if quadrature_name == "log-normal"
            LogNormalQuad()
        elseif quadrature_name == "gaussian"
            GaussianQuad()
        else
            error("Invalid thermodynamics quadrature $(quadrature_name)")
        end
        N = quadrature_order
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

abstract type AbstractSurfaceThermoState end
struct GCMSurfaceThermoState <: AbstractSurfaceThermoState end

# Define broadcasting for types
Base.broadcastable(x::AbstractSurfaceThermoState) = tuple(x)
Base.broadcastable(x::AbstractMoistureModel) = tuple(x)
Base.broadcastable(x::AbstractEnergyFormulation) = tuple(x)
Base.broadcastable(x::AbstractPrecipitationModel) = tuple(x)
Base.broadcastable(x::AbstractForcing) = tuple(x)
Base.broadcastable(x::EDMFX) = tuple(x)
Base.broadcastable(x::DiagnosticEDMFX) = tuple(x)
Base.broadcastable(x::AbstractEntrainmentModel) = tuple(x)
Base.broadcastable(x::AbstractDetrainmentModel) = tuple(x)
Base.broadcastable(x::AbstractEnvThermo) = tuple(x)

Base.@kwdef struct RadiationDYCOMS_RF01{FT}
    "Large-scale divergence"
    divergence::FT = 3.75e-6
    alpha_z::FT = 1.0
    kappa::FT = 85.0
    F0::FT = 70.0
    F1::FT = 22.0
end
import AtmosphericProfilesLibrary as APL

struct RadiationTRMM_LBA{R}
    rad_profile::R
    function RadiationTRMM_LBA(::Type{FT}) where {FT}
        rad_profile = APL.TRMM_LBA_radiation(FT)
        return new{typeof(rad_profile)}(rad_profile)
    end
end

# TODO: remove AbstractPerformanceMode and all subtypes
# This is temporarily needed to investigate performance of
# our handling of tracers.
abstract type AbstractPerformanceMode end
struct PerfExperimental <: AbstractPerformanceMode end
struct PerfStandard <: AbstractPerformanceMode end
struct TestDycoreConsistency end

Base.broadcastable(x::AbstractPerformanceMode) = tuple(x)

Base.@kwdef struct AtmosModel{
    MC,
    PEM,
    MM,
    EF,
    PM,
    F,
    S,
    RM,
    LA,
    EC,
    AT,
    EEM,
    EDM,
    ESMF,
    ESDF,
    ENP,
    TCM,
    NOGW,
    OGW,
    HD,
    VD,
    VS,
    RS,
    ST,
    SM,
}
    model_config::MC = nothing
    perf_mode::PEM = nothing
    moisture_model::MM = nothing
    energy_form::EF = nothing
    precip_model::PM = nothing
    forcing_type::F = nothing
    subsidence::S = nothing
    radiation_mode::RM = nothing
    ls_adv::LA = nothing
    edmf_coriolis::EC = nothing
    advection_test::AT = nothing
    edmfx_entr_model::EEM = nothing
    edmfx_detr_model::EDM = nothing
    edmfx_sgs_mass_flux::ESMF = nothing
    edmfx_sgs_diffusive_flux::ESDF = nothing
    edmfx_nh_pressure::ENP = nothing
    turbconv_model::TCM = nothing
    non_orographic_gravity_wave::NOGW = nothing
    orographic_gravity_wave::OGW = nothing
    hyperdiff::HD = nothing
    vert_diff::VD = nothing
    viscous_sponge::VS = nothing
    rayleigh_sponge::RS = nothing
    sfc_temperature::ST = nothing
    surface_model::SM = nothing
end

Base.broadcastable(x::AtmosModel) = tuple(x)

function Base.summary(io::IO, atmos::AtmosModel)
    pns = string.(propertynames(atmos))
    buf = maximum(length.(pns))
    keys = propertynames(atmos)
    vals = repeat.(" ", map(s -> buf - length(s) + 2, pns))
    bufs = (; zip(keys, vals)...)
    print(io, '\n')
    for pn in propertynames(atmos)
        prop = getproperty(atmos, pn)
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

abstract type AbstractCallbackFrequency end
struct EveryNSteps <: AbstractCallbackFrequency
    n::Int
end
struct EveryΔt{FT} <: AbstractCallbackFrequency
    Δt::FT
end
struct AtmosCallback{F, CBF <: AbstractCallbackFrequency, VI <: Vector{Int}}
    f!::F
    cbf::CBF
    n_measured_calls::VI
end
callback_frequency(cb::AtmosCallback) = cb.cbf
prescribed_every_n_steps(x::EveryNSteps) = x.n
prescribed_every_n_steps(cb::AtmosCallback) = prescribed_every_n_steps(cb.cbf)

prescribed_every_Δt_steps(x::EveryΔt) = x.Δt
prescribed_every_Δt_steps(cb::AtmosCallback) = prescribed_every_Δt_steps(cb.cbf)

# TODO: improve accuracy
n_expected_calls(cbf::EveryΔt, dt, tspan) = (tspan[2] - tspan[1]) / cbf.Δt
n_expected_calls(cbf::EveryNSteps, dt, tspan) =
    ((tspan[2] - tspan[1]) / dt) / cbf.n
n_expected_calls(cb::AtmosCallback, dt, tspan) =
    n_expected_calls(cb.cbf, dt, tspan)

AtmosCallback(f!, cbf) = AtmosCallback(f!, cbf, Int[0])
function (cb::AtmosCallback)(integrator)
    cb.f!(integrator)
    cb.n_measured_calls[] += 1
    return nothing
end
n_measured_calls(cb::AtmosCallback) = cb.n_measured_calls[]

struct AtmosConfig{FT, TD, PA, C}
    toml_dict::TD
    parsed_args::PA
    comms_ctx::C
end

Base.eltype(::AtmosConfig{FT}) where {FT} = FT

"""
    AtmosConfig(config_file::String)

Helper function for the AtmosConfig constructor. Reads a YAML file into a Dict
and passes it to the AtmosConfig constructor.
"""
function AtmosConfig(config_file::String; comms_ctx = nothing)
    config = YAML.load_file(config_file)
    return AtmosConfig(config; comms_ctx)
end

"""
    AtmosConfig(; comms_ctx = nothing)
Helper function for the AtmosConfig constructor.
Reads the `config_file` from the command line into a Dict
and passes it to the AtmosConfig constructor.
"""
function AtmosConfig(; comms_ctx = nothing)
    parsed_args = parse_commandline(argparse_settings())
    return AtmosConfig(parsed_args["config_file"]; comms_ctx)
end

AtmosConfig(::Nothing; comms_ctx = nothing) = AtmosConfig(Dict(); comms_ctx)

"""
    AtmosConfig(config::Dict; comms_ctx = nothing)
Constructs the AtmosConfig from the Dict passed in. This Dict overrides all of 
the default configurations set in `default_config_dict()`.
"""
function AtmosConfig(config::Dict; comms_ctx = nothing)
    config = override_default_config(config)
    FT = config["FLOAT_TYPE"] == "Float64" ? Float64 : Float32
    toml_dict = CP.create_toml_dict(
        FT;
        override_file = CP.merge_toml_files(config["toml"]),
    )
    # TODO: is there a better way? We need a better
    #       mechanism on the ClimaCore side.
    if config["trunc_stack_traces"]
        @eval Main begin
            import ClimaCore
            ClimaCore.Fields.truncate_printing_field_types() = true
        end
    end
    comms_ctx = isnothing(comms_ctx) ? get_comms_context(config) : comms_ctx
    device = ClimaComms.device(comms_ctx)
    if device isa ClimaComms.CPUMultiThreaded
        @info "Running ClimaCore in threaded mode, with $(Threads.nthreads()) threads."
    else
        @info "Running ClimaCore in unthreaded mode."
    end

    C = typeof(comms_ctx)
    TD = typeof(toml_dict)
    PA = typeof(config)
    return AtmosConfig{FT, TD, PA, C}(toml_dict, config, comms_ctx)
end
