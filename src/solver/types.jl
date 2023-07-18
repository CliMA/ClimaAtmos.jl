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

# define type for smagorinsky_lily
abstract type AbstractEddyViscosityModel end
Base.@kwdef struct SmagorinskyLily{FT} <: AbstractEddyViscosityModel
    Cs::FT = 0.2
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
    EAT,
    EED,
    ESF,
    ENP,
    TCM,
    SS,
    NOGW,
    OGW,
    HD,
    VD,
    VS,
    RS,
    ST,
    SM,
    SL
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
    edmfx_adv_test::EAT = nothing
    edmfx_entr_detr::EED = nothing
    edmfx_sgs_flux::ESF = nothing
    edmfx_nh_pressure::ENP = nothing
    turbconv_model::TCM = nothing
    surface_scheme::SS = nothing
    non_orographic_gravity_wave::NOGW = nothing
    orographic_gravity_wave::OGW = nothing
    hyperdiff::HD = nothing
    vert_diff::VD = nothing
    viscous_sponge::VS = nothing
    rayleigh_sponge::RS = nothing
    sfc_temperature::ST = nothing
    surface_model::SM = nothing
    smagorinsky_lily::SL = nothing
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

import ClimaCore

function AtmosTargetParsedArgs(
    s = argparse_settings();
    target_job,
    dict = parsed_args_per_job_id(; filter_name = "driver.jl"),
)
    parsed_args_defaults = cli_defaults(s)

    # Start with performance target, and override anything provided in ARGS
    parsed_args_prescribed = parsed_args_from_ARGS(ARGS)

    _target_job = get(parsed_args_prescribed, "target_job", nothing)
    if _target_job ≠ nothing && target_job ≠ nothing
        error("Target job specified multiple times")
    end
    _target_job ≠ nothing && (target_job = _target_job)
    parsed_args_perf_target = isnothing(target_job) ? Dict() : dict[target_job]

    parsed_args = merge(
        parsed_args_defaults,
        parsed_args_perf_target,
        parsed_args_prescribed,
    )
    return parsed_args
end


"""
    AtmosCoveragePerfParsedArgs()

Define an atmos config for performance runs, and allows
options to be overridden in several ways. In short the
precedence for defining `parsed_args` is

    - Highest precedence: args defined in `ARGS`
    - Mid     precedence: args defined in `parsed_args_perf_target` (below)
    - Lowest  precedence: args defined in `cli_defaults(s)`

# Example usage:

Launch with `julia --project=perf/`

```julia
import ClimaAtmos as CA
import Random
Random.seed!(1234)
parsed_args = CA.AtmosCoveragePerfParsedArgs(;moist="dry")
config = CA.AtmosConfig(;parsed_args)
```
"""
function AtmosCoveragePerfParsedArgs(s = argparse_settings())
    @info "Using coverage performance parameters + provided CL arguments."
    parsed_args_defaults = cli_defaults(s)
    dict = parsed_args_per_job_id(; filter_name = "driver.jl")

    # Start with performance target, and override anything provided in ARGS
    parsed_args_prescribed = parsed_args_from_ARGS(ARGS)

    target_job = get(parsed_args_prescribed, "target_job", nothing)
    parsed_args_perf_target = isnothing(target_job) ? Dict() : dict[target_job]

    parsed_args_perf_target["forcing"] = "held_suarez"
    parsed_args_perf_target["vert_diff"] = true
    parsed_args_perf_target["surface_setup"] = "DefaultExchangeCoefficients"
    parsed_args_perf_target["moist"] = "equil"
    parsed_args_perf_target["rad"] = "allskywithclear"
    parsed_args_perf_target["precip_model"] = "0M"
    parsed_args_perf_target["dt"] = "1secs"
    parsed_args_perf_target["t_end"] = "10secs"
    parsed_args_perf_target["dt_save_to_sol"] = Inf
    parsed_args_perf_target["z_elem"] = 25
    parsed_args_perf_target["h_elem"] = 12

    parsed_args = merge(
        parsed_args_defaults,
        parsed_args_perf_target,
        parsed_args_prescribed,
    )
    return parsed_args
end

AtmosCoveragePerfConfig(s = argparse_settings()) =
    AtmosConfig(s; parsed_args = AtmosCoveragePerfParsedArgs(s))

function AtmosConfigArgs(
    s = argparse_settings();
    args = String[],
    parsed_args = parse_commandline(args, s),
    comms_ctx = get_comms_context(parsed_args),
)
    @info "Running ClimaAtmos with default argparse settings + $args"
    return AtmosConfig(s; parsed_args, comms_ctx)
end

function AtmosConfig(
    s = argparse_settings();
    parsed_args = parse_commandline(s),
    comms_ctx = get_comms_context(parsed_args),
)
    FT = parsed_args["FLOAT_TYPE"] == "Float64" ? Float64 : Float32
    toml_dict = CP.create_toml_dict(
        FT;
        override_file = parsed_args["toml"],
        dict_type = "alias",
    )
    toml_dict, parsed_args =
        merge_parsed_args_with_toml(toml_dict, parsed_args, cli_defaults(s))

    # TODO: is there a better way? We need a better
    #       mechanism on the ClimaCore side.
    if parsed_args["trunc_stack_traces"]
        @eval Main begin
            import ClimaCore
            ClimaCore.Fields.truncate_printing_field_types() = true
        end
    end
    device = ClimaComms.device(comms_ctx)
    if device isa ClimaComms.CPUMultiThreaded
        @info "Running ClimaCore in threaded mode, with $(Threads.nthreads()) threads."
    else
        @info "Running ClimaCore in unthreaded mode."
    end

    C = typeof(comms_ctx)
    TD = typeof(toml_dict)
    PA = typeof(parsed_args)
    return AtmosConfig{FT, TD, PA, C}(toml_dict, parsed_args, comms_ctx)
end
Base.eltype(::AtmosConfig{FT}) where {FT} = FT

"""
Merges parsed_args with the toml_dict generated from CLIMAParameters.
Priority for clashes: parsed_args > toml_dict > default_args
Converts `nothing` to empty string, since CLIMAParameters does not support type Nothing.
The dictionary overrides existing toml_dict values if there are clashes.
"""
function merge_parsed_args_with_toml(toml_dict, parsed_args, default_args)
    toml_type(val::AbstractFloat) = "float"
    toml_type(val::Integer) = "integer"
    toml_type(val::Bool) = "bool"
    toml_type(val::String) = "string"
    toml_type(val::Symbol) = "string"
    toml_type(val::Nothing) = "string"
    toml_value(val::Nothing) = ""
    toml_value(val::Symbol) = String(val)
    toml_value(val) = val

    for (key, value) in parsed_args
        if haskey(default_args, key)
            if parsed_args[key] != default_args[key] ||
               !haskey(toml_dict.data, key)
                toml_dict.data[key] = Dict(
                    "type" => toml_type(value),
                    "value" => toml_value(value),
                    "alias" => key,
                )
            end
            parsed_args[key] = if toml_dict.data[key]["value"] == ""
                nothing
            elseif parsed_args[key] isa Symbol
                Symbol(toml_dict.data[key]["value"])
            else
                toml_dict.data[key]["value"]
            end
        end
    end
    return toml_dict, parsed_args
end
