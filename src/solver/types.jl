import FastGaussQuadrature
import StaticArrays as SA
import Thermodynamics as TD

import ClimaUtilities.ClimaArtifacts: @clima_artifact
import LazyArtifacts

abstract type AbstractMoistureModel end
struct DryModel <: AbstractMoistureModel end
struct EquilMoistModel <: AbstractMoistureModel end
struct NonEquilMoistModel <: AbstractMoistureModel end

abstract type AbstractPrecipitationModel end
struct NoPrecipitation <: AbstractPrecipitationModel end
struct Microphysics0Moment <: AbstractPrecipitationModel end
struct Microphysics1Moment <: AbstractPrecipitationModel end

abstract type AbstractCloudModel end
struct GridScaleCloud <: AbstractCloudModel end
struct QuadratureCloud <: AbstractCloudModel end
struct SGSQuadratureCloud <: AbstractCloudModel end

abstract type AbstractModelConfig end
struct SingleColumnModel <: AbstractModelConfig end
struct SphericalModel <: AbstractModelConfig end
struct BoxModel <: AbstractModelConfig end
struct PlaneModel <: AbstractModelConfig end

abstract type AbstractSST end
struct ZonallySymmetricSST <: AbstractSST end
struct ZonallyAsymmetricSST <: AbstractSST end
struct RCEMIPIISST <: AbstractSST end

abstract type AbstractInsolation end
struct IdealizedInsolation <: AbstractInsolation end
struct TimeVaryingInsolation <: AbstractInsolation end
struct RCEMIPIIInsolation <: AbstractInsolation end
struct GCMDrivenInsolation <: AbstractInsolation end

"""
    AbstractOzone

Describe how ozone concentration should be set.
"""
abstract type AbstractOzone end

"""
    IdealizedOzone

Implement a static (not varying in time) idealized ozone profile as described by
`idealized_ozone`.
"""
struct IdealizedOzone <: AbstractOzone end

"""
    PrescribedOzone

Implement a time-varying ozone profile as read from disk.

The CMIP6 forcing dataset is used. For production runs, you should acquire the
high-resolution, multi-year `ozone_concentrations` artifact. If this is not available, a low
resolution, single-year version will be used.

Refer to ClimaArtifacts for more information on how to obtain the artifact.
"""
struct PrescribedOzone <: AbstractOzone end


abstract type AbstractSurfaceTemperature end
struct PrescribedSurfaceTemperature <: AbstractSurfaceTemperature end
Base.@kwdef struct PrognosticSurfaceTemperature{FT} <:
                   AbstractSurfaceTemperature
    # optional slab ocean parameters:
    depth_ocean::FT = 40 # ocean mixed layer depth [m]
    ρ_ocean::FT = 1020 # ocean density [kg / m³]
    cp_ocean::FT = 4184 # ocean heat capacity [J/(kg * K)]
    q_flux::Bool = false # use Q-flux (parameterization of horizontal ocean mixing of energy)
    Q₀::FT = -20 # Q-flux maximum mplitude [W/m²]
    ϕ₀::FT = 16 # Q-flux meridional scale [deg]
end

abstract type AbstractHyperdiffusion end
Base.@kwdef struct ClimaHyperdiffusion{FT} <: AbstractHyperdiffusion
    ν₄_vorticity_coeff::FT
    ν₄_scalar_coeff::FT
    divergence_damping_factor::FT
end

abstract type AbstractVerticalDiffusion end
Base.@kwdef struct VerticalDiffusion{DM, FT} <: AbstractVerticalDiffusion
    C_E::FT
end
diffuse_momentum(::VerticalDiffusion{DM}) where {DM} = DM
Base.@kwdef struct FriersonDiffusion{DM, FT} <: AbstractVerticalDiffusion
    C_E::FT
end
diffuse_momentum(::FriersonDiffusion{DM}) where {DM} = DM
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
    dc::FT = 0.8
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
# maybe need to <: AbstractForcing
struct GCMForcing{FT}
    external_forcing_file::String
    cfsite_number::String
end

struct ISDACForcing end

struct EDMFCoriolis{U, V, FT}
    prof_ug::U
    prof_vg::V
    coriolis_param::FT
end

abstract type AbstractEnvBuoyGradClosure end
struct BuoyGradMean <: AbstractEnvBuoyGradClosure end

Base.broadcastable(x::BuoyGradMean) = tuple(x)

"""
    EnvBuoyGradVars

Variables used in the environmental buoyancy gradient computation.
"""
Base.@kwdef struct EnvBuoyGradVars{FT, TS}
    ts::TS
    ∂θv∂z_unsat::FT
    ∂qt∂z_sat::FT
    ∂θl∂z_sat::FT
end

function EnvBuoyGradVars(
    ts::TD.ThermodynamicState,
    ∂θv∂z_unsat_∂qt∂z_sat_∂θl∂z_sat,
)
    (; ∂θv∂z_unsat, ∂qt∂z_sat, ∂θl∂z_sat) = ∂θv∂z_unsat_∂qt∂z_sat_∂θl∂z_sat
    return EnvBuoyGradVars(ts, ∂θv∂z_unsat, ∂qt∂z_sat, ∂θl∂z_sat)
end

Base.eltype(::EnvBuoyGradVars{FT}) where {FT} = FT
Base.broadcastable(x::EnvBuoyGradVars) = tuple(x)

struct MixingLength{FT}
    master::FT
    wall::FT
    tke::FT
    buoy::FT
end
MixingLength(args...) = MixingLength{promote_type(typeof.(args)...)}(args...)

abstract type AbstractEDMF end

struct PrognosticEDMFX{N, TKE, FT} <: AbstractEDMF
    a_half::FT # WARNING: this should never be used outside of divide_by_ρa
end
PrognosticEDMFX{N, TKE}(a_half::FT) where {N, TKE, FT} =
    PrognosticEDMFX{N, TKE, FT}(a_half)

struct DiagnosticEDMFX{N, TKE, FT} <: AbstractEDMF
    a_half::FT # WARNING: this should never be used outside of divide_by_ρa
end
DiagnosticEDMFX{N, TKE}(a_half::FT) where {N, TKE, FT} =
    DiagnosticEDMFX{N, TKE, FT}(a_half)

n_mass_flux_subdomains(::PrognosticEDMFX{N}) where {N} = N
n_mass_flux_subdomains(::DiagnosticEDMFX{N}) where {N} = N
n_mass_flux_subdomains(::Any) = 0

n_prognostic_mass_flux_subdomains(::PrognosticEDMFX{N}) where {N} = N
n_prognostic_mass_flux_subdomains(::Any) = 0

use_prognostic_tke(::PrognosticEDMFX{N, TKE}) where {N, TKE} = TKE
use_prognostic_tke(::DiagnosticEDMFX{N, TKE}) where {N, TKE} = TKE
use_prognostic_tke(::Any) = false

abstract type AbstractEntrainmentModel end
struct NoEntrainment <: AbstractEntrainmentModel end
struct PiGroupsEntrainment <: AbstractEntrainmentModel end
struct GeneralizedEntrainment <: AbstractEntrainmentModel end
struct GeneralizedHarmonicsEntrainment <: AbstractEntrainmentModel end

abstract type AbstractDetrainmentModel end

struct NoDetrainment <: AbstractDetrainmentModel end
struct PiGroupsDetrainment <: AbstractDetrainmentModel end
struct GeneralizedDetrainment <: AbstractDetrainmentModel end
struct GeneralizedHarmonicsDetrainment <: AbstractDetrainmentModel end
struct ConstantAreaDetrainment <: AbstractDetrainmentModel end

abstract type AbstractQuadratureType end
struct LogNormalQuad <: AbstractQuadratureType end
struct GaussianQuad <: AbstractQuadratureType end

abstract type AbstractSGSamplingType end
struct SGSMean <: AbstractSGSamplingType end
struct SGSQuadrature{N, QT, A, W} <: AbstractSGSamplingType
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
quad_type(::SGSQuadrature{N}) where {N} = N #TODO - this seems wrong?

abstract type AbstractSurfaceThermoState end
struct GCMSurfaceThermoState <: AbstractSurfaceThermoState end

abstract type AbstractTendencyModel end
struct UseAllTendency <: AbstractTendencyModel end
struct NoGridScaleTendency <: AbstractTendencyModel end
struct NoSubgridScaleTendency <: AbstractTendencyModel end

# Define broadcasting for types
Base.broadcastable(x::AbstractSurfaceThermoState) = tuple(x)
Base.broadcastable(x::AbstractMoistureModel) = tuple(x)
Base.broadcastable(x::AbstractPrecipitationModel) = tuple(x)
Base.broadcastable(x::AbstractForcing) = tuple(x)
Base.broadcastable(x::PrognosticEDMFX) = tuple(x)
Base.broadcastable(x::DiagnosticEDMFX) = tuple(x)
Base.broadcastable(x::AbstractEntrainmentModel) = tuple(x)
Base.broadcastable(x::AbstractDetrainmentModel) = tuple(x)
Base.broadcastable(x::AbstractSGSamplingType) = tuple(x)
Base.broadcastable(x::AbstractTendencyModel) = tuple(x)

Base.@kwdef struct RadiationDYCOMS{FT}
    "Large-scale divergence"
    divergence::FT = 3.75e-6
    alpha_z::FT = 1.0
    kappa::FT = 85.0
    F0::FT = 70.0
    F1::FT = 22.0
end

Base.@kwdef struct RadiationISDAC{FT}
    F₀::FT = 72  # W/m²
    F₁::FT = 15  # W/m²
    κ::FT = 170  # m²/kg
end

import AtmosphericProfilesLibrary as APL

struct RadiationTRMM_LBA{R}
    rad_profile::R
    function RadiationTRMM_LBA(::Type{FT}) where {FT}
        rad_profile = APL.TRMM_LBA_radiation(FT)
        return new{typeof(rad_profile)}(rad_profile)
    end
end

struct TestDycoreConsistency end

abstract type AbstractTimesteppingMode end
struct Explicit <: AbstractTimesteppingMode end
struct Implicit <: AbstractTimesteppingMode end

struct QuasiMonotoneLimiter end # For dispatching to use the ClimaCore QuasiMonotoneLimiter.

Base.@kwdef struct AtmosNumerics{EN_UP, TR_UP, ED_UP, ED_SG_UP, DYCORE, LIM}

    """Enable specific upwinding schemes for specific equations"""
    energy_upwinding::EN_UP
    tracer_upwinding::TR_UP
    edmfx_upwinding::ED_UP
    edmfx_sgsflux_upwinding::ED_SG_UP

    """Add NaNs to certain equations to track down problems"""
    test_dycore_consistency::DYCORE

    limiter::LIM
end
Base.broadcastable(x::AtmosNumerics) = tuple(x)

function Base.summary(io::IO, numerics::AtmosNumerics)
    pns = string.(propertynames(numerics))
    buf = maximum(length.(pns))
    keys = propertynames(numerics)
    vals = repeat.(" ", map(s -> buf - length(s) + 2, pns))
    bufs = (; zip(keys, vals)...)
    print(io, '\n')
    for pn in propertynames(numerics)
        prop = getproperty(numerics, pn)
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

const ValTF = Union{Val{true}, Val{false}}

Base.@kwdef struct EDMFXModel{
    EEM,
    EDM,
    ESMF <: ValTF,
    ESDF <: ValTF,
    ENP <: ValTF,
    EVR <: ValTF,
}
    entr_model::EEM = nothing
    detr_model::EDM = nothing
    sgs_mass_flux::ESMF = Val(false)
    sgs_diffusive_flux::ESDF = Val(false)
    nh_pressure::ENP = Val(false)
    filter::EVR = Val(false)
end

Base.@kwdef struct AtmosModel{
    MC,
    MM,
    PM,
    CM,
    CCDPS,
    F,
    S,
    OZ,
    RM,
    LA,
    EXTFORCING,
    EC,
    AT,
    TM,
    EDMFX,
    TCM,
    NOGW,
    OGW,
    HD,
    VD,
    DM,
    SAM,
    VS,
    RS,
    ST,
    IN,
    SM,
    SA,
    NUM,
}
    model_config::MC = nothing
    moisture_model::MM = nothing
    precip_model::PM = nothing
    cloud_model::CM = nothing
    call_cloud_diagnostics_per_stage::CCDPS = nothing
    forcing_type::F = nothing
    subsidence::S = nothing

    """What to do with ozone for radiation (when using RRTGMP)"""
    ozone::OZ = nothing

    radiation_mode::RM = nothing
    ls_adv::LA = nothing
    external_forcing::EXTFORCING = nothing
    edmf_coriolis::EC = nothing
    advection_test::AT = nothing
    tendency_model::TM = nothing
    edmfx_model::EDMFX = nothing
    turbconv_model::TCM = nothing
    non_orographic_gravity_wave::NOGW = nothing
    orographic_gravity_wave::OGW = nothing
    hyperdiff::HD = nothing
    vert_diff::VD = nothing
    diff_mode::DM = nothing
    sgs_adv_mode::SAM = nothing
    viscous_sponge::VS = nothing
    rayleigh_sponge::RS = nothing
    sfc_temperature::ST = nothing
    insolation::IN = nothing
    surface_model::SM = nothing
    surface_albedo::SA = nothing
    numerics::NUM = nothing
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

struct AtmosConfig{FT, TD, PA, C, CF}
    toml_dict::TD
    parsed_args::PA
    comms_ctx::C
    config_files::CF
    job_id::String
end

Base.eltype(::AtmosConfig{FT}) where {FT} = FT

TupleOrVector(T) = Union{Tuple{<:T, Vararg{T}}, Vector{<:T}}

# Use short, relative paths, if possible.
function normrelpath(file)
    rfile = normpath(relpath(file, dirname(config_path)))
    return if isfile(rfile) && samefile(rfile, file)
        rfile
    else
        file
    end
end

function maybe_add_default(config_files, default_config_file)
    return if any(x -> samefile(x, default_config_file), config_files)
        config_files
    else
        (default_config_file, config_files...)
    end
end

"""
    AtmosConfig(
        config_file::String = default_config_file;
        job_id = config_id_from_config_file(config_file),
        comms_ctx = nothing,
    )
    AtmosConfig(
        config_files::Union{NTuple{<:Any, String} ,Vector{String}};
        job_id = config_id_from_config_files(config_files),
        comms_ctx = nothing,
    )

Helper function for the AtmosConfig constructor. Reads a YAML file into a Dict
and passes it to the AtmosConfig constructor.
"""
AtmosConfig(
    config_file::String = default_config_file;
    job_id = config_id_from_config_file(config_file),
    comms_ctx = nothing,
) = AtmosConfig((config_file,); job_id, comms_ctx)

function AtmosConfig(
    config_files::TupleOrVector(String);
    job_id = config_id_from_config_files(config_files),
    comms_ctx = nothing,
)

    all_config_files =
        normrelpath.(maybe_add_default(config_files, default_config_file))

    configs = map(all_config_files) do config_file
        @info "Loading yaml file $config_file"
        strip_help_messages(load_yaml_file(config_file))
    end
    return AtmosConfig(
        configs;
        comms_ctx,
        config_files = all_config_files,
        job_id,
    )
end

"""
    AtmosConfig(
        configs::Union{NTuple{<:Any, Dict} ,Vector{Dict}};
        comms_ctx = nothing,
        config_files,
        job_id
    )

Constructs the AtmosConfig from the Dicts passed in. This Dict overrides all of
the default configurations set in `default_config_dict()`.
"""
AtmosConfig(configs::AbstractDict; kwargs...) =
    AtmosConfig((configs,); kwargs...)
function AtmosConfig(
    configs::TupleOrVector(AbstractDict);
    comms_ctx = nothing,
    config_files = [default_config_file],
    job_id = "",
)
    config_files = map(x -> normrelpath(x), config_files)

    # using config_files = [default_config_file] as a default
    # relies on the fact that override_default_config uses
    # default_config_file.
    config = override_default_config(configs)
    FT = config["FLOAT_TYPE"] == "Float64" ? Float64 : Float32
    toml_dict = CP.create_toml_dict(
        FT;
        override_file = CP.merge_toml_files(config["toml"]),
    )
    comms_ctx = isnothing(comms_ctx) ? get_comms_context(config) : comms_ctx

    config = config_with_resolved_and_acquired_artifacts(config, comms_ctx)
    device = ClimaComms.device(comms_ctx)
    if device isa ClimaComms.CPUMultiThreaded
        @info "Running ClimaCore in threaded mode, with $(Threads.nthreads()) threads."
    else
        @info "Running ClimaCore in unthreaded mode."
    end

    isempty(job_id) &&
        @warn "`job_id` is empty and likely not passed to AtmosConfig."

    @info "Making AtmosConfig with config files: $(sprint(config_summary, config_files))"

    C = typeof(comms_ctx)
    TD = typeof(toml_dict)
    PA = typeof(config)
    CF = typeof(config_files)
    return AtmosConfig{FT, TD, PA, C, CF}(
        toml_dict,
        config,
        comms_ctx,
        config_files,
        job_id,
    )
end

"""
    maybe_resolve_and_acquire_artifacts(input_str::AbstractString, context::ClimaComms.AbstractCommsContext)

When given a string of the form `artifact"name"/something/else`, resolve the
artifact path and download it (if not already available).

In all the other cases, return the input unchanged.
"""
function maybe_resolve_and_acquire_artifacts(
    input_str::AbstractString,
    context::ClimaComms.AbstractCommsContext,
)
    matched = match(r"artifact\"([a-zA-Z0-9_]+)\"(\/.*)?", input_str)
    if isnothing(matched)
        return input_str
    else
        artifact_name, other_path = matched
        return joinpath(
            @clima_artifact(string(artifact_name), context),
            lstrip(other_path, '/'),
        )
    end
end

function maybe_resolve_and_acquire_artifacts(
    input,
    _::ClimaComms.AbstractCommsContext,
)
    return input
end

"""
    config_with_resolved_and_acquired_artifacts(input_str::AbstractString, context::ClimaComms.AbstractCommsContext)

Substitute strings of the form `artifact"name"/something/else` with the actual
artifact path.
"""
function config_with_resolved_and_acquired_artifacts(
    config::AbstractDict,
    context::ClimaComms.AbstractCommsContext,
)
    return Dict(
        k => maybe_resolve_and_acquire_artifacts(v, context) for
        (k, v) in config
    )
end

function config_summary(io::IO, config_files)
    print(io, '\n')
    for x in config_files
        println(io, "   $x")
    end
end
