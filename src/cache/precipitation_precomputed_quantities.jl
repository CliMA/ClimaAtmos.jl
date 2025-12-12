#####
##### Precomputed quantities for precipitation processes
#####

import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Microphysics2M as CM2
import CloudMicrophysics.P3Scheme as CMP3

import Thermodynamics as TD
import ClimaCore.Operators as Operators
import ClimaCore.Fields as Fields
import ClimaCore.MatrixFields: get_field, extract_first

"""
    Smallest mass value that is different than zero for the purpose of mass_weigthed
    averaging of terminal velocities.
"""
ϵ_numerics(FT) = sqrt(floatmin(FT))

"""
   Kin(w_precip, u_air_uvw)

Helper function to compute the kinetic energy of cloud condensate and precipitation.

# Arguments
- `w_precip`: teminal velocity of cloud consensate or precipitation
- `u_air_uvw`: air velocity as a `UVW` field
"""
Kin(w_precip, u_air_uvw) = norm_sqr(UVW(0, 0, -w_precip) + u_air_uvw) / 2

internal_energy_fn(qᵪ_name) =
    qᵪ_name ∈ (@name(q_liq), @name(q_rai)) ? TD.internal_energy_liquid :
    qᵪ_name ∈ (@name(q_ice), @name(q_sno)) ? TD.internal_energy_ice : nothing #error("Invalid tracer name $qᵪ_name")

ρξw_div_ρξ_bounded(ρqw, ρq) = ifelse(ρq > ϵ_numerics(eltype(ρq)), ρqw / ρq, zero(ρq))

function total_energy_fn(qᵪ_name)
    I = internal_energy_fn(qᵪ_name)
    total_energy(thp, ᶜts, ᶜΦ, ᶜwᵪ, ᶜu_uvw) = I(thp, ᶜts) + ᶜΦ + Kin(ᶜwᵪ, ᶜu_uvw)
end
function internal_plus_geopotential_energy_fn(qᵪ_name)
    I = internal_energy_fn(qᵪ_name)
    internal_plus_geopotential_energy(thp, ᶜts, ᶜΦ) = I(thp, ᶜts) + ᶜΦ
end

"""
    set_precipitation_velocities!(Y, p, moisture_model, microphysics_model, turbconv_model)

Updates the precipitation terminal velocity, cloud sedimentation velocity,
and their contribution to total water and energy advection.
"""
function set_precipitation_velocities!(_, p, _, _, _)
    (; ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed
    @. ᶜwₜqₜ = WVec(0)
    @. ᶜwₕhₜ = WVec(0)
    return nothing
end
function set_precipitation_velocities!(Y, p, ::NonEquilMoistModel, ::Microphysics1Moment, _)
    (; ᶜwₜqₜ, ᶜwₕhₜ, ᶜts, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core
    cmc = CAP.microphysics_cloud_params(p.params)
    cmp = CAP.microphysics_1m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    ᶜu_uvw = @. lazy(UVW(ᶜu))

    qᵪ_names = (@name(q_liq), @name(q_ice), @name(q_rai), @name(q_sno))
    @. ᶜwₜqₜ = WVec(0)
    @. ᶜwₕhₜ = WVec(0)
    for qᵪ_name in qᵪ_names
        ᶜwᵪ = get_field(p.precomputed, get_ᶜwᵪ_name_from_qᵪ_name(qᵪ_name))
        ᶜqᵪ = ᶜspecific_gs_tracer(Y, qᵪ_name)
        wᵪ = terminal_velocity_func_1M(cmc, cmp, qᵪ_name)
        calc_hₜ = total_energy_func(qᵪ_name)
        @. ᶜwᵪ = wᵪ(Y.c.ρ, clip(ᶜqᵪ))
        @. ᶜwₜqₜ += WVec(ᶜwᵪ * ᶜqᵪ)
        @. ᶜwₕhₜ += WVec(ᶜwᵪ * ᶜqᵪ * calc_hₜ(thp, ᶜts, ᶜΦ, ᶜwᵪ, ᶜu_uvw))
    end
    return nothing
end
function set_precipitation_velocities!(Y, p,
    ::NonEquilMoistModel, ::Microphysics1Moment, turbconv_model::PrognosticEDMFX,
)
    (; ᶜΦ) = p.core
    (; ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed
    (; ᶜts⁰, ᶜtsʲs, ᶜρʲs) = p.precomputed
    ᶜsgsʲs = Y.c.sgsʲs
    ᶜρ = Y.c.ρ
    cmc = CAP.microphysics_cloud_params(p.params)
    cmp = CAP.microphysics_1m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    # Environment quantities
    ᶜρa⁰ = @. lazy(ρa⁰(ᶜρ, ᶜsgsʲs, turbconv_model))

    @. ᶜwₕhₜ = WVec(0)
    @. ᶜwₜqₜ = WVec(0)

    qᵪ_names = (@name(q_liq), @name(q_ice), @name(q_rai), @name(q_sno))
    for qᵪ_name in qᵪ_names
        # Functions that depend on qᵪ_name
        wᵪ_fn = terminal_velocity_func_1M(cmc, cmp, qᵪ_name)  # terminal velocity for tracer `q`
        _ρaqwʲ(ρaʲ, qᵪʲ, tsʲ) = begin  # Calculate `ρa ⋅ qᵪ ⋅ wᵪ` for subdomain `j`
            ρʲ = TD.air_density(thp, tsʲ)
            wᵪ = wᵪ_fn(ρʲ, qᵪʲ)
            ρaʲ * qᵪʲ * wᵪ
        end
        ρaqwʲ(sgsʲ, tsʲ) = _ρaqwʲ(sgsʲ.ρa, get_field(sgsʲ, qᵪ_name), tsʲ)
        ρaqwʲ_clip(sgsʲ, tsʲ) = ρaqwʲ(clip(sgsʲ), tsʲ)
        
        ρaqʲ(sgsʲ) = sgsʲ.ρa * get_field(sgsʲ, qᵪ_name)  # Calculate `ρa ⋅ qᵪ` for subdomain `j`
        ρaqʲ_clip(ᶜsgsʲ) = ρaqʲ(clip(ᶜsgsʲ))

        calc_hₜ = internal_plus_geopotential_energy_fn(qᵪ_name)  # TODO: do we need to add kinetic energy of subdomains?
        # Get grid-scale quantities
        ᶜqᵪ = ᶜspecific_gs_tracer(Y, qᵪ_name)
        ᶜwᵪ = get_field(p.precomputed, get_ᶜwᵪ_name_from_qᵪ_name(qᵪ_name))
        # Get environment quantities
        ᶜqᵪ⁰ = ᶜspecific_env_value(qᵪ_name, Y, p)

        # Compute gs sedimentation velocity based on subdomain velocities 
        #   (assuming gs flux equals sum of sgs fluxes)
        ᶜρaqwᵪ⁰ = @. lazy(_ρaqwʲ(clip(ᶜρa⁰), clip(ᶜqᵪ⁰), ᶜts⁰))
        ᶜρqᵪ = @. lazy(clip(ᶜρa⁰) * clip(ᶜqᵪ⁰) + draft_sum(ρaqʲ_clip, ᶜsgsʲs)) # subdomain sum
        @. ᶜwᵪ = ρξw_div_ρξ_bounded(ᶜρaqwᵪ⁰ + draft_sum(ρaqwʲ_clip, ᶜsgsʲs, ᶜtsʲs), ᶜρqᵪ)

        # implied environment mass flux: ρqᵪ⋅wᵪ (gm) - ∑ⱼ ρaʲ⋅qᵪʲ⋅wᵪʲ (drafts)
        ᶜρqᵪ_gm = get_field(Y.c, get_ᶜρqᵪ_name_from_qᵪ_name(qᵪ_name))  # gm
        ᶜρaqwᵪ⁰_implied = @. lazy(ᶜρqᵪ_gm * ᶜwᵪ) - draft_sum(ρaqwʲ, ᶜsgsʲs, ᶜtsʲs)  # not clipped

        # Calculate contribution of tracer `q` to total water and energy advection
        ρaqwₕhₜʲ(Φ) = (ᶜsgsʲ, ᶜtsʲ) -> ρaqwʲ(ᶜsgsʲ, ᶜtsʲ) * calc_hₜ(thp, ᶜtsʲ, Φ)
        ᶜρaqwᵪhₜ⁰ = @. lazy(ᶜρaqwᵪ⁰_implied * calc_hₜ(thp, ᶜts⁰, ᶜΦ))
        @. ᶜwₕhₜ += WVec(ᶜρaqwᵪhₜ⁰ + draft_sum(ρaqwₕhₜʲ(ᶜΦ), ᶜsgsʲs, ᶜtsʲs)) / ᶜρ
        @. ᶜwₜqₜ += WVec(ᶜwᵪ * ᶜqᵪ)
    end
    return nothing
end
function set_precipitation_velocities!(Y, p, ::NonEquilMoistModel, ::Microphysics2Moment, _)
    (; ᶜwₜqₜ, ᶜwₕhₜ, ᶜts, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core
    ᶜu_uvw = @. lazy(UVW(ᶜu))

    cmc = CAP.microphysics_cloud_params(p.params)
    cm1p = CAP.microphysics_1m_params(p.params)
    cm2p = CAP.microphysics_2m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    ᶜρ = Y.c.ρ

    qᵪ_names = (@name(q_liq), @name(q_ice), @name(q_rai), @name(q_sno))
    for qᵪ_name in qᵪ_names
        # Get velocity fields and functions
        ᶜwᵪ = get_field(p.precomputed, get_ᶜwᵪ_name_from_qᵪ_name(qᵪ_name))
        ᶜwₙᵪ = get_field(p.precomputed, get_ᶜwₙᵪ_name_from_qᵪ_name(qᵪ_name))
        wᵪ_func = terminal_velocity_mass_func_2M(cm2p, cmc, cm1p, qᵪ_name)
        wₙᵪ_func = terminal_velocity_number_func_2M(cm2p, cmc, cm1p, qᵪ_name)
        calc_hₜ = total_energy_func(qᵪ_name)
        # Get tracer fields
        ᶜqᵪ = ᶜspecific_gs_tracer(Y, qᵪ_name)
        ρnᵪ_name = get_ρnᵪ_name_from_qᵪ_name(qᵪ_name)
        ρnᵪ = get_field(Y.c, ρnᵪ_name)  # this is `Y.c` if ρnᵪ is not tracked for qᵪ
        # Compute velocity
        @. ᶜwᵪ = wᵪ_func(ᶜρ, ᶜqᵪ, ρnᵪ)  # `ρnᵪ` is not used if number is not tracked for qᵪ
        if ρnᵪ_name != @name()  # if number is tracked for qᵪ
            @. ᶜwₙᵪ = wₙᵪ_func(ᶜρ, ᶜqᵪ, ρnᵪ)
        end
        ᶜwqᵪ = @. lazy(WVec(ᶜwᵪ * ᶜqᵪ))
        @. ᶜwₜqₜ += ᶜwqᵪ
        @. ᶜwₕhₜ += ᶜwqᵪ * calc_hₜ(thp, ᶜts, ᶜΦ, ᶜwᵪ, ᶜu_uvw)
    end
    return nothing
end
function set_precipitation_velocities!(Y, p,
    ::NonEquilMoistModel, ::Microphysics2Moment, turbconv_model::PrognosticEDMFX,
)
    (; ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed
    (; ᶜΦ) = p.core
    (; ᶜts⁰, ᶜtsʲs, ᶜρʲs) = p.precomputed
    ᶜρ = Y.c.ρ
    ᶜsgsʲs = Y.c.sgsʲs
    cmc = CAP.microphysics_cloud_params(p.params)
    cm1p = CAP.microphysics_1m_params(p.params)
    cm2p = CAP.microphysics_2m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    ᶜρ⁰ = @. lazy(TD.air_density(thp, ᶜts⁰))
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))

    qᵪ_names = (@name(q_liq), @name(q_ice), @name(q_rai), @name(q_sno))
    for qᵪ_name in qᵪ_names
        nᵪ_name = get_nᵪ_name_from_qᵪ_name(qᵪ_name)
        nᵪ_exists = nᵪ_name != @name()
        # Functions that depend on `qᵪ_name`
        wᵪ_func = terminal_velocity_mass_func_2M(cm2p, cmc, cm1p, qᵪ_name)
        ρaqʲ(ᶜsgsʲ) = begin  # Calculate `ρa ⋅ qᵪ` for subdomain `j`
            qᵪʲ = get_field(ᶜsgsʲ, qᵪ_name)
            ᶜsgsʲ.ρa * qᵪʲ
        end
        ρaqwʲ(ᶜsgsʲ, ᶜρʲ) = begin  # Calculate `ρa ⋅ qᵪ ⋅ wᵪ` for subdomain `j`
            qᵪʲ = get_field(ᶜsgsʲ, qᵪ_name)
            nᵪʲ = nᵪ_exists ? get_field(ᶜsgsʲ, nᵪ_name) : zero(qᵪʲ)
            wᵪʲ = wᵪ_func(ᶜρʲ, clip(qᵪʲ), clip(ᶜρʲ * nᵪʲ))  # TODO: Should consistently clip nᵪ⁰ and nᵪʲ
            ᶜsgsʲ.ρa * qᵪʲ * wᵪʲ
        end
        wₙᵪ_func = terminal_velocity_number_func_2M(cm2p, cmc, cm1p, qᵪ_name)
        ρanʲ(ᶜsgsʲ) = begin  # Calculate `ρa ⋅ nᵪ` for subdomain `j`
            nᵪʲ = get_field(ᶜsgsʲ, nᵪ_name)
            ᶜsgsʲ.ρa * nᵪʲ
        end
        ρanwʲ(ᶜsgsʲ, ᶜρʲ) = begin  # Calculate `ρa ⋅ nᵪ ⋅ wᵪ` for subdomain `j`
            qᵪʲ = get_field(ᶜsgsʲ, qᵪ_name)
            nᵪʲ = nᵪ_exists ? get_field(ᶜsgsʲ, nᵪ_name) : zero(qᵪʲ)
            wₙᵪʲ = wₙᵪ_func(ᶜρʲ, clip(qᵪʲ), clip(ᶜρʲ * nᵪʲ))  # TODO: Should consistently clip nᵪ⁰ and nᵪʲ
            ᶜsgsʲ.ρa * nᵪʲ * wₙᵪʲ
        end
        calc_hₜ = internal_plus_geopotential_energy_fn(qᵪ_name)
        # Get grid-scale quantities
        ᶜqᵪ = ᶜspecific_gs_tracer(Y, qᵪ_name)
        ᶜwᵪ = get_field(p.precomputed, get_ᶜwᵪ_name_from_qᵪ_name(qᵪ_name))
        ᶜwₙᵪ = get_field(p.precomputed, get_ᶜwₙᵪ_name_from_qᵪ_name(qᵪ_name))
        # Get environment quantities
        ᶜqᵪ⁰ = ᶜspecific_env_value(qᵪ_name, Y, p)
        ᶜnᵪ⁰ = nᵪ_exists ? ᶜspecific_env_value(nᵪ_name, Y, p) : (@. lazy(zero(ᶜqᵪ⁰)))

        # Mass
        clip_ρaqʲ(ᶜsgsʲ) = clip(ρaqʲ(ᶜsgsʲ))
        ᶜρqᵪ = @. lazy(clip(ᶜρ⁰ * ᶜqᵪ⁰) + draft_sum(clip_ρaqʲ, ᶜsgsʲs))
        ᶜwᵪ⁰ = @. lazy(wᵪ_func(ᶜρ⁰, clip(ᶜqᵪ⁰), clip(ᶜρ⁰ * clip(ᶜnᵪ⁰))))  # TODO: Should consistently clip nᵪ⁰ and nᵪʲ
        ᶜρaqwᵪ⁰ = @. lazy(ᶜρa⁰ * ᶜqᵪ⁰ * ᶜwᵪ⁰)
        @. ᶜwᵪ = ρξw_div_ρξ_bounded(ᶜρaqwᵪ⁰ + draft_sum(ρaqwʲ, ᶜsgsʲs, ᶜρʲs), ᶜρqᵪ)

        # Number
        if nᵪ_exists
            clip_ρanʲ(ᶜsgsʲ) = clip(ρanʲ(ᶜsgsʲ))
            ᶜρnᵪ = @. lazy(clip(ᶜρ⁰ * ᶜnᵪ⁰) + draft_sum(clip_ρanʲ, ᶜsgsʲs))
            ᶜwₙᵪ⁰ = @. lazy(wₙᵪ_func(ᶜρ⁰, clip(ᶜqᵪ⁰), clip(ᶜρ⁰ * clip(ᶜnᵪ⁰))))  # TODO: Should consistently clip nᵪ⁰ and nᵪʲ
            ᶜρanwᵪ⁰ = @. lazy(ᶜρa⁰ * ᶜnᵪ⁰ * ᶜwₙᵪ⁰)
            @. ᶜwₙᵪ = ρξw_div_ρξ_bounded(ᶜρanwᵪ⁰ + draft_sum(ρanwʲ, ᶜsgsʲs, ᶜρʲs), ᶜρnᵪ)
        end

        # Calculate contribution of tracer `q` to total water and energy advection
        ρaqwₕhₜʲ(Φ) = (ᶜsgsʲ, ᶜρʲ, ᶜtsʲ) -> ρaqwʲ(ᶜsgsʲ, ᶜρʲ) * calc_hₜ(thp, ᶜtsʲ, Φ)
        ᶜρaqwᵪhₜ⁰ = @. lazy(ᶜρaqwᵪ⁰ * calc_hₜ(thp, ᶜts⁰, ᶜΦ))
        @. ᶜwₕhₜ += WVec(ᶜρaqwᵪhₜ⁰ + draft_sum(ρaqwₕhₜʲ(ᶜΦ), ᶜsgsʲs, ᶜρʲs, ᶜtsʲs)) / ᶜρ
        @. ᶜwₜqₜ += WVec(ᶜwᵪ * ᶜqᵪ)
    end
    return nothing
end

function set_precipitation_velocities!(
    Y, p, ::NonEquilMoistModel, ::Microphysics2MomentP3,
)
    ## liquid quantities (2M warm rain)
    (; ᶜwₗ, ᶜwᵣ, ᶜwnₗ, ᶜwnᵣ, ᶜwₜqₜ, ᶜwₕhₜ, ᶜts, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core

    (; ρ, ρq_liq, ρn_liq, ρq_rai, ρn_rai) = Y.c
    (; sb, rtv, ctv) = p.params.microphysics_2mp3_params.warm
    thp = CAP.thermodynamics_params(p.params)

    # Number- and mass weighted rain terminal velocity [m/s]
    ᶜrai_w_terms = @. lazy(
        CM2.rain_terminal_velocity(
            sb, rtv,
            max(zero(ρ), specific(ρq_rai, ρ)),
            ρ, max(zero(ρ), ρn_rai),
        ),
    )
    @. ᶜwnᵣ = getindex(ᶜrai_w_terms, 1)
    @. ᶜwᵣ = getindex(ᶜrai_w_terms, 2)
    # Number- and mass weighted cloud liquid terminal velocity [m/s]
    ᶜliq_w_terms = @. lazy(
        CM2.cloud_terminal_velocity(
            sb.pdf_c, ctv,
            max(zero(ρ), specific(ρq_liq, ρ)),
            ρ, max(zero(ρ), ρn_liq),
        ),
    )
    @. ᶜwnₗ = getindex(ᶜliq_w_terms, 1)
    @. ᶜwₗ = getindex(ᶜliq_w_terms, 2)

    ## Ice quantities
    (; ρq_ice, ρn_ice, ρq_rim, ρb_rim) = Y.c
    (; ᶜwᵢ) = p.precomputed
    (; cold) = CAP.microphysics_2mp3_params(p.params)

    # Number- and mass weighted ice terminal velocity [m/s]
    # Calculate terminal velocities
    (; ᶜlogλ, ᶜwnᵢ) = p.precomputed
    use_aspect_ratio = true  # TODO: Make a config option
    ᶜF_rim = @. lazy(ρq_rim / ρq_ice)
    ᶜρ_rim = @. lazy(ρq_rim / ρb_rim)
    ᶜstate_p3 = @. lazy(CMP3.P3State(cold.params,
        max(0, ρq_ice), max(0, ρn_ice), ᶜF_rim, ᶜρ_rim,
    ))
    @. ᶜlogλ = CMP3.get_distribution_logλ(ᶜstate_p3)
    args = (cold.velocity_params, ρ, ᶜstate_p3, ᶜlogλ)
    @. ᶜwnᵢ = CMP3.ice_terminal_velocity_number_weighted(args...; use_aspect_ratio)
    @. ᶜwᵢ = CMP3.ice_terminal_velocity_mass_weighted(args...; use_aspect_ratio)

    # compute their contributions to energy and total water advection
    @. ᶜwₜqₜ = WVec(ᶜwₗ * ρq_liq + ᶜwᵢ * ρq_ice + ᶜwᵣ * ρq_rai) / ρ
    calc_hₗ = total_energy_func(@name(q_liq))
    calc_hᵢ = total_energy_func(@name(q_ice))
    calc_hᵣ = total_energy_func(@name(q_rai))
    @. ᶜwₕhₜ =
        WVec(
            ᶜwₗ * ρq_liq * (calc_hₗ(thp, ᶜts, ᶜΦ, ᶜwₗ, UVW(ᶜu))) +
            ᶜwᵢ * ρq_ice * (calc_hᵢ(thp, ᶜts, ᶜΦ, ᶜwᵢ, UVW(ᶜu))) +
            ᶜwᵣ * ρq_rai * (calc_hᵣ(thp, ᶜts, ᶜΦ, ᶜwᵣ, UVW(ᶜu))),
        ) / ρ
    return nothing
end

"""
    set_precipitation_cache!(Y, p, microphysics_model, turbconv_model)

Computes the cache needed for precipitation tendencies. When run without edmf
model this involves computing precipitation sources based on the grid mean
properties. When running with edmf model this means summing the precipitation
sources from the sub-domains.
"""
set_precipitation_cache!(Y, p, _, _) = nothing
function set_precipitation_cache!(Y, p, ::Microphysics0Moment, _)
    (; params, dt) = p
    (; ᶜts) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed
    (; ᶜΦ) = p.core
    cm_params = CAP.microphysics_0m_params(params)
    thermo_params = CAP.thermodynamics_params(params)
    @. ᶜS_ρq_tot =
        Y.c.ρ * q_tot_0M_precipitation_sources(
            thermo_params, cm_params, dt, Y.c.ρq_tot / Y.c.ρ, ᶜts,
        )
    @. ᶜS_ρe_tot = ᶜS_ρq_tot * e_tot_0M_precipitation_sources_helper(thermo_params, ᶜts, ᶜΦ)
    return nothing
end
function set_precipitation_cache!(Y, p,
    ::Microphysics0Moment, turbconv::Union{DiagnosticEDMFX, PrognosticEDMFX},
)
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ⁰, ᶜSqₜᵖʲs) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed
    (; ᶜtsʲs) = p.precomputed
    thp = CAP.thermodynamics_params(p.params)
    ᶜρ = Y.c.ρ

    # Draft and environment area-weighted densities
    ᶜsgs_ρaʲs = ᶜρaʲs_list(Y, p, turbconv)
    ## For environment we multiply by grid mean `ρ` and not by `ᶜρa⁰`, assuming `a⁰ = 1`
    ᶜρa⁰ = turbconv isa PrognosticEDMFX ? (@. lazy(ρa⁰(ᶜρ, Y.c.sgsʲs, turbconv))) : ᶜρ
    ᶜts⁰ = turbconv isa PrognosticEDMFX ? p.precomputed.ᶜts⁰ : p.precomputed.ᶜts

    e_tot_src(Sqₜ, ρ, ts, Φ) = Sqₜ * ρ * e_tot_0M_precipitation_sources_helper(thp, ts, Φ)
    e_tot_srcʲ(Φ) = (Sqₜ, ρ, ts) -> e_tot_src(Sqₜ, ρ, ts, Φ)
    @. ᶜS_ρq_tot = ᶜSqₜᵖ⁰ * ᶜρa⁰ + draft_sum((S, ρ) -> S * ρ, ᶜSqₜᵖʲs, ᶜsgs_ρaʲs)
    ᶜS_ρe_tot⁰ = @. lazy(e_tot_src(ᶜSqₜᵖ⁰, ᶜρa⁰, ᶜts⁰, ᶜΦ))
    @. ᶜS_ρe_tot = ᶜS_ρe_tot⁰ + draft_sum(e_tot_srcʲ(ᶜΦ), ᶜSqₜᵖʲs, ᶜsgs_ρaʲs, ᶜtsʲs)
    return nothing
end
function set_precipitation_cache!(Y, p, ::Microphysics1Moment, _)
    (; dt) = p
    (; ᶜts) = p.precomputed
    (; ᶜSqₗᵖ, ᶜSqᵢᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ) = p.precomputed

    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
    ᶜq_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))
    ᶜq_liq = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
    ᶜq_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))

    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜSᵖ_snow = p.scratch.ᶜtemp_scalar_2

    # get thermodynamics and 1-moment microphysics params
    (; params) = p
    cmp = CAP.microphysics_1m_params(params)
    thp = CAP.thermodynamics_params(params)

    # compute precipitation source terms on the grid mean
    compute_precipitation_sources!(
        ᶜSᵖ, ᶜSᵖ_snow, ᶜSqₗᵖ, ᶜSqᵢᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ,
        Y.c.ρ, ᶜq_tot, ᶜq_liq, ᶜq_ice, ᶜq_rai, ᶜq_sno,
        ᶜts, dt, cmp, thp,
    )

    # compute precipitation sinks on the grid mean
    compute_precipitation_sinks!(
        ᶜSᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ,
        Y.c.ρ, ᶜq_tot, ᶜq_liq, ᶜq_ice, ᶜq_rai, ᶜq_sno,
        ᶜts, dt, cmp, thp,
    )
    return nothing
end
function set_precipitation_cache!(Y, p, ::Microphysics1Moment, ::DiagnosticEDMFX)
    # Nothing needs to be done on the grid mean. The Sources are computed
    # in edmf sub-domains.
    return nothing
end
function set_precipitation_cache!(Y, p, ::Microphysics1Moment, ::PrognosticEDMFX)
    # Nothing needs to be done on the grid mean. The Sources are computed
    # in edmf sub-domains.
    return nothing
end
function set_precipitation_cache!(Y, p, ::Microphysics2Moment, _)
    (; dt) = p
    (; ᶜts) = p.precomputed
    (; ᶜSqₗᵖ, ᶜSqᵢᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ) = p.precomputed
    (; ᶜSnₗᵖ, ᶜSnᵣᵖ) = p.precomputed
    ᶜρ = Y.c.ρ

    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜS₂ᵖ = p.scratch.ᶜtemp_scalar_2

    # get thermodynamics and microphysics params
    (; params) = p
    cmp = CAP.microphysics_2m_params(params)
    thp = CAP.thermodynamics_params(params)

    ᶜn_liq = @. lazy(specific(Y.c.ρn_liq, ᶜρ))
    ᶜn_rai = @. lazy(specific(Y.c.ρn_rai, ᶜρ))
    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, ᶜρ))
    ᶜq_liq = @. lazy(specific(Y.c.ρq_liq, ᶜρ))
    ᶜq_ice = @. lazy(specific(Y.c.ρq_ice, ᶜρ))
    ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, ᶜρ))
    ᶜq_sno = @. lazy(specific(Y.c.ρq_sno, ᶜρ))

    # compute warm precipitation sources on the grid mean (based on SB2006 2M scheme)
    compute_warm_precipitation_sources_2M!(
        ᶜSᵖ, ᶜS₂ᵖ, ᶜSnₗᵖ, ᶜSnᵣᵖ, ᶜSqₗᵖ, ᶜSqᵣᵖ,
        ᶜρ, ᶜn_liq, ᶜn_rai, ᶜq_tot, ᶜq_liq, ᶜq_ice, ᶜq_rai, ᶜq_sno,
        ᶜts, dt, cmp, thp,
    )

    #TODO - implement 2M cold processes!
    @. ᶜSqᵢᵖ = 0
    @. ᶜSqₛᵖ = 0

    return nothing
end
function set_precipitation_cache!(Y, p, ::Microphysics2Moment, ::DiagnosticEDMFX)
    error("Not implemented yet")
    return nothing
end
function set_precipitation_cache!(Y, p, ::Microphysics2Moment, ::PrognosticEDMFX)
    # Nothing needs to be done on the grid mean. The Sources are computed
    # in edmf sub-domains.
    return nothing
end

function set_precipitation_cache!(Y, p, ::Microphysics2MomentP3, ::Nothing)
    ### Rainy processes (2M)
    (; turbconv_model) = p.atmos
    set_precipitation_cache!(Y, p, Microphysics2Moment(), turbconv_model)
    # NOTE: the above function sets `ᶜSqᵢᵖ` to `0`. For P3, need to update `ᶜSqᵢᵖ` below!!

    ### Icy processes (P3)
    (; ᶜScoll, ᶜts, ᶜlogλ) = p.precomputed

    # get thermodynamics and microphysics params
    (; params) = p
    params_2mp3 = CAP.microphysics_2mp3_params(params)
    thermo_params = CAP.thermodynamics_params(params)

    ᶜY_reduced = (;
        Y.c.ρ,
        # condensate
        Y.c.ρq_liq, Y.c.ρn_liq, Y.c.ρq_rai, Y.c.ρn_rai,
        # ice
        Y.c.ρq_ice, Y.c.ρn_ice, Y.c.ρq_rim, Y.c.ρb_rim,
    )

    # compute warm precipitation sources on the grid mean (based on SB2006 2M scheme)
    compute_cold_precipitation_sources_P3!(
        ᶜScoll, params_2mp3, thermo_params, ᶜY_reduced, ᶜts, ᶜlogλ,
    )

    return nothing

end

"""
    set_precipitation_surface_fluxes!(Y, p, precipitation model)

Computes the flux of rain and snow at the surface. 

For the 0-moment microphysics it is an integral of the source terms in the column. 
For 1-moment microphysics it is the flux through the bottom cell face.
"""
set_precipitation_surface_fluxes!(Y, p, _) = nothing
function set_precipitation_surface_fluxes!(Y, p, ::Microphysics0Moment)
    (; ᶜts) = p.precomputed  # assume ᶜts has been updated
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed
    (; surface_rain_flux, surface_snow_flux) = p.precomputed
    (; col_integrated_precip_energy_tendency) = p.conservation_check

    # update total column energy source for surface energy balance
    Operators.column_integral_definite!(
        col_integrated_precip_energy_tendency, ᶜS_ρe_tot,
    )
    # update surface precipitation fluxes in cache for coupler's use
    thermo_params = CAP.thermodynamics_params(p.params)
    T_freeze = TD.Parameters.T_freeze(thermo_params)
    FT = eltype(p.params)
    ᶜT = @. lazy(TD.air_temperature(thermo_params, ᶜts))
    ᶜ3d_rain = @. lazy(ifelse(ᶜT >= T_freeze, ᶜS_ρq_tot, FT(0)))
    ᶜ3d_snow = @. lazy(ifelse(ᶜT < T_freeze, ᶜS_ρq_tot, FT(0)))
    Operators.column_integral_definite!(surface_rain_flux, ᶜ3d_rain)
    Operators.column_integral_definite!(surface_snow_flux, ᶜ3d_snow)
    return nothing
end
function set_precipitation_surface_fluxes!(Y, p,
    ::Union{Microphysics1Moment, Microphysics2Moment},
)
    (; surface_rain_flux, surface_snow_flux) = p.precomputed
    (; col_integrated_precip_energy_tendency) = p.conservation_check
    (; ᶜwᵣ, ᶜwₛ, ᶜwₗ, ᶜwᵢ, ᶜwₕhₜ) = p.precomputed
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    sfc_J = Fields.level(ᶠJ, Fields.half)
    sfc_space = axes(sfc_J)

    # Jacobian-weighted extrapolation from interior to surface, consistent with
    # the reconstruction of density on cell faces, ᶠρ = ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
    sfc_lev(x) = Fields.Field(Fields.field_values(Fields.level(x, 1)), sfc_space)
    int_J = sfc_lev(ᶜJ)
    int_ρ = sfc_lev(Y.c.ρ)
    sfc_ρ = @. lazy(int_ρ * int_J / sfc_J)

    # Constant extrapolation to surface, consistent with simple downwinding
    # Temporary scratch variables are used here until CC.field_values supports <lazy> fields
    ᶜq_rai = p.scratch.ᶜtemp_scalar
    ᶜq_sno = p.scratch.ᶜtemp_scalar_2
    ᶜq_liq = p.scratch.ᶜtemp_scalar_3
    ᶜq_ice = p.scratch.ᶜtemp_scalar_4
    @. ᶜq_rai = specific(Y.c.ρq_rai, Y.c.ρ)
    @. ᶜq_sno = specific(Y.c.ρq_sno, Y.c.ρ)
    @. ᶜq_liq = specific(Y.c.ρq_liq, Y.c.ρ)
    @. ᶜq_ice = specific(Y.c.ρq_ice, Y.c.ρ)
    sfc_qᵣ = Fields.Field(Fields.field_values(Fields.level(ᶜq_rai, 1)), sfc_space)
    sfc_qₛ = Fields.Field(Fields.field_values(Fields.level(ᶜq_sno, 1)), sfc_space)
    sfc_qₗ = Fields.Field(Fields.field_values(Fields.level(ᶜq_liq, 1)), sfc_space)
    sfc_qᵢ = Fields.Field(Fields.field_values(Fields.level(ᶜq_ice, 1)), sfc_space)
    sfc_wᵣ = Fields.Field(Fields.field_values(Fields.level(ᶜwᵣ, 1)), sfc_space)
    sfc_wₛ = Fields.Field(Fields.field_values(Fields.level(ᶜwₛ, 1)), sfc_space)
    sfc_wₗ = Fields.Field(Fields.field_values(Fields.level(ᶜwₗ, 1)), sfc_space)
    sfc_wᵢ = Fields.Field(Fields.field_values(Fields.level(ᶜwᵢ, 1)), sfc_space)
    sfc_wₕhₜ = Fields.Field(
        Fields.field_values(Fields.level(ᶜwₕhₜ.components.data.:1, 1)), sfc_space,
    )

    @. surface_rain_flux = sfc_ρ * (sfc_qᵣ * (-sfc_wᵣ) + sfc_qₗ * (-sfc_wₗ))
    @. surface_snow_flux = sfc_ρ * (sfc_qₛ * (-sfc_wₛ) + sfc_qᵢ * (-sfc_wᵢ))
    @. col_integrated_precip_energy_tendency = sfc_ρ * (-sfc_wₕhₜ)

    return nothing
end

function set_precipitation_surface_fluxes!(Y, p, ::Microphysics2MomentP3)
    set_precipitation_surface_fluxes!(Y, p, Microphysics2Moment())
    # TODO: Figure out what to do for ρn_ice, ρq_rim, ρb_rim
end
