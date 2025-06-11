
NVTX.@annotate function hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    prep_tracer_hyperdiffusion_tendency!(Yₜ_lim, Y, p, t)
    prep_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    if do_dss(axes(Y.c)) && !isnothing(p.atmos.hyperdiff)
        pairs = dss_hyperdiffusion_tendency_pairs(p)
        Spaces.weighted_dss!(pairs...)
    end
    apply_tracer_hyperdiffusion_tendency!(Yₜ_lim, Y, p, t)
    apply_hyperdiffusion_tendency!(Yₜ, Y, p, t)
end

@generated function sorted_nt(::Val{snames}, ::Val{unames}, vals...) where {snames,unames}
    svals_exprs = []
    for sn in snames
        i = findfirst(un -> un == sn, unames)::Int
        push!(svals_exprs, :(getfield(vals, $i)))
    end
    return quote
        NamedTuple{snames}(($(svals_exprs...),))
    end
end

prognostic_nt(::Val{names}, ::Val{K}, vals...) where {names, K} =
    sorted_nt(Val(names), Val(K), vals...)

@generated function construct_tends(::Val{names}, f, ᶜY, ᶠY, p, t) where {names}
    calls = []
    for name in names
        push!(calls, :(f(Val($(QuoteNode(name))), ᶜY, ᶠY, p, t)))
    end
    return quote
        ($(calls...),)
    end
end

function ᶜremaining_tendency(ᶜY, ᶠY, p, t)
    names = propertynames(ᶜY)
    tends = construct_tends(Val(names), ᶜremaining_tendency, ᶜY, ᶠY, p, t)
    return lazy.(foo.(Val(names), tends...))
end
foo(::Val{names}, tends...) where {names} = NamedTuple{names}(tends)

function ᶠremaining_tendency(ᶜY, ᶠY, p, t)
    names = propertynames(ᶠY)
    tends = construct_tends(Val(names), ᶠremaining_tendency, ᶜY, ᶠY, p, t)
    return lazy.(foo.(Val(names), tends...))
end
using ClimaCore.RecursiveApply: rzero
function ᶜremaining_tendency(::Val{:ρ}, ᶜY, ᶠY, p, t)
    :ρ in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρ))
    ᶜJ = Fields.local_geometry_field(ᶜY).J
    ᶠJ = Fields.local_geometry_field(ᶠY).J
    ᶜρ = ᶜY.ρ

    if !(p.atmos.moisture_model isa DryModel)
        ᶜwₜqₜ = compute_ᶜwₜqₜ(ᶜY, ᶠY, p, t)
        ∑tendencies = sub_tend(∑tendencies, water_adv(ᶜρ, ᶜJ, ᶠJ, ᶜwₜqₜ))
    end
    return ∑tendencies
end

add_tend(∑tends, t) = lazy.(∑tends .+ t)
add_tend(∑tends, ::NullBroadcasted) = ∑tends
sub_tend(∑tends, ::NullBroadcasted) = ∑tends
sub_tend(∑tends, t) = lazy.(∑tends .- t)

function ᶜremaining_tendency(::Val{:uₕ}, ᶜY, ᶠY, p, t)
    :uₕ in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.uₕ))
    (; zmax) = p
    (; viscous_sponge, rayleigh_sponge) = p.atmos
    ᶜz = Fields.coordinate_field(ᶜY).z
    ᶠz = Fields.coordinate_field(ᶠY).z
    ᶜuₕ = ᶜY.uₕ
    ∑tendencies = add_tend(∑tendencies, viscous_sponge_tendency_uₕ(ᶜuₕ, viscous_sponge))
    ∑tendencies = add_tend(∑tendencies, rayleigh_sponge_tendency_uₕ(ᶜuₕ, rayleigh_sponge, ᶜz, ᶠz, zmax))

    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρe_tot}, ᶜY, ᶠY, p, t)
    :ρe_tot in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρe_tot))

    (; moisture_model, viscous_sponge, precip_model) = p.atmos
    (; energy_upwinding) = p.atmos.numerics
    thermo_params = CAP.thermodynamics_params(p.params)
    thermo_args = (thermo_params, moisture_model, precip_model)
    ᶜz = Fields.coordinate_field(ᶜY).z
    ᶜJ = Fields.local_geometry_field(ᶜY).J
    ᶠJ = Fields.local_geometry_field(ᶠY).J
    ᶜρ = ᶜY.ρ
    ᶜuₕ = ᶜY.uₕ
    ᶜρe_tot = ᶜY.ρe_tot
    ᶜts = compute_ᶜts(ᶜY, ᶠY, p, t)
    ᶜp = @. lazy(TD.air_pressure(thermo_params, ᶜts))
    ᶜe_tot = @. lazy(ᶜρe_tot / ᶜρ)
    ᶜh_tot = @. lazy(TD.total_specific_enthalpy(thermo_params, ᶜts, ᶜe_tot))

    if !(p.atmos.moisture_model isa DryModel)
        ᶜwₕhₜ = compute_ᶜwₕhₜ(ᶜY, ᶠY, p, t)
        ∑tendencies = sub_tend(∑tendencies, water_adv(ᶜρ, ᶜJ, ᶠJ, ᶜwₕhₜ))
    end
    if energy_upwinding != Val(:none)
        (; dt) = p
        ᶠu³ = compute_ᶠu³(ᶜY, ᶠY)
        vtt = vertical_transport(ᶜρ, ᶠu³, ᶜh_tot, float(dt), energy_upwinding)
        ∑tendencies = add_tend(∑tendencies, vtt)
        vtt_central = vertical_transport(ᶜρ, ᶠu³, ᶜh_tot, float(dt), Val(:none))
        # need to improve NullBroadcast support for this.
        ∑tendencies = sub_tend(∑tendencies, vtt_central)
    end

    ∑tendencies = add_tend(∑tendencies, viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, viscous_sponge))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρq_tot}, ᶜY, ᶠY, p, t)
    :ρq_tot in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρq_tot))
    ᶜJ = Fields.local_geometry_field(ᶜY).J
    ᶠJ = Fields.local_geometry_field(ᶠY).J
    ᶜρ = ᶜY.ρ
    if !(p.atmos.moisture_model isa DryModel)
        ᶜwₜqₜ = compute_ᶜwₜqₜ(ᶜY, ᶠY, p, t)
        cmc = CAP.microphysics_cloud_params(p.params)
        cmp = CAP.microphysics_1m_params(p.params)
        thp = CAP.thermodynamics_params(p.params)
        ∑tendencies = sub_tend(∑tendencies, water_adv(ᶜρ, ᶜJ, ᶠJ, ᶜwₜqₜ))
    end
    (; tracer_upwinding) = p.atmos.numerics
    if !(p.atmos.moisture_model isa DryModel) && tracer_upwinding != Val(:none)
        (; dt) = p
        ᶜq_tot = @. lazy(ᶜY.ρq_tot / ᶜY.ρ)
        ᶠu³ = compute_ᶠu³(ᶜY, ᶠY)
        vtt = vertical_transport(ᶜρ, ᶠu³, ᶜq_tot, float(dt), tracer_upwinding)
        vtt_central = vertical_transport(ᶜρ, ᶠu³, ᶜq_tot, float(dt), Val(:none))
        ∑tendencies = add_tend(∑tendencies, vtt)
        ∑tendencies = sub_tend(∑tendencies, vtt_central)
    end

    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρq_liq}, ᶜY, ᶠY, p, t)
    :ρq_liq in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρq_liq))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρq_ice}, ᶜY, ᶠY, p, t)
    :ρq_ice in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρq_ice))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρq_rai}, ᶜY, ᶠY, p, t)
    :ρq_rai in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρq_rai))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρq_sno}, ᶜY, ᶠY, p, t)
    :ρq_sno in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρq_sno))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:sgsʲs}, ᶜY, ᶠY, p, t)
    :sgsʲs in propertynames(ᶜY) || return ()
    ∑tendencies = rzero(eltype(ᶜY.sgsʲs))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:sgs⁰}, ᶜY, ᶠY, p, t)
    :sgs⁰ in propertynames(ᶜY) || return ()
    ∑tendencies = rzero(eltype(ᶜY.sgs⁰))
    return ∑tendencies
end
function ᶠremaining_tendency(::Val{:u₃}, ᶜY, ᶠY, p, t)
    :u₃ in propertynames(ᶠY) || return ()
    ∑tendencies = zero(eltype(ᶠY.u₃))
    (; viscous_sponge) = p.atmos
    ᶜρ = ᶜY.ρ
    ᶜuₕ = ᶜY.uₕ
    ᶠuₕ³ = compute_ᶠuₕ³(ᶜuₕ, ᶜρ)
    ᶠu₃ = compute_ᶠu₃_with_bcs(ᶠY.u₃, ᶠuₕ³)
    ∑tendencies = add_tend(∑tendencies, viscous_sponge_tendency_u₃(ᶠu₃, viscous_sponge))
    return ∑tendencies
end
function ᶠremaining_tendency(::Val{:sgsʲs}, ᶜY, ᶠY, p, t)
    :sgsʲs in propertynames(ᶠY) || return ()
    ∑tendencies = rzero(eltype(ᶠY.sgsʲs))
    return ∑tendencies
end


water_adv(ᶜρ, ᶜJ, ᶠJ, ᶜχ::Real) = zero(eltype(ᶜρ))
water_adv(ᶜρ, ᶜJ, ᶠJ, ᶜχ) = # only valid when ᶜχ is a field
    @. lazy(ᶜprecipdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠright_bias(-(ᶜχ))))

function surface_velocity_full(ᶠu₃, ᶠuₕ³)
    assert_eltype(ᶠuₕ³, Geometry.Contravariant3Vector)
    assert_eltype(ᶠu₃, Geometry.Covariant3Vector)
    ᶠlg = Fields.local_geometry_field(axes(ᶠu₃))
    sfc_u₃ = ᶠu₃ # Fields.level(ᶠu₃.components.data.:1, half)
    sfc_uₕ³ = ᶠuₕ³ # Fields.level(ᶠuₕ³.components.data.:1, half)
    sfc_g³³ = g³³_field(axes(sfc_u₃))
    w₃ = @. lazy(- C3(sfc_uₕ³ / sfc_g³³, ᶠlg)) # u³ = uₕ³ + w³ = uₕ³ + w₃ * g³³
    assert_eltype(w₃, Geometry.Covariant3Vector)
    return w₃
end

function compute_ᶜts(ᶜY, ᶠY, p, t)
    (; moisture_model, precip_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    thermo_args = (thermo_params, moisture_model, precip_model)
    ᶜz = Fields.coordinate_field(ᶜY).z
    FT = Spaces.undertype(axes(ᶜY))
    grav = FT(CAP.grav(p.params))
    ᶜΦ = @. lazy(grav * ᶜz)

    ᶜρ = ᶜY.ρ
    ᶜuₕ = ᶜY.uₕ
    ᶠuₕ³ = compute_ᶠuₕ³(ᶜuₕ, ᶜρ)
    ᶠu₃ = compute_ᶠu₃_with_bcs(ᶠY.u₃, ᶠuₕ³)
    ᶜK = compute_kinetic(ᶜuₕ, ᶠu₃)
    ᶜspecific = @. lazy(specific_gs(ᶜY))
    return @. lazy(ts_gs(thermo_args..., ᶜspecific, ᶜK, ᶜΦ, ᶜρ))
end

assert_eltype(bc::Base.AbstractBroadcasted, ::Type{T}) where {T} =
    assert_eltype(eltype(bc), T)
assert_eltype(f::Fields.Field, ::Type{T}) where {T} =
    assert_eltype(Fields.field_values(f), T)
assert_eltype(data::DataLayouts.AbstractData, ::Type{T}) where {T} =
    assert_eltype(eltype(data), T)
assert_eltype(::Type{S}, ::Type{T}) where {S, T} =
    @assert S <: T "Type $S should be a subtype of $T"

function compute_ᶠu₃_with_bcs(ᶠu₃, ᶠuₕ³)
    assert_eltype(ᶠu₃, Geometry.Covariant3Vector)
    assert_eltype(ᶠuₕ³, Geometry.Contravariant3Vector)
    ᶠz = Fields.coordinate_field(axes(ᶠu₃)).z
    sfc_u₃ = surface_velocity_full(ᶠu₃, ᶠuₕ³)
    # todo: generalize this with z_min
    return @. lazy(ifelse(iszero(ᶠz), sfc_u₃, ᶠu₃))
end
function compute_ᶠu³(ᶜY, ᶠY)
    ᶜρ = ᶜY.ρ
    ᶜuₕ = ᶜY.uₕ
    ᶠuₕ³ = compute_ᶠuₕ³(ᶜuₕ, ᶜρ)
    ᶠu₃ = compute_ᶠu₃_with_bcs(ᶠY.u₃, ᶠuₕ³)
    return @. lazy(ᶠuₕ³ + CT3(ᶠu₃))
end

NVTX.@annotate function remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ_lim .= zero(eltype(Yₜ_lim))
    device = ClimaComms.device(axes(Y.c))
    (localmem_lg, localmem_state) = if device isa ClimaComms.CUDADevice
        Val(false), Val(true)
    else
        Val(false), Val(false)
    end
    (; moisture_model, viscous_sponge, precip_model) = p.atmos
    p_kernel = (;
        zmax = Spaces.z_max(axes(Y.f)),
        atmos = p.atmos,
        params = p.params,
        dt = p.dt,
    )
    if :sfc in propertynames(Y) # columnwise! does not yet handle .sfc
        parent(Yₜ.sfc) .= zero(Spaces.undertype(axes(Y.c)))
    end
    Operators.columnwise!(
        device,
        ᶜremaining_tendency,
        ᶠremaining_tendency,
        Yₜ.c,
        Yₜ.f,
        Y.c,
        Y.f,
        p_kernel,
        t,
        localmem_lg,
        localmem_state,
    )
    horizontal_tracer_advection_tendency!(Yₜ_lim, Y, p, t)
    fill_with_nans!(p)
    horizontal_advection_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    additional_tendency!(Yₜ, Y, p, t)
    return Yₜ
end

import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces

function z_coordinate_fields(space::Spaces.AbstractSpace)
    ᶜz = Fields.coordinate_field(Spaces.center_space(space)).z
    ᶠz = Fields.coordinate_field(Spaces.face_space(space)).z
    return (; ᶜz, ᶠz)
end


NVTX.@annotate function additional_tendency!(Yₜ, Y, p, t)

    (; ᶜh_tot, ᶜspecific) = p.precomputed
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜρ = Y.c.ρ
    (; forcing_type, moisture_model, rayleigh_sponge, viscous_sponge) = p.atmos
    (; ls_adv, edmf_coriolis) = p.atmos
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜp, sfc_conditions, ᶜts) = p.precomputed

    hs_args = (ᶜuₕ, ᶜp, params, sfc_conditions.ts, moisture_model, forcing_type)
    hs_tendency_uₕ = held_suarez_forcing_tendency_uₕ(hs_args...)
    hs_tendency_ρe_tot = held_suarez_forcing_tendency_ρe_tot(ᶜρ, hs_args...)
    edmf_cor_tend_uₕ = edmf_coriolis_tendency_uₕ(ᶜuₕ, edmf_coriolis)
    lsa_args = (ᶜρ, thermo_params, ᶜts, t, ls_adv)
    bc_lsa_tend_ρe_tot = large_scale_advection_tendency_ρe_tot(lsa_args...)

    # TODO: can we write this out explicitly?
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        vst_tracer = viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, viscous_sponge)
        @. ᶜρχₜ += vst_tracer
        @. Yₜ.c.ρ += vst_tracer
    end

    # Held Suarez tendencies
    @. Yₜ.c.uₕ += hs_tendency_uₕ
    @. Yₜ.c.ρe_tot += hs_tendency_ρe_tot

    subsidence_tendency!(Yₜ, Y, p, t, p.atmos.subsidence)

    @. Yₜ.c.ρe_tot += bc_lsa_tend_ρe_tot
    if moisture_model isa AbstractMoistModel
        bc_lsa_tend_ρq_tot = large_scale_advection_tendency_ρq_tot(lsa_args...)
        @. Yₜ.c.ρq_tot += bc_lsa_tend_ρq_tot
    end

    @. Yₜ.c.uₕ += edmf_cor_tend_uₕ

    external_forcing_tendency!(Yₜ, Y, p, t, p.atmos.external_forcing)

    if p.atmos.sgs_adv_mode == Explicit()
        edmfx_sgs_vertical_advection_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.turbconv_model,
        )
    end

    if p.atmos.diff_mode == Explicit()
        vertical_diffusion_boundary_layer_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.vert_diff,
        )
        edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end

    surface_flux_tendency!(Yₜ, Y, p, t)

    radiation_tendency!(Yₜ, Y, p, t, p.atmos.radiation_mode)
    if p.atmos.sgs_entr_detr_mode == Explicit()
        edmfx_entr_detr_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    if p.atmos.sgs_mf_mode == Explicit()
        edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    if p.atmos.sgs_nh_pressure_mode == Explicit()
        edmfx_nh_pressure_drag_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    edmfx_filter_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    edmfx_tke_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    if p.atmos.noneq_cloud_formation_mode == Explicit()
        cloud_condensate_tendency!(
            Yₜ,
            Y,
            p,
            p.atmos.moisture_model,
            p.atmos.precip_model,
        )
    end

    edmfx_precipitation_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.turbconv_model,
        p.atmos.precip_model,
    )
    precipitation_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.moisture_model,
        p.atmos.precip_model,
        p.atmos.turbconv_model,
    )

    # NOTE: Precipitation tendencies should be applied before calling this function,
    # because precipitation cache is used in this function
    surface_temp_tendency!(Yₜ, Y, p, t, p.atmos.surface_model)

    # NOTE: All ρa tendencies should be applied before calling this function
    pressure_work_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    sl = p.atmos.smagorinsky_lilly
    horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl)
    vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl)

    # NOTE: This will zero out all momentum tendencies in the edmfx advection test
    # please DO NOT add additional velocity tendencies after this function
    zero_velocity_tendency!(Yₜ, Y, p, t)

    # TODO: make bycolumn-able
    non_orographic_gravity_wave_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.non_orographic_gravity_wave,
    )
    orographic_gravity_wave_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.orographic_gravity_wave,
    )
end
