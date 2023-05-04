#####
##### Precomputed quantities
#####

import Thermodynamics as TD
import ClimaCore: Spaces, Fields

"""
    precomputed_quantities(Y, atmos)

Allocates and returns the precomputed quantities:
    - `ᶜspecific`: the specific quantities on cell centers (for every prognostic
        quantity `ρχ`, there is a corresponding specific quantity `χ`)
    - `ᶜu`: the covariant velocity on cell centers
    - `ᶠu³`: the third component of contravariant velocity on cell faces
    - `ᶜK`: the kinetic energy on cell centers
    - `ᶜts`: the thermodynamic state on cell centers
    - `ᶜp`: the air pressure on cell centers

If the `turbconv_model` is EDMFX, there also two SGS versions of every quantity
except for `ᶜp` (we assume that the pressure is the same across all subdomains):
    - `_⁰`: the value for the environment
    - `_ʲs`: a tuple of the values for the mass-flux subdomains
In addition, there are several other SGS quantities for the EDMFX model:
    - `ᶜρa⁰`: the area-weighted air density of the environment on cell centers
    - `ᶠu₃⁰`: the vertical component of the covariant velocity of the environment
        on cell faces
    - `ᶜρ⁰`: the air density of the environment on cell centers
    - `ᶜρʲs`: a tuple of the air densities of the mass-flux subdomains on cell
        centers

TODO: Rename `ᶜK` to `ᶜκ`.
"""
function precomputed_quantities(Y, atmos)
    FT = eltype(Y)
    TST = thermo_state_type(atmos.moisture_model, FT)
    n = n_mass_flux_subdomains(atmos.turbconv_model)
    gs_quantities = (;
        ᶜspecific = specific_gs.(Y.c),
        ᶜu = similar(Y.c, C123{FT}),
        ᶠu³ = similar(Y.f, CT3{FT}),
        ᶜK = similar(Y.c, FT),
        ᶜts = similar(Y.c, TST),
        ᶜp = similar(Y.c, FT),
    )
    sgs_quantities =
        n == 0 ? (;) :
        (;
            ᶜspecific⁰ = specific_full_sgs⁰.(Y.c, atmos.turbconv_model),
            ᶜρa⁰ = similar(Y.c, FT),
            ᶠu₃⁰ = similar(Y.f, C3{FT}),
            ᶜu⁰ = similar(Y.c, C123{FT}),
            ᶠu³⁰ = similar(Y.f, CT3{FT}),
            ᶜK⁰ = similar(Y.c, FT),
            ᶜts⁰ = similar(Y.c, TST),
            ᶜρ⁰ = similar(Y.c, FT),
            ᶜspecificʲs = specific_sgsʲs.(Y.c, atmos.turbconv_model),
            ᶜuʲs = similar(Y.c, NTuple{n, C123{FT}}),
            ᶠu³ʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶜKʲs = similar(Y.c, NTuple{n, FT}),
            ᶜtsʲs = similar(Y.c, NTuple{n, TST}),
            ᶜρʲs = similar(Y.c, NTuple{n, FT}),
        )
    return (; gs_quantities..., sgs_quantities...)
end

# Interpolates the third contravariant component of Y.c.uₕ to cell faces.
function set_ᶠuₕ³!(ᶠuₕ³, Y)
    Fields.bycolumn(axes(Y.c)) do colidx
        ᶜJ = Fields.local_geometry_field(Y.c).J
        @. ᶠuₕ³[colidx] =
            ᶠwinterp(Y.c.ρ[colidx] * ᶜJ[colidx], CT3(Y.c.uₕ[colidx]))
    end
end

"""
    set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)

Modifies `Y.f.u₃` so that `ᶠu³` is 0 at the surface. If `turbconv_model` is
EDMFX, also modifies `Y.f.sgsʲs` so that each `u₃ʲ` is equal to `u₃` at the
surface.
"""
function set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)
    sfc_u₃ = Fields.level(Y.f.u₃.components.data.:1, half)
    sfc_uₕ³ = Fields.level(ᶠuₕ³.components.data.:1, half)
    sfc_g = Fields.local_geometry_field(sfc_u₃).gⁱʲ.components.data
    end_index = fieldcount(eltype(sfc_g)) # This will be 4 in 2D and 9 in 3D.
    sfc_g³³ = sfc_g.:($end_index) # For both 2D and 3D spaces, g³³ = g[end].
    @. sfc_u₃ = -sfc_uₕ³ / sfc_g³³ # u³ = uₕ³ + w³ = uₕ³ + w₃ * g³³
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        sfc_u₃ʲ = Fields.level(Y.f.sgsʲs.:($j).u₃.components.data.:1, half)
        @. sfc_u₃ʲ = sfc_u₃
    end
end

# This is used to set the grid-scale velocity quantities ᶜu, ᶠu³, ᶜK based on
# ᶠu₃, and it is also used to set the SGS quantities based on ᶠu₃⁰ and ᶠu₃ʲ.
function set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, ᶠu₃, ᶜuₕ, ᶠuₕ³)
    Fields.bycolumn(axes(ᶜu)) do colidx
        @. ᶜu[colidx] = C123(ᶜuₕ[colidx]) + ᶜinterp(C123(ᶠu₃[colidx]))
        @. ᶠu³[colidx] = ᶠuₕ³[colidx] + CT3(ᶠu₃[colidx])
        compute_kinetic!(ᶜK[colidx], ᶜuₕ[colidx], ᶠu₃[colidx])
    end
end

function set_sgs_ᶠu₃!(w_function, ᶠu₃, Y, turbconv_model)
    ρaʲs(sgsʲs) = map(sgsʲ -> sgsʲ.ρa, sgsʲs)
    u₃ʲs(sgsʲs) = map(sgsʲ -> sgsʲ.u₃, sgsʲs)
    Fields.bycolumn(axes(Y.c)) do colidx
        @. ᶠu₃[colidx] = w_function(
            ᶠinterp(ρaʲs(Y.c.sgsʲs[colidx])),
            u₃ʲs(Y.f.sgsʲs[colidx]),
            ᶠinterp(Y.c.ρ[colidx]),
            Y.f.u₃[colidx],
            turbconv_model,
        )
    end
end

function add_sgs_ᶜK!(ᶜK, Y, ᶜρa⁰, ᶠu₃⁰, turbconv_model)
    function do_col!(ᶜK, Yc, Yf, ᶜρa⁰, ᶠu₃⁰)
        @. ᶜK += ᶜρa⁰ * ᶜinterp(dot(ᶠu₃⁰ - Yf.u₃, CT3(ᶠu₃⁰ - Yf.u₃))) / 2 / Yc.ρ
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            ᶜρaʲ = Yc.sgsʲs.:($j).ρa
            ᶠu₃ʲ = Yf.sgsʲs.:($j).u₃
            @. ᶜK +=
                ᶜρaʲ * ᶜinterp(dot(ᶠu₃ʲ - Yf.u₃, CT3(ᶠu₃ʲ - Yf.u₃))) / 2 / Yc.ρ
        end
    end
    Fields.bycolumn(axes(Y.c)) do colidx
        do_col!(
            ᶜK[colidx],
            Y.c[colidx],
            Y.f[colidx],
            ᶜρa⁰[colidx],
            ᶠu₃⁰[colidx],
        )
    end
end

function thermo_state(
    thermo_params;
    ρ = nothing,
    p = nothing,
    θ = nothing,
    e_int = nothing,
    q_tot = nothing,
    q_pt = nothing,
)
    get_ts(ρ::Real, ::Nothing, θ::Real, ::Nothing, ::Nothing, ::Nothing) =
        TD.PhaseDry_ρθ(thermo_params, ρ, θ)
    get_ts(ρ::Real, ::Nothing, θ::Real, ::Nothing, q_tot::Real, ::Nothing) =
        TD.PhaseEquil_ρθq(thermo_params, ρ, θ, q_tot)
    get_ts(ρ::Real, ::Nothing, θ::Real, ::Nothing, ::Nothing, q_pt) =
        TD.PhaseNonEquil_ρθq(thermo_params, ρ, θ, q_pt)
    get_ts(ρ::Real, ::Nothing, ::Nothing, e_int::Real, ::Nothing, ::Nothing) =
        TD.PhaseDry_ρe(thermo_params, ρ, e_int)
    get_ts(ρ::Real, ::Nothing, ::Nothing, e_int::Real, q_tot::Real, ::Nothing) =
        TD.PhaseEquil_ρeq(thermo_params, ρ, e_int, q_tot)
    get_ts(ρ::Real, ::Nothing, ::Nothing, e_int::Real, ::Nothing, q_pt) =
        TD.PhaseNonEquil_ρeq(thermo_params, ρ, e_int, q_pt)
    get_ts(::Nothing, p::Real, θ::Real, ::Nothing, ::Nothing, ::Nothing) =
        TD.PhaseDry_pθ(thermo_params, p, θ)
    get_ts(::Nothing, p::Real, θ::Real, ::Nothing, q_tot::Real, ::Nothing) =
        TD.PhaseEquil_pθq(thermo_params, p, θ, q_tot)
    get_ts(::Nothing, p::Real, θ::Real, ::Nothing, ::Nothing, q_pt) =
        TD.PhaseNonEquil_pθq(thermo_params, p, θ, q_pt)
    get_ts(::Nothing, p::Real, ::Nothing, e_int::Real, ::Nothing, ::Nothing) =
        TD.PhaseDry_pe(thermo_params, p, e_int)
    get_ts(::Nothing, p::Real, ::Nothing, e_int::Real, q_tot::Real, ::Nothing) =
        TD.PhaseEquil_peq(thermo_params, p, e_int, q_tot)
    get_ts(::Nothing, p::Real, ::Nothing, e_int::Real, ::Nothing, q_pt) =
        TD.PhaseNonEquil_peq(thermo_params, p, e_int, q_pt)
    return get_ts(ρ, p, θ, e_int, q_tot, q_pt)
end

function thermo_vars(energy_form, moisture_model, specific, K, Φ)
    energy_var = if energy_form isa PotentialTemperature
        (; specific.θ)
    elseif energy_form isa TotalEnergy
        (; e_int = specific.e_tot - K - Φ)
    end
    moisture_var = if moisture_model isa DryModel
        (;)
    elseif moisture_model isa EquilMoistModel
        (; specific.q_tot)
    elseif moisture_model isa NonEquilMoistModel
        q_pt_args = (specific.q_tot, specific.q_liq, specific.q_ice)
        (; q_pt = TD.PhasePartition(q_pt_args...))
    end
    return (; energy_var..., moisture_var...)
end

ts_gs(thermo_params, energy_form, moisture_model, specific, K, Φ, ρ) =
    thermo_state(
        thermo_params;
        thermo_vars(energy_form, moisture_model, specific, K, Φ)...,
        ρ,
    )

ts_sgs(thermo_params, energy_form, moisture_model, specific, K, Φ, p) =
    thermo_state(
        thermo_params;
        thermo_vars(energy_form, moisture_model, specific, K, Φ)...,
        p,
    )

"""
    set_precomputed_quantities!(Y, p, t)

Updates the precomputed quantities stored in `p` based on the current state `Y`.

This function also applies a "filter" to `Y` in order to ensure that `ᶠu³` is 0
at the surface (i.e., to enforce the impenetrable boundary condition). If the
`turbconv_model` is EDMFX, the filter also ensures that `ᶠu³⁰` and `ᶠu³ʲs` are 0
at the surface. In the future, we will probably want to move this filtering
elsewhere, but doing it here ensures that it occurs whenever the precomputed
quantities are updated.

Note: If you need to use any of the precomputed quantities, please call this
function instead of recomputing the value yourself. Otherwise, it will be
difficult to ensure that the duplicated computations are consistent.
"""
function set_precomputed_quantities!(Y, p, t)
    (; energy_form, moisture_model, turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    n = n_mass_flux_subdomains(turbconv_model)
    thermo_args = (thermo_params, energy_form, moisture_model)
    (; ᶜspecific, ᶜu, ᶠu³, ᶜK, ᶜts, ᶜp, ᶜΦ) = p
    ᶠuₕ³ = p.ᶠtemp_CT3

    @. ᶜspecific = specific_gs(Y.c)
    set_ᶠuₕ³!(ᶠuₕ³, Y)

    # TODO: We might want to move this to dss! (and rename dss! to something
    # like enforce_constraints!).
    set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)

    set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, Y.f.u₃, Y.c.uₕ, ᶠuₕ³)
    if n > 0
        # TODO: In the following increments to ᶜK, we actually need to add
        # quantities of the form ᶜρaχ⁰ / ᶜρ⁰ and ᶜρaχʲ / ᶜρʲ to ᶜK, rather than
        # quantities of the form ᶜρaχ⁰ / ᶜρ and ᶜρaχʲ / ᶜρ. However, we cannot
        # compute ᶜρ⁰ and ᶜρʲ without first computing ᶜts⁰ and ᶜtsʲ, both of
        # which depend on the value of ᶜp, which in turn depends on ᶜts. Since
        # ᶜts depends on ᶜK (at least when the energy_form is TotalEnergy), this
        # means that the amount by which ᶜK needs to be incremented is a
        # function of ᶜK itself. So, unless we run a nonlinear solver here, this
        # circular dependency will prevent us from computing the exact value of
        # ᶜK. For now, we will make the anelastic approximation ᶜρ⁰ ≈ ᶜρʲ ≈ ᶜρ.
        # add_sgs_ᶜK!(ᶜK, Y, ᶜρa⁰, ᶠu₃⁰, turbconv_model)
        # @. ᶜK += Y.c.sgs⁰.ρatke / Y.c.ρ
        # TODO: We should think more about these increments before we use them.
    end
    @. ᶜts = ts_gs(thermo_args..., ᶜspecific, ᶜK, ᶜΦ, Y.c.ρ)
    @. ᶜp = TD.air_pressure(thermo_params, ᶜts)

    if n > 0
        (; ᶜspecific⁰, ᶜρa⁰, ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜts⁰, ᶜρ⁰) = p
        (; ᶜspecificʲs, ᶜuʲs, ᶠu³ʲs, ᶜKʲs, ᶜtsʲs, ᶜρʲs) = p
        @. ᶜspecific⁰ = specific_full_sgs⁰(Y.c, turbconv_model)
        @. ᶜρa⁰ = ρa⁰(Y.c)
        set_sgs_ᶠu₃!(u₃⁰, ᶠu₃⁰, Y, turbconv_model)
        set_velocity_quantities!(ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶠu₃⁰, Y.c.uₕ, ᶠuₕ³)
        @. ᶜK⁰ += ᶜspecific⁰.tke
        @. ᶜts⁰ = ts_sgs(thermo_args..., ᶜspecific⁰, ᶜK⁰, ᶜΦ, ᶜp)
        @. ᶜρ⁰ = TD.air_density(thermo_params, ᶜts⁰)
        @. ᶜspecificʲs = specific_sgsʲs(Y.c, turbconv_model)
        for j in 1:n
            ᶜuʲ = ᶜuʲs.:($j)
            ᶠu³ʲ = ᶠu³ʲs.:($j)
            ᶜKʲ = ᶜKʲs.:($j)
            ᶠu₃ʲ = Y.f.sgsʲs.:($j).u₃
            ᶜspecificʲ = ᶜspecificʲs.:($j)
            ᶜtsʲ = ᶜtsʲs.:($j)
            ᶜρʲ = ᶜρʲs.:($j)
            set_velocity_quantities!(ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠu₃ʲ, Y.c.uₕ, ᶠuₕ³)
            @. ᶜtsʲ = ts_sgs(thermo_args..., ᶜspecificʲ, ᶜKʲ, ᶜΦ, ᶜp)
            @. ᶜρʲ = TD.air_density(thermo_params, ᶜtsʲ)

            # When ᶜe_intʲ = ᶜe_int and ᶜq_totʲ = ᶜq_tot, we still observe that
            # ᶜρʲ != ᶜρ. This is because the conversion from ᶜρ to ᶜp to ᶜρʲ
            # introduces a tiny round-off error of order epsilon to ᶜρʲ. If left
            # unchecked, this round-off error then changes the tendency of ᶠu₃ʲ,
            # which in turn introduces an error to ᶠu³ʲ, which then increases
            # the error in ᶜρʲ. For now, we will filter ᶜρʲ to fix this. Note
            # that this will no longer be necessary after we add diffusion.
            @. ᶜρʲ = ifelse(abs(ᶜρʲ - Y.c.ρ) <= 2 * eps(Y.c.ρ), Y.c.ρ, ᶜρʲ)
        end
    end

    return nothing
end

"""
    diagnostic_sgs_quantities(Y, p, t)

Allocates, sets, and returns `ᶜspecific⁺`, `ᶠu₃⁺`, `ᶜu⁺`, `ᶠu³⁺`, `ᶜK⁺`, `ᶜts⁺`,
`ᶜa⁺`, and `ᶜa⁰` in a way that is consistent with `set_precomputed_quantities!`.
This function assumes that `set_precomputed_quantities!` has already been
called.
"""
function diagnostic_sgs_quantities(Y, p, t)
    (; energy_form, moisture_model, turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    thermo_args = (thermo_params, energy_form, moisture_model)
    (; ᶜp, ᶜρa⁰, ᶜρ⁰, ᶜΦ) = p
    ᶠuₕ³ = p.ᶠtemp_CT3
    set_ᶠuₕ³!(ᶠuₕ³, Y)
    ᶜspecific⁺ = @. specific_sgs⁺(Y.c, turbconv_model)
    (ᶠu₃⁺, ᶜu⁺, ᶠu³⁺, ᶜK⁺) = similar.((p.ᶠu₃⁰, p.ᶜu⁰, p.ᶠu³⁰, p.ᶜK⁰))
    set_sgs_ᶠu₃!(u₃⁺, ᶠu₃⁺, Y, turbconv_model)
    set_velocity_quantities!(ᶜu⁺, ᶠu³⁺, ᶜK⁺, ᶠu₃⁺, Y.c.uₕ, ᶠuₕ³)
    ᶜts⁺ = @. ts_sgs(thermo_args..., ᶜspecific⁺, ᶜK⁺, ᶜΦ, ᶜp)
    ᶜa⁺ = @. ρa⁺(Y.c) / TD.air_density(thermo_params, ᶜts⁺)
    ᶜa⁰ = @. ᶜρa⁰ / ᶜρ⁰
    return (; ᶜspecific⁺, ᶠu₃⁺, ᶜu⁺, ᶠu³⁺, ᶜK⁺, ᶜts⁺, ᶜa⁺, ᶜa⁰)
end
