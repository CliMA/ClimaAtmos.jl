#####
##### Advection
#####

using LinearAlgebra: ×, dot
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

function horizontal_advection_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜu, ᶜK, ᶜp, ᶜΦ, ᶜp_ref) = p
    if p.atmos.turbconv_model isa EDMFX
        (; ᶜu⁰, ᶜuʲs) = p
    end

    @. Yₜ.c.ρ -= divₕ(Y.c.ρ * ᶜu)
    if p.atmos.turbconv_model isa EDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).ρa -= divₕ(Y.c.sgsʲs.:($$j).ρa * ᶜuʲs.:($$j))
        end
    end

    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ -= divₕ(Y.c.ρθ * ᶜu)
    elseif :ρe_tot in propertynames(Y.c)
        (; ᶜh_tot) = p
        @. Yₜ.c.ρe_tot -= divₕ(Y.c.ρ * ᶜh_tot * ᶜu)
    end
    if p.atmos.turbconv_model isa EDMFX
        for j in 1:n
            if :ρθ in propertynames(Y.c)
                @. Yₜ.c.sgsʲs.:($$j).ρaθ -=
                    divₕ(Y.c.sgsʲs.:($$j).ρaθ * ᶜuʲs.:($$j))
            elseif :ρe_tot in propertynames(Y.c)
                (; ᶜh_totʲs) = p
                @. Yₜ.c.sgsʲs.:($$j).ρae_tot -=
                    divₕ(Y.c.sgsʲs.:($$j).ρa * ᶜh_totʲs.:($$j) * ᶜuʲs.:($$j))
            end
        end
    end

    if p.atmos.turbconv_model isa EDMFX
        @. Yₜ.c.sgs⁰.ρatke -= divₕ(Y.c.sgs⁰.ρatke * ᶜu⁰)
    end

    @. Yₜ.c.uₕ -= C12(gradₕ(ᶜp - ᶜp_ref) / Y.c.ρ + gradₕ(ᶜK + ᶜΦ))
    # Without the C12(), the right-hand side would be a C1 or C2 in 2D space.
end

function horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜu) = p
    if p.atmos.turbconv_model isa EDMFX
        (; ᶜuʲs) = p
    end

    for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
        @. Yₜ.c.:($$ρχ_name) -= divₕ(Y.c.:($$ρχ_name) * ᶜu)
    end

    if p.atmos.turbconv_model isa EDMFX
        for j in 1:n
            for ρaχ_name in
                filter(is_tracer_var, propertynames(Y.c.sgsʲs.:($j)))
                @. Yₜ.c.sgsʲs.:($$j).:($$ρaχ_name) -=
                    divₕ(Y.c.sgsʲs.:($$j).:($$ρaχ_name) * ᶜuʲs.:($$j))
            end
        end
    end
end

function explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    point_type = eltype(Fields.coordinate_field(Y.c))
    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶜu, ᶠu³, ᶜK, ᶜf) = p
    ᶜω³ = p.ᶜtemp_CT3
    ᶠω¹² = p.ᶠtemp_CT12
    if p.atmos.turbconv_model isa EDMFX
        ᶠω¹²ʲs = p.ᶠtemp_CT12ʲs
    end

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = curlₕ(Y.c.uₕ)
    elseif point_type <: Geometry.Abstract2DPoint
        @. ᶜω³ = zero(ᶜω³)
    end

    @. ᶠω¹² = ᶠcurlᵥ(Y.c.uₕ)
    if p.atmos.turbconv_model isa EDMFX
        for j in 1:n
            @. ᶠω¹²ʲs.:($$j) = ᶠω¹²
        end
    end
    @. ᶠω¹² += CT12(curlₕ(Y.f.u₃))
    if p.atmos.turbconv_model isa EDMFX
        for j in 1:n
            @. ᶠω¹²ʲs.:($$j) += CT12(curlₕ(Y.f.sgsʲs.:($$j).u₃))
        end
    end
    # Without the CT12(), the right-hand side would be a CT1 or CT2 in 2D space.

    @. Yₜ.c.uₕ -=
        ᶜinterp(ᶠω¹² × (ᶠinterp(Y.c.ρ * ᶜJ) * ᶠu³)) / (Y.c.ρ * ᶜJ) +
        (ᶜf + ᶜω³) × CT12(ᶜu)
    @. Yₜ.f.u₃ -= ᶠω¹² × ᶠinterp(CT12(ᶜu)) + ᶠgradᵥ(ᶜK)

    if p.atmos.turbconv_model isa EDMFX
        (; ᶜp, ᶜp_ref, ᶜρ_ref, ᶠgradᵥ_ᶜΦ) = p
        (; ᶠu³ʲs, ᶜuʲs, ᶜKʲs, ᶜρʲs, ᶜspecificʲs, edmfx_upwinding) = p
        (; dt) = p.simulation
        ᶜ1 = p.ᶜtemp_scalar
        ᶜh_totʲs = p.atmos.energy_form isa TotalEnergy ? p.ᶜh_totʲs : nothing

        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                ᶠω¹²ʲs.:($$j) × ᶠinterp(CT12(ᶜuʲs.:($$j))) + ᶠgradᵥ(ᶜKʲs.:($$j))
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                (
                    ᶠgradᵥ(ᶜp - ᶜp_ref) +
                    ᶠinterp(ᶜρʲs.:($$j) - ᶜρ_ref) * ᶠgradᵥ_ᶜΦ
                ) / ᶠinterp(ᶜρʲs.:($$j))
        end

        Fields.bycolumn(axes(Y.c.ρ)) do colidx
            for j in 1:n

                @. ᶜ1[colidx] = one(Y.c.ρ[colidx])
                vertical_transport!(
                    Yₜ.c.sgsʲs.:($j).ρa[colidx],
                    ᶜJ[colidx],
                    Y.c.sgsʲs.:($j).ρa[colidx],
                    ᶠu³ʲs.:($j)[colidx],
                    ᶜ1[colidx],
                    dt,
                    edmfx_upwinding,
                )

                if :ρae_tot in propertynames(Yₜ.c.sgsʲs.:($j))
                    vertical_transport!(
                        Yₜ.c.sgsʲs.:($j).ρae_tot[colidx],
                        ᶜJ[colidx],
                        Y.c.sgsʲs.:($j).ρa[colidx],
                        ᶠu³ʲs.:($j)[colidx],
                        ᶜh_totʲs.:($j)[colidx],
                        dt,
                        edmfx_upwinding,
                    )
                end

                for (ᶜρaχʲₜ, ᶜχʲ, χ_name) in
                    matching_subfields(Yₜ.c.sgsʲs.:($j), ᶜspecificʲs.:($j))
                    χ_name == :e_tot && continue
                    vertical_transport!(
                        ᶜρaχʲₜ[colidx],
                        ᶜJ[colidx],
                        Y.c.sgsʲs.:($j).ρa[colidx],
                        ᶠu³ʲs.:($j)[colidx],
                        ᶜχʲ[colidx],
                        dt,
                        edmfx_upwinding,
                    )
                end
            end
        end
    end
end
