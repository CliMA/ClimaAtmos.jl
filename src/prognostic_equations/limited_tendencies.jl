using LinearAlgebra: ×, norm, dot

import ClimaAtmos.Parameters as CAP
using ClimaCore: Operators, Fields, Limiters, Geometry, Spaces

import ClimaComms
using ClimaCore.Geometry: ⊗

import Thermodynamics as TD

using ClimaCore.Utilities: half

import ClimaCore.Fields: ColumnField

function limited_tendency!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    set_precomputed_quantities!(Y, p, t)
    horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)
    NVTX.@range "tracer hyperdiffusion tendency" color = colorant"yellow" begin
        tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    end
end

function limiters_func!(Y, p, t, ref_Y)
    (; limiter) = p
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    if !isnothing(limiter)
        for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
            Limiters.compute_bounds!(limiter, ref_Y.c.:($ρχ_name), ref_Y.c.ρ)
            Limiters.apply_limiter!(Y.c.:($ρχ_name), Y.c.ρ, limiter)
        end
        if p.atmos.turbconv_model isa EDMFX
            for j in 1:n
                for ρaχ_name in
                    filter(is_tracer_var, propertynames(Y.c.sgsʲs.:($j)))
                    ᶜρaχ_ref = ref_Y.c.sgsʲs.:($j).:($ρaχ_name)
                    ᶜρa_ref = ref_Y.c.sgsʲs.:($j).ρa
                    ᶜρaχ = Y.c.sgsʲs.:($j).:($ρaχ_name)
                    ᶜρa = Y.c.sgsʲs.:($j).ρa
                    Limiters.compute_bounds!(limiter, ᶜρaχ_ref, ᶜρa_ref)
                    Limiters.apply_limiter!(ᶜρaχ, ᶜρa, limiter)
                end
            end
        end
    end
end
