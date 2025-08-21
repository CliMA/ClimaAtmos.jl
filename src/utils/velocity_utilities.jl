"""
    Velocity computation utilities for ClimaAtmos

This module provides functions to compute velocity quantities on demand
instead of storing them as precomputed variables.
"""

import LazyBroadcast: lazy
import LinearAlgebra: norm_sqr

"""
    compute_ᶠuₕ³(ᶜuₕ, ᶜρ)

Computes the third contravariant component of horizontal velocity on cell faces.
"""
function compute_ᶠuₕ³(ᶜuₕ, ᶜρ)
    ᶜJ = Fields.local_geometry_field(ᶜρ).J
    return @. lazy(ᶠwinterp(ᶜρ * ᶜJ, CT3(ᶜuₕ)))
end

"""
    compute_ᶜu(Y, ᶠuₕ³)

Computes the covariant velocity on cell centers from the horizontal velocity
and vertical velocity components.
"""
function compute_ᶜu(Y, ᶠuₕ³)
    return @. C123(Y.c.uₕ) + ᶜinterp(C123(Y.f.u₃))
end

"""
    compute_ᶠu³(Y, ᶠuₕ³)

Computes the contravariant velocity on cell faces from the horizontal velocity
and vertical velocity components.
"""
function compute_ᶠu³(Y, ᶠuₕ³)
    return @. ᶠuₕ³ + CT3(Y.f.u₃)
end

"""
    compute_ᶜK(ᶜu)

Computes the kinetic energy on cell centers from the velocity field.
"""
function compute_ᶜK(ᶜu)
    return @. norm_sqr(ᶜu) / 2
end

"""
    compute_ᶠu(Y, ᶜu, ᶠu³)

Computes the full velocity on faces from the cell-center velocity and
face velocity components.
"""
function compute_ᶠu(Y, ᶜu, ᶠu³)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    return @. CT123(ᶠwinterp(Y.c.ρ * ᶜJ, CT12(ᶜu))) + CT123(ᶠu³)
end

"""
    compute_sgs_velocity_quantities(Y, ᶠuₕ³, turbconv_model, j)

Computes the subgrid-scale velocity quantities for a specific mass-flux subdomain.
"""
function compute_sgs_velocity_quantities(Y, ᶠuₕ³, turbconv_model, j)
    ᶜuʲ = @. C123(Y.c.uₕ) + ᶜinterp(C123(Y.f.sgsʲs.:($j).u₃))
    ᶠu³ʲ = @. ᶠuₕ³ + CT3(Y.f.sgsʲs.:($j).u₃)
    ᶜKʲ = @. norm_sqr(ᶜuʲ) / 2
    return ᶜuʲ, ᶠu³ʲ, ᶜKʲ
end

"""
    compute_environment_velocity_quantities(Y, ᶠuₕ³, turbconv_model)

Computes the environment velocity quantities for EDMFX models.
"""
function compute_environment_velocity_quantities(Y, ᶠuₕ³, turbconv_model)
    if turbconv_model isa EDOnlyEDMFX
        ᶜu⁰ = @. C123(Y.c.uₕ) + ᶜinterp(C123(Y.f.u₃))
        ᶠu³⁰ = @. ᶠuₕ³ + CT3(Y.f.u₃)
        ᶜK⁰ = @. norm_sqr(ᶜu⁰) / 2
    else
        # For other EDMFX models, compute from sgs velocities
        ᶜu⁰ = @. C123(Y.c.uₕ) + ᶜinterp(C123(Y.f.u₃))
        ᶠu³⁰ = @. ᶠuₕ³ + CT3(Y.f.u₃)
        ᶜK⁰ = @. norm_sqr(ᶜu⁰) / 2
    end
    return ᶜu⁰, ᶠu³⁰, ᶜK⁰
end 
