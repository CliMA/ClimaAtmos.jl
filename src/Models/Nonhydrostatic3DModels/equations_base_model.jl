@inline function rhs_base_model!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

function rhs_base_model!(
    dY,
    Y,
    Ya,
    t,
    p,
    Φ,
    ::AdvectiveForm,
    bc_base,
    params,
    hyperdiffusivity,
    flux_correction,
    FT,
)
    # relevant parameters
    Ω::FT = CLIMAParameters.Planet.Omega(params)
    g::FT = CLIMAParameters.Planet.grav(params)
    κ₄::FT = hyperdiffusivity

    # unpack boundary boundary_conditions
    bc_ρ = bc_base.ρ
    bc_uh = bc_base.uh
    bc_w = bc_base.w

    # operators
    # spectral horizontal operators
    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.WeakGradient()
    hcurl = Operators.Curl()
    hwcurl = Operators.WeakCurl()

    # vertical FD operators with BC's
    # interpolators
    interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    interp_f2c = Operators.InterpolateF2C()

    # flux correction aka upwinding
    flux_correction_center = Operators.FluxCorrectionC2C(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    # base components
    dYm = dY.base
    dρ = dYm.ρ # scalar on centers
    duh = dYm.uh # Covariant12Vector on centers
    dw = dYm.w # Covariant3Vector on faces
    Ym = Y.base
    ρ = Ym.ρ
    uh = Ym.uh
    w = Ym.w

    # TODO!: Initialize all tendencies to zero for good practice!

    # hyperdiffusion
    χuh = @. duh =
        hwgrad(hdiv(uh)) -
        Geometry.Covariant12Vector(hwcurl(Geometry.Covariant3Vector(hcurl(uh))),)
    Spaces.weighted_dss!(duh)
    @. duh =
        -κ₄ * (
            hwgrad(hdiv(χuh)) -
            Geometry.Covariant12Vector(hwcurl(Geometry.Covariant3Vector(hcurl(
                χuh,
            ))),)
        )

    # density
    # the vector is split into horizontal and vertical components so that they can be
    # applied individually
    flux_bottom = get_boundary_flux(bc_ρ.bottom, ρ, Y, Ya)
    flux_top = get_boundary_flux(bc_ρ.top, ρ, Y, Ya)
    vector_vdiv_f2c = Operators.DivergenceF2C(
        top = Operators.SetValue(flux_top),
        bottom = Operators.SetValue(flux_bottom),
    )
    uvw = @. Geometry.Covariant123Vector(uh) +
       Geometry.Covariant123Vector(interp_f2c(w))
    @. dρ = -hdiv(ρ * uvw) # horizontal divergence
    @. dρ -= vector_vdiv_f2c(interp_c2f(ρ * uh)) # explicit vertical part
    @. dρ -= vector_vdiv_f2c(interp_c2f(ρ) * w) # TODO: implicit vertical part

    # horizontal momentum
    # TODO: hmm, same problem as in the 2d model
    flux_bottom = get_boundary_flux(bc_uh.bottom, uh, Y, Ya)
    flux_top = get_boundary_flux(bc_uh.top, uh, Y, Ya)
    vector_vcurl_c2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant12Vector(
            FT(0),
            FT(0),
        )),
        top = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
    )

    ω³ = @. hcurl(uh) # Contravariant3Vector
    ω¹² = @. hcurl(w) # Contravariant12Vector
    @. ω¹² += vector_vcurl_c2f(uh) # Contravariant12Vector
    u¹² = # TODO!: Will need to be changed with topography
        @. Geometry.Contravariant12Vector(Geometry.Covariant123Vector(interp_c2f(
            uh,
        )),) # Contravariant12Vector in 3D
    u³ = @. Geometry.Contravariant3Vector(Geometry.Covariant123Vector(w))  # TODO!: Will need to be changed with topography
    # coriolis
    if Ω != 0
        lat = Fields.coordinate_field(axes(ρ)).lat
        f = @. Geometry.Contravariant3Vector(Geometry.WVector(
            2 * Ω * sind(lat),
        ))  # TODO!: Will need to be changed with topography
    else
        y = Fields.coordinate_field(axes(ρ)).y
        f = @. Geometry.Contravariant3Vector(Geometry.WVector(Ω * y))
    end
    E = @. (norm(uvw)^2) / 2 + Φ

    @. duh -= interp_f2c(ω¹² × u³)
    @. duh -=
        (f + ω³) ×
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(uh))
    @. duh -= hgrad(p) / ρ
    @. duh -= hgrad(E)

    # vertical momentum
    flux_bottom = get_boundary_flux(bc_w.bottom, w, Y, Ya)
    flux_top = get_boundary_flux(bc_w.top, w, Y, Ya)
    # TODO: hack in a flux bc??
    scalar_vgrad_c2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
        top = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    )
    @. dw = ω¹² × u¹² # Covariant3Vector on faces
    @. dw -= scalar_vgrad_c2f(p) / interp_c2f(ρ)
    @. dw -= scalar_vgrad_c2f(E)

    if flux_correction
        @. dρ += flux_correction_center(w, ρ)
    end

    # discrete stiffness summation for spectral operations
    Spaces.weighted_dss!(dρ)
    Spaces.weighted_dss!(duh)
    Spaces.weighted_dss!(dw)

    return dY
end
