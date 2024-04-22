
function set_stochastic_entrainment_with_exponential_solver!(integrator)
    (; params, precomputed, atmos, scratch) = integrator.p
    sde_params = params.stochastic_params.entr
    (; ᶜentrʲs) = precomputed
    n = n_mass_flux_subdomains(atmos.turbconv_model)
    for colⱼ in 1:n
        entr_μ = scratch.ᶜtemp_scalar_3
        entr_μ = get_deterministic_entrainment(integrator, colⱼ)
        AR1_step!(ᶜentrʲs, sde_params, integrator, colⱼ, entr_μ)
    end
end

function set_stochastic_detrainment_with_exponential_solver!(integrator)
    (; params, precomputed, atmos, scratch) = integrator.p
    sde_params = params.stochastic_params.detr
    (; ᶜdetrʲs) = precomputed
    n = n_mass_flux_subdomains(atmos.turbconv_model)
    for colⱼ in 1:n
        detr_μ = scratch.ᶜtemp_scalar_3
        detr_μ = get_deterministic_detrainment(integrator, colⱼ)
        AR1_step!(ᶜdetrʲs, sde_params, integrator, colⱼ, detr_μ)
    end
end

"""
AR1_step!(u, sde_params, integrator)

Given the continuous-time OU process, 
    `` du = -τ( u - μ )dt + σ dW_t, ``
solves the corresponding discrete-time AR(1) process, with the "exponential" solve,
    `` u_{n+1} = u_n e^{-τ dt} + μ (1 - e^{-τ dt}) + σ √{ (1 - e^{-2τ dt} ) / (2τ) ) z_n, ``
where `z_n` is a random number drawn from a standard normal distribution.

# Arguments
- `u`: The state to in-place evolve the AR(1) process, e.g. entrainment or detrainment.
- `sde_params`: The AR(1) parameters, a struct/NamedTuple with fields `μ`, `τ`, and `σ`.
- `integrator`: The integrator

Notes:
(;μ, θ, σ) = p
exp_minus_θh = @. exp(-θ * h)
umean = uprev * exp_minus_θh + μ * (1 - exp_minus_θh)
unoise = √(σ^2 / 2θ * (1 - exp_minus_θh^2)) * z
unext = umean + unoise
unext
"""
function AR1_step!(u, sde_params, integrator, colⱼ, μ)
    (; atmos, dt) = integrator.p
    # n = n_mass_flux_subdomains(atmos.turbconv_model)
    nlevels = Spaces.nlevels(axes(integrator.u.c))
    # closure
    # (; μ, τ, σ) = sde_params
    (; τ, σ) = sde_params
    σ_z = σ√( (1 - exp(-2τ * dt)) / 2τ )
    z = integrator.p.scratch.ᶜtemp_scalar
    # for colⱼ in 1:n
    uₙ₊₁ = uₙ = u.:($colⱼ)
    # Set z to a random number drawn from a standard normal distribution, for each column
    for level in 1:nlevels  # Note: only works in SCM
        z_level = Fields.field_values(Fields.level(z, level))
        z_level .= randn()
    end

    @. uₙ₊₁ = uₙ * exp(-τ * dt) + μ * (1 - exp(-τ * dt)) + √uₙ * σ_z * z
    @. uₙ₊₁ = max(uₙ₊₁, 0)  # limit with `limit_entrainment`?
    # end
end


function get_deterministic_entrainment(integrator, colⱼ)
    (; u, p) = integrator
    Y = u
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜtke⁰, ᶜu, ᶜp, ᶜts⁰, ᶜuʲs, ᶜtsʲs, ᶜρʲs) = p.precomputed
    FT = eltype(Y)
    FT = Float32

    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜlg = Fields.local_geometry_field(Y.c)

    ca_entr = ClimaAtmos.GeneralizedEntrainment()
    # create entrainment(..., ::SDEntrModel) = [...] ᶜentrʲs.:($$j) (use if/else to pass entrjs...)
    @. entrainment(
        params,
        ᶜz,
        z_sfc,
        ᶜp,
        Y.c.ρ,
        draft_area(Y.c.sgsʲs.:($$colⱼ).ρa, ᶜρʲs.:($$colⱼ)),
        get_physical_w(ᶜuʲs.:($$colⱼ), ᶜlg),
        TD.relative_humidity(thermo_params, ᶜtsʲs.:($$colⱼ)),
        ᶜphysical_buoyancy(params, Y.c.ρ, ᶜρʲs.:($$colⱼ)),
        get_physical_w(ᶜu, ᶜlg),
        TD.relative_humidity(thermo_params, ᶜts⁰),
        FT(0),
        max(ᶜtke⁰, 0),
        ca_entr, # p.atmos.edmfx_entr_model,
    )
end

function get_deterministic_detrainment(integrator, colⱼ)
    (; u, p) = integrator
    Y = u
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜtke⁰, ᶜu, ᶜp, ᶜts⁰, ᶜuʲs, ᶜtsʲs, ᶠu³ʲs, ᶜρʲs) = p.precomputed
    # FT = eltype(Y)
    FT = Float32

    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜlg = Fields.local_geometry_field(Y.c)

    # sgsʲρa = Y.c.sgsʲs.:($colⱼ).ρa

    ᶜvert_div = p.scratch.ᶜtemp_scalar
    @. ᶜvert_div = ᶜdivᵥ(ᶠinterp(ᶜρʲs.:($$colⱼ)) * ᶠu³ʲs.:($$colⱼ)) / ᶜρʲs.:($$colⱼ)
    ᶜmassflux_vert_div = p.scratch.ᶜtemp_scalar_2
    @. ᶜmassflux_vert_div =
        ᶜdivᵥ(ᶠinterp(Y.c.sgsʲs.:($$colⱼ).ρa) * ᶠu³ʲs.:($$colⱼ))

    zero_scalar = FT(0)

    ca_entr = ClimaAtmos.GeneralizedDetrainment()
    @. detrainment(
        params,
        ᶜz,
        z_sfc,
        ᶜp,
        Y.c.ρ,
        Y.c.sgsʲs.:($$colⱼ).ρa,
        draft_area(Y.c.sgsʲs.:($$colⱼ).ρa, ᶜρʲs.:($$colⱼ)),
        get_physical_w(ᶜuʲs.:($$colⱼ), ᶜlg),
        TD.relative_humidity(thermo_params, ᶜtsʲs.:($$colⱼ)),
        ᶜphysical_buoyancy(params, Y.c.ρ, ᶜρʲs.:($$colⱼ)),
        get_physical_w(ᶜu, ᶜlg),
        TD.relative_humidity(thermo_params, ᶜts⁰),
        zero_scalar,
        zero_scalar,
        ᶜvert_div,
        ᶜmassflux_vert_div,
        ᶜtke⁰,
        ca_entr, # p.atmos.edmfx_detr_model,
    )
end