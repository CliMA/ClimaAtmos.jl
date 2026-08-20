# we are ignoring the volume reflectance for now
"""
    SurfaceAlbedoModel

Strategy for setting the direct and diffuse shortwave surface reflectivities
seen by the radiation scheme (via `set_surface_albedo!`).

Subtypes:

  - [`ConstantAlbedo`](@ref): a single constant albedo for idealized experiments.
  - [`RegressionFunctionAlbedo`](@ref): the ocean-albedo regression of [Jin2011](@cite).
  - [`CouplerAlbedo`](@ref): albedo supplied externally by the coupler.
"""
abstract type SurfaceAlbedoModel end

"""
    CouplerAlbedo()

Surface albedo supplied by an external driver (the coupler), which writes the
direct/diffuse shortwave albedos into the radiation cache. ClimaAtmos performs
no albedo computation of its own in this mode.
"""
struct CouplerAlbedo <: SurfaceAlbedoModel end

"""
    ConstantAlbedo{FT} <: SurfaceAlbedoModel

Spatially and temporally constant surface albedo, used for idealized experiments.

The same value is applied to the direct and diffuse shortwave albedos. The field has
no default: the Julia-API default in `AtmosModel` is `ConstantAlbedo(; α = 0.07)`,
while the YAML configuration path constructs it from the ClimaParams parameter
`idealized_ocean_albedo` (0.38, from O'Gorman and Schneider, 2008).

# Fields

  - `α`: Surface albedo for both direct and diffuse shortwave radiation [-].
"""
@kwdef struct ConstantAlbedo{FT} <: SurfaceAlbedoModel
    α::FT
end

"""
    RegressionFunctionAlbedo{FT}(; n, n0, p, q_clear, q_cloud, wave_slope)

Ocean surface albedo from the regression functions of [Jin2011](@cite) (J11),
with direct and diffuse components computed separately. Volume reflectance
(foam and subsurface scattering) is currently ignored.

# Fields

  - `n`: Relative refractive index of water and air, `n = n_w/n_a` [-].
  - `n0`: Refractive index of water for visible light [-].
  - `p`: Regression coefficients for the direct albedo (J11 eq. 4) [-].
  - `q_clear`: Regression coefficients for the clear-sky diffuse albedo (J11 eq. 5a) [-].
  - `q_cloud`: Regression coefficients for the cloudy-sky diffuse albedo (J11 eq. 5b) [-].
  - `wave_slope`: Function of wind speed returning the mean wave-slope distribution
    width of the Cox-Munk model (J11 eq. 2) [-].

# Constructor

The keyword constructor supplies the J11 regression coefficients as defaults, so
`RegressionFunctionAlbedo{FT}()` is the standard usage.
"""
struct RegressionFunctionAlbedo{FT, F <: Function} <: SurfaceAlbedoModel
    n::FT                           # relative refractive index of water and air (n = n_w/n_a) TODO: f(wavelength) for a spectrally dependent scheme
    n0::FT                          # refractive index of water for visible light
    p::SA.SVector{11, FT}           # regression coefficients of J11 for eq. 4
    q_clear::SA.SVector{4, FT}      # regression coefficients of J11 for eq. 5a
    q_cloud::SA.SVector{3, FT}      # regression coefficients of J11 for eq. 5b
    wave_slope::F                   # mean wave slope distribution width (Cox-Munk model, J11 Eq. 2)
end
function RegressionFunctionAlbedo{FT}(;
    n = FT(1.34),
    n0 = FT(1.34),
    p = SA.SVector{11, FT}(
        0.0152,
        -1.7873,
        6.8972,
        -8.5778,
        4.071,
        -7.6446,
        0.1643,
        -7.8409,
        -3.5639,
        -2.3588,
        10.0538,
    ),
    q_clear = SA.SVector{4, FT}(-0.1482, -0.012, 0.1608, -0.0244),
    q_cloud = SA.SVector{3, FT}(-0.1479, 0.1502, -0.016),
    wave_slope = u -> sqrt(FT(0.003) + FT(0.00512) * u),
) where {FT <: AbstractFloat}
    return RegressionFunctionAlbedo(n, n0, p, q_clear, q_cloud, wave_slope)
end

import RRTMGP

"""
    set_surface_albedo!(Y, p, t, α_model::ConstantAlbedo)

Set the direct and diffuse shortwave surface albedos to the constant `α_model.α`.

Mutates the direct and diffuse albedo arrays of `p.radiation.rrtmgp_solver`.
"""
function set_surface_albedo!(Y, p, t, α_model::ConstantAlbedo{FT}) where {FT}

    direct_sw_surface_albedo = RRTMGP.direct_sw_surface_albedo(p.radiation.rrtmgp_solver)
    diffuse_sw_surface_albedo = RRTMGP.diffuse_sw_surface_albedo(p.radiation.rrtmgp_solver)

    @. direct_sw_surface_albedo = α_model.α
    @. diffuse_sw_surface_albedo = α_model.α
end

"""
    set_surface_albedo!(Y, p, t, α_model::RegressionFunctionAlbedo)

Set the direct and diffuse shortwave surface albedos of the ocean from the
regression functions of [Jin2011](@cite), evaluated at the current solar zenith
angle and near-surface wind speed.

Mutates the direct and diffuse albedo arrays of `p.radiation.rrtmgp_solver` and
uses `p.scratch.temp_field_level` as scratch. Reads the cosine of the solar
zenith angle from the RRTMGP solver and the wind speed from level 1 of `Y.c.uₕ`.
"""
function set_surface_albedo!(
    Y,
    p,
    t,
    α_model::RegressionFunctionAlbedo{FT},
) where {FT}

    direct_sw_surface_albedo = RRTMGP.direct_sw_surface_albedo(p.radiation.rrtmgp_solver)
    diffuse_sw_surface_albedo = RRTMGP.diffuse_sw_surface_albedo(p.radiation.rrtmgp_solver)
    cos_zenith = RRTMGP.cos_zenith(p.radiation.rrtmgp_solver)

    λ = FT(0) # spectral wavelength (not used for now)
    μ = cos_zenith

    surface_albedo = p.scratch.temp_field_level
    f_direct = surface_albedo_direct(α_model)
    surface_albedo .=
        f_direct.(
            λ,
            Fields.array2field(μ, axes(surface_albedo)),
            norm.(Fields.level(Y.c.uₕ, 1)),
        )
    direct_sw_surface_albedo .= Fields.field2array(surface_albedo)'

    f_diffuse = surface_albedo_diffuse(α_model)
    surface_albedo .=
        f_diffuse.(
            λ,
            Fields.array2field(μ, axes(surface_albedo)),
            norm.(Fields.level(Y.c.uₕ, 1)),
        )
    diffuse_sw_surface_albedo .= Fields.field2array(surface_albedo)'
end

"""
    set_surface_albedo!(Y, p, t, ::CouplerAlbedo)

Skip setting the surface albedo, which is handled by the coupler.

To avoid NaNs or invalid values in the first radiation call, the coupler retrieves
the albedo initial conditions from the surface models and provides them to the
atmosphere model before stepping. At `t == 0`, this method initializes the
insolation variables (unless the insolation is `IdealizedInsolation`).
"""
function set_surface_albedo!(Y, p, t, ::CouplerAlbedo)
    FT = eltype(Y)
    if FT(t) == 0
        # set initial insolation initial conditions
        !(p.atmos.insolation isa IdealizedInsolation) &&
            set_insolation_variables!(Y, p, t, p.atmos.insolation)
    else
        nothing
    end
end

"""
    surface_albedo_direct(α_model::RegressionFunctionAlbedo)

Return a function `(λ, cosθ, u) -> α` that computes the direct ocean surface
albedo from the regression function of [Jin2011](@cite) (eqs. 1 and 4), given the
wavelength `λ` (currently unused) [m], the cosine of the solar zenith angle
`cosθ` [-], and the near-surface wind speed `u` [m/s]. The result is clamped to
[0, 1] and is zero when the sun is below the horizon (`cosθ ≤ 0`).
"""
function surface_albedo_direct(α_model::RegressionFunctionAlbedo{FT}) where {FT}
    α_dir =
        (λ, cosθ, u) -> begin
            if cosθ <= 0
                return zero(FT)
            else
                # relative refractive index of water and air (n = n_w/n_a)
                n = α_model.n

                # refractive index of water for visible light
                n0 = α_model.n0

                # mean wave slope distribution width
                σ = α_model.wave_slope(u)

                # Fresnel reflectance (assuming equal contribution of the p-polorized and s-polarized components, and using the perfect dielectric medium approximation)
                sinθ(cosθ) = sqrt(1 - cosθ^2)
                rf_p(n, cosθ) =
                    (
                        (n^2 * cosθ - sqrt(n^2 - sinθ(cosθ)^2)) /
                        (n^2 * cosθ + sqrt(n^2 - sinθ(cosθ)^2))
                    )^2
                rf_s(n, cosθ) =
                    (
                        (cosθ - sqrt(n^2 - sinθ(cosθ)^2)) /
                        (cosθ + sqrt(n^2 - sinθ(cosθ)^2))
                    )^2
                rf(n, cosθ) = (rf_p(n, cosθ) + rf_s(n, cosθ)) / 2

                # regression coefficients
                p = α_model.p

                # the regression function (J11, eq. 4)
                f(cosθ, σ) =
                    (
                        p[1] +
                        p[2] * cosθ +
                        p[3] * cosθ^2 +
                        p[4] * cosθ^3 +
                        p[5] * σ +
                        p[6] * σ * cosθ
                    ) * exp(
                        p[7] +
                        p[8] * cosθ +
                        p[9] * cosθ^2 +
                        p[10] * σ +
                        p[11] * σ * cosθ,
                    )

                # return the albedo (J11, eq. 1)
                return min(
                    one(FT),
                    max(
                        zero(FT),
                        rf(n, cosθ) - rf(n, cosθ) / rf(n0, cosθ) * f(cosθ, σ),
                    ),
                )
            end
        end
    return α_dir
end

"""
    surface_albedo_diffuse(α_model::RegressionFunctionAlbedo)

Return a function `(λ, cosθ, u) -> α` that computes the diffuse ocean surface
albedo from the regression functions of [Jin2011](@cite) (eqs. 5a and 5b), given
the wavelength `λ` (currently unused) [m], the cosine of the solar zenith angle
`cosθ` [-], and the near-surface wind speed `u` [m/s]. The result is clamped to
[0, 1] and is zero when the sun is below the horizon (`cosθ ≤ 0`).

!!! note

    For now the cloud fraction is assumed to be 0, so only the clear-sky
    branch (eq. 5a) contributes.
"""
function surface_albedo_diffuse(
    α_model::RegressionFunctionAlbedo{FT},
) where {FT}
    α_diff =
        (λ, cosθ, u) -> begin
            if cosθ <= 0
                return zero(FT)
            else
                cloud_fraction = 0 # TODO: connect this to the EDMF

                # clear sky (J11, eq. 5a)
                n = α_model.n
                σ = α_model.wave_slope(u)

                q_clear = α_model.q_clear
                α_clear =
                    q_clear[1] +
                    q_clear[2] * σ +
                    q_clear[3] * n +
                    q_clear[4] * n * σ

                # cloudy sky (J11, eq. 5b)
                q_cloud = α_model.q_cloud
                α_cloud = q_cloud[1] + q_cloud[2] * n + q_cloud[3] * n * σ

                # return the albedo
                return min(
                    one(FT),
                    max(
                        zero(FT),
                        cloud_fraction * α_cloud +
                        (1 - cloud_fraction) * α_clear,
                    ),
                )
            end
        end
    return α_diff
end
