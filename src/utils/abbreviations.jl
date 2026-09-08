#####
##### Shorthands for ClimaCore vector types and operators
#####
##### These abbreviations are used throughout ClimaAtmos, so the naming
##### convention is worth learning once:
#####
##### - `C<i>` is a `Covariant<i>Vector` and `CT<i>` a `Contravariant<i>Vector`;
#####   the digits list the components carried, e.g. `C12` is horizontal and
#####   `C3` vertical. `UVec`, `VVec`, `WVec`, `UV`, and `UVW` are the
#####   corresponding local (physical) vectors.
##### - A `ᶜ` prefix means the operator *outputs* cell centers, a `ᶠ` prefix
#####   cell faces. An operator therefore reads the opposite staggering:
#####   `ᶜdivᵥ` is face-to-center, `ᶠgradᵥ` center-to-face.
##### - A `ᵥ` suffix marks a vertical finite-difference operator and a `ₕ`
#####   suffix a horizontal spectral-element operator; a leading `w` on a
#####   horizontal operator selects its weak form.
##### - A `_matrix` suffix is the `MatrixFields.operator_matrix` of the
#####   operator, i.e. its linearization, used to assemble Jacobian blocks.
#####
##### The vertical operators differ mainly in their boundary conditions, which
##### is what the individual docstrings below record.
#####

using ClimaCore: Geometry, Operators, MatrixFields
import ClimaCore

# Alternatively, we could use Vec₁₂₃, Vec³, etc., if that is more readable.
"""
    C1, C2, C12, C3, C123
    CT1, CT2, CT12, CT3, CT123
    UVec, VVec, WVec, UV, UVW

Shorthands for the `ClimaCore.Geometry` vector types.

`C<i>` and `CT<i>` are covariant and contravariant vectors whose digits list the
components carried, so `C12` is horizontal and `C3` vertical. The `U`/`V`/`W`
names are the local physical vectors, with components in [m/s] for velocities.
Each name is also the constructor of its vector type, so it doubles as a
conversion, e.g. `C12(u)`.
"""
const C1 = Geometry.Covariant1Vector
const C2 = Geometry.Covariant2Vector
const C12 = Geometry.Covariant12Vector
const C3 = Geometry.Covariant3Vector
const C123 = Geometry.Covariant123Vector
const CT1 = Geometry.Contravariant1Vector
const CT2 = Geometry.Contravariant2Vector
const CT12 = Geometry.Contravariant12Vector
const CT3 = Geometry.Contravariant3Vector
const CT123 = Geometry.Contravariant123Vector
const UVec = Geometry.UVector
const VVec = Geometry.VVector
const WVec = Geometry.WVector
const UV = Geometry.UVVector
const UVW = Geometry.UVWVector

"""
    divₕ
    wdivₕ
    split_divₕ
    gradₕ
    wgradₕ
    curlₕ
    wcurlₕ

Horizontal spectral-element operators: divergence, gradient, and curl, each in a
strong form and (with the `w` prefix) a weak form.

`split_divₕ(ρu, χ)` is the entropy-stable split-form discretization of
`∇ₕ·(ρu χ)`, used for the horizontal advection of scalars.
"""
const divₕ = Operators.Divergence()
const wdivₕ = Operators.WeakDivergence()
const split_divₕ = Operators.SplitDivergence()
const gradₕ = Operators.Gradient()
const wgradₕ = Operators.WeakGradient()
const curlₕ = Operators.Curl()
const wcurlₕ = Operators.WeakCurl()

"""
    ᶜinterp
    ᶜdivᵥ
    ᶜgradᵥ

Face-to-center vertical interpolation, divergence, and gradient, without
boundary conditions.

The boundary values of the operand are used as they are, so these operators are
appropriate only where the operand is already defined on the boundary faces;
see `ᶜadvdivᵥ`, `ᶜprecipdivᵥ`, and `ᶜdiffdivᵥ` for the constrained divergences.
"""
const ᶜinterp = Operators.InterpolateF2C()
const ᶜdivᵥ = Operators.DivergenceF2C()
const ᶜgradᵥ = Operators.GradientF2C()

"""
    ᶜadvdivᵥ

Face-to-center vertical divergence of an advective flux, with zero flux imposed
at the top and bottom faces.

Mass and tracers are not advected through the model top or through the surface,
so their advective flux divergence uses this operator rather than the
unconstrained `ᶜdivᵥ`.
"""
const ᶜadvdivᵥ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(CT3(0)),
    top = Operators.SetValue(CT3(0)),
)

"""
    ᶜprecipdivᵥ

Face-to-center vertical divergence of a precipitation flux, with zero flux at
the top and free outflow at the bottom.

Leaving the bottom boundary unconstrained lets the extrapolated interior flux
carry precipitation out of the domain; `ᶠtop_bias` supplies that
reconstruction.
"""
const ᶜprecipdivᵥ = Operators.DivergenceF2C(top = Operators.SetValue(CT3(0)))

"""
    ᶜdiffdivᵥ

Face-to-center vertical divergence of a diffusive scalar flux, with zero flux at
the top and bottom faces.

Surface fluxes are added as explicit tendencies rather than through this
boundary condition.
"""
const ᶜdiffdivᵥ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(C3(0)),
    top = Operators.SetValue(C3(0)),
)

"""
    ᶠdiffdivᵥ_u₃

Center-to-face vertical divergence of the diffusive flux of vertical momentum
`u₃`, with zero divergence imposed at the top and bottom faces.
"""
const ᶠdiffdivᵥ_u₃ = Operators.DivergenceC2F(
    bottom = Operators.SetDivergence(0),
    top = Operators.SetDivergence(0),
)

"""
    ᶠbottom_bias
    ᶠtop_bias
    ᶜbottom_bias
    ᶜtop_bias

One-sided reconstructions that take the value from the neighbor below
(`bottom`) or above (`top`) the target point.

`ᶠtop_bias` also supplies the free-outflow boundary reconstruction used with
`ᶜprecipdivᵥ`.
"""
const ᶠbottom_bias = Operators.BottomBiasedC2F()
const ᶠtop_bias = Operators.TopBiasedC2F() # for free outflow in ᶜprecipdivᵥ
const ᶜbottom_bias = Operators.BottomBiasedF2C()
const ᶜtop_bias = Operators.TopBiasedF2C()

# TODO: Implement proper extrapolation instead of simply reusing the first
# interior value at the surface.
"""
    ᶠinterp

Center-to-face interpolation, extrapolating to the top and bottom boundary
faces by reusing the nearest interior value.
"""
const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

"""
    ᶠwinterp

Center-to-face interpolation weighted by a first argument (typically a volume or
mass weight), with the same boundary extrapolation as `ᶠinterp`.
"""
const ᶠwinterp = Operators.WeightedInterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

# TODO: Replace these boundary conditions with NaN's, since they are
# meaningless and we only need to specify them in order to be able to
# materialize broadcasts. Any effect these boundary conditions have on the
# boundary values of Y.f.u₃ is overwritten when we call set_velocity_at_surface!.
# Ideally, we would enforce the boundary conditions on Y.f.u₃ by filtering it
# immediately after adding the tendency to it. However, this is not currently
# possible because our implicit solver is unable to handle filtering, which is
# why these boundary conditions are 0's rather than NaN's.
"""
    ᶠgradᵥ

Center-to-face vertical gradient, with zero gradient at the top and bottom
faces.

The boundary values are placeholders needed only so that broadcasts can be
materialized; see the comment above this definition.
"""
const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(C3(0)),
    top = Operators.SetGradient(C3(0)),
)

"""
    ᶠcurlᵥ

Center-to-face vertical curl, with zero curl at the top and bottom faces.
"""
const ᶠcurlᵥ = Operators.CurlC2F(
    bottom = Operators.SetCurl(CT12(0, 0)),
    top = Operators.SetCurl(CT12(0, 0)),
)

"""
    ᶠupwind1

First-order upwind reconstruction of the product of a center scalar with a face
velocity.
"""
const ᶠupwind1 = Operators.UpwindBiasedProductC2F()

"""
    ᶠupwind3

Third-order upwind-biased reconstruction of the product of a center scalar with
a face velocity, with the interior stencil's ghost points at the top and bottom
boundaries filled by linear extrapolation from the interior.
"""
const ᶠupwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
    bottom = Operators.Extrapolate{1}(),
    top = Operators.Extrapolate{1}(),
)

"""
    ᶠlin_vanleer

Linear van Leer reconstruction of the product of a center scalar with a face
velocity, with the `MonotoneLocalExtrema` (Mono5) slope constraint and
closest-value ghost points at the boundaries.
"""
const ᶠlin_vanleer = Operators.LinVanLeerC2F(
    bottom = Operators.Extrapolate{0}(),
    top = Operators.Extrapolate{0}(),
    constraint = Operators.MonotoneLocalExtrema(), # (Mono5)
)

"""
    ᶜinterp_matrix
    ᶜbottom_bias_matrix
    ᶜtop_bias_matrix
    ᶜdivᵥ_matrix
    ᶜadvdivᵥ_matrix
    ᶜprecipdivᵥ_matrix
    ᶠtop_bias_matrix
    ᶠinterp_matrix
    ᶠwinterp_matrix
    ᶠgradᵥ_matrix
    ᶠupwind1_matrix
    ᶠupwind3_matrix

Operator matrices of the correspondingly named vertical operators, produced by
`MatrixFields.operator_matrix`.

Applying one of these to a field yields the banded matrix that represents the
linear action of the operator, including its boundary conditions. They are the
building blocks of the implicit Jacobian in `manual_sparse_jacobian.jl`.
"""
const ᶜinterp_matrix = MatrixFields.operator_matrix(ᶜinterp)
const ᶜbottom_bias_matrix = MatrixFields.operator_matrix(ᶜbottom_bias)
const ᶜtop_bias_matrix = MatrixFields.operator_matrix(ᶜtop_bias)
const ᶜdivᵥ_matrix = MatrixFields.operator_matrix(ᶜdivᵥ)
const ᶜadvdivᵥ_matrix = MatrixFields.operator_matrix(ᶜadvdivᵥ)
const ᶜprecipdivᵥ_matrix = MatrixFields.operator_matrix(ᶜprecipdivᵥ)
const ᶠtop_bias_matrix = MatrixFields.operator_matrix(ᶠtop_bias)
const ᶠinterp_matrix = MatrixFields.operator_matrix(ᶠinterp)
const ᶠwinterp_matrix = MatrixFields.operator_matrix(ᶠwinterp)
const ᶠgradᵥ_matrix = MatrixFields.operator_matrix(ᶠgradᵥ)
const ᶠupwind1_matrix = MatrixFields.operator_matrix(ᶠupwind1)
const ᶠupwind3_matrix = MatrixFields.operator_matrix(ᶠupwind3)

"""
    u_component(u::Geometry.LocalVector)
    v_component(u::Geometry.LocalVector)
    w_component(u::Geometry.LocalVector)

Extract the zonal, meridional, or vertical physical component of a local
velocity vector [m/s].
"""
u_component(u::Geometry.LocalVector) = u.u
v_component(u::Geometry.LocalVector) = u.v
w_component(u::Geometry.LocalVector) = u.w
