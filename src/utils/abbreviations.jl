using ClimaCore: Geometry, Operators
using ClimaCore.MatrixFields

# Alternatively, we could use Vec₁₂₃, Vec³, etc., if that is more readable.
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
const UVW = Geometry.UVWVector

const divₕ = Operators.Divergence()
const wdivₕ = Operators.WeakDivergence()
const gradₕ = Operators.Gradient()
const wgradₕ = Operators.WeakGradient()
const curlₕ = Operators.Curl()
const wcurlₕ = Operators.WeakCurl()

const ᶜinterp = Operators.InterpolateF2C()
const ᶜdivᵥ = Operators.DivergenceF2C()
const ᶜgradᵥ = Operators.GradientF2C()

# Tracers do not have advective fluxes through the top and bottom cell faces.
const ᶜadvdivᵥ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(CT3(0)),
    top = Operators.SetValue(CT3(0)),
)

# Precipitation has no flux at the top, but it has free outflow at the bottom.
const ᶜprecipdivᵥ = Operators.DivergenceF2C(top = Operators.SetValue(CT3(0)))

const ᶠright_bias = Operators.RightBiasedC2F() # for free outflow in ᶜprecipdivᵥ

# TODO: Implement proper extrapolation instead of simply reusing the first
# interior value at the surface.
const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
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
const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(C3(0)),
    top = Operators.SetGradient(C3(0)),
)
const ᶠcurlᵥ = Operators.CurlC2F(
    bottom = Operators.SetCurl(CT12(0, 0)),
    top = Operators.SetCurl(CT12(0, 0)),
)

const ᶠupwind1 = Operators.UpwindBiasedProductC2F()
const ᶠupwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
    bottom = Operators.ThirdOrderOneSided(),
    top = Operators.ThirdOrderOneSided(),
)
const ᶠfct_boris_book = Operators.FCTBorisBook(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
)
const ᶠfct_zalesak = Operators.FCTZalesak(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
)

const ᶜinterp_matrix = MatrixFields.operator_matrix(ᶜinterp)
const ᶜdivᵥ_matrix = MatrixFields.operator_matrix(ᶜdivᵥ)
const ᶜadvdivᵥ_matrix = MatrixFields.operator_matrix(ᶜadvdivᵥ)
const ᶜprecipdivᵥ_matrix = MatrixFields.operator_matrix(ᶜprecipdivᵥ)
const ᶠright_bias_matrix = MatrixFields.operator_matrix(ᶠright_bias)
const ᶠinterp_matrix = MatrixFields.operator_matrix(ᶠinterp)
const ᶠwinterp_matrix = MatrixFields.operator_matrix(ᶠwinterp)
const ᶠgradᵥ_matrix = MatrixFields.operator_matrix(ᶠgradᵥ)
const ᶠupwind1_matrix = MatrixFields.operator_matrix(ᶠupwind1)
const ᶠupwind3_matrix = MatrixFields.operator_matrix(ᶠupwind3)

# Helper functions to extract components of vectors
u_component(u::Geometry.LocalVector) = u.u
v_component(u::Geometry.LocalVector) = u.v
w_component(u::Geometry.LocalVector) = u.w
