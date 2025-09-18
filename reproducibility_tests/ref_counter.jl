264

# **README**
#
# What is the ref_counter?
#
# The ref_counter is part of reproduciability tests. The counter identifies a
# particular snapshot of our code, a "reference". Each PR is tested against this
# reference to check if it reproduces the expected behavior. This allows us to
# check that PRs that are expected to not modify the behavior do indeed preserve
# the previous behavior.
#
# When am I allowed to increase the ref_counter?
#
# If you know that your PR is changing some behavior (e.g., you are changing
# parameters, or how things are computed, or added a new component), you should
# increase the number on top of this file and add an explanation on why it has
# changed in the comments below. Increasing the ref_counter will make your PR
# the new reference that other PRs will be compared against.


#=
264
- Swap vertical averages in reconstructions of uʰ and ωʰ×uᵛ

263
- Make vertical diffusion of updrafts implicit

262
- Add vertical diffusion to prognostic EDMF updrafts

261
- Ignore the implicit solver Jacobian for the tracer-density block from diffusion

260
- Change environment TKE to grid-mean TKE

259
- Fix env boundary values when setting implicit and explicit cache

258
- Update deps, specifically ClimaParams.jl and Thermodynamics.jl

257
- Redefine sedimentation velocity for Prognostic EDMF with 1-moment or 2-moment microphysics
  on the grid scale; fix a bug in EDMFx SGS mass flux.

256
- Fix a bug in EDMF diffusive flux

255
- Add radiation tendency to prognostic edmf updrafts

254
- Use rayleigh and viscous sponges in the prognostic_edmfx_aquaplanet

253
- Update deps, specifically ClimaParams.jl and Thermodynamics.jl

252
- Update deps, specifically RootSolvers.jl

252
- Limit the noise in ice and snow 1M microphysics scheme in cloud formation and
  limiter formulation.

251
- Remove ᶜtke⁰, ᶜh_tot, ᶜmse⁰, ᶜρ⁰, and specific quantities from precomputed quantities cache.
  Introduce helper functions to compute sums over draft, environmental volumetric variables,
  and specific env variables.

250
- Add ARG aerosol activation for 2M microphysics; increase the allocation limit for `flame_callbacks`
  from 391864 to 391942 to account for additional allocations when constructing `ClimaAtmosParameters`
  with the new aerosol parameters in `microphysics_2m_parameters`.

249
- Remove viscosity, diffusivity, and mixing length from precomputed quantities

248
- update deps: climacore 0.14.35

247
- Hyperdiffusion of enthalpy perturbation

246
- Prognostic edmf docstrings + refactor + scm_coriolis change

245
- Add 2M test case to the reproducibility job_id list

244
- Add a new test case: aquaplanet with 2M microphysics

243
- Consolidate eddy diffusivity logic in eddy_diffusion_closures.jl and mass flux
  logic in mass_flux_closures.jl, update buoyancy calculations to be consistent throughout
  (use geopotential gradient instead of g parameter), add helper functions for computing
  diffusivity/viscosity, adds l_grid to MixingLength output struct. Includes more docstrings
  and comments.

242
- Use ustar^3 surface tke boundary condition

241
- Add triangle inequality limiter from Horn (2012) and apply it in microphysics

240
- Fix a bug in viscous sponge

239
- Update mixing length formulation

238
- Limit by Pr_max = 10 (new default in ClimaParams v0.10.30)

237
- Changed the formulation of the Richardson and Prandtl numbers. We are now limiting by
  Pr_max = 100 instead of Ri_max = 0.25. Different behavior for stable and neutral conditions.

236
- Radiation fluxes in runs with the `deep_atmosphere` configuration now take into account the
  column expansion with height. Affects only cases with `DeepSphericalGlobalGeometry`.

235
- Related to #3775, the computation and update of non-orographic gravity wave (NOGW)
  are now separated into callback and tendencies update, affecting the NOGW-related
  tests.

234
- Move the virtual mass term in pressure closure to prognostic edmf momentum equation
  (This only affects prognostic edmf)

233
- modify the derivative of p and \rho with respect to qt

232
- Use lazy broadcasting instead of temp scalar in implicit solver for kappa_m vars,
  which fixes a bug that the temp scalar is updated before it is reused.
  (This only affects prognostic edmf)

231
- Add mass flux derivatives with respect to grid-mean u_3
  (This only affects prognostic edmf)

230
- Add u_{3,m} (updraft) Jacobians to updraft MSE, rho*a, and q_tot prognostic equations. Move sgs ∂ᶠu₃ʲ derivatives to BlockLowerTriangularSolve.
  (This only affects prognostic edmf)

229
- Remove derivatives with respect to grid mean rho in edmf implicit solver
  (This only affects prognostic edmf)

228
- Only treat the drag term in edmf pressure closure implicitly
  (This only affects prognostic edmf)

227
- Move nonhydrostatic pressure drag calculation to implicit precomputed quantities
  (This only affects prognostic edmf)

226
- Add updraft rho*a and u_{3,m} jacobian terms
  (This only affects prognostic edmf)

225
- Move nonhydrostatic pressure drag calculation to precomputed quantities and
  remove one reproducibility job
  (This only affects prognostic edmf)

224
- Machine precision differences due to https://github.com/CliMA/ClimaCore.jl/pull/2232

223
- Add some more jobs for reproductibility tests

222
- We don't know why, but aquaplanet_nonequil_allsky_gw_res and aquaplanet_equil_allsky_gw_raw_zonalasym
  seemed to not produce the same result between the merge-queue branch and the main branch.

221
- Change the way cloud liquid effective radius is computed for radiation

220
- Split out cached variables that should be treated implicitly, so that all
  other cached variables are no longer updated by the implicit solver

219
- Change the operations order in vertical advection upwinding

218
- Change surface flux tendency to fully explicit

217
- Change reconstruction of density on cell faces

216
- Change prescribed aerosols in `aquaplanet_nonequil_allsky_gw_res.yml`

215
- Update dependencies, including ClimaCore, which updated how metric terms are
  computed

214
- Rename some config files

213
- Update to deep-atmos eqns by default, fix vorticity diagnostic

212
- Update RRTMGP, which changes aerosol optics calculation

211
- Remove `FriersonDiffusion` and use `DecayWithHeightDiffusion` instead.

210
- Change prescribed aerosol dataset to MERRA2

209
- Floating point changes, with minor adjustments to how to sponge tendencies are
  added.

208
- Update RRTMGP to v0.19.2, which changes cloud optics slightly

207
- Changes in the CI for the working fluid

206
- Change default timestepper to ARS343

205
- A ClimaTimeSteppers update

204
- Change cloud ice effective radius

203
- Remove `override_precip_timescale`

202
- Slightly changed CO2 prescription

201
- Updated ClimaTimeSteppers to 0.7.39, slightly improving conservation properties

200
- Set NO2 in radiation to zero because there is potentially a bug in RRTMGP

199
- Moved CI to Julia 1.11

198
- Added terms to the implicit solver that result in changes in the
aquaplanet (ρe_tot) equil allsky monin_obukhov varying insol gravity wave (gfdl_restart) high top 1-moment

197
- Added single column hydrostatic balance reproducibility test

196
- Set bubble correction to false as default

195
- Use `vanleer_limiter` as default.

194
- Reproducibility infrastructure fixes.

193
- More reproducibility infrastructure fixes.

192
- Reproducibility infrastructure fixes.

191
- Reproducibility infrastructure debugging.

190
- Updated to new reproducibility infrastructure.

189
- We don't know why, but the regression tests are failing because there is no reference dataset.
  We are bumping the reference counter to see if this fixes the issue.

188
- Updated dependencies

187
- Change the model top for a few ci cases

186
- Topography dataset has been modified to the 60 arc-second ETOPO2022 dataset.
  This is behaviour changing for the gravity-wave (raw-topo) parameterization
  when computing `hmax` and `T tensor`.

185

184
- Changed default ozone profile.
  Jobs that failed:
  - Diagnostic EDMFX aquaplanet with TKE

183
- Change model top and vertical resolution

182
- We don't know why, but sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric
  seemed to not produce the same result between the merge-queue branch and the main branch,
  but now seems to be reproducing the same results.

181
- Ensure limit turbulent entrainment is positive

180
- Update Thermodynamics, which changes the definition of internal energy
  and enthalpy

179:
- Update Pi entr groups/ add parameter vectors

178:
- Added aerosol to one of the reproducibility tests

177:
- change numerics of non-orographic gravity waves

176:
- Switch to hyperbolic tangent stretching

175:
- Updated dependencies, DSS refactor may have different order of operations.

174:
- Reprodicibility test failed in main, try bumping the reference counter again

173:
- Refactor gravity wave parameterization

172:
- Updated dependencies

171:
- Changed start date

170:
- Moved precomputed quantities update of SSPKnoth from beginning of each timestep to end of previous timestep

169:
- Moved from Dierckx to Interpolations

168: Updated RRTMGP to v0.15.0
- Updated RRTMGP artifact
- Split solver into longwave and shortwave solvers

167:
- Removed the filter for radiative fluxes when zenith angle is close to 90 degrees

166:
- Move to SSPKnoth

165:
- Removed reference state for the dycore

164:
- Changed approximation for erf in calculation of some boundary conditions for EDMF.

163:
- Fixed bug introduced in 162

162:
- Changed the order of operations in surface conditions calculation.

161:
- Change domain top to 55 km in simulations with high top

160:
- Introduces initial conditions for the baroclinic-wave
  test case in a deep-atmosphere configuration. Modifies
  existing config to use `deep_atmosphere` mode.

159:
- Changed the boundary condition of edmf updraft properties
  to be dependent on the surface area

158:
 - Switched back the precipitation threshold definition in the
   0-moment scheme to specific humidity

157:
 - For the grid mean precipitation tendency in the 0-moment scheme:
    - added limiting by q_tot/dt
    - switched the precipitation threshold definition
      from specific humidity based to supersaturation based

156:
 - Changed start date (changes insolation)
=#
