204

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
