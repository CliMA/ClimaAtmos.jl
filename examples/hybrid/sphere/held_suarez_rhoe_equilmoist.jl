using ClimaCorePlots, Plots
using ClimaCore.DataLayouts

include("baroclinic_wave_utilities.jl")

const sponge = false

# Variables required for driver.jl (modify as needed)
params = BaroclinicWaveParameterSet()
horizontal_mesh = baroclinic_wave_mesh(; params, h_elem = 4)
npoly = 4
z_max = FT(30e3)
z_elem = 10
dt = FT(400)
dt_save_to_sol = FT(60 * 60 * 24)
dt_save_to_disk = FT(0) # 0 means don't save to disk
ode_algorithm = OrdinaryDiffEq.Rosenbrock23
jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

additional_cache(Y, params, dt) = merge(
    hyperdiffusion_cache(Y; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(Y, dt) : NamedTuple(),
    held_suarez_cache(Y),
    vertical_diffusion_boundary_layer_cache(Y),
    zero_moment_microphysics_cache(Y),
)
function additional_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    held_suarez_tendency!(Yₜ, Y, p, t)
    vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t)
    zero_moment_microphysics_tendency!(Yₜ, Y, p, t)
end

center_initial_condition(local_geometry, params) = center_initial_condition(
    local_geometry,
    params,
    Val(:ρe);
    moisture_mode = Val(:equil),
)

function postprocessing(sol, output_dir)
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].c.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].c.ρe))"

    anim = Plots.@animate for Y in sol.u
        ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
        Plots.plot(ᶜv, level = 3, clim = (-6, 6))
    end
    Plots.mp4(anim, joinpath(output_dir, "v.mp4"), fps = 5)

    anim = Plots.@animate for Y in sol.u
        ᶜq_tot = Y.c.ρq_tot ./ Y.c.ρ
        Plots.plot(ᶜq_tot .* FT(1e3), level = 3, clim = (0, 1))
    end
    Plots.mp4(anim, joinpath(output_dir, "q_tot.mp4"), fps = 5)
end
