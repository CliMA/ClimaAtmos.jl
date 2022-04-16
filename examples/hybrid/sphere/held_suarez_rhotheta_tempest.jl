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
dt_save_to_disk = FT(0) # 0 means don't save to disk
ode_algorithm = OrdinaryDiffEq.Rosenbrock23
jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

additional_cache(Y, params, dt) = merge(
    hyperdiffusion_cache(Y; κ₄ = FT(2e17), use_tempest_mode = true),
    sponge ? rayleigh_sponge_cache(Y, dt) : NamedTuple(),
    held_suarez_cache(Y),
)
function additional_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    held_suarez_tendency!(Yₜ, Y, p, t)
end

center_initial_condition(local_geometry, params) =
    center_initial_condition(local_geometry, params, Val(:ρθ))

function postprocessing(sol, output_dir)
    @info "L₂ norm of ρθ at t = $(sol.t[1]): $(norm(sol.u[1].c.ρθ))"
    @info "L₂ norm of ρθ at t = $(sol.t[end]): $(norm(sol.u[end].c.ρθ))"

    anim = Plots.@animate for Y in sol.u
        ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
        Plots.plot(ᶜv, level = 3, clim = (-6, 6))
    end
    Plots.mp4(anim, joinpath(output_dir, "v.mp4"), fps = 5)
end
