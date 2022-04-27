using PrettyTables

include("../radiation_utilities.jl")

const 𝔼_name = :ρe

struct EarthParameterSet <: AbstractEarthParameterSet end

Δx = FT(1) # Note: This value shouldn't matter, since we only have 1 column.

params = EarthParameterSet()
horizontal_mesh =
    periodic_rectangle_mesh(; x_max = Δx, y_max = Δx, x_elem = 1, y_elem = 1)
quad = Spaces.Quadratures.GL{1}()
z_max = FT(100e3)
z_elem = 100
t_end = FT(60 * 60 * 24 * 365.25 * 5)
dt = FT(60 * 60 * 3)
dt_save_to_sol = 100 * dt
ode_algorithm = OrdinaryDiffEq.Rosenbrock23
jacobian_flags = (;
    ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = 𝔼_name == :ρe ? :no_∂ᶜp∂ᶜK : :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact,
)

additional_cache(Y, params, dt) = rrtmgp_model_cache(Y, params)
additional_tendency!(Yₜ, Y, p, t) = rrtmgp_model_tendency!(Yₜ, Y, p, t)
additional_callbacks = (PeriodicCallback(
    rrtmgp_model_callback!,
    dt; # this will usually be bigger than dt, but for this example it can be dt
    initial_affect = true, # run callback at t = 0
    save_positions = (false, false), # do not save Y before and after callback
),)

function center_initial_condition(local_geometry, params)
    R_d = FT(Planet.R_d(params))
    MSLP = FT(Planet.MSLP(params))
    grav = FT(Planet.grav(params))

    T₀ = FT(300)

    z = local_geometry.coordinates.z
    p = MSLP * exp(-z * grav / (R_d * T₀))
    ρ = p / (R_d * T₀)
    ts = TD.PhaseDry_ρp(params, ρ, p)

    if 𝔼_name == :ρθ
        𝔼_kwarg = (; ρθ = ρ * TD.liquid_ice_pottemp(params, ts))
    elseif 𝔼_name == :ρe
        𝔼_kwarg = (; ρe = ρ * (TD.internal_energy(params, ts) + grav * z))
    elseif 𝔼_name == :ρe_int
        𝔼_kwarg = (; ρe_int = ρ * TD.internal_energy(params, ts))
    end
    return (; ρ, 𝔼_kwarg..., uₕ = Geometry.Covariant12Vector(FT(0), FT(0)))
end
face_initial_condition(local_geometry, params) =
    (; w = Geometry.Covariant3Vector(FT(0)))

function custom_postprocessing(sol, output_dir)
    get_var(i, var) = Fields.single_field(sol.u[i], var)
    n = length(sol.u)
    #! format: off
    get_row(var) = [
        "Y.$(join(var, '.'))";;
        "$(norm(get_var(1, var), 2)) → $(norm(get_var(n, var), 2))";;
        "$(mean(get_var(1, var))) → $(mean(get_var(n, var)))";;
        "$(maximum(abs, get_var(1, var))) → $(maximum(abs, get_var(n, var)))";;
        "$(minimum(abs, get_var(1, var))) → $(minimum(abs, get_var(n, var)))";;
    ]
    #! format: on
    pretty_table(
        vcat(map(get_row, Fields.property_chains(sol.u[1]))...);
        title = "Change in Y from t = $(sol.t[1]) to t = $(sol.t[n]):",
        header = ["var", "‖var‖₂", "mean(var)", "max(∣var∣)", "min(∣var∣)"],
        alignment = :c,
    )

    anim = @animate for Y in sol.u
        if :ρθ in propertynames(Y.c)
            ᶜts = @. thermo_state_ρθ(Y.c.ρθ, Y.c, params)
        elseif :ρe in propertynames(Y.c)
            grav = FT(Planet.grav(params))
            ᶜK = @. norm_sqr(C123(Y.c.uₕ) + C123(ᶜinterp(Y.f.w))) / 2
            ᶜΦ = grav .* Fields.coordinate_field(Y.c).z
            ᶜts = @. thermo_state_ρe(Y.c.ρe, Y.c, ᶜK, ᶜΦ, params)
        elseif :ρe_int in propertynames(Y.c)
            ᶜts = @. thermo_state_ρe_int(Y.c.ρe_int, Y.c, params)
        end
        plot(
            vec(TD.air_temperature.(params, ᶜts)),
            vec(Fields.coordinate_field(Y.c).z ./ 1000);
            xlabel = "T [K]",
            ylabel = "z [km]",
            xlims = (100, 300),
            legend = false,
        )
    end
    Plots.mp4(anim, joinpath(output_dir, "T.mp4"), fps = 10)
end
