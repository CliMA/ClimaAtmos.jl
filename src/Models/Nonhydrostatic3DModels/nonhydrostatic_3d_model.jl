"""
    Nonhydrostatic3DModel <: AbstractModel

A three-dimensional non-hydrostatic model, which is typically used for simulating
the Euler equations. Required fields are `domain`, `boundary_conditions`, and
`parameters`.
"""
Base.@kwdef struct Nonhydrostatic3DModel{D, B, T, M, VD, F, BC, P, FT, C} <:
                   AbstractNonhydrostatic3DModel
    domain::D
    base::B = AdvectiveForm()
    thermodynamics::T = TotalEnergy()
    moisture::M = Dry()
    vertical_diffusion::VD = NoVerticalDiffusion()
    flux_corr::F = true
    hyperdiffusivity::FT
    boundary_conditions::BC
    parameters::P
    cache::C = CacheEmpty()
    transform_wfact::Bool = false
    test_implicit_solver::Bool = false
end

function Models.components(model::Nonhydrostatic3DModel)
    (
        base = model.base,
        thermodynamics = model.thermodynamics,
        moisture = model.moisture,
        vertical_diffusion = model.vertical_diffusion,
    )
end

function Models.default_initial_conditions(
    model::Nonhydrostatic3DModel,
    space_center,
    space_face,
)
    # we need to provide default initial conditions for the model, because the ode solver
    # requires inital conditions when getting instantiated, but we also want to support the `set!` function
    # interface for initialization and re-initialization.
    local_geometry_center = Fields.local_geometry_field(space_center)
    local_geometry_face = Fields.local_geometry_field(space_face)

    # initialize everything to zeros of the correct types on the correct spaces
    FT = Spaces.undertype(space_center)
    zero_inits = map(Models.components(model)) do component
        variable_names = Models.variable_names(component)
        if !isnothing(variable_names) # e.g., a Dry() doesn't have moisture variable names
            variable_types = Models.variable_types(component, model, FT)
            variable_space_types = Models.variable_spaces(component, model)
            zero_inits =
                map(zip(variable_types, variable_space_types)) do (T, ST)
                    zero_instance = zero(T) # somehow need this, otherwise eltype inference error
                    if space_center isa ST
                        map(_ -> zero_instance, local_geometry_center)
                    elseif space_face isa ST
                        map(_ -> zero_instance, local_geometry_face)
                    else
                        error("$ST is neither a $space_center nor a $space_face.")
                    end
                end
            tmp = NamedTuple{variable_names}(zero_inits)
            Fields.FieldVector(; tmp...)
        end
    end

    # filter out the nothing subcomponents (e.g., a Dry() doesn't have moisture variables)
    zero_inits = NamedTuple(
        c => zero_inits[c]
        for c in keys(zero_inits) if !isnothing(zero_inits[c])
    )

    # ensure that the tendency function doesn't crash when evaluated on the initial conditions
    # ʕノ•ᴥ•ʔノ ︵ ┻━┻
    p_0 = FT(1.0e5)
    R_d = FT(287.0)
    κ = FT(2 / 7)
    T_tri = FT(273.16)
    grav = FT(9.80616)
    cp_d = R_d / κ
    cv_d = cp_d - R_d
    T₀ = FT(250)
    z = local_geometry_center.coordinates.z
    p = @. p_0 * exp(-grav / (R_d * T₀) * z)
    ρ = @. p / (R_d * T₀)
    zero_inits.base.ρ .= ρ
    if :ρθ in propertynames(zero_inits.thermodynamics)
        @. zero_inits.thermodynamics.ρθ = ρ * T₀ * (p_0 / p)^(R_d / cp_d)
    elseif :ρe_tot in propertynames(zero_inits.thermodynamics)
        @. zero_inits.thermodynamics.ρe_tot =
            ρ * (cv_d * (T₀ - T_tri) + grav * z)
    elseif :ρe_int in propertynames(zero_inits.thermodynamics)
        @. zero_inits.thermodynamics.ρe_int = ρ * cv_d * (T₀ - T_tri)
    end

    return Fields.FieldVector(; zero_inits...)
end

function Models.default_ode_cache(
    model::Nonhydrostatic3DModel,
    cache::CacheEmpty,
    space_center,
    space_face,
)
    return nothing
end

function Models.make_ode_function(
    model::Nonhydrostatic3DModel,
    space_center,
    space_face,
)
    FT = eltype(model.domain) # model works on different float types

    # shorthands for model components & model styles
    base_style = model.base
    thermo_style = model.thermodynamics
    moisture_style = model.moisture
    params = model.parameters
    hyperdiffusivity = model.hyperdiffusivity
    flux_correction = model.flux_corr
    vert_diffusion_style = model.vertical_diffusion

    function implicit_rhs!(dY, Y, Ya, t)
        Φ = calculate_gravitational_potential(Y, Ya, params, FT)
        p = calculate_pressure(
            Y,
            Ya,
            base_style,
            thermo_style,
            moisture_style,
            params,
            FT,
        )
        implicit_rhs_base_model!(dY, Y, Ya, t, p, Φ, base_style, params, FT)
        implicit_rhs_thermodynamics!(
            dY,
            Y,
            Ya,
            t,
            p,
            base_style,
            thermo_style,
            params,
            FT,
        )
    end

    function explicit_rhs!(dY, Y, Ya, t)
        # auxiliary calculation is done here so we don't
        # redo it all the time and can cache the values
        Φ = calculate_gravitational_potential(Y, Ya, params, FT)
        p = calculate_pressure(
            Y,
            Ya,
            base_style,
            thermo_style,
            moisture_style,
            params,
            FT,
        )

        # base model equations
        # Ex.: ∂ₜρ = ..., ∂ₜρuh = ..., etc.
        explicit_rhs_base_model!(
            dY,
            Y,
            Ya,
            t,
            p,
            Φ,
            base_style,
            params,
            hyperdiffusivity,
            flux_correction,
            FT,
        )
        # Ex.: ∂ₜρθ = ...
        explicit_rhs_thermodynamics!(
            dY,
            Y,
            Ya,
            t,
            p,
            base_style,
            thermo_style,
            params,
            hyperdiffusivity,
            flux_correction,
            FT,
        )
        # Ex.: ∂ₜρq_tot = ...
        rhs_moisture!(
            dY,
            Y,
            Ya,
            t,
            p,
            base_style,
            moisture_style,
            params,
            hyperdiffusivity,
            FT,
        )

        # vertical diffusion
        rhs_vertical_diffusion!(
            dY,
            Y,
            Ya,
            t,
            p,
            base_style,
            thermo_style,
            moisture_style,
            vert_diffusion_style,
            params,
            FT,
        )

        # rhs_tracer!
        # rhs_edmf!
    end
    
    function Wfact!(W, Y, p, dtγ, t)
        jacobian!(W, Y, p, dtγ, t, FT)

        if W.test
            (; flags, ∂dρ∂M, ∂dE∂M, ∂dM∂E, ∂dM∂ρ, ∂dM∂M) = W

            # Checking every column takes too long, so just check one.
            i, j, h = 1, 1, 1
            if :ρθ in propertynames(Y.thermodynamics)
                E_name = (:thermodynamics, :ρθ)
            elseif :ρe_tot in propertynames(Y.thermodynamics)
                E_name = (:thermodynamics, :ρe_tot)
            elseif :ρe_int in propertynames(Y.thermodynamics)
                E_name = (:thermodynamics, :ρe_int)
            end
            w_name = (:base, :w)
            args = (implicit_rhs!, Y, p, t, i, j, h)
            @assert matrix_column(∂dρ∂M, space_face, i, j, h) ==
                    exact_column_jacobian_block(args..., (:base, :ρ), w_name)
            @assert matrix_column(∂dM∂E, space_center, i, j, h) ≈
                    exact_column_jacobian_block(args..., w_name, E_name)
            @assert matrix_column(∂dM∂M, space_face, i, j, h) ≈
                    exact_column_jacobian_block(args..., w_name, w_name)
            ∂dE∂M_approx = matrix_column(∂dE∂M, space_face, i, j, h)
            ∂dE∂M_exact =
                exact_column_jacobian_block(args..., E_name, w_name)
            if flags.∂dE∂M_mode == :exact
                @assert ∂dE∂M_approx ≈ ∂dE∂M_exact
            else
                err = norm(∂dE∂M_approx .- ∂dE∂M_exact) / norm(∂dE∂M_exact)
                @assert err < 1e-6
                # Note: the highest value seen so far is ~3e-7 (only applies to ρe_tot)
            end
            ∂dM∂ρ_approx = matrix_column(∂dM∂ρ, space_center, i, j, h)
            ∂dM∂ρ_exact =
                exact_column_jacobian_block(args..., w_name, (:base, :ρ))
            if flags.∂dM∂ρ_mode == :exact
                @assert ∂dM∂ρ_approx ≈ ∂dM∂ρ_exact
            else
                err = norm(∂dM∂ρ_approx .- ∂dM∂ρ_exact) / norm(∂dM∂ρ_exact)
                @assert err < 0.03
                # Note: the highest value seen so far for ρe_tot is ~0.01, and the
                # highest value seen so far for ρθ is ~0.02
            end
        end
    end

    W = SchurComplementW(
        space_center,
        space_face,
        thermo_style,
        params,
        model.transform_wfact,
        (;
            ∂dE∂M_mode = thermo_style isa TotalEnergy ? :no_∂p∂K : :exact,
            ∂dM∂ρ_mode = :exact,
        ),
        model.test_implicit_solver,
    )
    jac_kwargs =
        model.transform_wfact ? (; jac_prototype = W, Wfact_t = Wfact!) :
        (; jac_prototype = W, Wfact = Wfact!)

    return SplitFunction(
        ODEFunction(
            implicit_rhs!;
            jac_kwargs...,
            tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= FT(0)),
        ),
        explicit_rhs!,
    )
end
