using Test
using Dates
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: DataLayouts, Domains, Fields, Geometry, Meshes, Operators, Spaces
import ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name
import ClimaTimeSteppers as CTS
const TD = CA.TD

@noinline _alloc_vt(Y, p, Δt, cache) =
    @allocated CA.step_vertical_scalar_transport!(Y, p, Δt, cache)

@noinline _alloc_record_imp(Y, p) =
    @allocated CA.record_implicit_stage_entry_energy!(Y, p)

@noinline _alloc_record_exp_imp(Yₜ, Y, p) =
    @allocated CA.maybe_record_explicit_imp_stage_energy!(Yₜ, Y, p)

function _stretched_z_mesh(::Type{FT}, z_elem::Int, z_max::Real) where {FT}
    domain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(z_max));
        boundary_names = (:bottom, :top),
    )
    return Meshes.IntervalMesh(
        domain,
        Meshes.ExponentialStretching(FT(z_max) / FT(2));
        nelems = z_elem,
    )
end

function _host_arr(f)
    f_contig = similar(f, eltype(f))
    f_contig .= f
    return Array(parent(f_contig))
end

function _set_col_scalar!(ᶜdest, vals::AbstractVector)
    f_contig = similar(ᶜdest, eltype(ᶜdest))
    copyto!(parent(f_contig), reshape(eltype(ᶜdest).(vals), size(parent(f_contig))))
    ᶜdest .= f_contig
    return ᶜdest
end

function _set_col_w!(ᶠu₃_dest, w_vals::AbstractVector)
    FT = Spaces.undertype(axes(ᶠu₃_dest))
    ᶠw = Fields.Field(FT, axes(ᶠu₃_dest))
    copyto!(parent(ᶠw), reshape(FT.(w_vals), size(parent(ᶠw))))
    @. ᶠu₃_dest = Geometry.Covariant3Vector(Geometry.WVector(ᶠw))
    return ᶠu₃_dest
end

function _imex_alg(alg_name)
    nm = CTS.NewtonsMethod(;
        max_iters = 1,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
    )
    return CTS.IMEXAlgorithm(alg_name, nm, CTS.Unconstrained())
end

function _imex_ssp_alg(alg_name)
    nm = CTS.NewtonsMethod(;
        max_iters = 1,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
    )
    return CTS.IMEXAlgorithm(alg_name, nm, CTS.Unconstrained())
end

function _make_ssprk_integrator(
    model,
    alg_name,
    dt::FT,
    t_end::FT;
    post_step_hook! = CA.step_vertical_scalar_transport!,
) where {FT}
    alg = CTS.IMEXAlgorithm(
        alg_name,
        CTS.NewtonsMethod(;
            max_iters = 1,
            update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
        ),
    )
    Y = CA.initial_state(model)
    p = CA.build_cache(
        Y,
        model,
        model.params,
        dt,
        DateTime(2010, 1, 1),
        nothing;
        ode_algo = alg,
        update_cache_every = "stage",
        post_step_hook!,
    )
    cb = CTS.CallbackSet(CA.post_step_discrete_callback())
    int_args, int_kwargs = CA.args_integrator(
        Y,
        p,
        (zero(FT), t_end),
        alg,
        cb,
        CA.ManualSparseJacobian(),
        false,
        model.prescribed_flow,
        dt,
        "stage",
        "stage",
    )
    return CTS.init(int_args...; int_kwargs...)
end

function _column_integral(ᶜf)
    return sum(ᶜf)
end

function _with_cache_atmos(p::CA.AtmosCache, atmos)
    vals = map(fieldnames(CA.AtmosCache)) do f
        f === :atmos ? atmos : getfield(p, f)
    end
    return CA.AtmosCache(vals...)
end

# Discrete Exner-hydrostatic pressure profile where `add_implicit_u₃_tendency!`
# vanishes to machine precision at `t = 0` for any `(T, q_tot, q_liq, q_ice)`.
function _solve_discrete_exner_hydrostatic_p!(
    p_vec::Vector{FT},
    T_vec::Vector{FT},
    qt_vec::Vector{FT},
    ql_vec::Vector{FT},
    qi_vec::Vector{FT},
    Φ_vec::Vector{FT},
    thp,
    cp_d::FT,
    p_bot::FT,
) where {FT}
    nz = length(p_vec)
    p_vec[1] = p_bot
    for k in 1:(nz - 1)
        pk = p_vec[k]
        Φk = Φ_vec[k]
        Φkp = Φ_vec[k + 1]
        dΦ = Φkp - Φk
        θvk = CA.theta_v(thp, T_vec[k], pk, qt_vec[k], ql_vec[k], qi_vec[k])
        θvrk = CA.theta_vr(thp, pk)
        Φrk = CA.phi_r(thp, pk)
        Πk = TD.exner_given_pressure(thp, pk)

        R_mk = TD.gas_constant_air(thp, qt_vec[k], ql_vec[k], qi_vec[k])
        p_guess = pk * exp(-dΦ / (R_mk * T_vec[k]))
        for _ in 1:6
            f_val = begin
                θvkp = CA.theta_v(
                    thp,
                    T_vec[k + 1],
                    p_guess,
                    qt_vec[k + 1],
                    ql_vec[k + 1],
                    qi_vec[k + 1],
                )
                θvrkp = CA.theta_vr(thp, p_guess)
                Φrkp = CA.phi_r(thp, p_guess)
                Πkp = TD.exner_given_pressure(thp, p_guess)
                (dΦ - (Φrkp - Φrk)) +
                cp_d * FT(0.5) * ((θvk - θvrk) + (θvkp - θvrkp)) * (Πkp - Πk)
            end
            dp = p_guess * FT(1e-6)
            p_hi = p_guess + dp
            θvkp_h = CA.theta_v(
                thp,
                T_vec[k + 1],
                p_hi,
                qt_vec[k + 1],
                ql_vec[k + 1],
                qi_vec[k + 1],
            )
            f_hi =
                (dΦ - (CA.phi_r(thp, p_hi) - Φrk)) +
                cp_d *
                FT(0.5) *
                ((θvk - θvrk) + (θvkp_h - CA.theta_vr(thp, p_hi))) *
                (TD.exner_given_pressure(thp, p_hi) - Πk)
            df_dp = (f_hi - f_val) / dp
            p_guess -= f_val / df_dp
        end
        p_vec[k + 1] = p_guess
    end
    return p_vec
end

# Exact 1-step linear response of per-stage backward-Euler upwind applied inside
# an S-stage ARK tableau `b_imp` (with diagonal `γ = a_imp[i, i]`):
# on `ARS343` (`b_3 < 0`), a unit spike goes `-12.8%` negative for `C >= 5`.
function _per_stage_ark_upwind_spike(tab, nz::Int, spike_k::Int, C::Float64)
    a = Matrix{Float64}(tab.a_imp.coeffs)
    b = Vector{Float64}(tab.b_imp.coeffs)
    s = length(b)
    B = zeros(Float64, nz, nz)
    for k in 1:nz
        if k < nz
            B[k, k] += 1.0
        end
        if k > 1
            B[k, k - 1] -= 1.0
        end
    end
    u0 = zeros(Float64, nz)
    u0[spike_k] = 1.0
    T_stages = [zeros(Float64, nz) for _ in 1:s]
    for i in 1:s
        rhs = copy(u0)
        for j in 1:(i - 1)
            rhs .+= a[i, j] .* T_stages[j]
        end
        γ_i = a[i, i]
        U_i = iszero(γ_i) ? rhs : (I + (γ_i * C) .* B) \ rhs
        T_stages[i] .= iszero(γ_i) ? (-C .* (B * U_i)) : ((U_i .- rhs) ./ γ_i)
    end
    u_end = copy(u0)
    for i in 1:s
        u_end .+= b[i] .* T_stages[i]
    end
    return u_end
end
