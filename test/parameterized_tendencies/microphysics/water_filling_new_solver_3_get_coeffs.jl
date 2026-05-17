"""Fixed path — always the same env on this machine (avoids hash(~/repo) ≠ hash(/net/.../repo))."""

"""`Pkg.activate` the plot env; first time only, `develop` ClimaAtmos + `add` CairoMakie."""
function wf_quad_activate_env!()
    env = joinpath(tempdir(), "climaatmos_wf_new_solver_3")
    if !ispath(joinpath(env, "Project.toml"))
        mkpath(env)
        Pkg.activate(env)
        Pkg.add(["LsqFit", "SpecialFunctions"])
        println("WF new solver 3: initialized ", env)
    else
        Pkg.activate(env)
    end
    return env
end

wf_quad_activate_env!()
using Pkg; Pkg
Pkg.instantiate()

using LsqFit
import SpecialFunctions: erf, erfc

 using LsqFit
using LinearAlgebra
using SpecialFunctions

# Set arbitrary precision to 256-bit for the deep dry tail matrix inversions
setprecision(BigFloat, 256)

# ==========================================
# 1. THE EXACT MATH 
# ==========================================
# Float64 versions for the Moist Regime
exact_phi_f64(x::Float64) = exp(-0.5 * x^2) / sqrt(2 * π)
exact_Phi_f64(x::Float64) = 0.5 * (1.0 + erf(x / sqrt(2.0)))
Q_from_x_f64(x::Float64) = x * exact_Phi_f64(x) + exact_phi_f64(x)

# BigFloat versions for the Dry Tail
function exact_phi_bf(x::BigFloat)
    return exp(-BigFloat("0.5") * x^2) / sqrt(BigFloat("2.0") * BigFloat(π))
end

function exact_Phi_bf(x::BigFloat)
    # High-precision trapezoidal integration for BigFloat CDF
    if x < -8; return BigFloat("0.0"); end
    steps = 10000
    dx = x / steps
    integral = BigFloat("0.0")
    for i in 1:steps
        t1 = (i - 1) * dx
        t2 = i * dx
        integral += (exact_phi_bf(t1) + exact_phi_bf(t2)) * dx / 2
    end
    return BigFloat("0.5") + integral
end

function Q_from_x_bf(x::BigFloat)
    return x * exact_Phi_bf(x) + exact_phi_bf(x)
end

# ==========================================
# 2. LOEB'S MINIMAX ALGORITHM (For Dry Tail)
# ==========================================
function minimax_rational_fit(X::Vector{BigFloat}, Y::Vector{BigFloat}, deg_num::Int, deg_den::Int; max_iter=50)
    N = length(X)
    cols = deg_num + 1 + deg_den
    W = ones(BigFloat, N)
    local c = zeros(BigFloat, cols)
    
    for iter in 1:max_iter
        A = zeros(BigFloat, N, cols)
        B = zeros(BigFloat, N)
        for i in 1:N
            for j in 0:deg_num
                A[i, j+1] = W[i] * (X[i]^j)
            end
            for j in 1:deg_den
                A[i, deg_num + 1 + j] = -W[i] * Y[i] * (X[i]^j)
            end
            B[i] = W[i] * Y[i]
        end
        
        c_new = (A' * A) \ (A' * B)
        if norm(c_new - c) < BigFloat("1e-20")
            break
        end
        c = c_new
        
        for i in 1:N
            den_val = BigFloat("1.0")
            for j in 1:deg_den
                den_val += c[deg_num + 1 + j] * (X[i]^j)
            end
            W[i] = BigFloat("1.0") / max(abs(den_val), BigFloat("1e-30"))
        end
    end
    return c[1:deg_num+1], c[deg_num+2:end]
end

# ==========================================
# 3. GENERATE & FIT
# ==========================================
println("Generating Moist Data (Float64)...")
x_moist = collect(range(0.0, 10.0, length=50000))
Q_moist = Q_from_x_f64.(x_moist)

println("Fitting Moist Rational (Degree 3/3)...")
function rational_3_3(Q, p)
    @. (p[1] + Q * (p[2] + Q * (p[3] + Q * p[4]))) / (1.0 + Q * (p[5] + Q * (p[6] + Q * p[7])))
end
fit_moist = curve_fit(rational_3_3, Q_moist, x_moist, ones(Float64, 7); maxIter=10000)
p_m = fit_moist.param

println("Generating Dry Data (256-bit BigFloat)...")
x_dry = [BigFloat("-6.0") + i * (BigFloat("6.0")/1000) for i in 0:1000]
Q_dry = Q_from_x_bf.(x_dry)

println("Fitting Dry Minimax Rational (Degree 3/3)...")
num_coeffs, den_coeffs = minimax_rational_fit(Q_dry, x_dry, 3, 3)

# ==========================================
# 4. OUTPUT CODE
# ==========================================
println("\n\n" * "="^60)
println("SUCCESS! COPY THIS INTO YOUR FASTWATERFILLING MODULE")
println("="^60 * "\n")

println("    # 1. BRANCHLESS RATIONAL GUESSES (Pole-Free, 256-bit safe)")
println("    num_m = FT($(p_m[1])) + Q * (FT($(p_m[2])) + Q * (FT($(p_m[3])) + Q * FT($(p_m[4]))))")
println("    den_m = FT(1.0) + Q * (FT($(p_m[5])) + Q * (FT($(p_m[6])) + Q * FT($(p_m[7]))))")
println("    x_moist = num_m / den_m")
println("")
println("    num_d = FT($(Float64(num_coeffs[1]))) + Q * (FT($(Float64(num_coeffs[2]))) + Q * (FT($(Float64(num_coeffs[3]))) + Q * FT($(Float64(num_coeffs[4])))))")
println("    den_d = FT(1.0) + Q * (FT($(Float64(den_coeffs[1]))) + Q * (FT($(Float64(den_coeffs[2]))) + Q * FT($(Float64(den_coeffs[3])))))")
println("    x_dry = num_d / den_d")