# usage: julia bins_n.jl e1 e2 ... (edges in um) -> prints 0M/3M arrays for a toml override
module FS
lognormal_spectrum(r_hat, modes) = sum(((F, r, σg),) -> (F / r_hat) * exp(-log(r_hat / r)^2 / (2 * log(σg)^2)), modes)
function bin_log_simpson(f, lo, hi; N = 4096)
    x0, h = log(lo), (log(hi) - log(lo)) / N
    s = f(lo) * lo + f(hi) * hi
    for j in 1:(N - 1); x = exp(x0 + j * h); s += (isodd(j) ? 4 : 2) * f(x) * x; end
    return s * h / 3
end
end
modes = ((0.2157, 0.05545, 17.02), (60.93, 0.0914, 1.813), (5.949, 0.776, 1.759))
c3 = 4 / 3 * π * 2170.0 * (1e-6)^3        # dry mass per unit r̂³ (validated: reproduces stored arrays)
edges = parse.(Float64, ARGS)
m(k, lo, hi) = FS.bin_log_simpson(r -> FS.lognormal_spectrum(r, modes) * r^k, lo, hi)
n = length(edges) - 1
m0 = [m(0, edges[i], edges[i+1]) for i in 1:n]
m3 = [m(3, edges[i], edges[i+1]) for i in 1:n] .* c3
fmt(v) = "[" * join(string.(round.(v; sigdigits = 4)), ", ") * "]"
println("edges_m = [", join(string.(edges .* 1e-6), ", "), "]")
println("0M = ", fmt(m0)); println("3M = ", fmt(m3)); println("total 3M = ", sum(m3))
