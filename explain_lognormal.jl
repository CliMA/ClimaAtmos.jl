using RootSolvers

# This script demonstrates exactly what happens with the LogNormal transform
# when z_q (χ1) is computed from the Rosenblatt δq.

function demonstrate()
    μ_q = 1.0e-4
    σ_q = 1.0e-6
    δq = 3.8e-4  # A typical large fluctuation from Brent (dominated by L = 1e-3)
    
    # How the code computes the Gaussian deviate z_q (χ1):
    z_q = δq / (sqrt(2.0) * σ_q)
    
    println("μ_q = ", μ_q)
    println("σ_q (SGS turbulence) = ", σ_q)
    println("δq (Physical fluctuation from Rosenblatt) = ", δq)
    println("Computed Gaussian deviate z_q = ", z_q)
    
    # Gaussian transform:
    q_gaussian = μ_q + sqrt(2.0) * σ_q * z_q
    println("\nGaussian output: ", q_gaussian)
    
    # LogNormal transform (from get_physical_point):
    ratio = σ_q / μ_q
    σ_ln = sqrt(log(1.0 + ratio^2))
    μ_ln = log(μ_q) - σ_ln^2 / 2
    
    q_lognormal = exp(μ_ln + sqrt(2.0) * σ_ln * z_q)
    println("\nLogNormal output: ", q_lognormal)
    println("Ratio of LogNormal output to mean: ", q_lognormal / μ_q)
end

demonstrate()
