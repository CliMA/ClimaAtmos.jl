# A demonstration to prove that the LogNormal transform on the combined
# Rosenblatt fluctuation exponentiates the macroscopic gradient, while the
# affine mapping (Python prototype) preserves it.

function demo()
    μ_q = 1.0e-4
    σ_q = 1.0e-6
    
    # At the top of the cell, the local mean is 6e-4.
    # The combined physical fluctuation δq from the Rosenblatt solver includes
    # this macroscopic shift.
    δq = 5.0e-4  # (i.e. 6e-4 - 1e-4)
    
    println("Cell Mean μ_q: ", μ_q)
    println("Combined Physical Fluctuation δq from Rosenblatt: ", δq)
    println("True local mean at top of cell: ", μ_q + δq)
    
    println("\n--- Python Prototype (Affine Mapping) ---")
    q_python = max(0.0, μ_q + δq)
    println("q_hat = ", q_python)
    
    println("\n--- Julia LogNormalSGS (Current Code) ---")
    z_q = δq / (sqrt(2.0) * σ_q)
    
    ratio = σ_q / μ_q
    σ_ln = sqrt(log(1.0 + ratio^2))
    μ_ln = log(μ_q) - σ_ln^2 / 2.0
    
    q_julia = exp(μ_ln + sqrt(2.0) * σ_ln * z_q)
    println("q_hat = ", q_julia)
    println("Notice how it exponentiates the 5x gradient into an exp(5) spike!")
end

demo()
