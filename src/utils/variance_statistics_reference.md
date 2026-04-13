
Gemini gave me these functions.
We need to clean um up, get em into proper Distribution.jl ondition, figure out how to get quadrature weights for em, etc.



# Modeling Bounded Linear Gradients: From Singular Gaussians to Bivariate Uniform-Normal Convolutions

## 1. Introduction and Motivation

In many physical and statistical modeling frameworks, continuous spatial or temporal domains are discretized into finite intervals (bins, grid cells, or time steps). A common mathematical challenge arises when evaluating non-linear functions over these discrete intervals. 

If a 2D property (e.g., a coordinate in state space) exhibits a continuous, constant gradient across a discrete interval, representing the entire interval with a single mean value artificially erases sub-interval variance. This leads to persistent underestimation of threshold-dependent processes. 

To recover this lost variance, one must construct a Probability Density Function (PDF) that accurately represents the distribution of states within that interval. This document derives the formulation for representing a 2D linear gradient subject to local variance, contrasting the naive Bivariate Gaussian approach with the mathematically robust Bivariate Uniform-Normal Convolution.

---

## 2. The Singular (Degenerate) Gaussian on a Line

Consider a distribution of points perfectly aligned along an infinite 1D line embedded in a 2D space. Let the line be defined by a mean vector $\boldsymbol{\mu}$ and a direction vector $\mathbf{v}$. 

The generative model for a point $\mathbf{X}$ on this line is:
$$\mathbf{X} = \boldsymbol{\mu} + Z\mathbf{v}$$
where $Z \sim \mathcal{N}(0, 1)$ is a standard normal random variable.

Because $\mathbf{X}$ is an affine transformation of a Gaussian variable, it is strictly a Bivariate Gaussian. Its moments are:
* **Mean:** $\mathbb{E}[\mathbf{X}] = \boldsymbol{\mu}$
* **Covariance:** $\Sigma = \text{Var}(Z\mathbf{v}) = \mathbf{v}\mathbf{v}^T$

The covariance matrix $\Sigma$ is mathematically singular; its determinant is exactly zero ($|\Sigma| = 0$). Consequently, the standard Bivariate Gaussian PDF formula cannot be evaluated because the inverse $\Sigma^{-1}$ does not exist. The distribution has entirely collapsed into a 1D subspace.

---

## 3. The Bounded Interval and the Naive Gaussian

In practical applications, gradients do not stretch to infinity; they are bounded within a discrete cell. Let the state vector at the start of the interval be $\mathbf{a}$ and the state vector at the end of the interval be $\mathbf{b}$. Let the difference vector be $\mathbf{d} = \mathbf{b} - \mathbf{a}$.

Furthermore, assume there is local, correlated Gaussian turbulence at every point along the segment, defined by a fully populated $2 \times 2$ covariance matrix $\Sigma_{local}$.

### The Naive Gaussian Assumption
A common, computationally inexpensive approach is to compute the total mean and total covariance of the interval and simply fit a standard Bivariate Gaussian to those moments. 

To match the variance of a uniform gradient spanning $\mathbf{d}$, one can construct a generative Gaussian model using a weight $W \sim \mathcal{N}\left(0, \frac{1}{12}\right)$, matching the variance of a Standard Uniform distribution:
$$\mathbf{X}_{naive} = \frac{\mathbf{a} + \mathbf{b}}{2} + W\mathbf{d} + \mathbf{Z}_{local}$$
where $\mathbf{Z}_{local} \sim \mathcal{N}(\mathbf{0}, \Sigma_{local})$.

### Shortcomings of the Naive Model
While $\mathbf{X}_{naive}$ preserves the correct first and second moments (mean and covariance) of the bounded interval, its higher-order moments and spatial structure are fundamentally flawed:

1.  **Central Clustering:** A Gaussian inherently clusters probability mass around its mean (the midpoint of the interval). A true constant gradient has equal probability mass everywhere along the segment.
2.  **Infinite Tails:** The Gaussian tails extend infinitely past the physical boundaries $\mathbf{a}$ and $\mathbf{b}$. This assigns non-zero probability to unphysical states that lie entirely outside the grid cell.
3.  **Failure under Non-Linearity:** When integrated against a convex non-linear function (e.g., a Heaviside step function or an exponential trigger), the naive model systematically distorts the expected value. It underestimates the frequency of boundary extremes while overestimating the frequency of intermediate states.

---

## 4. The Corrected Distribution: Bivariate Uniform-Normal Convolution

To accurately model the physics of a constant gradient subjected to local Gaussian noise, the discrete interval must be modeled as a continuous mixture. The structural gradient is represented by a Uniform distribution, and the local variance is represented by a Normal distribution. 

### Generative Formulation
Let $U \sim \text{Uniform}(0, 1)$ represent the normalized coordinate along the vector $\mathbf{d}$. 
Let $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \Sigma_{local})$ represent the local turbulent variance, where $\Sigma_{local}$ may contain non-zero off-diagonal covariance terms.

The true state $\mathbf{X}$ is the sum of these independent random variables:
$$\mathbf{X} = \mathbf{a} + U(\mathbf{b} - \mathbf{a}) + \mathbf{Z}$$

This produces a distribution that maintains a flat, uniform plateau of probability mass smoothly stretching between $\mathbf{a}$ and $\mathbf{b}$, while the local covariance $\Sigma_{local}$ dictates the cross-sectional shape of the "pill".

---

## 5. Derivation of Statistical Moments

Because $\mathbf{X}$ is a linear combination of independent random variables ($U$ is strictly independent of the instantaneous eddy $\mathbf{Z}$), its moments can be derived exactly without complex integration.

### The Mean Vector
Using the linearity of expectation and the fact that $\mathbb{E}[U] = 0.5$ and $\mathbb{E}[\mathbf{Z}] = \mathbf{0}$:
$$\mathbb{E}[\mathbf{X}] = \mathbf{a} + \mathbb{E}[U](\mathbf{b} - \mathbf{a}) + \mathbb{E}[\mathbf{Z}]$$
$$\mathbb{E}[\mathbf{X}] = \mathbf{a} + 0.5(\mathbf{b} - \mathbf{a}) = \frac{\mathbf{a} + \mathbf{b}}{2}$$

### The Covariance Matrix
Since $U$ and $\mathbf{Z}$ are independent of each other, the variance of their sum is the sum of their variances. The variance of a Standard Uniform distribution is $\text{Var}(U) = \frac{1}{12}$.
$$\text{Var}(\mathbf{X}) = \text{Var}(U(\mathbf{b} - \mathbf{a})) + \text{Var}(\mathbf{Z})$$
$$\Sigma_{total} = \frac{1}{12}(\mathbf{b} - \mathbf{a})(\mathbf{b} - \mathbf{a})^T + \Sigma_{local}$$

This yields a profound geometric result: the total sub-grid covariance is the exact linear superposition of the local turbulent covariance and the structural covariance mathematically forced by the background gradient.

---

## 6. Derivation of the Exact Probability Density Function (PDF)

To evaluate the PDF $f_{\mathbf{X}}(\mathbf{x})$ at an arbitrary point $\mathbf{x} \in \mathbb{R}^2$, we must account for the fact that $\Sigma_{local}$ is generally correlated. We cannot simply use orthogonal dot-products in the physical space, because the local variance forms an angled ellipse rather than a perfect circle. 

To solve this, we apply a **Whitening (Mahalanobis) Transformation** to temporarily warp the coordinate space, rendering the local variance perfectly isotropic, evaluating the orthogonal components, and then mapping the density back to physical space.

### Step 1: The Whitening Transformation
Let $\mathbf{L}$ be the lower triangular matrix resulting from the Cholesky decomposition of the local covariance matrix: $\Sigma_{local} = \mathbf{L}\mathbf{L}^T$.

We transform all spatial vectors by multiplying them by the inverse matrix $\mathbf{L}^{-1}$. Let the "tilde" notation represent coordinates in this newly warped space:
$$\tilde{\mathbf{x}} = \mathbf{L}^{-1}\mathbf{x}$$
$$\tilde{\mathbf{a}} = \mathbf{L}^{-1}\mathbf{a}$$
$$\tilde{\mathbf{b}} = \mathbf{L}^{-1}\mathbf{b}$$

In this transformed space, the local Gaussian variance is mathematically forced into the Identity matrix ($\tilde{\Sigma}_{local} = \mathbf{I}$). Therefore, the local noise is now perfectly isotropic with standard deviation $\sigma = 1$.

### Step 2: Orthogonal Decomposition in Warped Space
Because the local variance is now isotropic, we can safely decouple the warped space orthogonally. Let $\tilde{\mathbf{d}} = \tilde{\mathbf{b}} - \tilde{\mathbf{a}}$ be the gradient vector, and $\tilde{L} = \|\tilde{\mathbf{d}}\|$ be its length. Let $\tilde{\mathbf{v}} = \tilde{\mathbf{x}} - \tilde{\mathbf{a}}$.

We calculate the parallel projection scalar $p$ and the squared perpendicular distance $d_{\perp}^2$:
$$p = \frac{\tilde{\mathbf{v}} \cdot \tilde{\mathbf{d}}}{\tilde{L}}$$
$$d_{\perp}^2 = \max(0, \|\tilde{\mathbf{v}}\|^2 - p^2)$$

### Step 3: Density in the Warped Space
The probability density in the warped space, $\tilde{f}(\tilde{\mathbf{x}})$, is the product of a standard 1D Normal distribution (perpendicular) and a 1D Uniform-Normal convolution (parallel), evaluated with $\sigma = 1$:

$$f_{\perp}(d_{\perp}) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{d_{\perp}^2}{2} \right)$$

$$f_{\parallel}(p) = \frac{1}{2\tilde{L}} \left[ \text{erf}\left(\frac{p}{\sqrt{2}}\right) - \text{erf}\left(\frac{p - \tilde{L}}{\sqrt{2}}\right) \right]$$

$$\tilde{f}(\tilde{\mathbf{x}}) = f_{\perp}(d_{\perp}) \cdot f_{\parallel}(p)$$

### Step 4: Mapping Back to Physical Space
To find the true probability density $f_{\mathbf{X}}(\mathbf{x})$ in the original physical space, we apply the change of variables formula. We multiply the warped density by the absolute value of the determinant of the transformation matrix, $|\det(\mathbf{L}^{-1})|$, to account for the spatial stretching.

Since $|\det(\mathbf{L}^{-1})| = \frac{1}{\sqrt{\det(\Sigma_{local})}}$, the exact continuous PDF is:

$$f_{\mathbf{X}}(\mathbf{x}) = \frac{\tilde{f}(\tilde{\mathbf{x}})}{\sqrt{\det(\Sigma_{local})}}$$

### Edge Case: Zero Gradient ($\tilde{L} = 0$)
If the gradient across the discrete interval is zero ($\mathbf{a} = \mathbf{b}$), then $\tilde{L} = 0$. By taking the limit of $f_{\parallel}(p)$ as $\tilde{L} \to 0$, the parallel component collapses to $\frac{1}{\sqrt{2\pi}} \exp(-p^2/2)$. The joint density $\tilde{f}(\tilde{\mathbf{x}})$ becomes a standard isotropic Gaussian in the warped space. Scaled by the determinant, the final formula smoothly and perfectly degenerates into the standard equation for a Bivariate Gaussian centered at $\mathbf{a}$ with covariance $\Sigma_{local}$.

#================================================ #

using SpecialFunctions
using LinearAlgebra





function uniform_normal_pdf(x::Float64, a::Float64, b::Float64, sigma::Float64)
    if a == b
        return (1.0 / (sigma * sqrt(2 * π))) * exp(-0.5 * ((x - a) / sigma)^2)
    end
    
    term1 = erf((x - a) / (sigma * sqrt(2.0)))
    term2 = erf((x - b) / (sigma * sqrt(2.0)))
    
    return (1.0 / (2.0 * (b - a))) * (term1 - term2)
end

# Example usage over a range (broadcasting the function)
x_vals = collect(-30.0:0.1:30.0)
pdf_vals = uniform_normal_pdf.(x_vals, -5.0, 5.0, 1.0)

function bivariate_uniform_normal_pdf(x::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}, sigma::Float64)
    d = b - a
    L = norm(d) # Length of the line segment
    
    # Edge case: If bounds are equal, it collapses to a standard 2D Gaussian
    if L == 0.0
        return (1.0 / (2.0 * π * sigma^2)) * exp(-norm(x - a)^2 / (2.0 * sigma^2))
    end
    
    # Vector from the start point 'a' to our evaluation point 'x'
    v = x - a
    
    # 1. Decompose the distance into parallel and perpendicular components
    # 'p' is the projection of 'x' along the line segment
    p = dot(v, d) / L 
    
    # 'd_perp_sq' is the squared perpendicular distance from 'x' to the line
    # (max is used to prevent tiny negative values caused by floating-point rounding)
    d_perp_sq = max(0.0, dot(v, v) - p^2)
    
    # 2. The Perpendicular Component (Standard 1D Gaussian)
    perp_comp = (1.0 / (sqrt(2.0 * π) * sigma)) * exp(-d_perp_sq / (2.0 * sigma^2))
    
    # 3. The Parallel Component (Your 1D Uniform-Normal Convolution)
    term1 = erf(p / (sigma * sqrt(2.0)))
    term2 = erf((p - L) / (sigma * sqrt(2.0)))
    parallel_comp = (1.0 / (2.0 * L)) * (term1 - term2)
    
    # The final 2D density is the product of the two orthogonal components
    return perp_comp * parallel_comp
end

# --- Example Usage ---
a_state = [-5.0, -5.0]
b_state = [5.0, 5.0]
sigma_local = 1.0

# Evaluate at a single point
test_point = [0.0, 1.0]
pdf_val = bivariate_uniform_normal_pdf(test_point, a_state, b_state, sigma_local)
println("PDF at $test_point: $pdf_val")

