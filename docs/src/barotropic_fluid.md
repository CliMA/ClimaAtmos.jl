# Barotropic Fluid
The Julia struct 
```julia
ThreeDimensionalCompressibleEulerWithBarotropicFluid()
```

is a shorthand for the following base set of equations

```math
    \begin{align}
    \partial_t \rho + \nabla \cdot (\rho \vec{u})  &= 0 
    \label{eq:continuity}
    \\
    \partial_t (\rho \vec{u}) + \nabla \cdot (\rho \vec{u} \otimes \vec{u} + p I)  &= 0 
    \label{eq:momentum}
    \\
    \partial_t (\rho \theta) + \nabla \cdot ( \vec{u}  \rho \theta )  &= 0 
    \label{eq:temperature}
    \end{align}
``` 

where the variables are
1. $\rho$  :  density
1. $\rho u $ : momentum
1. $\rho \theta $ : potential temperature
1. $p$  : pressure

Pressure is given by
```math
    \begin{align}
    p =  \frac{c_s^2}{\rho_0}\rho^2
    \label{eq:barotropic_pressure}
    \end{align}
```
where $c_s$ is a constant (related to the soundspeed) and $\rho_0$ is a reference density
