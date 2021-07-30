# Moist Compressible Euler with Total Energy
The Julia struct 
```julia
ThreeDimensionalMoistCompressibleEulerWithTotalEnergy()
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
        \partial_t (\rho e) + \nabla \cdot ( \vec{u} [\rho e + p])  &= 0 
    \label{eq:energy}
    \\
    \partial_t (\rho q) + \nabla \cdot ( \vec{u}  \rho q )  &= 0 
    \label{eq:moisture}
    \end{align}
``` 

where the variables are
1. $$\rho $$  :  density
1. $$\rho \vec{u} $$ : momentum
1. $$\rho e $$ : total energy
1. $$\rho q $$ : total moisture
1. $$p $$  : pressure

Pressure is given by the ideal gas law
```math
    \begin{align}
    p = R_d T
    \label{eq:ideal_gas_moist_total_energy}
    \end{align}
```
and temperature is diagnosed from the prognostic variables via
```math
    \begin{align}
    T  = T_0 +  \frac{\rho e -  \frac{1}{2} \rho \vec{u} \cdot \vec{u} - \rho \phi - \rho q e_0  }{cv_m \rho  }
    \label{eq:temperature_to_energy_moist}
    \end{align}
```
Here $T_0$ is a reference temperature.

## Additional Sources

The momentum equations may be augmented with source terms such as a coriolis force as well as gravity term

```math
    \begin{align}
    \partial_t \rho + \nabla \cdot (\rho \vec{u})  &= 0 
    \\
    \partial_t (\rho \vec{u}) + \nabla \cdot (\rho \vec{u} \otimes \vec{u} + p I)  &= - \vec{\Omega} \times \rho u + \rho \nabla \phi
    \\
        \partial_t (\rho e) + \nabla \cdot ( \vec{u} [\rho e + p])  &= 0 
    \\ 
    \partial_t (\rho q) + \nabla \cdot ( \vec{u}  \rho q )  &= 0 
    \end{align}
```