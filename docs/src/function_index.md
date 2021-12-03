# API
## Domains
```@autodocs
Modules = [ClimaAtmos.Domains]
```

## Boundary Conditions
```@autodocs
Modules = [ClimaAtmos.BoundaryConditions]
```

## Models Interface
```@autodocs
Modules = [ClimaAtmos.Models]
```

## Models
```@autodocs
Modules = (m = ClimaAtmos.Models; filter(x -> x isa Module && x != m, map(name -> getproperty(m, name), names(m; all = true)))) # all submodules of ClimaAtmos.Models
```

## Callbacks
```@autodocs
Modules = [ClimaAtmos.Callbacks]
```

## Simulations
```@autodocs
Modules = [ClimaAtmos.Simulations]
```
