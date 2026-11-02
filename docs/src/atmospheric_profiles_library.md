# Atmospheric Profiles Library

`ClimaAtmos.AtmosphericProfilesLibrary` holds the published reference profiles that
the single-column [setups](setups.md) build on.

```@docs
ClimaAtmos.AtmosphericProfilesLibrary
```

Each section below plots every profile of one case, evaluated in `Float64`, with
heights in km and times in hours. Profiles that also take the Exner function, such
as `Bomex_dTdt`, are not plotted.

```@example apl
include("atmospheric_profiles_library_plots.jl") # hide
nothing # hide
```

## ARM SGP (`ARM_SGP_*`)

Profiles from the case definition [Brown2002](@cite).

```@example apl
plot_case("ARM_SGP") # hide
```

## BOMEX (`Bomex_*`)

Profiles from the case definition [Siebesma2003](@cite).

```@example apl
plot_case("Bomex") # hide
```

## Dry bubble (`DryBubble_*`)

```@example apl
plot_case("DryBubble") # hide
```

## DYCOMS-II RF01 (`Dycoms_RF01_*`)

Profiles from the case definition [Stevens2005](@cite).

```@example apl
plot_case("Dycoms_RF01") # hide
```

## DYCOMS-II RF02 (`Dycoms_RF02_*`)

Profiles from the case definition [Ackerman2009](@cite).

```@example apl
plot_case("Dycoms_RF02") # hide
```

## GABLS (`GABLS_*`)

```@example apl
plot_case("GABLS"; z_max = 400) # hide
```

## GATE III (`GATE_III_*`)

Profiles from the case definition [Khairoutdinov2009](@cite).

```@example apl
plot_case("GATE_III"; z_max = 18000) # hide
```

## ISDAC (`ISDAC_*`)

Profiles from the case definition [ovchinnikov_intercomparison_2014](@cite).

```@example apl
plot_case("ISDAC") # hide
```

## Larcform 1 (`Larcform1_*`)

Profiles from the case definition [Pithan2016](@cite).

```@example apl
plot_case("Larcform1"; z_max = 25000) # hide
```

## Life cycle (`LifeCycleTan2018_*`)

Profiles from the case definition [Tan2018](@cite).

```@example apl
plot_case("LifeCycleTan2018") # hide
```

## Nieuwstadt (`Nieuwstadt_*`)

Profiles from the case definition [Nieuwstadt1993](@cite).

```@example apl
plot_case("Nieuwstadt") # hide
```

## RICO (`Rico_*`)

Profiles from the case definition [vanZanten2011](@cite).

```@example apl
plot_case("Rico") # hide
```

## SP (`SP_*`)

```@example apl
plot_case("SP") # hide
```

## Soares (`Soares_*`)

Profiles from the case definition [Soares2004](@cite).

```@example apl
plot_case("Soares") # hide
```

## TRMM-LBA (`TRMM_LBA_*`)

Profiles from the case definition [Grabowski2006](@cite).

```@example apl
plot_case("TRMM_LBA"; z_max = 25000, t_max = 6 * 3600) # hide
```
