160

# 160: 
# - Introduces initial conditions for the baroclinic-wave
#   test case in a deep-atmosphere configuration. Modifies
#   existing config to use `deep_atmosphere` mode. 
#
# 159:
# - Changed the boundary condition of edmf updraft properties
#   to be dependent on the surface area

# 158:
#  - Switched back the precipitation threshold defintion in the
#    0-moment scheme to specific humidity

# 157:
#  - For the grid mean precipitation tendency in the 0-moment scheme:
#     - added limiting by q_tot/dt
#     - switched the precipitation threshold defintion
#       from specific humidity based to supersaturation based

# 156:
#  - Changed start date (changes insolation)
