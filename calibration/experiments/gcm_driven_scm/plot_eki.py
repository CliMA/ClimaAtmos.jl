import xarray as xr
import matplotlib.pyplot as plt
import os
import h5py


eki_path = "./output/gcm_driven_scm/iteration_001/eki_file.jld2"
file = h5py.File(eki_path, "r")

