import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

types = 2

vel_exp = True
H_0 = 70 / 1e3  # en km/s/Kpc
dens_m = 0.3
dens_l = 0.7
z = 4

r = 150
vmax = 0
n = 4100

n_type1 = int(6 * n / np.pi)  # nº particulas segun su tipo

n_type2 = 3000
n_types = [0, n_type1]

mass_type1 = 1e2 / n  # en 10^10 M_sun
mass_type2 = 0.5
mass_types = [0, mass_type1]

f = h5py.File("initial_conditions_equi_exp_3k.hdf5", "w")

for i in range(types):
    PartType = f.create_group(f"PartType{i:01d}")
    dimension = int(n_types[i] ** (1 / 3))
    X, Y, Z = np.meshgrid(
        np.linspace(-r, r, dimension),
        np.linspace(-r, r, dimension),
        np.linspace(-r, r, dimension),
    )
    cua_part = np.vstack(
        [
            X.flatten()[: n_types[i]],
            Y.flatten()[: n_types[i]],
            Z.flatten()[: n_types[i]],
        ]
    ).T
    pos = np.array(
        cua_part[
            np.sqrt(cua_part[:, 0] ** 2 + cua_part[:, 1] ** 2 + cua_part[:, 2] ** 2) < r
        ],
        dtype=np.float32,
    )

    PartType.create_dataset("Coordinates", data=pos)
    n_types[i] = len(pos)
    print(n_types[i])
    if vel_exp:
        v = H_0 * np.sqrt(dens_m * (1 + z) ** 3 + dens_l) * pos
    else:
        v = np.array(np.random.uniform(-vmax, vmax, (n_types[i], 3)), dtype=np.int32)

    PartType.create_dataset(
        "Velocities",
        data=v,
    )
    PartType.create_dataset("ParticleIDs", data=np.arange(n_types[i], dtype=np.int32))

print(n_types)
Header = f.create_group("Header")
Header.attrs["NumPart_ThisFile"] = np.array(n_types, dtype=np.int32)
Header.attrs["NumPart_Total"] = np.array(n_types, dtype=np.int64)
Header.attrs["MassTable"] = np.array(mass_types, dtype=np.float32)
Header.attrs["NumFilesPerSnapshot"] = np.array([1], dtype=np.int32)
Header.attrs["Time"] = np.array([0], dtype=np.int32)
Header.attrs["Redshift"] = np.array([20], dtype=np.int32)
Header.attrs["BoxSize"] = np.array([1], dtype=np.int32)
Header.attrs["Omega0"] = 1.0

f.close()
