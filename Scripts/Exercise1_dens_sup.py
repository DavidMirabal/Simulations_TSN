import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import ticker, cm

plt.style.use(["seaborn-v0_8-colorblind", "D:/plstyle_dmb2.mplstyle"])

# Directorio:
dic = '../Simulations_for_fun/G2-galaxy/output_0_3_300/'
# snapshot:
frame = 100

data = h5py.File(dic + f"snapshot_{frame:03d}.hdf5")
masa = data["Header"].attrs["MassTable"]
keys = list(data.keys())

# coordenadas de todos los tipos de partículos:
coord = [None] * (len(keys) - 3)
dens_sup = [None] * (len(keys) - 3)
xedges = [None] * (len(keys) - 3)
yedges = [None] * (len(keys) - 3)

for i in range(3, len(keys)):
    coord[i - 3] = data[keys[i]]["Coordinates"]
    coord[i - 3] = np.array(coord[i - 3])

    dens_sup[i - 3] = [None] * 3  # 3 proyecciones
    yedges[i - 3] = [None] * 3
    xedges[i - 3] = [None] * 3

    dens_sup[i - 3][0], xedges[i - 3][0], yedges[i - 3][0] = np.histogram2d(
        coord[i - 3][:, 0],
        coord[i - 3][:, 1],
        bins=150,
        range=((-50, 50), (-50, 50)),
    )

    dens_sup[i - 3][0] = (
        dens_sup[i - 3][0]
        * masa[i - 2]
        * 1e10
        / (
            (xedges[i - 3][0][1] - xedges[i - 3][0][0])
            * (yedges[i - 3][0][1] - yedges[i - 3][0][0])
        )
    )

    dens_sup[i - 3][1], xedges[i - 3][1], yedges[i - 3][1] = np.histogram2d(
        coord[i - 3][:, 0],
        coord[i - 3][:, 2],
        bins=150,
        range=((-50, 50), (-50, 50)),
    )

    dens_sup[i - 3][1] = (
        dens_sup[i - 3][1]
        * 1e10
        * masa[i - 2]
        / (
            (xedges[i - 3][1][1] - xedges[i - 3][1][0])
            * (yedges[i - 3][1][1] - yedges[i - 3][1][0])
        )
    )

    dens_sup[i - 3][2], xedges[i - 3][2], yedges[i - 3][2] = np.histogram2d(
        coord[i - 3][:, 1],
        coord[i - 3][:, 2],
        bins=150,
        range=((-50, 50), (-50, 50)),
    )

    dens_sup[i - 3][2] = (
        dens_sup[i - 3][2]
        * masa[i - 2]
        * 1e10
        / (
            (xedges[i - 3][2][1] - xedges[i - 3][2][0])
            * (yedges[i - 3][2][1] - yedges[i - 3][2][0])
        )
    )

fig, ax = plt.subplots(figsize=(18, 6), ncols=3)
label = [
    (r"${\rm X\,[kpc]}$", r"${\rm Y\,[kpc]}$"),
    (r"${\rm X\,[kpc]}$", r"${\rm Z\,[kpc]}$"),
    (r"${\rm Z\,[kpc]}$", r"${\rm Y\,[kpc]}$"),
]
extent = (-50, 50, -50, 50)

for i in range(3):
    im = ax[i].imshow(
        dens_sup[1][i],
        cmap="terrain_r",
        extent=extent,
        aspect="equal",
    )
    ax[i].set_xlabel(label[i][0])
    ax[i].set_ylabel(label[i][1])
cbar = fig.colorbar(
    cm.ScalarMappable(norm=Normalize(0, 6e8), cmap="terrain_r"),
    orientation="horizontal",
    ax=ax,
    aspect=150,
    location="top",
)

cbar.set_label(r"${\rm Mass\;density\,[M_{\odot}/kpc^2]}$")
cbar.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.set_position([0.2, 0.9, 0.6, 0.1])
fig.subplots_adjust(wspace=0.2, bottom=0, hspace=0)
plt.savefig('exercise1_3.pdf')
plt.show()
