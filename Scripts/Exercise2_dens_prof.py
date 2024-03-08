import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

import os

plt.style.use(["seaborn-v0_8-colorblind", "D:/plstyle_dmb2.mplstyle"])


def count(coord, r, x):
    x0, y0, z0 = x[0], x[1], x[2]
    particles = coord[
        np.sqrt(
            (coord[:, 0] - x0) ** 2 + (coord[:, 1] - y0) ** 2 + (coord[:, 2] - z0) ** 2
        )
        < r
    ]
    return len(particles), particles


def center_mass(coord, n, m):
    # x0,y0,z0 = x[0],x[1],x[2]
    # particles = coord[np.sqrt((coord[:,0] -x0)**2 + (coord[:,1] -y0)**2 + (coord[:,2] -z0)**2) < r]
    # n = count(coord, r, x)[0]
    M = n * m
    return (
        np.sum(m * coord[:, 0]) / M,
        np.sum(m * coord[:, 1]) / M,
        np.sum(m * coord[:, 2]) / M,
    )


def perfil_r(n_r, r_max, coord, x0, m):
    centros = coord[:, 0] - x0[-1][0], coord[:, 1] - x0[-1][1], coord[:, 2] - x0[-1][2]
    radios = np.logspace(-0.4, r_max, n_r)
    dens = np.empty_like(np.linspace(0, 1, n_r))
    for i in range(len(radios) - 1):
        par = coord[
            (
                np.sqrt((centros[0]) ** 2 + (centros[1]) ** 2 + (centros[2]) ** 2)
                < radios[i + 1]
            )
            & (
                np.sqrt((centros[0]) ** 2 + (centros[1]) ** 2 + (centros[2]) ** 2)
                > radios[i]
            )
        ]
        dens[i] = (
            len(par)
            * m
            / ((4 * np.pi * radios[i + 1] ** 3 / 3) - (4 * np.pi * radios[i] ** 3 / 3))
        )

    return radios, dens


def cent(r, coord, m):
    x0 = [(0, 0, 0)]
    l = np.arange(r, 0, -10)
    for index, i in enumerate(l):
        n, particles = count(coord, i, x0[index])
        if n <= 200:
            break
        x0.append(center_mass(particles, n, m))

    return x0


def NFW(r, rho_0, R_s):
    return rho_0 / (((r / R_s) * (1 + r / R_s) ** 2))


# Directorio:
dic = '../output_col_3k/'

archives = np.array(os.listdir(dic))
snapshots = archives[["snapshot" in archivo for archivo in archives]]
data = h5py.File(dic + snapshots[-1])
keys = list(data.keys())
m = data['Header'].attrs['MassTable'][1]
coord = [None] * (len(keys) - 3)
for i in range(3, len(keys)):
    coord[i - 3] = data[keys[i]]["Coordinates"]
    coord[i - 3] = np.array(coord[i - 3])

data.close()
x0 = cent(150, coord[0], m)
fig = plt.figure(figsize=(6, 5))

ax = fig.add_subplot(111)
hist = ax.hist2d(
    coord[0][:, 0],
    coord[0][:, 1],
    bins=100,
    range=((-150, 150), (-150, 150)),
    cmap='terrain_r',
)
plt.colorbar(hist[3], ax=ax)

cmap = matplotlib.colormaps.get_cmap('hot')
colors = cmap(np.linspace(0, 1, len(x0)))

for i in range(len(x0)):
    ax.scatter(x0[i][0], x0[i][1], s=50, marker='x', color=colors[i])
    if i != len(x0) - 1:
        ax.arrow(
            x0[i][0],
            x0[i][1],
            x0[i + 1][0] - x0[i][0],
            x0[i + 1][1] - x0[i][1],
            head_width=0.1,
            head_length=0.1,
            fc='blue',
            ec='blue',
        )

ax.axis('equal')
ax.set_xlabel(r'X {\rm [kpc]}')
ax.set_ylabel(r'Y {\rm [kpc]}')
plt.tight_layout()
plt.show()

dic = '../output_cluster/'
archives = np.array(os.listdir(dic))
snapshots = archives[["snapshot" in archivo for archivo in archives]]
data2 = h5py.File(dic + snapshots[-1])
keys2 = list(data2.keys())
m2 = data2['Header'].attrs['MassTable'][1]

coord2 = [None] * (len(keys2) - 3)

for i in range(3, len(keys)):
    coord2[i - 3] = data2[keys2[i]]["Coordinates"]
    coord2[i - 3] = np.array(coord2[i - 3])
del data2

dic = '../output_equi_col_3k/'
archives = np.array(os.listdir(dic))
snapshots = archives[["snapshot" in archivo for archivo in archives]]
data3 = h5py.File(dic + snapshots[-1])
keys3 = list(data3.keys())
m3 = data3['Header'].attrs['MassTable'][1]

coord3 = [None] * (len(keys3) - 3)

for i in range(3, len(keys3)):
    coord3[i - 3] = data3[keys3[i]]["Coordinates"]
    coord3[i - 3] = np.array(coord3[i - 3])
del data3


x02 = cent(10000, coord2[0], m2)
x03 = cent(150, coord3[0], m3)

n_r = 20
r_max = np.log10(150)
radios, dens = perfil_r(n_r, r_max, coord[0], x0, m)
radios2, dens2 = perfil_r(n_r, 4, coord2[0], x02, m2)
radios3, dens3 = perfil_r(n_r, r_max, coord3[0], x03, m3)
dens, dens3, radios, radios3 = dens[:-1], dens3[:-1], radios[:-1], radios3[:-1]

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].plot(
    np.log10(radios), np.log10(dens * 1e10), color='k', ls='-', label='Pseudo\nUniform'
)
ax[0].plot(np.log10(radios3), np.log10(dens3 * 1e10), color='k', ls='--', label='Grid')

dic = '../output_exp_3k/'

archives = np.array(os.listdir(dic))
snapshots = archives[["snapshot" in archivo for archivo in archives]]
data = h5py.File(dic + snapshots[-1])
keys = list(data.keys())
m = data['Header'].attrs['MassTable'][1]
coord = [None] * (len(keys) - 3)
for i in range(3, len(keys)):
    coord[i - 3] = data[keys[i]]["Coordinates"]
    coord[i - 3] = np.array(coord[i - 3])

data.close()
x0 = cent(150, coord[0], m)
n_r = 20
r_max = np.log10(150)
radios, dens = perfil_r(n_r, r_max, coord[0], x0, m)
ax[0].plot(
    np.log10(radios[:-1]),
    np.log10(dens[:-1] * 1e10),
    color='b',
    ls='-',
    label='Hubble\nexpansion',
)

ax[1].plot(np.log10(radios2), np.log10(dens2 * 1e10), color='k', ls='-')

dic = '../output_equi_exp_3k/'

archives = np.array(os.listdir(dic))
snapshots = archives[["snapshot" in archivo for archivo in archives]]
data = h5py.File(dic + snapshots[-1])
keys = list(data.keys())
m = data['Header'].attrs['MassTable'][1]
coord = [None] * (len(keys) - 3)
for i in range(3, len(keys)):
    coord[i - 3] = data[keys[i]]["Coordinates"]
    coord[i - 3] = np.array(coord[i - 3])

data.close()
x0 = cent(150, coord[0], m)
n_r = 20
r_max = np.log10(150)
radios, dens = perfil_r(n_r, r_max, coord[0], x0, m)
ax[0].plot(
    np.log10(radios[:-1]),
    np.log10(dens[:-1] * 1e10),
    color='b',
    ls=':',
    label='Hubble\nexpansion\n(grid)',
)

ax[1].plot(np.log10(radios2), np.log10(dens2 * 1e10), color='k', ls='-')

mask_1 = (radios > 0.444) & (dens > 0) & (radios < 100)
param1, _ = curve_fit(NFW, radios[mask_1], dens[mask_1], p0=(100, 1), maxfev=12000)
nfw_1 = NFW(radios, param1[0], param1[1]) * 1e10

mask_2 = (radios2 > 72) & (dens2 > 0)
param2, _ = curve_fit(NFW, radios2[mask_2], dens2[mask_2], p0=(1e6, 100), maxfev=12000)
nfw_2 = NFW(radios2, param2[0], param2[1]) * 1e10

mask_3 = (radios3 > 0.444) & (dens3 > 0) & (radios3 < 100)
param3, _ = curve_fit(NFW, radios3[mask_3], dens3[mask_3], p0=(100, 1), maxfev=12000)
nfw_3 = NFW(radios3, param3[0], param3[1]) * 1e10


print(param1)
print(param2)
print(param3)
ax[0].plot(
    np.log10(radios), np.log10(nfw_1), color='r', ls='-', alpha=0.6, label='NFW fitted'
)
ax[1].plot(np.log10(radios2), np.log10(nfw_2), color='r', ls='-', alpha=0.6)
ax[0].plot(
    np.log10(radios3),
    np.log10(nfw_3),
    color='r',
    ls='--',
    alpha=0.6,
    label='NFW fitted\n(grid)',
)

ax[0].legend(fontsize=11, labelcolor='k', loc='lower left', ncols=2)
ax[0].set_ylabel(r'$\log (\rho\, {\rm [M_{\odot}/kpc^3]})$')
ax[0].set_xlabel(r'$\log (r\, {\rm [kpc]})$')
ax[1].set_xlabel(r'$\log (r\, {\rm [kpc]})$')

ax[1].set_yticklabels([])
ax[1].yaxis.tick_right()
ax[1].set_yticklabels([])

ax[0].set_ylim(2, 12)
ax[1].set_ylim(2, 12)

ax[0].set_xticklabels(ax[0].get_xticks(), fontsize=12)
ax[0].set_yticklabels(ax[0].get_yticks(), fontsize=12)

ax[1].set_xticklabels(ax[1].get_xticks(), fontsize=12)
ax[1].set_yticklabels(ax[1].get_yticks(), fontsize=12)
xlim = ax[0].get_xlim()
xlim1 = ax[1].get_xlim()


ax[0].axvspan(xlim[0], np.log10(0.444), facecolor='gray', alpha=0.3)
ax[1].axvspan(xlim1[0], np.log10(72), facecolor='gray', alpha=0.3)

ax[0].set_xlim(xlim[0], xlim[1])
ax[1].set_xlim(xlim1[0], xlim1[1])

ax[0].set_title('Collapsing Sphere')
ax[1].set_title('G2-Cluster')


plt.tight_layout()
fig.subplots_adjust(wspace=0)
plt.savefig('exercise2_5.pdf')
plt.show()
