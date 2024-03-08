import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os

plt.style.use(["seaborn-v0_8-colorblind", "D:/plstyle_dmb2.mplstyle"])


def sch_fun(M, phi, M_1, alpha):
    return phi * np.exp(-M / M_1) * (M / M_1) ** (alpha)


dic = '../output_mod/'

fig, ax = plt.subplots()
c = ['red', 'gold', 'blue']
frame = [430, 500, 600]
for i, frame in enumerate(frame):
    data = h5py.File(dic + f"fof_subhalo_tab_{frame:03d}.hdf5")
    mass = np.array(data['Subhalo']['SubhaloMass'])
    time = data['Header'].attrs['Time']
    z = data['Header'].attrs['Redshift']
    data.close()

    bins = np.logspace(11, np.log10(np.max(mass * 1e10)), 15)
    hist, bins = np.histogram(mass * 1e10, bins=bins)
    dbins = np.log(bins[1:] - bins[:-1])
    hist = hist / dbins
    hist = hist / 50**3
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    print(f'Z = {round(z, 2)}')

    param, _ = curve_fit(
        sch_fun, bin_centers[1:], hist[1:], p0=(1e9, 0.5e15, 1), maxfev=120000
    )
    print(param)
    print('--------------')
    ax.plot(bin_centers[1:], hist[1:], c=c[i], label=f'$z = {round(z, 2)}$', lw=1)
    ax.plot(
        bin_centers[1:],
        sch_fun(bin_centers[1:], param[0], param[1], param[2]),
        c=c[i],
        ls='--',
        alpha=0.7,
        lw=2,
    )

ax.legend()
ax.set_ylim(1e-7, 0.0005)
ax.set_xlabel(r'$M\;{\rm [M_\odot]}$')
ax.set_ylabel(r'$dn/d\ln M\;{\rm [}h^3{\rm Mpc^{-3} \ln^{-1} (M_\odot /}h) ]$')
ax.set_xscale('log')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('exercise3_1.pdf')
plt.show()
