import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib
from scipy.optimize import curve_fit


import os

plt.style.use(["seaborn-v0_8-colorblind", "D:/plstyle_dmb2.mplstyle"])

dic = '../output_col_3k/'

data = np.loadtxt(dic + 'energy.txt')
t = data[:, 0] * 978.56  # en Myr
E_t = data[:, 1] * 1.989e46  # 3.113e40
E_p = data[:, 2] * 1.989e46
E_k = data[:, 3] * 1.989e46
E_T = E_k + E_p

fig, ax = plt.subplots(figsize=(8, 8), nrows=2, ncols=2, sharex=True, sharey=True)
ax[1, 1].tick_params(axis='both', labelsize=16)
ax[1, 0].tick_params(axis='both', labelsize=16)
ax[0, 0].tick_params(axis='both', labelsize=16)
ax[0, 1].tick_params(axis='both', labelsize=16)

ax[0, 0].text(0.02, 0.93, 'a)', transform=ax[0, 0].transAxes, weight="bold", size=20)
ax[0, 0].plot(t, E_k, c='r', ls='-', label=r'$E_{\rm kin}$')
ax[0, 0].plot(t, E_p, c='b', ls='-', label=r'$E_{\rm pot}$')
ax[0, 0].plot(t, E_T, c='k', ls='-', label=r'$E_{\rm tot}$')

ax[0, 0].axvline(960, alpha=0.6, color='k', ls='--', lw=1)
ax[0, 0].text(600, 0.85e54, r'$t_{ff}$', color='k')
ax[0, 0].axvline(3 * 960, alpha=0.6, color='k', ls='--', lw=1)
ax[0, 0].text(775 * 3, 0.85e54, r'$3t_{ff}$', color='k')


ax[0, 0].yaxis.set_minor_locator(AutoMinorLocator())
ax[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax[0, 0].set_xlim(0, 5e3)
ax[0, 0].grid()
ax[0, 0].set_ylabel(r'$E\,{\rm[J]}$', fontsize=22)

dic = '../output_exp_3k/'

data = np.loadtxt(dic + 'energy.txt')
t = data[:, 0] * 978.56  # en Myr
E_t = data[:, 1] * 1.989e46
E_p = data[:, 2] * 1.989e46
E_k = data[:, 3] * 1.989e46
E_T = E_k + E_p

ax[1, 0].text(0.02, 0.93, 'c)', transform=ax[1, 0].transAxes, weight="bold", size=20)
ax[1, 0].plot(t, E_k, c='r', ls='-', label=r'$E_{\rm kin}$')
ax[1, 0].plot(t, E_p, c='b', ls='-', label=r'$E_{\rm pot}$')
ax[1, 0].plot(t, E_T, c='k', ls='-', label=r'$E_{\rm tot}$')

ax[1, 0].legend(fontsize='medium', labelcolor='k', loc='upper right')
ax[1, 0].yaxis.set_minor_locator(AutoMinorLocator())
ax[1, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax[1, 0].set_xlim(0, 5e3)
ax[1, 0].grid()
ax[1, 0].set_xlabel(r'${\rm Time\, [Myr]}$', fontsize=22)
ax[1, 0].set_ylabel(r'$E\,{\rm[J]}$', fontsize=22)

dic = '../output_final_equi/'

data = np.loadtxt(dic + 'energy.txt')
t = data[:, 0] * 978.56  # en Myr
E_t = data[:, 1] * 1.989e46
E_p = data[:, 2] * 1.989e46
E_k = data[:, 3] * 1.989e46
E_T = E_k + E_p


ax[0, 1].text(0.02, 0.93, 'b)', transform=ax[0, 1].transAxes, weight="bold", size=20)
ax[0, 1].plot(t, E_k, c='r', ls='-', label=r'$E_{\rm kin}$')
ax[0, 1].plot(t, E_p, c='b', ls='-', label=r'$E_{\rm pot}$')
ax[0, 1].plot(t, E_T, c='k', ls='-', label=r'$E_{\rm tot}$')

ax[0, 1].axvline(960, alpha=0.6, color='k', ls='--', lw=1)
ax[0, 1].text(600, 0.85e54, r'$t_{ff}$', color='k')
ax[0, 1].axvline(3 * 960, alpha=0.6, color='k', ls='--', lw=1)
ax[0, 1].text(775 * 3, 0.85e54, r'$3t_{ff}$', color='k')

ax[0, 1].yaxis.set_minor_locator(AutoMinorLocator())
ax[0, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax[0, 1].set_xlim(0, 5e3)
ax[0, 1].grid()

dic = '../output_equi_exp_3k/'

data = np.loadtxt(dic + 'energy.txt')
t = data[:, 0] * 978.56  # en Myr
E_t = data[:, 1] * 1.989e46
E_p = data[:, 2] * 1.989e46
E_k = data[:, 3] * 1.989e46
E_T = E_k + E_p

ax[1, 1].text(0.02, 0.93, 'd)', transform=ax[1, 1].transAxes, weight="bold", size=20)
ax[1, 1].plot(t, E_k, c='r', ls='-', label=r'$E_{\rm kin}$')
ax[1, 1].plot(t, E_p, c='b', ls='-', label=r'$E_{\rm pot}$')
ax[1, 1].plot(t, E_T, c='k', ls='-', label=r'$E_{\rm tot}$')

ax[1, 1].yaxis.set_minor_locator(AutoMinorLocator())
ax[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax[1, 1].set_xlim(0, 5e3)
ax[1, 1].set_ylim(-1.4e54, 1.4e54)
ax[1, 1].grid()
ax[1, 1].set_xlabel(r'${\rm Time\, [Myr]}$', fontsize=22)
ax[1, 1].set_xticks([0, 1000, 2000, 3000, 4000])


plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('exercise2_4.pdf')
plt.show()
