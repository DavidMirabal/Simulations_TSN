import h5py
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import os
import sys
import time
import sys
from tqdm import tqdm
import colorsys
import traceback
import dmb_figures as dmb
import psutil

if len(sys.argv) != 5:
    print(
        "Uso: python generator.py directorio_output/ papi limites_snapshots1 lmites_snapshots2"
    )
    sys.exit(1)

dic = str(sys.argv[1])
papi = int(sys.argv[2])  # 0 normalmente
lim_snap1 = int(sys.argv[3])
lim_snap2 = int(sys.argv[4])

archives = np.array(os.listdir(dic))
snapshots = archives[["snapshot" in archivo for archivo in archives]]

snapshots = np.array(snapshots)[lim_snap1:lim_snap2]


def figs(frame, x, y, z, papi=papi):
    plt.cla()
    data = h5py.File(dic + f"snapshot_{frame:03d}.hdf5")
    keys = list(data.keys())
    redshift = data['Header'].attrs['Redshift']
    # print('Hola')
    # print(f'snapshot_{frame:03d}.hdf5')
    coord = [None] * (len(keys) - 3)
    for j in range(3, len(keys)):
        coord[j - 3] = data[keys[j]]["Coordinates"]
        coord[j - 3] = np.array(coord[j - 3])
    data.close()
    azim = [0, 0, 0]
    elev = [90, 45, 0]

    FIGURA = dmb.Figura(
        ancho=16, ticks=('yes', 'yes'), lw_spine=1, ratio=16 / 9, fontsize=22, s_text=12
    )
    fig, ax = FIGURA.axs(2, 1, ('3d', '2d'))
    ax[1].set_ylim(1, 1e4)
    ax[1].set_xlim(1e11, 0.5e15)
    funcion_halo = False
    try:
        data_fof = h5py.File(dic + f"fof_subhalo_tab_{frame:03d}.hdf5")
        fof_len = np.array(data_fof['Subhalo']['SubhaloLen'])
        fof_mass = np.array(data_fof['Subhalo']['SubhaloMass'])
        fof_n = np.array(np.array(data_fof['Subhalo']['SubhaloGroupNr']))
        data_fof.close()

        colors1 = dmb.generar_colores(len(fof_len))
        bins = np.logspace(11, np.log10(0.5e15), 20)
        hist, bins = np.histogram(fof_mass * 1e10, bins=bins)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax[1].plot(bin_centers[1:], hist[1:], color='w')
        funcion_halo = True

    except:
        pass

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    FIGURA.ylabel(1, r'N_{\rm halos}')
    FIGURA.xlabel(1, r'{\rm Mass\, [M_\odot]}')
    j = 0
    for k in range(papi, len(keys) - 3):
        # print(np.max(a[k]-1))

        ax[j].scatter(
            coord[k][:, 0],
            coord[k][:, 1],
            coord[k][:, 2],
            marker="o",
            s=0.001,
            c='w',
            alpha=0.1,
        )
        # long_pre = 0
        # if funcion_halo:
        #     for index, i in enumerate(fof_len):
        #         ax[j].scatter(
        #             coord[k][int(long_pre) : int(long_pre + i), 0],
        #             coord[k][int(long_pre) : int(long_pre + i), 1],
        #             coord[k][int(long_pre) : int(long_pre + i), 2],
        #             marker="o",
        #             s=0.05,
        #             c=colors1[index],
        #             alpha=(-0.1 + 0.001) / (len(fof_len)) * index + 0.1,
        #         )
        #         long_pre = long_pre + i
        if funcion_halo:
            long_pre = 0
            for i in np.arange(fof_n[-1] + 1):
                mask = np.where(fof_n == i)[0]
                for l in mask:
                    ax[j].scatter(
                        coord[k][long_pre : long_pre + fof_len[l], 0],
                        coord[k][long_pre : long_pre + fof_len[l], 1],
                        coord[k][long_pre : long_pre + fof_len[l], 2],
                        marker='o',
                        s=0.01,
                        c=colors1[i],
                        alpha=(-0.1 + 0.001) / (fof_n[-1] + 1) * i + 0.1,
                    )
                    long_pre = long_pre + fof_len[l]

    # ax[j].view_init(elev=elev[j], azim=azim[j])

    ax[j].set_axis_off()
    ax[j].set_xlim(x[0], x[1])
    ax[j].set_ylim(y[0], y[1])
    ax[j].set_zlim(z[0], z[1])

    # ax[j].text(
    #     0.5,
    #     0.5,
    #     -0.1,
    #     r'$z=$' + str(round(redshift, 1)),
    #     horizontalalignment='center',
    #     verticalalignment='center',
    #     transform=plt.gca().transAxes,
    #     color='w',
    # )

    ax[j].view_init(elev=30, azim=1 * frame * (3 / (2 * np.pi)))
    ax[j].set_aspect('equal', adjustable='box')
    ax[j].set_title(
        f'$z={round(redshift, 2)}$',
        color='w',
    )
    #
    fig.subplots_adjust(left=0.1, right=0.98, top=1, bottom=0.05, wspace=0)
    fig.tight_layout()
    fig.savefig(dic + "images3/im_" + str(frame) + ".png", dpi=500)

    del coord

    plt.close()
    del fig


def process_frame(frame):
    try:
        figs(
            frame,
            (0, 52),
            (0, 52),
            (0, 52),
        )
    except Exception as e:
        traceback.print_exc()  # Imprime la traza de la excepción
        print(f"{e}")


def main():
    # desired_cores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # current_process = psutil.Process(os.getpid())

    # current_process.cpu_affinity(desired_cores)

    with concurrent.futures.ProcessPoolExecutor(max_workers=18) as executor:
        # Ejecutar las llamadas a process_frame en paralelo
        # executor.map(process_frame, range(len(snapshots)))
        results = list(
            tqdm(
                executor.map(process_frame, range(lim_snap1, lim_snap2 + 1)),
                total=len(snapshots),
            )
        )


if __name__ == "__main__":
    main()
