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


if len(sys.argv) != 3:
    print("Uso: python generator.py directorio_output/ papi")
    sys.exit(1)

dic = str(sys.argv[1])
papi = int(sys.argv[2])  # 0 normalmente

data = h5py.File(dic + "snapshot_000.hdf5")
keys = list(data.keys())
coord = [None] * (len(keys) - 3)
izq = [None] * (len(keys) - 3)
der = [None] * (len(keys) - 3)

for i in range(3, len(keys)):
    coord[i - 3] = data[keys[i]]["Coordinates"]
    coord[i - 3] = np.array(coord[i - 3])

    izq[i - 3] = np.array(data[keys[i]]["ParticleIDs"])[coord[i - 3][:, 0] < 0]
    der[i - 3] = np.array(data[keys[i]]["ParticleIDs"])[coord[i - 3][:, 0] > 0]

archives = np.array(os.listdir(dic))
snapshots = archives[["snapshot" in archivo for archivo in archives]]


def generate_tinted_colors(base_color, n):
    """
    Generate n tinted colors with decreasing saturation based on a given base color.

    Parameters:
    - base_color (tuple): RGB values of the base color (e.g., (R, G, B)).
    - n (int): Number of tinted colors to generate.

    Returns:
    - List of hexadecimal color strings compatible with Matplotlib.
    """
    # Convert the base color to HSV
    hsv_base_color = colorsys.rgb_to_hsv(*[x / 255.0 for x in base_color])

    # Generate n tinted colors with decreasing saturation
    tinted_colors = [
        colorsys.hsv_to_rgb(hsv_base_color[0], i / n, hsv_base_color[2])
        for i in range(1, n + 1)
    ]

    # Convert the colors back to RGB tuples and then to hexadecimal strings
    hex_tinted_colors = [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in tinted_colors
    ]

    return hex_tinted_colors


# Ejemplo de uso
base_color1 = (255, 0, 0)
base_color2 = (0, 0, 255)
num_colors = len(keys) - 3

colors1 = generate_tinted_colors(base_color1, num_colors)
colors2 = generate_tinted_colors(base_color2, num_colors)


def figs(frame, fig, ax, x, y, z, papi=papi):
    # frame = frame+1
    plt.cla()
    data = h5py.File(dic + f"snapshot_{frame:03d}.hdf5")
    keys = list(data.keys())
    # print('Hola')
    # print(f'snapshot_{frame:03d}.hdf5')
    coord = [None] * (len(keys) - 3)
    for j in range(3, len(keys)):
        coord[j - 3] = data[keys[j]]["Coordinates"]
        coord[j - 3] = np.array(coord[j - 3])

    a = [None] * (len(keys) - 3)
    b = [None] * (len(keys) - 3)

    for j in range(3, len(keys)):
        """
        # Para izq[j-3]
        izq_indices = np.searchsorted(data[keys[j]]['ParticleIDs'], izq[j-3])
        a[j-3] = izq_indices

        # Para der[j-3]
        der_indices = np.searchsorted(data[keys[j]]['ParticleIDs'], der[j-3])
        b[j-3] = der_indices
        """
        mask = np.isin(data[keys[j]]["ParticleIDs"], izq[j - 3])
        a[j - 3] = np.where(mask)[0].astype(int)

        mask = np.isin(data[keys[j]]["ParticleIDs"], der[j - 3])
        b[j - 3] = np.where(mask)[0].astype(int)

    sc1 = [[None] * (len(keys) - 3)] * 3
    sc2 = [[None] * (len(keys) - 3)] * 3

    azim = [0, 0, 0]
    elev = [90, 45, 0]

    for j in range(3):
        for k in range(papi, len(keys) - 3):
            # print(np.max(a[k]-1))

            sc1[j][k] = ax[j].scatter(
                coord[k][:, 0][a[k]],
                coord[k][:, 1][a[k]],
                coord[k][:, 2][a[k]],
                marker="o",
                s=1,
                c=colors1[k],
                alpha=1,
            )
            sc2[j][k] = ax[j].scatter(
                coord[k][:, 0][b[k]],
                coord[k][:, 1][b[k]],
                coord[k][:, 2][b[k]],
                marker="o",
                s=1,
                c=colors2[k],
                alpha=1,
            )

        ax[j].view_init(elev=elev[j], azim=azim[j])
        ax[j].axis("equal")
        ax[j].set_axis_off()
        ax[j].set_xlim(x[0], x[1])
        ax[j].set_ylim(y[0], y[1])
        ax[j].set_zlim(z[0], z[1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    if os.path.exists(dic + "images/"):
        fig.savefig(dic + "images/im_" + str(frame) + ".png", dpi=500)
    else:
        os.makedirs(dic + "images/")
        fig.savefig(dic + "images/im_" + str(frame) + ".png", dpi=500)
    # print(len(sc1))
    # print(sc1[0][0])
    # for j in range(3):
    #     for k in range(papi, len(keys) - 3):
    #         sc1[j][k].cla()
    #         sc2[j][k].cla()

    for j in range(3):
        ax[j].cla()

    plt.close()


color = "black"
fig = plt.figure(figsize=(9, 16))
fig.set_facecolor(color)
ax1 = fig.add_subplot(311, projection="3d")
ax2 = fig.add_subplot(312, projection="3d")
ax3 = fig.add_subplot(313, projection="3d")
ax1.set_facecolor(color)
ax2.set_facecolor(color)
ax3.set_facecolor(color)


def process_frame(frame):
    try:
        figs(
            frame,
            fig,
            [ax1, ax2, ax3],
            (-200, 200),
            (-200, 200),
            (-200, 200),
        )
    except Exception as e:
        traceback.print_exc()  # Imprime la traza de la excepción
        print(f"{e}")


def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Ejecutar las llamadas a process_frame en paralelo
        # executor.map(process_frame, range(len(snapshots)))
        results = list(
            tqdm(
                executor.map(process_frame, range(len(snapshots))), total=len(snapshots)
            )
        )


if __name__ == "__main__":
    main()
