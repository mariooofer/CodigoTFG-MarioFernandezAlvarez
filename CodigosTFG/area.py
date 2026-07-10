"""
CALCULO DE AREAS RELATIVAS DE LOS CLUSTERS EN FUNCIÓN DEL TIEMPO

Guarda un fichero areas_por_tiempov7.txt con las áreas relativas de todos los clusters
en todos los frames de todas las simulaciones, organizadas por tiempo (una línea por frame)
para poder hacer los histogramas de áreas por tiempo en verareas.py

INSTRUCCIONES DE USO:
    python areas.py <directorio_de_simulacion> <modo>

    modo: 0 = núcleo rígido (sin deformación)
          1 = núcleo deformable (con deformación)
"""

from sys import argv
from pathlib import Path
import glob

import numpy as np
from scipy.ndimage import gaussian_filter

from funciones import (read_frame_bin, construir_heatmap, filtromedian,
                       construir_lista, get_labels, aplicar_histeresis,
                       leer_limitesxy, leer_limitesz, leer_percentil_bajo,
                       N_GRID_DEFAULT, MCS, MS, METHOD, PERCENTIL_ALTO,SIGMA)



#  PARÁMETROS


N_GRID = N_GRID_DEFAULT
PERCENTIL_MODO1 = 75      # percentil fijo para el modo dinámico



#  CONTROL DE ENTRADA


if len(argv) == 3:
    simdir = Path(argv[1])
    modo   = int(argv[2])
    if modo not in (0, 1):
        print("Error: el modo debe ser 0 o 1.")
        exit()
else:
    print("Uso: python areas.py <directorio_de_simulacion> <modo>")
    print("     modo: 0 = núcleo rígido, 1 = núcleo deformable")
    exit()

if not simdir.exists():
    print(f"Error: El directorio {simdir} no existe.")
    exit()

N_PARTICULAS = 50000 if modo == 0 else 59860

filepath = sorted(glob.glob(str(simdir / "Sim*")))
if not filepath:
    print("No se encontraron simulaciones en el directorio especificado.")
    exit()



#  PREPROCESADO (mediana + gaussiano + histéresis)


def preprocesar(heatmap, percentil_bajo):
    """Mediana → gaussiano → histéresis + opening + remove_small."""
    suave_median = filtromedian(heatmap)
    suave_mg     = gaussian_filter(suave_median.astype(float), sigma=SIGMA)
    return aplicar_histeresis(suave_mg, percentil_bajo, PERCENTIL_ALTO)


#  BUCLE PRINCIPAL


areas_por_frame = []

for idx, path in enumerate(filepath):
    clusterpath = sorted(glob.glob(str(Path(path) / "trajectory-*-*.trr")))

    # Leer límites desde fichero 
    limites = leer_limitesxy(path)

    if modo == 0:
        limzmin, limzmax = leer_limitesz(path)
        xmax, xmin, ymax, ymin = limites[0]
        x_bins = np.linspace(xmin, xmax, N_GRID + 1)
        y_bins = np.linspace(ymin, ymax, N_GRID + 1)
        radio_fisico = (xmax - xmin) / 2
        area_circulo = np.pi * radio_fisico**2
        area_pixel   = ((xmax - xmin) / N_GRID) * ((ymax - ymin) / N_GRID)

    # Leer percentil óptimo 
    if modo == 0:
        percentil_bajo = leer_percentil_bajo(path)
    else:
        percentil_bajo = PERCENTIL_MODO1

    print(f"Simulación {idx}: mcs={MCS}  ms={MS}  method={METHOD}  "
          f"percentil_bajo={percentil_bajo}")

    #Clustering frame a frame y áreas relativas
    areas_simulacion = []

    for t, p in enumerate(clusterpath):
        if modo == 1:
            xmax, xmin, ymax, ymin = limites[t]
            x_bins = np.linspace(xmin, xmax, N_GRID + 1)
            y_bins = np.linspace(ymin, ymax, N_GRID + 1)
            radio_fisico = (xmax - xmin) / 2
            area_circulo = np.pi * radio_fisico**2
            area_pixel   = ((xmax - xmin) / N_GRID) * ((ymax - ymin) / N_GRID)

        positions = read_frame_bin(p, N_PARTICULAS)["positions"]
        if modo == 0:
            heatmap = construir_heatmap(positions, x_bins, y_bins,
                                        limzmin, limzmax, N_GRID)
        else:
            heatmap = construir_heatmap(positions, x_bins, y_bins,
                                        n_grid=N_GRID)

        suave  = preprocesar(heatmap, percentil_bajo)
        lista  = construir_lista(suave)
        labels = get_labels(lista, MCS, MS, METHOD, n_grid=N_GRID)

        etiquetas_reales = np.unique(labels[labels > 0])
        areas_frame = [
            (np.sum(labels == cl) * area_pixel) / area_circulo
            for cl in etiquetas_reales
        ]
        areas_simulacion.append(areas_frame)

    areas_por_frame.append(areas_simulacion)



#  REORGANIZAR POR TIEMPO


n_frames = len(areas_por_frame[0])

areas_por_tiempo = [
    [area for sim in areas_por_frame for area in sim[t]]
    for t in range(n_frames)
]



#  GUARDAR TXT


output_path = simdir / "areas_por_tiempov7.txt"
with open(output_path, "w") as f:
    for areas_frame in areas_por_tiempo:
        f.write(" ".join(f"{a:.8f}" for a in areas_frame) + "\n")

print(f"\nÁreas guardadas en: {output_path}")