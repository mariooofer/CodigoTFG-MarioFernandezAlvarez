"""
CÁLCULO DE AMI ENTRE FRAMES PARA UNA SIMULACIÓN DADA Y TODOS LOS LAGS POSIBLES
 (ELIMINACIÓN DE LOS ÚLTIMOS 15 LAGS POR POCOS DATOS)

INSTRUCCIONES DE USO (SOLO PARA EL MODO 0, NÚCLEO RÍGIDO):

    python calcular_ami.py <directorio_de_simulacion>
"""

from sys import argv
from pathlib import Path
import glob
import re

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import filters, morphology
from sklearn.metrics import adjusted_mutual_info_score


from funciones import (read_frame_bin, construir_heatmap, construir_lista,
                       get_labels, filtro_sin_ruido, filtromedian,N_GRID_DEFAULT,MCS,MS,METHOD,PERCENTIL_ALTO,
                       leer_limitesxy, leer_limitesz,leer_percentil_bajo,aplicar_histeresis)


# ═══════════════════════════════════════════════════════════════
#  CONTROL DE ENTRADA
# ══════════════════════════════════════════

if len(argv) == 2:
    simdir = Path(argv[1])


else:
    print("Uso: python calcular_ami.py <directorio_de_simulacion> ")

    exit()

if not simdir.exists():
    print(f"Error: El directorio {simdir} no existe.")
    exit()

N_PARTICULAS = 50000 








#  CONTROL DE ENTRADA

filepath = sorted(glob.glob(str(simdir / "trajectory-*-*.trr")))


if not filepath:
    # Puede que se pase directamente una carpeta de simulación
    filepath = [str(simdir)]

resultados = []   # (sim, percentil, mcs, ms, RI, ARI)
   


#LECTURA LÍMITES Y PERCENTIL BAJO

limitesxy = leer_limitesxy(simdir)
limitesz  = leer_limitesz(simdir)
xmax_g, xmin_g, ymax_g, ymin_g = np.max(limitesxy, axis=0)[[0, 1, 2, 3]]
limzminimo, limzmaximo = limitesz
x_bins_g = np.linspace(xmin_g, xmax_g, N_GRID_DEFAULT + 1)
y_bins_g = np.linspace(ymin_g, ymax_g, N_GRID_DEFAULT + 1)
percentil_bajo = leer_percentil_bajo(simdir)


#CLUSTERING DE TODOS LOS FRAMES

labels_por_frame = []
for p in filepath:
    print(f"Procesando frame: {Path(p).name}")
    positions = read_frame_bin(p, N_PARTICULAS)["positions"]

    heatmap = construir_heatmap(positions, x_bins_g, y_bins_g, limzminimo, limzmaximo)



    suave_median = filtromedian(heatmap)
    suave_mg     = gaussian_filter(suave_median.astype(float), sigma=0.5)
    suave        = aplicar_histeresis(suave_mg, percentil_bajo, PERCENTIL_ALTO)
    labels       = get_labels(construir_lista(suave), MCS, MS, METHOD)
    labels_por_frame.append(labels)

#AMI EN FUNCIÓN DEL LAG


n_fr = len(labels_por_frame)
lag_max = n_fr - 1 - 15          # no calcular los últimos 15 lags
if lag_max < 1:
    lag_max = n_fr - 1

lags, ami_lag, ami_lag_std = [], [], []
for lag in range(1, lag_max + 1):
    print("lag actual:", lag)
    ami_l = []
    for t in range(n_fr - lag):
        a, b = filtro_sin_ruido(labels_por_frame[t], labels_por_frame[t + lag])
        if len(a) > 0:
            ami_l.append(adjusted_mutual_info_score(a, b))
    if ami_l:
        lags.append(lag)
        ami_lag.append(np.nanmean(ami_l))
        ami_lag_std.append(np.nanstd(ami_l))

resultados.append((simdir.name, percentil_bajo, MCS, MS,
                       np.array(lags), np.array(ami_lag), np.array(ami_lag_std)))

print(f"{simdir.name}: percentil={percentil_bajo}  mcs={MCS} ms={MS}  "
          f"lags=1..{lag_max}")


#  GUARDAR Y GRAFICAR

if resultados:
    # Conversión del lag a tiempo real según el modo
    DT = 0.0094           # s/frame
    etiqueta_lag = "Lag (s)"
    unidad = "s"

    out_txt = simdir / "ami_vs_lag.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for sim, pb, mcs, ms, lags, ami, ami_std in resultados:
            f.write(f"# {sim}  percentil={pb}  mcs={mcs}  ms={ms}\n")
            f.write(f"{'lag':<8}{'tiempo(' + unidad + ')':<14}{'AMI':<12}{'AMI_std':<12}\n")
            for L, m, s in zip(lags, ami, ami_std):
                f.write(f"{L:<8}{L * DT:<14.4f}{m:<12.4f}{s:<12.4f}\n")
            f.write("\n")
    print(f"\nTabla guardada en: {out_txt}")

    # Gráfica AMI vs lag (una curva con barras de error por simulación)
    fig, ax = plt.subplots(figsize=(11, 5))
    for sim, pb, mcs, ms, lags, ami, ami_std in resultados:
        ax.errorbar(lags * DT, ami, yerr=ami_std,
                    marker="o", markersize=3, linewidth=1.2, capsize=2,
                    alpha=0.85, label=sim)
    ax.set_xlabel(etiqueta_lag, fontsize=20)
    ax.set_ylabel("AMI medio", fontsize=20)

    if len(resultados) > 1:
        ax.legend(fontsize=12, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    out_fig = simdir / "ami_vs_lag.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {out_fig}")
else:
    print("No se procesó ninguna simulación.")