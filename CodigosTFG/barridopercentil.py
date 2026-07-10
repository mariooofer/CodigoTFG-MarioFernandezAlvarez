"""

BARRIDO DEL PERCENTIL BAJO DE LA UMBRALIZACIÓN POR HISTÉRESIS

Se sacan 3 ficheros de salida con los siguiente contenidos por cada valor de percentil bajo: 

  1) el AMI medio paso a paso (entre frames consecutivos) y su desviación,
  2) el número medio de clústeres por frame y su desviación.
  3) el número medio de partículas retenidas tras el preprocesado y su desviación.

Genera 3 gráficas frente al percentil bajo:
  - AMI paso a paso (media ± std)
  - Número de clústeres (media ± std)
- Número de partículas retenidas (media ± std)

El preprocesado base (mediana + gaussiano) se calcula UNA sola vez por
frame; solo la histéresis se repite para cada percentil (más eficiente).

INSTRUCCIONES DE USO (ESTE CODIGO SOLO FUNCIONA PARA EL MODO 0, NÚCLEO RÍGIDO):
    python barrido_percentil.py <directorio_de_simulacion>


"""



from os import mkdir
from sys import argv
from pathlib import Path
import glob

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import filters, morphology
from sklearn.metrics import adjusted_mutual_info_score

from funciones import (read_frame_bin, construir_heatmap, construir_lista,
                       get_labels, filtro_sin_ruido, filtromedian, aplicar_histeresis,
                       leer_limitesxy, leer_limitesz, N_GRID_DEFAULT,MCS,MS,METHOD,PERCENTIL_ALTO,SIGMA)

plt.rcParams.update({
    "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
})


#  CONFIGURACIÓN DEL BARRIDO


PERCENTILES = list(range(65, 91))   # 65, 66, ..., 90



#  CONTROL DE ENTRADA


if len(argv) != 2:
    print("Uso: python barrido_percentil.py <directorio_de_simulacion>")
    exit()

simdir = Path(argv[1])
if not simdir.exists():
    print(f"Error: El directorio {simdir} no existe.")
    exit()

N_PARTICULAS = 50000

outdir = Path(simdir/"barrido_percentil")
outdir.mkdir(parents=True, exist_ok=True)

#  PRECÁLCULO DEL PREPROCESADO BASE (mediana + gaussiano) POR FRAME


clusterpath = sorted(glob.glob(str(simdir / "trajectory-*-*.trr")))
if not clusterpath:
    print(f"Error: no se encontraron .trr en {simdir}")
    exit()

print(f"Frames: {len(clusterpath)}")

# Límites globales
limitesxy = leer_limitesxy(simdir)
limitesz  = leer_limitesz(simdir)
xmax_g, xmin_g, ymax_g, ymin_g = np.max(limitesxy, axis=0)[[0, 1, 2, 3]]
limzminimo, limzmaximo = limitesz
x_bins_g = np.linspace(xmin_g, xmax_g, N_GRID_DEFAULT + 1)
y_bins_g = np.linspace(ymin_g, ymax_g, N_GRID_DEFAULT + 1)


print("Precomputando preprocesado base (mediana + gaussiano) por frame...")
suaves_mg = []   # un suave_mg por frame (antes de la histéresis)
for p in clusterpath:
    positions = read_frame_bin(p, N_PARTICULAS)["positions"]
    heatmap = construir_heatmap(positions, x_bins_g, y_bins_g, limzminimo, limzmaximo)
    suave_median = filtromedian(heatmap)
    suave_mg     = gaussian_filter(suave_median.astype(float), sigma=SIGMA)
    suaves_mg.append(suave_mg)

n_fr = len(suaves_mg)



#  BARRIDO DEL PERCENTIL BAJO


ami_media,  ami_std  = [], []
nc_media,   nc_std   = [], []
np_media,   np_std   = [], []   # partículas retenidas (suma del heatmap tras preprocesado)

for pb in PERCENTILES:
    # Clustering de todos los frames con este percentil
    labels_por_frame = []
    n_clusters_frame = []
    n_particulas_frame = []
    for suave_mg in suaves_mg:
        suave  = aplicar_histeresis(suave_mg, pb,PERCENTIL_ALTO)   
        labels = get_labels(construir_lista(suave), MCS, MS, METHOD)
        labels_por_frame.append(labels)
        n_clusters_frame.append(len(np.unique(labels[labels > 0])))
        n_particulas_frame.append(int(suave.sum()))   

    # AMI paso a paso (frames consecutivos)
    ami_paso = []
    for t in range(n_fr - 1):
        a, b = filtro_sin_ruido(labels_por_frame[t], labels_por_frame[t + 1])
        if len(a) > 0:
            ami_paso.append(adjusted_mutual_info_score(a, b))

    ami_media.append(np.nanmean(ami_paso) if ami_paso else np.nan)
    ami_std.append(np.nanstd(ami_paso)   if ami_paso else np.nan)
    nc_media.append(np.mean(n_clusters_frame))
    nc_std.append(np.std(n_clusters_frame))
    np_media.append(np.mean(n_particulas_frame))
    np_std.append(np.std(n_particulas_frame))

    print(f"  Percentil {pb}:  AMI={ami_media[-1]:.4f}±{ami_std[-1]:.4f}   "
          f"NC={nc_media[-1]:.2f}±{nc_std[-1]:.2f}   "
          f"Npart={np_media[-1]:.0f}±{np_std[-1]:.0f}")

PERCENTILES = np.array(PERCENTILES)
ami_media = np.array(ami_media); ami_std = np.array(ami_std)
nc_media  = np.array(nc_media);  nc_std  = np.array(nc_std)
np_media  = np.array(np_media);  np_std  = np.array(np_std)



#  GUARDAR TABLA


out_txt = outdir / "barrido_percentil.txt"
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(f"{'percentil':<12}{'AMI':<12}{'AMI_std':<12}{'NC':<12}{'NC_std':<12}"
            f"{'Npart':<12}{'Npart_std':<12}\n")
    for pb, am, ams, nc, ncs, npt, npts in zip(
            PERCENTILES, ami_media, ami_std, nc_media, nc_std, np_media, np_std):
        f.write(f"{pb:<12}{am:<12.4f}{ams:<12.4f}{nc:<12.4f}{ncs:<12.4f}"
                f"{npt:<12.1f}{npts:<12.1f}\n")
print(f"\nTabla guardada en: {out_txt}")


#  GRÁFICA 1: AMI paso a paso vs percentil


fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(PERCENTILES, ami_media, yerr=ami_std,
            color="#1f77b4", marker="o", markersize=5, linewidth=1.5,
            capsize=3, label="AMI paso a paso")
ax.set_xlabel("Percentil bajo (%)")
ax.set_ylabel("AMI medio (frames consecutivos)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
out1 = outdir / "barrido_ami.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figura AMI guardada en: {out1}")


#  GRÁFICA 2: Número de clústeres vs percentil


fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(PERCENTILES, nc_media, yerr=nc_std,
            color="#2e8b57", marker="s", markersize=5, linewidth=1.5,
            capsize=3, label="Nº de clústeres")
ax.set_xlabel("Percentil bajo (%)")
ax.set_ylabel("Número medio de clústeres")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
out2 = outdir / "barrido_nc.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figura NC guardada en: {out2}")



#  GRÁFICA 3: Partículas retenidas vs percentil


fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(PERCENTILES, np_media, yerr=np_std,
            color="#d4820a", marker="D", markersize=5, linewidth=1.5,
            capsize=3, label="Partículas retenidas")
ax.set_xlabel("Percentil bajo (%)")
ax.set_ylabel("Nº medio de partículas tras el preprocesado")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
out3 = outdir / "barrido_particulas.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figura partículas guardada en: {out3}")

print("\nBarrido finalizado.")