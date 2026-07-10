"""
CÁLCULO DE PERCENTIL ÓPTIMO Y CLUSTERING HDBSCAN


Usa un percentil de histéresis ÓPTIMO :
  · Modo 0 (rígido):     se elige con optimizar_percentiles() a partir de
                         barrido_percentil.txt (generado previamente).
  · Modo 1 (deformable): percentil fijo = 75,

Genera, por combinación de parámetros HDBSCAN:
  · Modo 0: figura NC y mapa de acumulación.
  · Modo 1: figura NC.


INSTRUCCIONES DE USO

    python suavizado_trr.py <directorio_de_simulacion> <modo>
    modo: 0 = núcleo rígido, 1 = núcleo deformable
"""

from sys import argv
from pathlib import Path
import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
from skimage import filters, morphology
from skimage.filters import median
from skimage.morphology import disk
from sklearn.metrics import adjusted_mutual_info_score

from funciones import (read_frame_bin, construir_heatmap, construir_lista,
                       get_labels, filtro_sin_ruido, ajuste_lineal,
                       filtromedian, leer_limitesxy, leer_limitesz,aplicar_histeresis,PERCENTIL_ALTO,N_GRID_DEFAULT,MCS,MS,METHOD)

# ═══════════════════════════════════════════════════════════════
#  PARÁMETROS (definidos aquí, sin config.py)
# ══════════════════════════════════════



EXCLUIR        = 0                # últimos lags excluidos del ajuste

SIGMA          = 0.5
PERCENTIL_MODO1 = 75              # percentil fijo para el modo dinámico

plt.rcParams.update({
    "axes.labelsize": 17, "axes.titlesize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
})


# ═══════════════════════════════════════════════════════════════
#  PREPROCESADO
# ═══════════════════════════════════════════════════════════════

def preprocesar_base(heatmap):
    """Mediana + gaussiano (parte del preprocesado que NO depende del percentil)."""
    suave_median = filtromedian(heatmap)
    return gaussian_filter(suave_median.astype(float), sigma=SIGMA)





def leerbarrido_percentil(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        encabezado = f.readline().strip().split()
        datos = []
        for linea in f:
            linea = linea.strip()
            if linea:
                valores = list(map(float, linea.split()))
                datos.append(dict(zip(encabezado, valores)))
    return datos


def optimizar_percentiles(datos):
    # 1. Los 10 percentiles con MAYOR AMI
    top_ami = sorted(datos, key=lambda x: x["AMI"], reverse=True)[:15]
    set_ami = {f["percentil"] for f in top_ami}
    # 2. Los 10 percentiles con MENOR NC_std
    top_nc_std = sorted(datos, key=lambda x: x["NC_std"])[:15]
    set_nc_std = {f["percentil"] for f in top_nc_std}
    # 3. Intersección
    interseccion = set_ami.intersection(set_nc_std)
    # 4. Selección
    if interseccion:
        filas = [f for f in datos if f["percentil"] in interseccion]
        ganador = max(filas, key=lambda x: x["Npart"])
    else:
        ganador = next((f for f in datos if f["percentil"] == 75.0), None)
    return ganador


# ═══════════════════════════════════════════════════════════════
#  CONTROL DE ENTRADA
# ═══════════════════════════════════════════════════════════════

if len(argv) == 3:
    simdir = Path(argv[1])
    modo   = int(argv[2])
    if modo not in (0, 1):
        print("Error: el modo debe ser 0 o 1.")
        exit()
else:
    print("Uso: python suavizado_trr.py <directorio_de_simulacion> <modo>")
    print("     modo: 0 = núcleo rígido, 1 = núcleo deformable")
    exit()

if not simdir.exists():
    print(f"Error: El directorio {simdir} no existe.")
    exit()

N_PARTICULAS = 50000 if modo == 0 else 59860

clusterfilepath = sorted(glob.glob(str(simdir / "trajectory-*-*.trr")))
if not clusterfilepath:
    print("No se encontraron archivos .trr en el directorio especificado.")
    exit()

print(f"Modo: {'núcleo rígido (0)' if modo == 0 else 'núcleo deformable (1)'}")



#  LÍMITES Y BINS


limitesxy = leer_limitesxy(simdir)
limzmin, limzmax = leer_limitesz(simdir)

if modo == 0:
    xmax_g, xmin_g, ymax_g, ymin_g = np.max(limitesxy, axis=0)[[0, 1, 2, 3]]
    x_bins_global = np.linspace(xmin_g, xmax_g, N_GRID_DEFAULT + 1)
    y_bins_global = np.linspace(ymin_g, ymax_g, N_GRID_DEFAULT + 1)
    print(f"  Límites X: [{xmin_g:.4f}, {xmax_g:.4f}]")
    print(f"  Límites Y: [{ymin_g:.4f}, {ymax_g:.4f}]")
    print(f"  Grid: {N_GRID_DEFAULT} x {N_GRID_DEFAULT} celdas")
else:
    print(f"Modo deformable: límites por frame ({len(limitesxy)} frames).")



#  PERCENTIL ÓPTIMO


if modo == 0:
    ruta_barrido = simdir / "barrido_percentil" / "barrido_percentil.txt"
    if not ruta_barrido.exists():
        ruta_barrido = simdir / "barrido_percentil.txt"
    resultados_barrido = leerbarrido_percentil(ruta_barrido)
    ganador = optimizar_percentiles(resultados_barrido)
    percentil_optimo = int(ganador["percentil"])
else:
    percentil_optimo = PERCENTIL_MODO1

print(f"\n>>> Percentil óptimo utilizado {percentil_optimo}\n")

outtxt = simdir/"percentil_optimo.txt"

with open(outtxt, "w", encoding="utf-8") as f:
    f.write(f"Percentil óptimo {percentil_optimo}\n")


#  PRECARGA Y PREPROCESADO DE FRAMES


print(f"Cargando y preprocesando {len(clusterfilepath)} frames...")
datos_pre = []

for idx, path in enumerate(clusterfilepath):
    positions = read_frame_bin(path, N_PARTICULAS)["positions"]

    if modo == 0:
        x_bins = x_bins_global
        y_bins = y_bins_global
    else:
        xmax, xmin, ymax, ymin = limitesxy[idx]
        x_bins = np.linspace(xmin, xmax, N_GRID_DEFAULT + 1)
        y_bins = np.linspace(ymin, ymax, N_GRID_DEFAULT + 1)

    heatmap = construir_heatmap(positions, x_bins, y_bins, limzmin, limzmax, N_GRID_DEFAULT)

    if modo == 1:
        # Excluir las celdas que valen 0 al inicio del preprocesado:
        # se conservan solo las regiones no nulas del heatmap.
        heatmap = np.where(heatmap > 0, heatmap, 0)

    suave_mg = preprocesar_base(heatmap)
    datos_pre.append((heatmap, suave_mg))

n_frames = len(datos_pre)


#  CLUSTERING CON EL PERCENTIL ÓPTIMO


# Parámetros HDBSCANs 


TAG    = f"mcs{MCS}_ms{MS}_{METHOD}"


def calcular_labels(percentil_bajo):
    """Devuelve la lista de matrices de labels (una por frame)."""
    labels_por_frame = []
    for t in range(n_frames):
        suave  = aplicar_histeresis(datos_pre[t][1], percentil_bajo, PERCENTIL_ALTO)
        labels = get_labels(construir_lista(suave), MCS, MS, METHOD, n_grid=N_GRID_DEFAULT)
        labels_por_frame.append(labels)
    return labels_por_frame

print("Calculando clustering de todos los frames...")
labels_por_frame = calcular_labels(percentil_optimo)


#  RESUMEN Y FIGURAS


outdir = simdir / "figuras_estabilidadv5"
outdir.mkdir(parents=True, exist_ok=True)
print(f"\nGuardando figuras en: {outdir}")

resumen_path = simdir / "resumen_estabilidadv5.txt"
with open(resumen_path, "w", encoding="utf-8") as f:
    f.write(f"Percentil bajo utilizado: {percentil_optimo}\n")
print(f"Resumen guardado en: {resumen_path}")

#Nº de clústeres por frame
nc     = [len(np.unique(lf[lf > 0])) for lf in labels_por_frame]
nc_arr = np.array(nc, dtype=float)
nc_media = np.nanmean(nc_arr)
nc_std   = np.nanstd(nc_arr)
print(f"  {TAG}  NC media={nc_media:.2f}  std={nc_std:.2f}")

# Figura Nº clústeres 
if modo == 0:
    DT = 0.0094           # s/frame
    etiqueta_tiempo = "Tiempo (s)"
else:
    DT = 8.25 / 60.0      # min/frame (8,25 s → minutos)
    etiqueta_tiempo = "Tiempo (min)"

tiempos = np.arange(n_frames) * DT

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tiempos, nc_arr,
        color="#2e8b57", marker="^", linestyle="-", linewidth=1.2, markersize=4,
        label=f"Nº clústeres — media: {nc_media:.2f}  std: {nc_std:.2f}")

if modo == 1:
    ax.axvline(15 * DT, color="#d62728", linestyle="--", linewidth=1.5,
               label="Inicio deformación")
    ax.axvline(273 * DT, color="#1f77b4", linestyle="--", linewidth=1.5,
               label="Final deformación")

ax.set_xlabel(etiqueta_tiempo)
ax.set_ylabel("Número de clústeres")
ax.legend(loc="upper right", framealpha=0.9, fontsize=13)
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
fig.savefig(outdir / f"NC_{TAG}.png", dpi=150, bbox_inches="tight")
plt.close(fig)



#  MAPA DE ACUMULACIÓN DE CLÚSTERES


print("\nGenerando mapa de acumulación de clústeres...")

if modo == 0:
    frecuenciacluster = np.zeros((N_GRID_DEFAULT, N_GRID_DEFAULT), dtype=int)
    for t in range(n_frames):
        cluster = labels_por_frame[t].copy()
        cluster[cluster != 0] = 1          # 1 si pertenece a algún clúster
        frecuenciacluster += cluster

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    c = ax.imshow(frecuenciacluster, cmap="viridis", origin="lower",
                  vmin=0, vmax=np.max(frecuenciacluster))
    ax.set_aspect("equal")
    plt.colorbar(c, ax=ax)
    circulo = Circle((N_GRID_DEFAULT // 2, N_GRID_DEFAULT // 2), radius=N_GRID_DEFAULT // 2 - 1,
                     edgecolor="white", facecolor="none", linestyle="--", linewidth=1.5)
    ax.add_patch(circulo)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    fig.savefig(outdir / "frecuenciacluster.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapa de acumulación guardado en: {outdir / 'frecuenciacluster.png'}")
