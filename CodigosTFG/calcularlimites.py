"""
CÁLCULO DE LÍMITES XY Y Z DE LA SIMULACIÓN PARA CONSTRUIR LOS MAPAS DE CALOR (HEATMAPS)

Los límites máximo y mínimo de z limitan cuanto del eje z se consideran para el heatmap,
en general el rango de z se limita a [-7, 7] tiene buena señal pero puede ser que en alguna simulación se tenga que ajustar.
Los límites xy se calculan de forma global para el modo 0 (núcleo rígido) y por frame para el modo 1 (núcleo deformable).

Tanto los límites xy como los límites z se guardan en ficheros de texto en el directorio de simulación, 
para poder usarlos a lo largo del proceso de análisis.


INSTRUCCIONES DE USO:

    python calcular_limites.py <directorio_de_simulacion> <modo>

    modo: 0 = estático (sin deformación)
          1 = dinámico (con deformación)

"""
import sys
import glob
import struct
from pathlib import Path
from funciones import read_frame_bin

import numpy as np

#PARÁMETROS

LIMZMIN = -7
LIMZMAX =  7

# Número de partículas según el modo
N_PART_ESTATICO = 50000
N_PART_DINAMICO = 59860




#  CONTROL DE ENTRADA

# MODO 0 SIN DEFORMACION 
# MODO 1 CON DEFORMACION


if len(sys.argv) != 3:
    print("Uso: python calcular_limites.py <directorio_de_simulacion> <modo>")
    print("     modo: 0 = estático, 1 = dinámico")
    sys.exit()

simdir = Path(sys.argv[1])
modo   = int(sys.argv[2])
if modo not in (0, 1):
    print("Error: el modo debe ser 0 o 1.")
    sys.exit()
if not simdir.exists():
    print(f"Error: el directorio {simdir} no existe.")
    sys.exit()

N_PARTICULAS = N_PART_ESTATICO if modo == 0 else N_PART_DINAMICO

clusterpath = sorted(glob.glob(str(simdir / "trajectory-*-*.trr")))
if not clusterpath:
    print(f"Error: no se encontraron .trr en {simdir}")
    sys.exit()

print(f"Frames encontrados: {len(clusterpath)}")
print(f"Modo: {'estático (0)' if modo == 0 else 'dinámico (1)'}")



#  CÁLCULO DE LÍMITES Y ESCRITUTRA EN FICHEROS 


def limites_frame(positions):
    """Devuelve (xmax, xmin, ymax, ymin) de las partículas con LIMZMIN<z<LIMZMAX."""
    z = positions[:, 2]
    mask = (z > LIMZMIN) & (z < LIMZMAX)
    x = positions[mask, 0]
    y = positions[mask, 1]
    return float(x.max()), float(x.min()), float(y.max()), float(y.min())


ruta_xy = simdir / "limitesxy.txt"

if modo == 0:
    # Límites GLOBALES de toda la simulación (un solo conjunto de 4)
    xmax_g = ymax_g = -np.inf
    xmin_g = ymin_g =  np.inf
    for p in clusterpath:
        positions = read_frame_bin(p, N_PARTICULAS)["positions"]
        xmax, xmin, ymax, ymin = limites_frame(positions)
        xmax_g = max(xmax_g, xmax); xmin_g = min(xmin_g, xmin)
        ymax_g = max(ymax_g, ymax); ymin_g = min(ymin_g, ymin)
    with open(ruta_xy, "w", encoding="utf-8") as f:
        f.write(f"{xmax_g} {xmin_g} {ymax_g} {ymin_g}\n")
    print(f"Límites globales: xmax={xmax_g:.4f} xmin={xmin_g:.4f} "
          f"ymax={ymax_g:.4f} ymin={ymin_g:.4f}")

else:
    with open(ruta_xy, "w", encoding="utf-8") as f:
        for i, p in enumerate(clusterpath):
            positions = read_frame_bin(p, N_PARTICULAS)["positions"]
            xmax, xmin, ymax, ymin = limites_frame(positions)
            f.write(f"{xmax} {xmin} {ymax} {ymin}\n")
    print(f"Límites por frame escritos ({len(clusterpath)} líneas).")

print(f"Fichero de límites XY guardado en: {ruta_xy}")



#  FICHERO CON LOS LÍMITES Z


ruta_z = simdir / "limitesz.txt"
with open(ruta_z, "w", encoding="utf-8") as f:
    f.write(f"{LIMZMIN} {LIMZMAX}\n")
print(f"Fichero de límites Z guardado en: {ruta_z}  (LIMZMIN={LIMZMIN}, LIMZMAX={LIMZMAX})")

print("\nCálculo de límites finalizado.")